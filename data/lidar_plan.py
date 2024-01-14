
import logging
import concurrent.futures
import multiprocessing
import pandas as pd
from functools import reduce
import operator
from tqdm import tqdm
import re
import signal

import lidar_downloader
import lidar_helper

LOG_EOF = 'EOF'
LOG_TASK_COMPLETE = 'Completed Task'
LOG_TASK_FAIL = 'Failed Task'


class QueueLogger:
    """Mock interface for logger. Puts the thing to be logged in a queue."""

    def __init__(self, queue):
        self.queue = queue
        self.prefix = ""

    def info(self, message):
        self.queue.put({'level': 'info', 'message': self.prefix + message})

    def warning(self, message):
        self.queue.put({'level': 'warning', 'message': self.prefix + message})

    def debug(self, message):
        self.queue.put({'level': 'debug', 'message': self.prefix + message})

    def error(self, message):
        self.queue.put({'level': 'error', 'message': self.prefix + message})

    def bind(self, prefix, id=0):
        self.prefix = f"[{prefix}::{id}] "


def handle_log_entry(log_entry, ignore_unprefixed=True):
    """Unpack the log entry put by the queue logger and actually log it."""
    level = log_entry['level']
    message = log_entry['message']

    if ignore_unprefixed and not message.startswith('['):
        return 0

    if level == 'info':
        logging.info(message)
    elif level == 'warning':
        logging.warning(message)
    elif level == 'debug':
        logging.debug(message)
    elif level == 'error':
        logging.error(message)
    else:
        logging.error(f"[QueueLogger] Unknown log level {level}")

    return 1 if message.endswith(LOG_EOF) else 0


class Task:
    """Base class for Pipeline task."""

    def __init__(self, **kwargs):
        self.task_name = 'BaseName'
        self.pool_type = kwargs['pool_type']
        self.max_workers = kwargs['max_workers']
        self.fan_out = kwargs['fan_out']
        self.timeout = kwargs['timeout']

    def handle_queue(self, id, queue_in, queue_out, logger):
        """
        Performs a task on items from an upstream queue, until 
        sentinel value task==None is reached. Once the task 
        is complete it is added to the downstream queue.
        """
        # TODO: make logger a member of Task and bind it in the constructor
        logger.bind(self.task_name, id)
        logger.info(f"Handling queue")

        class TaskTimeOut(Exception):
            pass

        def handle_timeout(signum, frame):
            raise TaskTimeOut("timeout")
        signal.signal(signal.SIGALRM, handle_timeout)

        while True:
            try:
                task = queue_in.get()

                logger.info(f"Got Task {task}")
                if task is not None:
                    signal.alarm(self.timeout)
                    self.perform_task(task, logger)
                    signal.alarm(0)

                    logger.info(f"{LOG_TASK_COMPLETE} {task}")

            except (TaskTimeOut, Exception) as e:
                logger.error(f"{LOG_TASK_FAIL} {task}: {e}")

            else:
                # Only send this downstream if it succeeded
                queue_out.put(task)

            finally:
                queue_in.task_done()

                if task is None:
                    # Sentinel value for actual logger
                    logger.info(f"{LOG_EOF}")
                    for _ in range(self.fan_out):
                        queue_out.put(None)
                    break

    def perform_task(self, task, logger):
        """Actual task to perform to be implemented by derived classes"""
        pass

    def create_task_queue(self, logger):
        """Inspect existing contents of relevant directories and report back a queue of tasks to be done

        Returns:
            tasks: List of tasks. Each element should be able to be passed to perform_task()
        """
        pass


class DownloadTask(Task):
    """Implements sftp download of LIDAR data."""

    def __init__(self, sftp_config, local_directory, remote_ls_file, **kwargs):
        super().__init__(**kwargs)
        self.task_name = 'Downloader'
        self.sftp_config = sftp_config
        self.local_directory = local_directory

        self.create_task_queue_fresh = False
        self.remote_ls_file = remote_ls_file

    def perform_task(self, list_of_files, logger):
        # print("Happily performing task")
        lidar_downloader.download_many(
            self.sftp_config, self.local_directory, list_of_files, logger, tqdm_disable=True)

    def create_task_queue(self, logger):
        logger.bind(self.task_name)

        if self.create_task_queue_fresh:
            lidar_downloader.list_files(self.sftp_config, self.remote_ls_file)

        done, todo = lidar_downloader.compare_remote_listing_to_already_downloaded(
            self.remote_ls_file, self.local_directory)

        logger.info(
            f"Found {len(done)} done and {len(todo)} to do. Last done: {done[-1:]}, next todo: {todo[:1]}.")
        return done, todo


class UnzipTask(Task):
    """Implements task of unzipping the lidar data"""

    def __init__(self, local_directory, **kwargs):
        super().__init__(**kwargs)
        self.local_directory = local_directory
        self.task_name = 'Unzipper'

        self.delete_zip = True

    def perform_task(self, list_of_files, logger):
        lidar_helper.unzip_files_in_directory(
            folder_path=self.local_directory, zip_files=list_of_files, logger=logger, delete_zip=self.delete_zip)

    def create_task_queue(self, logger):
        logger.bind(self.task_name)

        done, todo = lidar_helper.files_to_unzip(self.local_directory)

        logger.info(
            f"Found {len(done)} done and {len(todo)} to do. Last done: {done[-1:]}, next todo: {todo[:1]}.")
        return done, todo


class PreprocessTask(Task):
    """Implements preprocessing steps to generate training data"""

    def __init__(self, local_directory, out_directory, shapes_directory, output_image_size=256, **kwargs):
        super().__init__(**kwargs)
        self.task_name = 'Preprocessor'
        self.local_directory = local_directory
        self.out_directory = out_directory
        self.shapes_directory = shapes_directory
        self.output_image_size = output_image_size

    def perform_task(self, list_of_directories, logger):
        lidar_helper.process_dtm(list_of_directories, self.local_directory, self.out_directory,
                                 self.shapes_directory, logger, output_image_size=self.output_image_size)

    def create_task_queue(self, logger):
        logger.bind(self.task_name)

        done, todo = lidar_helper.files_to_process(
            self.local_directory, self.out_directory, self.output_image_size)

        logger.info(
            f"Found {len(done)} done and {len(todo)} to do. Last done: {done[-1:]}, next todo: {todo[:1]}.")
        return done, todo


class DataPipeline:
    """Pipeline manager.
    Responsible for:
        1) constructing task queues based on expected state of data/ 
        2) dispatching tasks to handlers concurrently
    """

    def __init__(self, data_raw_path, data_out_path, shape_path, remote_ls_file):
        sftp_config = lidar_downloader.sftp_cfg('sftpconfig.json')
        self.remote_ls_file = remote_ls_file

        self.downloader = DownloadTask(
            sftp_config,
            data_raw_path,
            remote_ls_file,
            max_workers=1,
            fan_out=1,
            timeout=60,
            pool_type='thread')

        self.unzipper = UnzipTask(
            data_raw_path,
            max_workers=1,
            fan_out=2,
            timeout=10,
            pool_type='process')

        self.preprocessor = PreprocessTask(
            data_raw_path,
            data_out_path,
            shape_path,
            output_image_size=5000,
            max_workers=2,
            fan_out=1,
            timeout=90,
            pool_type='process')

        self.tasks = [self.downloader, self.unzipper, self.preprocessor]

    def total_num_workers(self):
        return sum([task.max_workers for task in self.tasks])

    def create_queue(self, logger, report=None):
        """Creates queue for each task
        Args:
            logger: logger to report to
        Returns:
            todos_for_tasks: List of todos for each task. Will be used to create task queue
        Raises: 
            AssertionError is raised if the `todo` list of one task does not match the `done` list of a preceeding task
        """
        todos_for_tasks = []
        done_for_tasks = []
        t_done_prev = None
        for task in self.tasks:
            logging.info(f"Generating task queue for task {task.task_name}.")
            t_done, t_todo = task.create_task_queue(logger)

            if t_done_prev is not None:
                assert set(t_done_prev) == set(t_todo + t_done)

            t_done_prev = t_done
            todos_for_tasks.append(t_todo)
            done_for_tasks.append(t_done)

        if report is not None:
            logging.info(f"Generating report in ./{report}.")

            all_files = lidar_downloader.read_ls_file(self.remote_ls_file)

            # Header row of report
            # ___________________________________________________________________
            # | Filename    | Size  | Downloader    | Unzipper  | Preprocessor  |
            # | DTM-1m-...  | 45123 | 1             | 0         | 0             |

            lines = []
            df_rows = []
            # lines.append(['Filename', 'Size'] +
            #             [task.task_name for task in self.tasks].join(','))
            # lines.append(f"Filename,Size," +
            #             ','.join([task.task_name for task in self.tasks]))
            for file, size in all_files.items():
                df_rows.append(
                    [file, size] + [int(file in done) for done in done_for_tasks])
                # lines.append(
                #    f"{file},{size}," + ','.join([f"{int(file in done)}" for done in done_for_tasks]))
            df = pd.DataFrame(df_rows, columns=[
                "Filename", "Size"] + [task.task_name for task in self.tasks])

            assert report.endswith(".csv")
            df.to_csv(report, index=False)

        return todos_for_tasks, df

    def print_summary_report(self, df_before, df_after):
        df_before['Tag'] = 'Before'
        df_after['Tag'] = 'After'
        df_full = pd.concat([df_before, df_after])
        summary = df_full.groupby('Tag').agg(
            TotalSize=('Size', 'sum'),
            TotalNumber=('Size', 'count'),
            NumberDownloaded=('Downloader', 'sum'),
            NumberUnzipped=('Unzipper', 'sum'),
            NumberProcessed=('Preprocessor', 'sum'),
            SizeDownloaded=(
                'Size', lambda x: (df_full.loc[x.index, 'Downloader'] * df_full.loc[x.index, 'Size']).sum()),
            SizeProcessed=(
                'Size', lambda x: (df_full.loc[x.index, 'Preprocessor'] * df_full.loc[x.index, 'Size']).sum())
        )

        summary['TotalSize'] /= 1e6
        summary['SizeDownloaded'] /= 1e6
        summary['SizeProcessed'] /= 1e6
        summary['DownloadProgress'] = 100 * \
            summary['SizeDownloaded'] / summary['TotalSize']
        summary['ProcessedProgress'] = 100 * \
            summary['SizeProcessed'] / summary['TotalSize']
        print(summary)

    def run(self, N):
        """Run Parallel pipeline

        Args:
            N: Number of task to enque
        """
        with multiprocessing.Manager() as manager:
            q_download = manager.Queue()
            q_unzip = manager.Queue()
            q_preprocess = manager.Queue()
            q_out = manager.Queue()
            q_log = manager.Queue()
            logger = QueueLogger(q_log)

            todos_for_tasks, df_before = self.create_queue(
                logger, report='report.csv')

            for task in todos_for_tasks[0][:N]:
                q_download.put([task])
            for task in todos_for_tasks[1][:N]:
                q_unzip.put([task])
            for task in todos_for_tasks[2][:N]:
                q_preprocess.put([task])

            q_download_size = q_download.qsize()
            q_unzip_size = q_unzip.qsize()
            q_preprocess_size = q_preprocess.qsize()
            q_download.put(None)

            download_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.downloader.max_workers)
            unzip_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.unzipper.max_workers)
            preprocess_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.preprocessor.max_workers)

            for id in range(self.downloader.max_workers):
                download_pool.submit(self.downloader.handle_queue,
                                     id, q_download, q_unzip, logger)

            for id in range(self.unzipper.max_workers):
                unzip_pool.submit(self.unzipper.handle_queue,
                                  id, q_unzip, q_preprocess, logger)

            for id in range(self.preprocessor.max_workers):
                preprocess_pool.submit(self.preprocessor.handle_queue,
                                       id, q_preprocess, q_out, logger)

            logging.info(f"[DataPipeline] Processing queue")

            counter = 0
            logging.info(f"[DataPipeline] Flushing output queue")
            with tqdm(total=q_download_size) as pbar_download, \
                    tqdm(total=q_download_size+q_unzip_size) as pbar_unzip, \
                    tqdm(total=q_download_size+q_unzip_size+q_preprocess_size) as pbar_process:
                while True:

                    log_entry = q_log.get()
                    counter += handle_log_entry(log_entry)

                    pattern = r"\[(?P<task_name>[a-zA-Z]+)::(?P<id>\d+)\]\s(?P<message>.*)"
                    match = re.match(pattern, log_entry['message'])
                    if LOG_TASK_COMPLETE in log_entry['message'] and match:
                        matchd = match.groupdict()
                        if matchd['task_name'] == 'Downloader':
                            pbar_download.update(1)
                        if matchd['task_name'] == 'Unzipper':
                            pbar_unzip.update(1)
                        if matchd['task_name'] == 'Preprocessor':
                            pbar_process.update(1)

                    q_log.task_done()

                    if counter == self.total_num_workers():
                        logging.info(f"[DataPipeline] All queues empty")
                        break

            # Handle output queue
            while True:
                out = q_out.get()
                q_out.task_done()

                if out == None:
                    break

            logging.info("[DataPipeline] Shutting down")
            download_pool.shutdown(wait=True)
            unzip_pool.shutdown(wait=True)

            _, df_after = self.create_queue(logger, report='report2.csv')

            self.print_summary_report(df_before, df_after)


if __name__ == '__main__':

    logging.basicConfig(filename='lidar_plan.log',  level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%y%m%d %H:%M:%S')
    logging.info("=================Starting...=========================")

    pipeline = DataPipeline(
        data_raw_path='/mnt/d/lidarnn_raw_new',
        data_out_path='/mnt/d/lidarnn',
        remote_ls_file='ls.txt',
        shape_path='/mnt/d/lidarnn_shapes')

    pipeline.run(N=200)

    logging.info("=================Ending...===========================")

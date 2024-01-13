
import logging
import concurrent.futures
import multiprocessing
import queue
import time  # Used for simulating work

import lidar_downloader
import lidar_helper

LOG_EOF = 'EOF'


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

    def bind(self, prefix):
        self.prefix = f"[{prefix}] "


def handle_log_entry(log_entry):
    """Unpack the log entry put by the queue logger and actually log it."""
    level = log_entry['level']
    message = log_entry['message']

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

    def handle_queue(self, queue_in, queue_out, logger):
        """
        Performs a task on items from an upstream queue, until 
        sentinel value task==None is reached. Once the task 
        is complete it is added to the downstream queue.
        """
        logger.bind(
            self.task_name)  # TODO: make logger a member of Task and bind it in the constructor
        logger.info(f"Handling queue")

        while True:
            task = queue_in.get()

            logger.info(f"Got Task {task}")
            if task is not None:
                try:
                    self.perform_task(task, logger)
                except Exception as e:
                    logger.error(e)

            queue_in.task_done()
            queue_out.put(task)

            if task is None:
                # Sentinel value for actual logger
                logger.info(f"{LOG_EOF}")
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

    def __init__(self, local_directory, out_directory, output_image_size=256, **kwargs):
        super().__init__(**kwargs)
        self.task_name = 'Preprocessor'
        self.local_directory = local_directory
        self.out_directory = out_directory
        self.output_image_size = output_image_size

    def perform_task(self, list_of_directories, logger):
        logger.bind(self.task_name)
        lidar_helper.process_dtm(list_of_directories, self.local_directory,
                                 self.out_directory, logger, output_image_size=self.output_image_size)

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

    def __init__(self, data_raw_path, data_out_path, remote_ls_file):
        sftp_config = lidar_downloader.sftp_cfg('sftpconfig.json')
        self.remote_ls_file = remote_ls_file

        self.downloader = DownloadTask(
            sftp_config,
            data_raw_path,
            remote_ls_file,
            max_workers=1,
            fan_out=1,
            pool_type='thread')

        self.unzipper = UnzipTask(
            data_raw_path,
            max_workers=1,
            fan_out=3,
            pool_type='process')

        self.preprocessor = PreprocessTask(
            data_raw_path,
            data_out_path,
            max_workers=3,
            fan_out=1,
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
            logger.info(f"Generating task queue for task {task.task_name}.")
            t_done, t_todo = task.create_task_queue(logger)

            if t_done_prev is not None:
                assert set(t_done_prev) == set(t_todo + t_done)

            t_done_prev = t_done
            todos_for_tasks.append(t_todo)
            done_for_tasks.append(t_done)

        if report is not None:
            logger.info(f"Generating report in ./{report}.")

            all_files = lidar_downloader.read_ls_file(self.remote_ls_file)

            # Header row of report
            # ___________________________________________________________________
            # | Filename    | Size  | Downloader    | Unzipper  | Preprocessor  |
            # | DTM-1m-...  | 45123 | 1             | 0         | 0             |

            lines = []
            # lines.append(['Filename', 'Size'] +
            #             [task.task_name for task in self.tasks].join(','))
            lines.append(f"Filename,Size," +
                         ','.join([task.task_name for task in self.tasks]))
            for file, size in all_files.items():
                lines.append(
                    f"{file},{size}," + ','.join([f"{int(file in done)}" for done in done_for_tasks]))

            with open(report, 'w') as f:
                f.write('\n'.join(lines))

        return todos_for_tasks

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

            todos_for_tasks = self.create_queue(logger, report='report.csv')

            for task in todos_for_tasks[0][:N]:
                q_download.put([task])
            for task in todos_for_tasks[1][:N]:
                q_unzip.put([task])
            for task in todos_for_tasks[2][:N]:
                q_preprocess.put([task])

            q_download.put(None)

            download_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.downloader.max_workers)
            unzip_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.unzipper.max_workers)
            preprocess_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.preprocessor.max_workers)

            for _ in range(self.downloader.max_workers):
                download_pool.submit(self.downloader.handle_queue,
                                     q_download, q_unzip, logger)

            for _ in range(self.unzipper.max_workers):
                unzip_pool.submit(self.unzipper.handle_queue,
                                  q_unzip, q_preprocess, logger)

            for _ in range(self.preprocessor.max_workers):
                preprocess_pool.submit(self.preprocessor.handle_queue,
                                       q_preprocess, q_out, logger)

            logging.info(f"[DataPipeline] Processing queue")

            counter = 0
            while True:
                log_entry = q_log.get()
                counter += handle_log_entry(log_entry)
                q_log.task_done()

                if counter == self.total_num_workers():
                    logging.info(f"[DataPipeline] All queues empty")
                    break

            # Handle output queue
            logging.info(f"[DataPipeline] Flushing output queue")
            while True:
                out = q_out.get()
                q_out.task_done()

                if out == None:
                    break

            logging.info("[DataPipeline] Shutting down")
            download_pool.shutdown(wait=True)
            unzip_pool.shutdown(wait=True)


if __name__ == '__main__':

    logging.basicConfig(filename='lidar_plan.log',  level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%y%m%d %H:%M:%S')
    logging.info("=================Starting...=========================")

    pipeline = DataPipeline(
        data_raw_path='/mnt/d/lidarnn_raw_new', data_out_path='/mnt/d/lidarnn', remote_ls_file='ls.txt')

    pipeline.run(N=10)

    logging.info("=================Ending...===========================")

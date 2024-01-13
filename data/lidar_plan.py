
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

    return message.endswith(LOG_EOF)


class Task:
    """Base class for Pipeline task."""

    def __init__(self, task_name):
        self.task_name = "BaseTask"

    def handle_queue(self, queue_in, queue_out, logger):
        """
        Performs a task on items from an upstream queue, until 
        sentinel value task==None is reached. Once the task 
        is complete it is added to the downstream queue.
        """
        logger.bind(self.task_name)
        logger.info(f"Handling queue")

        while True:
            task = queue_in.get()

            if task is not None:
                self.perform_task(task, logger)

            queue_in.task_done()
            queue_out.put(task)

            if task is None:
                # Sentinel value for actual logger
                logger.info(f"{LOG_EOF}")
                break

    def perform_task(task, logger):
        """Actual task to perform to be implemented by derived classes"""
        pass


class DownloadTask(Task):
    """Implements sftp download of LIDAR data."""

    def __init__(self, sftp_config, local_directory):
        self.task_name = "Downloader"
        self.sftp_config = sftp_config
        self.local_directory = local_directory

    def perform_task(self, list_of_files, logger):
        # print("Happily performing task")
        lidar_downloader.download_many(
            self.sftp_config, self.local_directory, list_of_files, logger, tqdm_disable=True)


class UnzipTask(Task):
    """Implements task of unzipping the lidar data"""

    def __init__(self, local_directory):
        self.local_directory = local_directory
        self.task_name = "Unzipper"

    def perform_task(self, list_of_files, logger):
        lidar_helper.unzip_files_in_directory(
            folder_path=self.local_directory, zip_files=list_of_files, logger=logger)


class DataPipeline:
    """Pipeline manager.
    Responsible for:
        1) constructing task queues based on expected state of data/ 
        2) dispatching tasks to handlers concurrently
    """

    def __init__(self, data_path):
        sftp_config = lidar_downloader.sftp_cfg('sftpconfig.json')
        self.downloader = DownloadTask(sftp_config, data_path)
        self.unzipper = UnzipTask(data_path)

    def start(self):
        with multiprocessing.Manager() as manager:
            q_download = manager.Queue()
            q_unzip = manager.Queue()
            q_process = manager.Queue()
            q_out = q_process  # TODO: implement Processor
            q_log = manager.Queue()

            logger = QueueLogger(q_log)
            logger.info(
                f"=====================================================")
            logger.info(f"[DataPipeline] Initialized logger")

            # Temporary manual queue set up
            download_list = [
                "LIDAR-DTM-1m-2022-NZ23ne.zip",
                "LIDAR-DTM-1m-2022-NZ23nw.zip",
                "LIDAR-DTM-1m-2022-NZ23se.zip",
                "LIDAR-DTM-1m-2022-NZ23sw.zip",
                "LIDAR-DTM-1m-2022-NZ24ne.zip",
                "LIDAR-DTM-1m-2022-NZ24nw.zip",
                "LIDAR-DTM-1m-2022-NZ24se.zip",
                "LIDAR-DTM-1m-2022-NZ24sw.zip"
            ]
            q_download.put(download_list)
            q_download.put(None)

            unzip_list = [

                "LIDAR-DTM-1m-2022-NZ09nw.zip",
                "LIDAR-DTM-1m-2022-NZ09se.zip",
                "LIDAR-DTM-1m-2022-NZ09sw.zip",
                "LIDAR-DTM-1m-2022-NZ10ne.zip",
            ]

            q_unzip.put(unzip_list)

            download_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1)
            download_pool.submit(self.downloader.handle_queue,
                                 q_download, q_unzip, logger)

            unzip_pool = concurrent.futures.ProcessPoolExecutor(max_workers=2)
            unzip_pool.submit(self.unzipper.handle_queue,
                              q_unzip, q_out, logger)

            logger.info(f"[DataPipeline] Processing queue")

            counter = 0
            while True:
                log_entry = q_log.get()
                counter += handle_log_entry(log_entry)
                q_log.task_done()

                if counter == 3:  # TODO : counter == total_num_workers
                    logger.info(f"[DataPipeline] All queues empty")
                    break

            logger.info(f"[DataPipeline] Joining queues")
            q_download.join()
            q_unzip.join()

            # Handle output queue
            logger.info(f"[DataPipeline] Flushing output queue")
            while True:
                out = q_out.get()
                q_out.task_done()

                if out == None:
                    break

            logger.info("Shutting down")
            download_pool.shutdown(wait=True)
            unzip_pool.shutdown(wait=True)


if __name__ == '__main__':

    logging.basicConfig(filename='lidar_plan.log',  level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%y%m%d %H:%M:%S')
    pipeline = DataPipeline(data_path='/mnt/d/lidarnn_raw_new/')

    pipeline.start()

    logging.info("Ending")

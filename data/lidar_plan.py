
import logging
import concurrent.futures
import multiprocessing
import queue
import time  # Used for simulating work

import lidar_downloader
import lidar_helper

LOG_EOF = 'EOF'


class QueueLogger:
    '''
    Not really a logger, but worker processes need not know this!
    '''

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
    '''
    unpack the log entry put by the queue logger and actually log it
    '''
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
    '''
    Base class for Pipeline task.
    '''

    def __init__(self, task_name):
        self.task_name = "BaseTask"

    def handle_queue(self, queue_in, queue_out, logger):
        '''
        Performs a task on items from an upstream queue, until 
        sentinel value task==None is reached. Once the task 
        is complete it is added to the downstream queue.
        '''
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

    def perform_task(task, log):
        '''
        Actual task to perform to be implemented by derived classes
        '''


class DownloadTask(Task):
    '''
    Implements sftp download of LIDAR data
    '''

    def __init__(self, sftp_config, local_directory):
        self.task_name = "Downloader"
        self.sftp_config = sftp_config
        self.local_directory = local_directory

    def perform_task(self, list_of_files, logger):
        # print("Happily performing task")
        lidar_downloader.download_many(
            self.sftp_config, self.local_directory, list_of_files, logger, tqdm_disable=True)


class DataPipeline:
    '''
    Pipeline manager responsible for:
        1) constructing task queues based on expected state of data/ 
        2) dispatching tasks to handlers concurrently
    '''

    def __init__(self):
        sftp_config = lidar_downloader.sftp_cfg('sftpconfig.json')
        self.downloader = DownloadTask(sftp_config, '/mnt/d/lidarnn_raw/')

    def start(self):
        with multiprocessing.Manager() as manager:
            queue = manager.Queue()
            queue_out = manager.Queue()

            log_queue = manager.Queue()
            logger = QueueLogger(log_queue)
            logger.info(
                f"=====================================================")
            logger.info(f"[DataPipeline] Initialized logger")

            queue.put(["LIDAR-DTM-1m-2022-NZ27se.zip"])
            queue.put(None)

            pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)
            pool.submit(self.downloader.handle_queue, queue, queue_out, logger)

            logger.info(f"[DataPipeline] Processing queue")

            counter = 0
            while True:
                log_entry = log_queue.get()
                counter += handle_log_entry(log_entry)

                if counter == 1:  # TODO : counter == total_num_workers
                    break

                log_queue.task_done()

            queue.join()

            x = queue_out.get()
            print(x)
            x = queue_out.get()
            print(x)

            print("Shutting down")
            pool.shutdown(wait=True)


if __name__ == '__main__':

    logging.basicConfig(filename='lidar_plan.log',  level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%y%m%d %H:%M:%S')
    pipeline = DataPipeline()

    pipeline.start()

    print("Ending.")

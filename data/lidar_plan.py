
import logging
import concurrent.futures
import multiprocessing
import queue
import time  # Used for simulating work

import lidar_downloader


class Task:
    '''
    Base class for Pipeline task.
    '''

    def __init__(self, task_name):
        self.task_name = "BaseTask"

    def handle_queue(self, queue_in, queue_out):
        '''
        Performs a task on items from an upstream queue, until 
        sentinel value task==None is reached. Once the task 
        is complete it is added to the downstream queue.
        '''
        logging.debug(f"[{self.task_name}] handling queue")

        while True:
            task = queue_in.get()

            if task is not None:
                self.perform_task(task)

            queue_in.task_done()
            queue_out.put(task)

            if task is None:
                break

    def perform_task(task):
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

    def perform_task(self, list_of_files):
        print("Hello from task")
        lidar_downloader.download_many(
            self.sftp_config, self.local_directory, list_of_files, tqdm_disable=True)


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

            queue.put(["LIDAR-DTM-1m-2022-NZ27se.zip"])
            queue.put(None)

            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            pool.submit(self.downloader.handle_queue, queue, queue_out)

            queue.join()

            x = queue_out.get()
            print(x)
            x = queue_out.get()
            print(x)

            print("Shutting down")
            pool.shutdown(wait=True)


if __name__ == '__main__':

    pipeline = DataPipeline()

    pipeline.start()

    print("Ending.")

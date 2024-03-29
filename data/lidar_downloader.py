import os
import socket
import paramiko
import json
import logging
from tqdm import tqdm

TQDM_FMT = '{desc:<32}: {percentage:3.0f}%|{bar}| {n:3.0f}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'


class ConnectionException(Exception):
    def __init__(self, host, port):
        Exception.__init__(self, host, port)
        self.message = f"Could not connect to {host}:{port}."


class SFTPConnection(object):
    '''
    Simple wrapper class to handle sftp connection. Adapted from pysftp
    '''

    def __init__(self, host, username, password, port=22):
        self._sftp_live = False
        self._sftp = None
        self._tconnect = {
            'username': username,
            'password': password,
            'hostkey': None,
        }

        try:
            self._transport = paramiko.Transport((host, port))
        except (AttributeError, socket.gaierror):
            raise ConnectionException(host, port)

        self._transport.connect(**self._tconnect)

        logging.getLogger("paramiko").setLevel(logging.WARNING)

    def _sftp_connect(self):
        '''Cached connection'''
        if not self._sftp_live:
            self._sftp = paramiko.SFTPClient.from_transport(self._transport)
            self._sftp_live = True

    def get(self, remotepath, localpath, callback=None, prefetch=False, max_concurrent_prefetch_requests=None):
        '''Wrapper for get()'''
        self._sftp_connect()
        self._sftp.get(remotepath, localpath, callback=callback,
                       prefetch=prefetch, max_concurrent_prefetch_requests=max_concurrent_prefetch_requests)

    def ls(self, remotepath='.'):
        '''Wrapper for listdir_attr'''
        self._sftp_connect()
        return self._sftp.listdir_attr(remotepath)

    def close(self):
        '''Close sftp connection and transport'''
        if self._sftp_live:
            self._sftp.close()
            self._sftp_live = False

        if self._transport:
            self._transport.close()
            self._transport = None

    @property
    def sftp_client(self):
        self._sftp_connect()
        return self._sftp

    def __enter__(self):
        return self

    def __del__(self):
        self.close()

    def __exit__(self, etype, value, traceback):
        self.close()


def sftp_cfg(json_fname):
    '''Read sftp config from file. Contents should be:
    {
        "SFTP_USER": "[username]",
        "SFTP_HOST": "[host]",
        "SFTP_PASSWORD": "[****]",
        "SFTP_REMOTE_DIRECTORY": "/composite/2022/Zip_Files/1m/DTM/"
    }
    '''
    with open(json_fname, 'r') as json_file:
        sftp_config = json.load(json_file)

    return sftp_config


def list_files(sftp_config, remote_ls_file=""):
    '''
    makes an sftp connection, lists the contents of sftp_config["SFTP_REMOTE_DIRECTORY"] in the following format:
        "{file_name}\t{file_size}\n"

    for instance:

    DTM-0001.zip    69100020
    DTM-0002.zip    70100100
    DTM-0003.zip    72100003

    if remote_ls_file=="", print to console, else write output to remote_ls_file 
    '''

    with SFTPConnection(host=sftp_config["SFTP_HOST"],
                        username=sftp_config["SFTP_USER"],
                        password=sftp_config["SFTP_PASSWORD"],
                        #                           cnopts=cnopts
                        ) as sftp:

        logging.info(f"[CONNECTED] to host {sftp_config['SFTP_HOST']}.")
        directory_structure = sftp.ls()

        if remote_ls_file:
            with open(remote_ls_file, 'w') as f:
                for attr in directory_structure:
                    f.write(f"{attr.filename}\t{attr.st_size}\n")
        else:
            for attr in directory_structure:
                logging.info(f"{attr.filename}\t{attr.st_size}\n")


def read_ls_file(remote_ls_file):
    with open(remote_ls_file, 'r') as f:
        remote_files = {line.split('\t')[0].rsplit('.', 1)[0]: int(
            line.split('\t')[1]) for line in f}

    return remote_files


def compare_remote_listing_to_already_downloaded(remote_ls_file, local_directory, download_queue=None):
    '''Compares the contents of remote_ls_file to the contents of local_directory, and enumerates a list
    of all files yet to be downloaded, and writes this list to download_queue    
    '''
    logging.info(
        f"[COMPARE] remote listing file to already downloaded zip files.")

    local_files = os.listdir(local_directory)
    local_files = set([f.split('.')[0] for f in local_files])

    remote_files = read_ls_file(remote_ls_file)

    # to_download = remote_files - local_files
    to_download = {file: size for (
        file, size) in remote_files.items() if file.split('.')[0] not in local_files}
    already_downloaded = {file: size for (
        file, size) in remote_files.items() if file.split('.')[0] in local_files}

    if download_queue is None:
        return list(already_downloaded.keys()), list(to_download.keys())

    with open(download_queue, 'w') as f:
        for file in to_download.keys():
            f.write(f"{file}\n")

    return {
        "to_download_size": sum(to_download.values()) * 1e-6,
        "to_download_count": len(to_download.values()),
        "already_downloaded_size": sum(already_downloaded.values()) * 1e-6,
        "already_downloaded_count": len(already_downloaded.values())
    }


def download_many(sftp_config, local_directory, download_queue, logger, remote_ls_file=None, N=None, tqdm_disable=False):
    '''
    Reads the contents of `download_queue` to establish a list of file names to download, and 
    proceeds to download each one

    This function is interruptible.
    When interrupted from the command line, the following happehs:
        -A message will be displayed on the console
        -The current download will finish
        -The connection gracefully closed
        -compare_remote_listing_to_already_downloaded() will be run to update the download_queue

    if N==None, the queue will be worked on until finished or an interrupt is received.
    if N > 0, the queue will be worked on until N files have been downloaded or an interrupt has been received.
    '''

    # read in list of files we need to download
    if isinstance(download_queue, list):
        files_to_download = download_queue
    else:
        with open(download_queue, 'r') as f:
            files_to_download = [line.strip() for line in f]

    if N is not None:
        files_to_download = files_to_download[:N]

    logger.info(f"Downloading {N} {len(files_to_download)} files.")
    count = 0

    with SFTPConnection(host=sftp_config["SFTP_HOST"],
                        username=sftp_config["SFTP_USER"],
                        password=sftp_config["SFTP_PASSWORD"],
                        ) as sftp:
        logger.info(f"[CONNECTED] to host {sftp_config['SFTP_HOST']}")

        with tqdm(total=len(files_to_download),
                  unit='file',
                  desc="Downloading files",
                  position=0,
                  disable=tqdm_disable,
                  bar_format=TQDM_FMT) as pbar:
            for file_name in files_to_download:
                with tqdm(total=100,
                          unit='%',
                          desc=f"{file_name}",
                          position=1,
                          leave=False,
                          disable=tqdm_disable,
                          bar_format=TQDM_FMT) as pbar_inner:
                    try:

                        # make sure we are downloading a zip
                        if not file_name.endswith('.zip'):
                            file_name = file_name + '.zip'

                        logger.info(f"[DOWNLOADING] {file_name}")
                        remote_file_path = os.path.join(
                            sftp_config["SFTP_REMOTE_DIRECTORY"], file_name)
                        local_file_path = os.path.join(
                            local_directory, file_name)

                        def _callback(size, file_size):
                            pbar_inner.update(
                                100 * size/file_size - pbar_inner.n)

                        sftp.get(remote_file_path, local_file_path,
                                 callback=_callback, prefetch=True, max_concurrent_prefetch_requests=64)

                        pbar.update(1)
                        count += 1
                    except KeyboardInterrupt:
                        logger.warning(
                            "\nDownload interrupted by the user. Cleaning up...")
                        break

                    except Exception as e:
                        logger.error(
                            f"Error downloading file {file_name}: {e}")

    logger.info(f"Updating queue file {download_queue}")

    if remote_ls_file is not None:
        # Update download queue
        res = compare_remote_listing_to_already_downloaded(
            remote_ls_file, local_directory, download_queue)

        print(
            f"Downloaded: {res['already_downloaded_size']:>12.0f} MB\t{res['already_downloaded_count']:>5} files")
        print(
            f"Remaining:  {res['to_download_size']:>12.0f} MB\t{res['to_download_count']:>5} files")

    if count < len(files_to_download):
        logger.info(f"Downloaded {count} files before interruption.")

    else:
        logger.info("All files downloaded successfully")


if __name__ == '__main__':
    logging.basicConfig(filename='lidar_downloader.log',  level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%y%m%d %H:%M')
    logger = logging.getLogger()
    compare_remote_listing_to_already_downloaded(
        'ls.txt', '/mnt/d/lidarnn_raw_new/', 'todo.txt')
    sftp_config = sftp_cfg('sftpconfig.json')
    download_many(sftp_config, '/mnt/d/lidarnn_raw/',
                  'todo.txt', logger, 'ls.txt', 2)

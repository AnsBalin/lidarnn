import os
from contextlib import contextmanager
from tqdm import tqdm
import json
# import pysftp
import time
import paramiko
import socket


class ConnectionException(Exception):
    def __init__(self, host, port):
        Exception.__init__(self, host, port)
        self.message = f"Could not connect to {host}:{port}."


class SFTPConnection(object):
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

    def _sftp_connect(self):
        if not self._sftp_live:
            self._sftp = paramiko.SFTPClient.from_transport(self._transport)
            self._sftp_live = True

    def get(self, remotepath, localpath, callback=None, prefetch=False, max_concurrent_prefetch_requests=None):
        self._sftp_connect()
        self._sftp.get(remotepath, localpath, callback=callback,
                       prefetch=prefetch, max_concurrent_prefetch_requests=max_concurrent_prefetch_requests)

    def ls(self, remotepath='.'):
        self._sftp_connect()
        return self._sftp.listdir_attr(remotepath)

    def close(self):
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
#    cnopts = pysftp.CnOpts()
#    cnopts.hostkeys = None

    with SFTPConnection(host=sftp_config["SFTP_HOST"],
                        username=sftp_config["SFTP_USER"],
                        password=sftp_config["SFTP_PASSWORD"],
                        #                           cnopts=cnopts
                        ) as sftp:
        print(f"Connection made to {sftp_config['SFTP_HOST']}.")
        directory_structure = sftp.ls()

        if remote_ls_file:
            with open(remote_ls_file, 'w') as f:
                for attr in directory_structure:
                    f.write(f"{attr.filename}\t{attr.st_size}\n")
        else:
            for attr in directory_structure:
                print(f"{attr.filename}\t{attr.st_size}\n")


def compare_remote_listing_to_already_downloaded(remote_ls_file, local_directory, download_queue):
    '''Compares the contents of remote_ls_file to the contents of local_directory, and enumerates a list
    of all files yet to be downloaded, and writes this list to download_queue    
    '''

    local_files = set(os.listdir(local_directory))

    with open(remote_ls_file, 'r') as f:
        remote_files = {line.split('\t')[0]: int(
            line.split('\t')[1]) for line in f}

    # to_download = remote_files - local_files
    to_download = {file: size for (
        file, size) in remote_files.items() if file not in local_files}
    already_downloaded = {file: size for (
        file, size) in remote_files.items() if file in local_files}

    with open(download_queue, 'w') as f:
        for file in to_download.keys():
            f.write(f"{file}\n")

    return {
        "to_download_size": sum(to_download.values()) * 1e-6,
        "to_download_count": len(to_download.values()),
        "already_downloaded_size": sum(already_downloaded.values()) * 1e-6,
        "already_downloaded_count": len(already_downloaded.values())
    }


def download_many(sftp_config, local_directory, download_queue, remote_ls_file, N=None):
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
    with open(download_queue, 'r') as f:
        files_to_download = [line.strip() for line in f]

    if N is not None:
        files_to_download = files_to_download[:N]

    print(f"Downloading {N} {len(files_to_download)} files.")
    count = 0

    with SFTPConnection(host=sftp_config["SFTP_HOST"],
                        username=sftp_config["SFTP_USER"],
                        password=sftp_config["SFTP_PASSWORD"],
                        ) as sftp:
        print(f"[CONNECTED] to host {sftp_config['SFTP_HOST']}")

        with tqdm(total=len(files_to_download), unit='file', desc="Downloading files") as pbar:
            for file_name in files_to_download:
                with tqdm(total=100, unit='%', desc=f"{file_name}") as pbar_inner:
                    try:
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
                        print("\nDownload interrupted by the user. Cleaning up...")
                        break

                    except Exception as e:
                        print(f"Error downloading file {file_name}: {e}")

    print(f"Updating queue file {download_queue}")
    # Update download queue
    res = compare_remote_listing_to_already_downloaded(
        remote_ls_file, local_directory, download_queue)

    print(
        f"Downloaded:\t{res['already_downloaded_size']:.0f}MB\t{res['already_downloaded_count']}")
    print(
        f"Remaining: \t{res['to_download_size']:.0f}MB\t{res['to_download_count']}")

    if count < len(files_to_download):
        print(f"Downloaded {count} files before interruption.")

    else:
        print("All files downloaded successfully")


if __name__ == '__main__':
    compare_remote_listing_to_already_downloaded(
        'ls.txt', '/mnt/d/lidarnn_raw/', 'todo.txt')
    sftp_config = sftp_cfg('sftpconfig.json')
    download_many(sftp_config, '/mnt/d/lidarnn_raw/', 'todo.txt', 'ls.txt', 10)

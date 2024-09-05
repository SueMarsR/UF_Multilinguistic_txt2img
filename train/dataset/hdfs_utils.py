import os
import shlex
import shutil
import argparse
import subprocess
from typing import List, Union
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor

_HADOOP_COMMAND_TEMPLATE = 'hadoop fs {command}'
_SUPPORTED_HDFS_PATH_PREFIXES = ('hdfs://', 'ufs://')


def has_hdfs_path_prefix(filepath):
    """Check if input filepath has hdfs prefix"""
    for prefix in _SUPPORTED_HDFS_PATH_PREFIXES:
        if filepath.startswith(prefix):
            return True
    return False


def upload_to_hdfs(src_path, dst_path, overwrite=False):
    """Upload src_path to hdfs dst_path"""
    if not os.path.exists(src_path):
        raise IOError('Input src_path {} not found in local storage'.format(src_path))
    if not has_hdfs_path_prefix(dst_path):
        raise ValueError('Input dst_path {} is not a hdfs path'.format(dst_path))

    try:
        cmd = '-put -f' if overwrite else '-put'
        cmd = '{} {} {}'.format(cmd, src_path, dst_path)
        check_call_hdfs_command(cmd)
        return True
    except Exception as e:
        msg = 'Failed to upload src {} to dst {}: {}'.format(src_path, dst_path, e)
        raise ValueError(msg)
    return False


def is_seq_of(seq, expected_type, seq_type=None):
    r"""Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    r"""Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)


def _get_hdfs_command(command):
    """Return hadoop fs command"""
    return _HADOOP_COMMAND_TEMPLATE.format(command=command)


def check_call_hdfs_command(command):
    """Check call hdfs command"""
    hdfs_command = _get_hdfs_command(command)
    subprocess.check_call(shlex.split(hdfs_command))


def download_from_hdfs(src_path: str,
                       dst_path: str,
                       overwrite: bool = False,
                       raise_exception: bool = False):
    """ Download src_path from hdfs to local dst_path

    Args:
        src_path: the source hdfs path
        dst_path: the local download destination
        overwrite: if True, the local file will be overwritten if it exists
        raise_exception: if True, error is raised when thing goes wrong
    """
    # Legality check
    assert isinstance(src_path, str) and has_hdfs_path_prefix(src_path), src_path
    assert isinstance(dst_path, str) and not has_hdfs_path_prefix(dst_path), dst_path

    # Get the targeted download path
    if os.path.isdir(dst_path):  # download to an existing folder
        download_path = os.path.join(dst_path, os.path.basename(src_path))
    else:  # download as a file
        download_path = dst_path

    if overwrite is True:  # Remove the targeted file/folder if it exists
        if os.path.isdir(download_path):
            shutil.rmtree(download_path)
        elif os.path.isfile(download_path):
            os.remove(download_path)
    else:  # skip downloading if the targeted file/folder exists
        if os.path.exists(download_path):
            return True

    # Download from hdfs
    try:
        cmd = '-get {} {}'.format(src_path, dst_path)
        check_call_hdfs_command(cmd)
        return True
    except Exception as e:
        msg = 'Failed to download src {} to dst {}: {}'.format(src_path, dst_path, e)
        raise ValueError(msg)
        return False


def batch_download_from_hdfs(src_paths: List[str],
                             dst_paths: Union[str, List[str]],
                             overwrite: bool = False,
                             raise_exception: bool = False,
                             mp_size: int = 1) -> bool:
    """ Batch download from hdfs with mp.Pool acceleration

    Args:
        src_paths: the source paths of hdfs files/folders
        dst_path: the local download destination
        overwrite: if True, the local file will be overwritten if it exists
        raise_exception: if True, error is raised when thing goes wrong
        mp_size: the max_workers for ProcessPoolExecutor
    Return:
        success: if True, all the downloads are successfully executed
    """
    # Legality check
    assert is_list_of(src_paths, str), f'src_paths {src_paths} must of a list of str'
    if isinstance(dst_paths, str):
        dst_paths = [dst_paths] * len(src_paths)
    else:
        assert len(dst_paths) == len(src_paths), \
            f'length of dst_paths {dst_paths} mismatches with src_paths {src_paths}'

    # Multiprocess download with ProcessPoolExecutor context manager
    with ProcessPoolExecutor(max_workers=mp_size) as executor:
        futures = [executor.submit(download_from_hdfs, src_path, dst_path, overwrite, raise_exception)
                   for src_path, dst_path in zip(src_paths, dst_paths)]
    download_success = [future.result() for future in futures]
    return all(download_success)


def popen_hdfs_command(command):
    """Call hdfs command with popen and return stdout result"""
    hdfs_command = _get_hdfs_command(command)
    p = subprocess.Popen(shlex.split(hdfs_command), stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    return stdout


def get_hdfs_list(filepath):
    """Glob hdfs path pattern"""
    try:
        cmd = '-ls {}'.format(filepath)
        stdout = popen_hdfs_command(cmd)
        lines = stdout.splitlines()
        if lines:
            # decode bytes string in python3 runtime
            lines = [line.decode('utf-8') for line in lines]
            return [line.split(' ')[-1] for line in lines]
        else:
            return []
    except Exception:
        return []


def get_hdfs_dir_children(hdfs_dir):
    r""" Find the children list and skip the illegal terms like 'items'
    """
    hdfs_path_list = get_hdfs_list(hdfs_dir)
    hdfs_path_list = [str(item) for item in hdfs_path_list if item.startswith('hdfs://')]
    return hdfs_path_list


def mkdir_hdfs(dirpath):
    """Mkdir hdfs directory"""
    try:
        cmd = '-mkdir -p {}'.format(dirpath)
        check_call_hdfs_command(cmd)
        return True
    except Exception as e:
        msg = 'Failed to mkdir {} in HDFS: {}'.format(dirpath, e)
        print(ValueError(msg))
        return False


def is_hdfs_folder_exists(remote_path):
    # 1: not exist 0: exist
    command = f"hdfs dfs -test -e {remote_path}; echo $?"
    filexistchk_output = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).communicate()
    return '1' not in str(filexistchk_output[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',
                        default='hdfs://harunava/home/byte_ecom_magellan_supply_chain/user/liushengzhe/train_shopee/597_data/')
    parser.add_argument('--dst', default='Goods_data')
    args = parser.parse_args()

    src = args.src
    dst = args.dst
    mp_size = 64
    src = get_hdfs_dir_children(src)
    batch_download_from_hdfs(src_paths=src, dst_paths=dst, mp_size=mp_size)

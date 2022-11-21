from typing import Union, Optional, List

import os
import sys
import shutil
from pathlib import Path

from termcolor import colored

import urllib
import wget
from urllib.request import urlopen
from urllib.parse import urlparse
import re
import requests
from requests.exceptions import RequestException
from zipfile import ZipFile, BadZipFile
from torchvision.datasets.utils import download_url, download_and_extract_archive
from imagedataset.imageloaders import LoaderPIL
from imagedataset.multidataset import AdvancedImageFolder, from_database
from imagedataset.operations import ImageLoader
from utils import load_config
from .gdrive import isGdrive, downloadAndExtractGdrive
from os.path import join, isdir
from os import listdir
import numpy as np


THISMODULE = sys.modules[__name__]

# CONSTANTS FROM CONFIG FILE
CONFIG = load_config("config", "datasets.yaml")
ROOT = CONFIG["dataset_root"]
DATABASE_ROOT = CONFIG["database_root"]


def get_dataset(dataset_name: str,
                loader: Optional[ImageLoader] = LoaderPIL(),
                from_database: Optional[bool] = False,            
                percentage: Optional[float] = None, 
                seed: Optional[int] = 123) -> AdvancedImageFolder:

    dataset_path = join(ROOT, dataset_name)

    if from_database:
        database_path = join(DATABASE_ROOT, dataset_name)
        return from_database(database_path).set_loader(loader)

    dataset = AdvancedImageFolder(dataset_path, load_percentage=percentage, seed=seed, 
                                  filter_image_formats=None)
    dataset.set_loader(loader)
    return dataset


def print_dataset_info(dataset_name: str):
    dataset_name = dataset_name.lower()

    path = join(ROOT, dataset_name)
    classes = get_subdirs(path, full_path=True)
    
    files = np.empty(len(classes))
    for i, c in enumerate(classes):
        files[i] = count_files(c)


    print(f"Dataset {dataset_name}")
    print(f"Number of classes: {len(classes)}")
    print(f"Total images     : {int(files.sum())}")
    print(f"Average images   : {files.mean():6.2f}")
    print(f"Std images       : {files.std():6.2f}")
    print(f"Maximum images   : {int(files.max())}")
    print(f"Minimum images   : {int(files.min())}")



def download_and_setup_whoi22():

    # TRAIN
    # get urls
    urls = list(dict(CONFIG["whoi22_train"]).values())
    out_dir = join(ROOT, "whoi22_train")

    # not downloaded yet.
    if not os.path.isdir(out_dir):
        Path(out_dir).mkdir(exist_ok=True)
        download(urls, out_dir)

    # TEST
    # get urls
    urls = list(dict(CONFIG["whoi22_test"]).values())
    out_dir = join(ROOT, "whoi22_test")

    # not downloaded yet.
    if not os.path.isdir(out_dir):
        Path(out_dir).mkdir(exist_ok=True)
        download(urls, out_dir)


def download_and_setup_zooscan():

    # get urls
    urls = list(dict(CONFIG["zooscan"]).values())
    out_dir = join(ROOT, "zooscan")
    
    # already downloaded...
    if os.path.isdir(out_dir):
        return
    
    # create dir and download
    Path(out_dir).mkdir(exist_ok=True)
    download(urls, out_dir)
    
    dirs = list(os.listdir(join(out_dir, "ZooScanSet", "imgs")))

    for dir in dirs:
        move_from = join(out_dir, "ZooScanSet", "imgs", dir)
        move_to = join(out_dir, dir)
        shutil.move(move_from, move_to)

    shutil.rmtree(join(out_dir, "ZooScanSet"))


def download_and_setup_zoolake():

    # get urls
    urls = list(dict(CONFIG["zoolake"]).values())
    tmp_dir = join(ROOT, "zoolake_tmp")
    
    # already downloaded...
    if os.path.isdir(join(ROOT, "zoolake_train")):
        return
    
    # create dir and download
    Path(tmp_dir).mkdir(exist_ok=True)
    download(urls, tmp_dir)

    train_txt = join(tmp_dir, "data", "zoolake_train_test_val_separated", "train_filenames.txt")
    val_txt = join(tmp_dir, "data", "zoolake_train_test_val_separated", "val_filenames.txt")
    test_txt = join(tmp_dir, "data", "zoolake_train_test_val_separated", "test_filenames.txt")

    train_lines = []
    test_lines = []

    with open(train_txt, "r") as f:
        train_lines += f.readlines()
    with open(val_txt, "r") as f:
        train_lines += f.readlines()
    with open(test_txt, "r") as f:
        test_lines += f.readlines()

    # get paths in correct format...
    train_lines = [join(*l.rstrip().split("/")[-4:]) for l in train_lines]
    test_lines = [join(*l.rstrip().split("/")[-4:]) for l in test_lines]

    outdir_train = join(ROOT, "zoolake_train")
    outdir_test = join(ROOT, "zoolake_test")

    for lines, out in [(train_lines, outdir_train), (test_lines, outdir_test)]:
        for line in lines:
    
            _, classname, _, filename = line.split("/")
            
            output_dir = join(out, classname)
            Path(output_dir).mkdir(exist_ok=True, parents=True)

            input_file = join(tmp_dir, "data", line)
            output_file = join(output_dir, filename)

            if os.path.isfile(input_file):
                try:
                    shutil.move(input_file, output_file)
                except Exception as e:
                    print(f"ERROR AT: {input_file}")
                    print(e)

    shutil.rmtree(tmp_dir)


def download_and_setup_kaggle_zooscan20():

    # get urls
    urls = list(dict(CONFIG["kaggle_zooscan20"]).values())
    out_dir = join(ROOT, "tmp")
    
    # already downloaded...
    if os.path.isdir(join(ROOT, "kaggle")):
        return
    
    # create dir and download
    Path(out_dir).mkdir(exist_ok=True)
    download(urls, out_dir)
    
    # setup files
    current_dirs = [join("Kaggle", "train"), join("ZooScan", "training"), join("ASLO", "training"), join("ASLO", "testing")]
    output_dirs = ["kaggle", "zooscan20", "aslo_train", "aslo_test"]

    for output_dir, current_dir in zip(output_dirs, current_dirs):
        current_path = join(ROOT, "tmp", "Dataset", current_dir)
        output_path = join(ROOT, output_dir)
        Path(output_path).mkdir(exist_ok=True)

        folders = [d for d in os.listdir(current_path) if os.path.isdir(join(current_path, d))]
        
        for folder in folders:
            shutil.move(join(current_path, folder), join(output_path, folder))

    shutil.rmtree(join(ROOT, "tmp"))


def download_and_setup_whoi():

    print("Start")
    # get urls
    urls = list(dict(CONFIG["whoi"]).values())
    out_dir = join(ROOT, "whoi")
    
    # already downloaded...
    if os.path.isdir(out_dir):
        return
    
    # # create dir and download
    Path(out_dir).mkdir(exist_ok=True)
    download(urls, out_dir)

    # setup files
    year_datasets = [yd for yd in os.listdir(out_dir) \
                               if os.path.isdir(join(out_dir, yd))]
    
    # merge all years
    for dataset in year_datasets:
        dataset_path = join(out_dir, dataset)
        classes = os.listdir(dataset_path)

        for class_dir in classes:
            print(class_dir)
            class_full_path = join(dataset_path, class_dir)
            out_class_dir = join(out_dir, class_dir)
            Path(out_class_dir).mkdir(exist_ok=True)

            images = os.listdir(class_full_path)

            for image in images:
                input_path = join(class_full_path, image)
                out_path = join(out_class_dir, f"{dataset}_{image}")
                shutil.move(input_path, out_path)

        shutil.rmtree(dataset_path)

    # delete useless files
    current_dirs = [d for d in os.listdir(out_dir) if isdir(join(out_dir, d))]
    for i, class_dir in enumerate(current_dirs):

        print(f"Status: {i+1}/{len(current_dirs)}")
        class_dir = join(out_dir, class_dir)

        for file in os.listdir(class_dir):
            if "Thumbs.db" in file:
                full_path = join(class_dir, file)
                os.remove(full_path)

    out_dir_unused = join(ROOT, "whoi_unused")
    key = ['mix' , 'flagellate', 'dactyliosolen', 'asterionellopsis', 'phaeocystis', 
           'other_lt20', 'euglena', 'dactfragcerataul', 'pennate', 'guinardia', 
           'thalassiosira', 'pleurosigma', 'chaetoceros', 'pseudonitzschia', 
           'skeletonema', 'ditylum', 'licmophora', 'cylindrotheca', 'dinoflagellate', 
           'ciliate', 'detritus', 'dinobryon', 'rhizosolenia']

    current_dirs = [d for d in os.listdir(out_dir) if isdir(join(out_dir, d))]
    for i, dir in enumerate(current_dirs):
        print(f"Status: {i+1}/{len(current_dirs)}")
        for k in key:
            if k.lower() in dir.lower():
                current_out_dir = join(out_dir_unused, dir)
                Path(current_out_dir).mkdir(exist_ok=True, parents=True)

                for file in os.listdir(join(out_dir, dir)):
                    input_path = join(out_dir, dir, file)
                    output_path = join(current_out_dir, file)
                    shutil.move(input_path, output_path)
                shutil.rmtree(join(out_dir, dir))
                break


def download(urls: Union[str, List[str]], out_dir: str):

    if isinstance(urls, str):
        urls = [urls]

    # if the dataset has not been downloaded yet
    if not os.path.exists(join(out_dir)):
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    for u in urls:
        try:
            file_path = wget.download(u, out=out_dir)
        except Exception:
            file_path = join(out_dir, "tmp.zip")
            headers = {'User-Agent': 'Interwebs Exploiter 4'}
            with requests.get(u, stream=True, allow_redirects=True, headers=headers) as r:
                r.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=100000): 
                        f.write(chunk)

    print(f"Extracting {file_path} to {out_dir}")
    extract_archive(file_path, out_dir)
    os.remove(file_path)


########################################################################################
def extract_archive(zip_file: str, output_dir: str):

    # tar file
    if zip_file.endswith("tar"):
        import tarfile
        with tarfile.open(zip_file) as tar:
            tar.extractall(path=output_dir)
        return 

    # zipfile
    z = ZipFile(zip_file)
    for name in z.namelist():
        try:
            z.extract(name, output_dir)
        except BadZipFile as e:
            print(f"Bad file {name}, skipping.")
            print(e)


def remove_subdir(path: str):
    """
        Remove an inner subdir.
        Example:
            remove_subdir('a/b'):

            a/b/c       -> a/c
            a/b/d.txt   -> a/d.txt
            a/e         -> a/e

    """

    # get parent directory "a/"
    final_path = Path(path).parents[0]

    # move all files/dirs from "a/b/" to "a/"
   
    for file_name in os.listdir(path):
        file_path_current = join(path, file_name)
        file_path_final   = join(final_path, file_name)
        shutil.move(file_path_current, file_path_final)

    # remove "b"
    os.rmdir(path)


def get_subdirs(path: str, full_path: Optional[bool] = True) -> List[str]:
    """
    Get the path of all subdirs of the specified path.

    Args:
        path (str): the path where to look for subdirs
        full_path (bool): if true the full path is returned, otherwise just the subdir
        names

    Returns:
        list of strings with the path of subdirs
    """

    if full_path:
        return [join(path, d) for d in os.listdir(path) 
                if os.path.isdir(join(path, d))]

    return [d for d in os.listdir(path) if os.path.isdir(join(path, d))]


def count_files(path: str) -> int:
    """
        Count the number of files in the specifier dir.

        Args:
            path (str): the dir where to look for files.
        
        Returns:
            the number of files in the directory specified by path
    """

    files = [f for f in os.listdir(path) if os.path.isfile(join(path, f))]
    return len(files)

# ERROR CLASSES

class DownloadError(Exception):
    """Raised when there is an error while downloading a dataset"""
    pass


class UnavailableDatasetError(Exception):
    """Raised when the dataset requested is not available"""
    pass


class NotDownloadedError(Exception):
    """Raised when the dataset requested is available, but correnly it has not been
       downloaded. 
    """
    pass


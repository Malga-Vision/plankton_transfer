import gdown
import shutil
import os

URL = "https://drive.google.com/uc?id="


def extract(filepath, remove_archive=False):
    extract_dir = os.path.dirname(filepath)
    shutil.unpack_archive(filepath, extract_dir)

    if remove_archive:
        os.remove(filepath)


def isGdrive(url):
    """
        Checks if the url is to goole drive.

        Args:
            url (string): the input url to check.
        Returns:
            True if it is a gdrive link.
    """
    
    return "drive.google.com/file/d/" in url


def getID(url):
    """
        Get file ID from a shareble gdrive url.
    """

    url = url.split("/")

    for i in range(len(url)):
        if url[i] == "d":
            id = url[i+1]
            return id
    return None


def downloadAndExtractGdrive(url, destination, remove_archive=False):
    FILE_NAME = "tmp.zip"
    downloadGdrive(url, os.path.join(destination, FILE_NAME))
    extract(os.path.join(destination, FILE_NAME), remove_archive=remove_archive)


def downloadGdrive(url, destination):
    """
        Download a file from gdrive shareble url.

        Args:
            url (string): sharable url to download.
            destination (string): destination file.
    """
    # extract id from shareble url
    id = getID(url)

    # append id to base url
    url = URL + id

    gdown.download(url, destination, quiet=False)

import cv2 as cv
import numpy as np
from collections import namedtuple
from typing import Optional, Tuple
from pathlib import Path
from os.path import join, isdir, isfile
from os import listdir

# template paths
THIS_DIRECTORY =  Path(__file__).parent.absolute()
TEMPL_TOP_LEFT = join(THIS_DIRECTORY, "templates", "top_left.jpg")
TEMPL_BOTTOM_RIGHT = join(THIS_DIRECTORY, "templates", "bottom_right.jpg")

# pixel offsets for template matching
OFFSET_TOPX =  2
OFFSET_TOPY =  2
OFFSET_BOTX = -1
OFFSET_BOTY = -1

Point = namedtuple("Point", ["x", "y"])
CropCoordinates = namedtuple("CropCoordinates", ["top_left", "bottom_right"])


def _create_template_mask(template: np.array) -> np.array:
    """
        Create a mask for the templates (to consider a non-rectangular region for 
        matching).
    """

    # erode the image a little bit
    kernel = np.ones((4, 4), np.uint8)
    template = cv.erode(template, kernel)

    # binarize
    thresh = 127
    template = cv.threshold(template, thresh, 255, cv.THRESH_BINARY)[1]

    # invert
    return 255 - template


def get_coordinates(image_path: str, method: Optional[int] = cv.TM_CCOEFF) \
                    -> Tuple[CropCoordinates, np.array]:
    """
        Find the top-left, bottom-right points on zooscan images using templates.

        NOTE:
            the templaes paths are set as constants on the top of the file!
        
        Args:
            image_path (str): the path to a zooscan input image.
            method (int, optional): the opencv method to match templates.
        
        Return:
            the CropCoordinates and the image
    """
    img = cv.imread(image_path,0)

    # templates
    left = cv.imread(TEMPL_TOP_LEFT,0)
    mask_left = _create_template_mask(left)
    right = cv.imread(TEMPL_BOTTOM_RIGHT,0)
    mask_right = _create_template_mask(right)

    w_right, h_right = right.shape[::-1]

    # match templates
    result_left = cv.matchTemplate(img, left, method, None, mask_left)
    result_right = cv.matchTemplate(img, right, method, None, mask_right)

    _, _, _, max_loc = cv.minMaxLoc(result_left)
    top_left = Point(x=max_loc[0]+OFFSET_TOPX, y=max_loc[1]+OFFSET_TOPY)
    _, _, _, max_loc = cv.minMaxLoc(result_right)
    bottom_right = Point(x=max_loc[0]+w_right+OFFSET_BOTX, 
                         y=max_loc[1]+h_right+OFFSET_BOTY)

    return CropCoordinates(top_left, bottom_right), img


def crop_image(image_path: str) -> np.array:
    """
        Loads the image from path, get the crop coordinates using templates and returns
        the cropped image.

        Args:
            image_path (str): the path to the zooscan image to crop.

        Returns:
            the cropped image.
    """
    crop_coordinates, img = get_coordinates(image_path)
    top_left, bottom_right = crop_coordinates

    width = bottom_right.x - top_left.x
    height = bottom_right.y - top_left.y
    img = img[top_left.y:top_left.y+height, top_left.x:top_left.x+width]

    return img


def crop_zooscan(input_path: str, output_path: str) -> None:

    """
        Crops the entire zooscan dataset and saves the output to given location.

        Args:
            input_path (str): the path to input zooscan dataset.
            output_path (str): the path to output cropped zooscan dataset.

        NOTE:
            the function creates automatically every output folder needed.
        
        NOTE: 
            the function prints to std output the paths of images where it fails.
    """

    Path(output_path).mkdir(exist_ok=True, parents=True)
    classes = [d for d in listdir(input_path) if isdir(join(input_path, d))]

    for c in classes:
        class_path = join(input_path, c)
        class_out_path = join(output_path, c)
        Path(class_out_path).mkdir(exist_ok=True)

        images = [f for f in listdir(class_path) if isfile(join(class_path, f))]

        for im in images:
            image_path = join(class_path, im)
            image_out_path = join(class_out_path, im)

            try:
                output_image = crop_image(image_path)
                cv.imwrite(image_out_path, output_image)
            except Exception as e:
                print(f"Failed crop on image: {image_path}")

import cv2 as cv
import numpy as np
from typing import Optional
from pathlib import Path
from os.path import join, isdir, isfile
from os import listdir


def pad_image_to_square(image_path: str, fill: Optional[int] = 255) -> np.array:
    """
        Loads the image from path and pads it with a constant value (specified by fill
        argument). The original image will be places approximately in the center of the 
        output.

        Args:
            image_path (str): the path to the zooscan image to crop.
            fill (int, optional): fill the padding pixels with this value.

        Returns:
            the padded squared image.
    """
    img = cv.imread(image_path,cv.IMREAD_UNCHANGED)
    height, width = img.shape[0], img.shape[1]

    channels = 1

    if len(img.shape) == 3:
        channels = img.shape[2]

    maximum_value = max([height, width])

    y_from = (maximum_value - height)//2
    x_from = (maximum_value - width)//2

    if channels > 1:
        new_image = np.full([maximum_value, maximum_value, channels], fill_value=fill)
    else:
        new_image = np.full([maximum_value, maximum_value], fill_value=fill)

    # copy old image inside
    new_image[y_from:y_from+height, x_from:x_from+width] = img

    return new_image


def pad_dataset(input_path: str, output_path: str, fill_value: Optional[int] = 255):

    """
        Pads all dataset images to squares with the value passed as "fill_value".

        Args:
            input_path (str): the path to input dataset.
            output_path (str): the path to output padded dataset.
            fill_value (int, optional): the value to pad images with.
        NOTE:
            the function creates automatically every output folder needed.
        
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

            ####
            image_out_path = image_out_path.split(".")[0] + ".jpg"
            print(image_out_path)
            ####

            try:
                output_image = pad_image_to_square(image_path, fill=fill_value)
                cv.imwrite(image_out_path, output_image)
            except Exception as e:
                print(f"Failed crop on image: {image_path}")
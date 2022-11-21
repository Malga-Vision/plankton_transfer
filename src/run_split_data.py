from os.path import join, isfile, isdir
from os import listdir
from shutil import move
import random
from pathlib import Path


def main(in_path, out_path):

    classes = [d for d in listdir(in_path) if isdir(join(in_path, d))]
    for class_name in classes:
        class_path = join(in_path, class_name)
        images = [i for i in listdir(class_path) if isfile(join(class_path, i))]
        random.shuffle(images)

        n_images = len(images)
        n_test_images = int(0.20 * n_images)
        test_images = images[0:n_test_images]
        
        for image in test_images:
            image_path = join(class_path, image)
            output_dir = join(out_path, class_name)
            Path(output_dir).mkdir(exist_ok=True, parents=True)
            move(image_path, join(output_dir, image))


if __name__ == "__main__":
    main("datasets/zooscan20_padded/", "datasets/zooscan20_padded_test")

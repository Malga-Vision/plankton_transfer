import argparse
from dataset import crop_zooscan

parser = argparse.ArgumentParser(description='Crop Zooscan Parser')
parser.add_argument('--input',type=str, default="datasets/zooscan20/")
parser.add_argument('--output',type=str, default="datasets/zooscan20_cropped/")


def main(input: str, output: str):
    crop_zooscan(input, output)

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
import argparse
from dataset import pad_dataset
from os.path import join

parser = argparse.ArgumentParser(description='Pad dataset Parser')
parser.add_argument('--root', type=str, default="datasets/")
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--fill', type=int, default=255)

def main(root: str, input: str, output: str, fill: int):
    input_path = join(root, input)
    output_path = join(root, output)
    pad_dataset(input_path, output_path, fill_value=fill)

if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)

import argparse
import timm

parser = argparse.ArgumentParser(description='Download architecture pretrained')
parser.add_argument('--name', type=str)

def main(name):
    timm.create_model(name, pretrained=True)
    
if __name__=='__main__':
    args = vars(parser.parse_args())
    main(**args)

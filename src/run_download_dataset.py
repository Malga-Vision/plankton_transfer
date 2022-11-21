from dataset import (download_and_setup_whoi, 
                     download_and_setup_whoi22, 
                     download_and_setup_zooscan,
                     download_and_setup_kaggle_zooscan20,
                     download_and_setup_zoolake)

def main():
    download_and_setup_whoi()
    download_and_setup_whoi22()
    download_and_setup_zooscan()
    download_and_setup_kaggle_zooscan20()
    download_and_setup_zoolake()


if __name__ == "__main__":
    main()
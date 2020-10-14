import argparse
from preprocess import BaeminDataset

def set_argument():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="./data", type=str, help="Training dataset directory")
    parser.add_argument("--logger", default=False, type=bool, help="Logger Type")
    parser.add_argument("--filepath", default="sample_data.csv", type=str, help="File Path")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = set_argument()
    order = BaeminDataset(args)
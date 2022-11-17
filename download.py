import argparse
from utils.downloads import download_data, download_mlruns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vehicle Color Classification Setup")
    parser.add_argument(
        "--download-data",
        type=bool,
        default=False,
        metavar="DD",
        help="Command to help whether you want to download dataset or not (default: True)"
    )
    parser.add_argument(
        "--download-mlruns",
        type=bool,
        default=False,
        metavar="DM",
        help="Command to help whether you want to download mlflow results or not (default: False)"
    )
    args = parser.parse_args()
    if args.download_data:
        print("Preparing download...")
        download_data()
    if args.download_mlruns:
        download_mlruns()

import os
import gdown
import zipfile
import patoolib
import logging


def download_data(dir_name: str = "data") -> None:
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)
    logging.info("Downloading data....")
    # gdown.download(
    #     "https://drive.google.com/uc?id=1n8Ja6g5eO82mbRlsTXkdVXNMPpApLd5K", quiet=False
    # )
    logging.info("Extracting zip file....")
    # with zipfile.ZipFile("./color.rar", 'r') as zip_ref:
    #     zip_ref.extractall("data")
    patoolib.extract_archive("./color.rar", outdir="./")
    os.remove("./color.rar")
    os.chdir("..")


def download_mlruns() -> None:
    logging.info("Downloading data....")
    gdown.download(
        "https://drive.google.com/uc?id=1yMkr0ABnUK3yNT3u5TNMvMQwYeK7iBwN", quiet=False
    )
    logging.info("Extracting zip file....")
    with zipfile.ZipFile("mlruns.zip", 'r') as zip_ref:
        zip_ref.extractall("")
    os.remove("mlruns.zip")

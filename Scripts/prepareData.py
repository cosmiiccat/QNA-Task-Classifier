import json

from config import (
    qasper_classification_path
)


def prepareData(data_path):

    data_fd = open(data_path)
    data = json.load(data_fd)

    print(data.keys())


def __run__():
    prepareData(qasper_classification_path)

if __name__ == "__main__":
    __run__()



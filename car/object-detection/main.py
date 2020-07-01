import pandas as pd
import numpy as np
import os


def read_csv_to_pkl():
    annotations = pd.read_csv(
        "/home/asventon/datasets/open_images/oidv6-train-annotations-bbox.csv")
    annotations.to_pickle(
        "/home/asventon/datasets/open_images/oidv6-train-annotations-bbox.pkl")


def remove_null_images():
    annotations = pd.read_pickle(
        "/home/asventon/datasets/open_images/oidv6-train-annotations-bbox.pkl")

    image_files = np.array(os.listdir(
        "/home/asventon/datasets/open_images/train_00/"))
    image_file_nums = []
    for i in range(len(image_files)):
        image_files[i] = image_files[i].split('.')[0]
        image_file_nums.append(int(image_files[i], 16))

    image_file_nums.sort()
    # print(image_file_nums)

    non_images = []

    for i, v in enumerate(annotations['ImageID']):
        if i % 1000000 == 0:
            print("i: {}".format(i))
        if int(v, 16) < image_file_nums[0] or int(v, 16) > image_file_nums[len(
                                                         image_file_nums)-1]:
            # if v in image_files:
            non_images.append(i)
    # print(non_images)
    print(len(annotations))
    annotations = annotations.drop(non_images)
    print(len(annotations))

    annotations.to_pickle(
        "/home/asventon/datasets/\
open_images/oidv6-train-annotations-bbox-cut.pkl")


remove_null_images()

annotations = pd.read_pickle(
    "/home/asventon/datasets/open_images/oidv6-train-annotations-bbox-cut.pkl")

print(len(annotations))

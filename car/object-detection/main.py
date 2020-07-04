import pandas as pd
import numpy as np
import os
from PIL import Image, ImageDraw

descriptions = {}
with open("/home/asventon/datasets/open_images/\
class-descriptions-boxable.csv") as f:
    t = f.read()
    ts = t.split("\n")
    for v in ts:
        if(v != ""):
            vs = v.split(',')
            descriptions[vs[0]] = vs[1]


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
        if i % 10000 == 0:
            print("i: {}".format(i))
        # if
        # int(v, 16) < image_file_nums[0] or int(v, 16) > image_file_nums[len(
        #                                                 image_file_nums)-1]:
        if v not in image_files:
            non_images.append(i)
    # print(non_images)
    print(len(annotations))
    annotations = annotations.drop(non_images)
    print(len(annotations))

    annotations.to_pickle(
        "/home/asventon/datasets/\
open_images/oidv6-train-annotations-bbox-cut.pkl")


def get_person_boxes():
    annotations = pd.read_pickle(
        "/home/asventon/datasets/open_images/o\
idv6-train-annotations-bbox-cut.pkl")

    p_anno = pd.DataFrame(columns=['ImageID', 'XMin', 'XMax', 'YMin', 'YMax'])

    i = 0
    fi = 0
    for v2 in annotations.iterrows():
        if(i % 10000 == 0):
            print("i: {}".format(i))
        if(i == 50000):
            break
        v = {
            "LabelName": v2[1][2],
            "ImageID": v2[1][0],
            "XMin": v2[1][4],
            "XMax": v2[1][5],
            "YMin": v2[1][6],
            "YMax": v2[1][7],
        }
        if(descriptions[v['LabelName']] == "Person"):
            if(len(p_anno.index) == 0):
                v.pop("LabelName")
                p_anno = p_anno.append(pd.DataFrame(v, index=[fi]))
                fi += 1
                continue
            if(p_anno.tail(1)['ImageID'].values[0] != v['ImageID']):
                v.pop("LabelName")
                p_anno = p_anno.append(pd.DataFrame(v, index=[fi]))
                fi += 1
                continue
            if(((v['XMax']-v['XMin']) * (v['YMax']-v['YMin'])) >
               ((p_anno.tail(1)['XMax'].values[0] -
                p_anno.tail(1)['XMin'].values[0]) *
                (p_anno.tail(1)['YMax'].values[0] -
                 p_anno.tail(1)['YMin'].values[0]))):
                v.pop("LabelName")
                p_anno = p_anno.append(pd.DataFrame(v, index=[fi]))
                fi += 1
        i += 1
    p_anno.to_pickle("person_boxes.pkl")
    print(p_anno)


def show_data():
    annotations = pd.read_pickle("person_boxes.pkl")
    image_files = np.array(os.listdir(
        "/home/asventon/datasets/open_images/train_00/"))
    for i in range(100):
        img = Image.open(
            "/home/asventon/datasets/open_images/train_00/"+image_files[i])

        draw = ImageDraw.Draw(img)
        xi = annotations['XMin'][i]*img.size[0]
        xa = annotations['XMax'][i]*img.size[0]
        yi = annotations['YMin'][i]*img.size[1]
        ya = annotations['YMax'][i]*img.size[1]
        draw.line((xi, yi, xi, ya), width=10, fill=0xffffff)
        draw.line((xa, yi, xa, ya), width=10, fill=0xffffff)
        draw.line((xi, yi, xa, yi), width=10, fill=0xffffff)
        draw.line((xi, ya, xa, ya), width=10, fill=0xffffff)

        img.save("outputs/output{}.png".format(i))

    os.system("feh outputs")


# get_person_boxes()
# show_data()


def show_full_data():
    annotations = pd.read_pickle("/home/asventon/datasets/\
open_images/oidv6-train-annotations-bbox-cut.pkl")
    image_files = np.array(os.listdir(
        "/home/asventon/datasets/open_images/train_00/"))
    for i in range(len(image_files)):
        image_files[i] = image_files[i].split(".")[0]
    for i in range(100):
        if(i % 10 == 0):
            print("i: " + str(i))
        img = Image.open(
         "/home/asventon/datasets/open_images/train_00/"+image_files[i]+".jpg")
        draw = ImageDraw.Draw(img)
        c_image_files =\
            annotations[annotations['ImageID'] == image_files[i]]
        # print(c_image_files)
        for c_box in c_image_files.iterrows():
            # print(c_box)
            cd_box = {
                "ImageID": c_box[1][0],
                "XMin": c_box[1][4],
                "XMax": c_box[1][5],
                "YMin": c_box[1][6],
                "YMax": c_box[1][7],
            }
            xi = cd_box['XMin']*img.size[0]
            xa = cd_box['XMax']*img.size[0]
            yi = cd_box['YMin']*img.size[1]
            ya = cd_box['YMax']*img.size[1]
            draw.line((xi, yi, xi, ya), width=10, fill=0xffffff)
            draw.line((xa, yi, xa, ya), width=10, fill=0xffffff)
            draw.line((xi, yi, xa, yi), width=10, fill=0xffffff)
            draw.line((xi, ya, xa, ya), width=10, fill=0xffffff)
        img.save("outputs/output{}.png".format(i))

    os.system("feh outputs")


show_full_data()

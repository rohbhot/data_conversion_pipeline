# column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'type']

import os
import os.path as osp
import shutil
import logging

import json
import pandas as pd

from .utils import file_exists


coco = {"images": [], "type": "instances", "annotations": [], "categories": []}

category_set = dict()
indices = {}
image_mapping = {}

category_item_id = 0
image_id = 20180000000
annotation_id = 0
not_found = []


def reset():
    coco["images"] = []
    coco["annotations"] = []


def addCatItem(name):
    global category_item_id
    if name in category_set:
        return category_set[name]

    category_item = dict()
    category_item["supercategory"] = "none"
    category_item_id += 1
    category_item["id"] = category_item_id
    category_item["name"] = name
    coco["categories"].append(category_item)
    category_set[name] = category_item_id
    indices[name] = category_item_id
    return category_item_id


def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception("Could not find filename tag in xml file.")
    if size["width"] is None:
        raise Exception("Could not find width tag in xml file.")
    if size["height"] is None:
        raise Exception("Could not find height tag in xml file.")
    image_id += 1
    image_item = dict()
    image_item["id"] = image_id
    image_item["file_name"] = file_name
    image_item["width"] = size["width"]
    image_item["height"] = size["height"]
    coco["images"].append(image_item)
    image_mapping[file_name] = image_id
    return image_id


def addAnnoItem(image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item["segmentation"] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item["segmentation"].append(seg)

    annotation_item["area"] = bbox[2] * bbox[3]
    annotation_item["iscrowd"] = 0
    annotation_item["ignore"] = 0
    annotation_item["image_id"] = image_id
    annotation_item["bbox"] = bbox
    annotation_item["category_id"] = category_id
    annotation_id += 1
    annotation_item["id"] = annotation_id
    coco["annotations"].append(annotation_item)


def process_df(df, type_, image_dirs, output_dir):
    output_image_dir = ["train2017", "val2017", "test2017"][type_]
    output_image_dir = osp.join(output_dir, output_image_dir)
    if not osp.exists(output_image_dir):
        os.makedirs(output_image_dir)

    for _, (
        image_name,
        width,
        height,
        class_,
        xmin,
        ymin,
        xmax,
        ymax,
        type_,
    ) in df.iterrows():
        if image_name not in image_mapping:
            size = {"width": width, "height": height}
            addImgItem(image_name, size)
            output_image_path = osp.join(output_image_dir, image_name)
            file_path = file_exists(image_name, image_dirs)
            if file_path:
                shutil.copyfile(osp.join(file_path), output_image_path)
            else:
                not_found.append(image_name)

        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
        addAnnoItem(image_mapping[image_name], indices[class_], bbox)


def intermediate_to_coco(df, image_dirs, output_dir):
    if isinstance(df, str):
        if osp.exists(df):
            df = pd.read_csv(df)
        else:
            print(f"CSV path {df} does not exist!")
            exit()

    if isinstance(image_dirs, str):
        image_dirs = [image_dirs]

    classes = df["class"].unique()

    if not osp.exists(osp.join(output_dir, "annotations")):
        os.makedirs(osp.join(output_dir, "annotations"))

    json_fnames = {
        0: "instances_train2017.json",
        1: "instances_val2017.json",
        2: "instances_test2017.json",
    }

    for name in classes:
        addCatItem(name)

    for type_ in range(3):
        split_df = df[df["type"] == type_]
        if len(split_df) > 0:
            split_df = split_df.sort_values("filename")
            print("Processing values")
            process_df(split_df, type_, image_dirs, output_dir)
            json_fname = json_fnames[type_]
            json_path = osp.join(output_dir, "annotations", json_fname)
            print(f"Writing to {json_path}")
            json.dump(coco, open(json_path, "w"))
            reset()

    nl = "\n"
    if not_found:
        logging.warning(f"Following file(s) were not found:\n{nl.join(not_found)}")


if __name__ == "__main__":
    intermediate_to_coco(
        "/home/aniket/Documents/Senquire/data_conversion/coco_output/labels.csv",
        [
            "/home/aniket/Documents/Senquire/data_conversion/coco/train2017",
            "/home/aniket/Documents/Senquire/data_conversion/coco/val2017",
            "/home/aniket/Documents/Senquire/data_conversion/coco/test2017",
        ],
        "/home/aniket/Documents/Senquire/data_conversion/coco_from_intermediate",
    )

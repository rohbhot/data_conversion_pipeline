# column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'type']

import os
import os.path as osp
import shutil
import logging
import pandas as pd
from .utils import file_exists


# def object_template(class_, xmin, ymin, xmax, ymax):
#     return f"""
#     <object>
# 		<name>{class_}</name>
# 		<pose>Unspecified</pose>
# 		<truncated>0</truncated>
# 		<difficult>0</difficult>
# 		<bndbox>
# 			<xmin>{xmin}</xmin>
# 			<ymin>{ymin}</ymin>
# 			<xmax>{xmax}</xmax>
# 			<ymax>{ymax}</ymax>
# 		</bndbox>
# 	</object>
# """

# def template(folder_name, file_path, width, height, object_list):
#     file_name = file_path.split('/')[-1]
#     obj_str = '\n'.join(object_list)
#     return f"""<annotation>
# 	<folder>{folder_name}</folder>
# 	<filename>{file_name}</filename>
# 	<path>{file_path}</path>
# 	<source>
# 		<database>Unknown</database>
# 	</source>
# 	<size>
# 		<width>{width}</width>
# 		<height>{height}</height>
# 		<depth>3</depth>
# 	</size>
# 	<segmented>0</segmented>
# {obj_str}
# </annotation>
# """
# #
# def append_split(output_dir, image_name, type_):
#     fname = ['train.txt','val.txt','test.txt'][type_]
#     image_name = '.'.join(image_name.split('.')[:-1])
#     with open(osp.join(output_dir, fname), 'a') as f:
#         f.write(image_name + '\n')


def intermediate_to_voc(df, image_dirs, output_dir):
    if isinstance(df, str):
        if osp.exists(df):
            df = pd.read_csv(df)
        else:
            logging.error(f"Dataframe path {df} does not exist!")
            return

    if isinstance(image_dirs, str):
        image_dirs = [image_dirs]

    df = df.sort_values("filename")

    output_image_dir = osp.join(output_dir, "VOCdevkit", "VOC2007", "JPEGImages")
    annotations_dir = osp.join(output_dir, "VOCdevkit", "VOC2007", "Annotations")
    split_dir = osp.join(output_dir, "VOCdevkit", "VOC2007", "ImageSets", "Main")

    required_dirs = [output_image_dir, annotations_dir, split_dir]

    for dir_name in required_dirs:
        if not osp.exists(osp.join(output_dir, dir_name)):
            os.makedirs(osp.join(output_dir, dir_name))

    curr_name = df.iloc[0]["filename"]
    objects = []
    not_found = []
    prev = None
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
        if curr_name != image_name:
            width, height, type_ = prev
            output_image_path = osp.join(output_image_dir, curr_name)
            xml_path = osp.join(
                annotations_dir, f"{'.'.join(curr_name.split('.')[:-1])}.xml"
            )
            append_split(split_dir, curr_name, type_)
            file_path = file_exists(curr_name, image_dirs)
            if file_path:
                shutil.copyfile(file_path, output_image_path)
            else:
                not_found.append(curr_name)
            xml_str = template(output_dir, output_image_path, width, height, objects)
            with open(xml_path, "w") as f:
                f.write(xml_str)
            objects = []
            curr_name = image_name

        prev = (width, height, type_)
        objects.append(object_template(class_, xmin, ymin, xmax, ymax))

    width, height, type_ = prev
    output_image_path = osp.join(output_image_dir, curr_name)
    xml_path = osp.join(annotations_dir, f"{'.'.join(curr_name.split('.')[:-1])}.xml")
    append_split(split_dir, curr_name, type_)
    file_path = file_exists(curr_name, image_dirs)

    if file_path:
        shutil.copyfile(file_path, output_image_path)
    else:
        not_found.append(curr_name)

    xml_str = template(output_dir, output_image_path, width, height, objects)

    with open(xml_path, "w") as f:
        f.write(xml_str)

    nl = "\n"
    if not_found:
        logging.warning(f"Following file(s) were not found:\n{nl.join(not_found)}")


if __name__ == "__main__":
    intermediate_to_voc(
        "intermediate_outputs/labels.csv",
        "VOCdevkit/VOC2007/JPEGImages",
        "voc_from_intermediate",
    )

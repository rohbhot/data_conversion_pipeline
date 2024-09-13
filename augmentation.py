import os
import cv2
import xml.etree.ElementTree as ET
import uuid


def generate_uuid():
    # Generate a UUID to add to the file names
    return str(uuid.uuid4().hex)


def flip_horizontal(image_path, annotation_path):
    image = cv2.imread(image_path)
    flipped_image = cv2.flip(image, 1)
    flipped_annotation = flip_xml_horizontal(annotation_path, image.shape[1])

    unique_id = generate_uuid()
    image_extension = os.path.splitext(image_path)[1]
    annotation_extension = os.path.splitext(annotation_path)[1]
    new_image_filename = unique_id + image_extension
    new_image_path = os.path.join(os.path.dirname(image_path), new_image_filename)
    new_annotation_path = os.path.join(os.path.dirname(annotation_path), unique_id + annotation_extension)

    cv2.imwrite(new_image_path, flipped_image)
    save_xml_annotation(flipped_annotation, new_annotation_path, new_image_filename)
    
    
def flip_xml_horizontal(annotation_path, dimension):
    # import pdb; pdb.set_trace();
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = bbox.find("xmin")
        xmax = bbox.find("xmax")
        new_xmin = str(dimension - int(xmax.text))
        new_xmax = str(dimension - int(xmin.text))
        xmin.text = new_xmin
        xmax.text = new_xmax
    return tree
    

def flip_vertical(image_path, annotation_path):
    # import pdb; pdb.set_trace();
    image = cv2.imread(image_path)
    flipped_image = cv2.flip(image, 0)
    flipped_annotation = flip_xml_vertical(annotation_path, image.shape[0])

    unique_id = generate_uuid()
    image_extension = os.path.splitext(image_path)[1]
    annotation_extension = os.path.splitext(annotation_path)[1]
    new_image_filename = unique_id + image_extension
    new_image_path = os.path.join(os.path.dirname(image_path), new_image_filename)
    new_annotation_path = os.path.join(os.path.dirname(annotation_path), unique_id + annotation_extension)

    cv2.imwrite(new_image_path, flipped_image)
    save_xml_annotation(flipped_annotation, new_annotation_path, new_image_filename)


def flip_xml_vertical(annotation_path, image_height):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    for obj in root.findall("object"):
        for bbox in obj.findall("bndbox"):
            xmin = int(bbox.find("xmin").text)
            xmax = int(bbox.find("xmax").text)
            ymin = int(bbox.find("ymin").text)
            ymax = int(bbox.find("ymax").text)

            new_xmin = xmin
            new_xmax = xmax
            new_ymin = image_height - ymax
            new_ymax = image_height - ymin

            bbox.find("xmin").text = str(new_xmin)
            bbox.find("xmax").text = str(new_xmax)
            bbox.find("ymin").text = str(new_ymin)
            bbox.find("ymax").text = str(new_ymax)

    return tree

def save_xml_annotation(annotation_tree, annotation_path, image_filename):
    root = annotation_tree.getroot()
    filename_element = root.find("filename")
    filename_element.text = image_filename

    annotation_tree.write(annotation_path)

    
def example_function(image_path, annotation_path, output_folder):
    # Example function, replace with your own implementation
    pass



'''
import os
import cv2
import xml.etree.ElementTree as ET
import uuid
import csv


def generate_uuid():
    # Generate a UUID to add to the file names
    return str(uuid.uuid4().hex)


def flip_horizontal(image_path, annotation_path):
    image = cv2.imread(image_path)
    flipped_image = cv2.flip(image, 1)
    flipped_annotation = flip_xml_annotation(annotation_path, image.shape[1])

    unique_id = generate_uuid()
    image_extension = os.path.splitext(image_path)[1]
    annotation_extension = os.path.splitext(annotation_path)[1]
    new_image_filename = unique_id + image_extension
    new_image_path = os.path.join(os.path.dirname(image_path), new_image_filename)
    new_annotation_path = os.path.join(os.path.dirname(annotation_path), unique_id + annotation_extension)

    cv2.imwrite(new_image_path, flipped_image)
    save_xml_annotation(flipped_annotation, new_annotation_path, new_image_filename)



def flip_xml_annotation(annotation_path, image_width):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = bbox.find("xmin")
        xmax = bbox.find("xmax")
        xmin.text = str(image_width - int(xmin.text))
        xmax.text = str(image_width - int(xmax.text))
    return tree


def save_xml_annotation(annotation_tree, annotation_path, image_filename):
    root = annotation_tree.getroot()
    filename_element = root.find("filename")
    filename_element.text = image_filename

    annotation_tree.write(annotation_path)



def write_csv(data, output_path):
    with open(output_path, "w", newline="") as file:
        import pdb; pdb.set_trace();
        writer = csv.writer(file)
        writer.writerows(data)


def read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        data.append([xmin, ymin, xmax, ymax])

    return data


def example_function(image_path, annotation_path):
    # Example function, replace with your own implementation
    pass
'''


'''
import os
import cv2
import xml.etree.ElementTree as ET
import uuid

def flip_horizontal(image_path, annotation_path):
    image = cv2.imread(image_path)
    flipped_image = cv2.flip(image, 1)
    flipped_annotation = flip_xml_annotation(annotation_path, image.shape[1])

    # Generate a UUID to add to the file names
    unique_id = str(uuid.uuid4().hex)
    image_extension = os.path.splitext(image_path)[1]
    annotation_extension = os.path.splitext(annotation_path)[1]
    new_image_path = os.path.join(os.path.dirname(image_path), unique_id + image_extension)
    new_annotation_path = os.path.join(os.path.dirname(annotation_path), unique_id + annotation_extension)

    cv2.imwrite(new_image_path, flipped_image)
    save_xml_annotation(flipped_annotation, new_annotation_path)

def flip_xml_annotation(annotation_path, image_width):
    # import pdb; pdb.set_trace();
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = bbox.find("xmin")
        xmax = bbox.find("xmax")
        xmin.text = str(image_width - int(xmin.text))
        xmax.text = str(image_width - int(xmax.text))
    return tree

def save_xml_annotation(annotation_tree, annotation_path):
    # import pdb; pdb.set_trace();
    annotation_tree.write(annotation_path)

def run_augmentation(input_folder, augmentation_type):
    # import pdb; pdb.set_trace();
    images = [
        f
        for f in os.listdir(input_folder)
        if f.endswith((".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP"))
    ]

    for image_file in images:
        image_path = os.path.join(input_folder, image_file)
        annotation_file = os.path.splitext(image_file)[0] + ".xml"
        annotation_path = os.path.join(input_folder, annotation_file)

        if augmentation_type == "flip_horizontal":
            flip_horizontal(image_path, annotation_path)
        # Add other augmentation functions here (e.g., rotation, vertical flip)
# 
if __name__ == "__main__":
    
    import argparse
    
    import pdb;pdb.set_trace();
    parser = argparse.ArgumentParser(description="Image augmentation")
    parser.add_argument("--input-folder", type=str, help="Path to the folder containing images and annotations")
    parser.add_argument("--aug-type", type=str, help="Type of augmentation to apply (e.g., flip_horizontal)")
    args = parser.parse_args()

    run_augmentation(args.input_folder, args.aug_type)
'''

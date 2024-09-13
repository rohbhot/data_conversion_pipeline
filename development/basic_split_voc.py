        
'''
import os
import shutil
from coco_converter.data_converter import DataConverter

    
# Define the source directories

train_dir = '/mnt/2tb/General/Anand/split_images/data_conversion_pipeline/output/train/'  # Replace with the path to your train directory
test_dir = '/mnt/2tb/General/Anand/split_images/data_conversion_pipeline/output/test/'  # Replace with the path to your test directory
val_dir = '/mnt/2tb/General/Anand/split_images/data_conversion_pipeline/output/val/'  # Replace with the path to your val directory

# Define the destination directory where the instances will be created
destination_directory = '/mnt/2tb/General/Anand/split_images/data_conversion_pipeline/output/'

# Define the names of the three instances
instances = ['VOCdevkit_train', 'VOCdevkit_val', 'VOCdevkit_test']

# Process each instance
for instance in instances:
    # Create the instance-specific destination directory
    instance_directory = os.path.join(destination_directory, instance, 'VOC2007')
    os.makedirs(instance_directory, exist_ok=True)

    # Move the train folder files to VOCdevkit_train
    if instance == 'VOCdevkit_train':
        train_jpeg_dir = os.path.join(instance_directory, 'JPEGImages')
        os.makedirs(train_jpeg_dir, exist_ok=True)

        train_annotations_dir = os.path.join(instance_directory, 'Annotations')
        os.makedirs(train_annotations_dir, exist_ok=True)

        os.makedirs(os.path.join(instance_directory, 'ImageSets', 'Main'), exist_ok=True)

        if os.path.exists(train_dir):
            for file_name in os.listdir(train_dir):
                if file_name.endswith('.jpg'):
                    shutil.move(os.path.join(train_dir, file_name), os.path.join(train_jpeg_dir, file_name))
                elif file_name.endswith('.xml'):
                    shutil.move(os.path.join(train_dir, file_name), os.path.join(train_annotations_dir, file_name))
        else:
            print(f"Error: Source directory '{train_dir}' does not exist.")

    # Move the test folder files to VOCdevkit_test
    elif instance == 'VOCdevkit_test':
        test_jpeg_dir = os.path.join(instance_directory, 'JPEGImages')
        os.makedirs(test_jpeg_dir, exist_ok=True)

        test_annotations_dir = os.path.join(instance_directory, 'Annotations')
        os.makedirs(test_annotations_dir, exist_ok=True)

        os.makedirs(os.path.join(instance_directory, 'ImageSets', 'Main'), exist_ok=True)

        if os.path.exists(test_dir):
            for file_name in os.listdir(test_dir):
                if file_name.endswith('.jpg'):
                    shutil.move(os.path.join(test_dir, file_name), os.path.join(test_jpeg_dir, file_name))
                elif file_name.endswith('.xml'):
                    shutil.move(os.path.join(test_dir, file_name), os.path.join(test_annotations_dir, file_name))

    # Move the val folder files to VOCdevkit_val
    elif instance == 'VOCdevkit_val':
        val_jpeg_dir = os.path.join(instance_directory, 'JPEGImages')
        os.makedirs(val_jpeg_dir, exist_ok=True)

        val_annotations_dir = os.path.join(instance_directory, 'Annotations')
        os.makedirs(val_annotations_dir, exist_ok=True)

        os.makedirs(os.path.join(instance_directory, 'ImageSets', 'Main'), exist_ok=True)

        if os.path.exists(val_dir):
            for file_name in os.listdir(val_dir):
                if file_name.endswith('.jpg'):
                    shutil.move(os.path.join(val_dir, file_name), os.path.join(val_jpeg_dir, file_name))
                elif file_name.endswith('.xml'):
                    shutil.move(os.path.join(val_dir, file_name), os.path.join(val_annotations_dir, file_name))




converter = DataConverter()
data_dirs = ['/mnt/2tb/General/Anand/split_images/data_conversion_pipeline/output/VOCdevkit_train/', '/mnt/2tb/General/Anand/split_images/data_conversion_pipeline/output/VOCdevkit_test/', '/mnt/2tb/General/Anand/split_images/data_conversion_pipeline/output/VOCdevkit_val/']
output_dirs = ['/mnt/2tb/General/Anand/split_images/data_conversion_pipeline/output/coco_train/', '/mnt/2tb/General/Anand/split_images/data_conversion_pipeline/output/coco_test/', '/mnt/2tb/General/Anand/split_images/data_conversion_pipeline/output/coco_val/']
for input_path, output_path in zip(data_dirs, output_dirs):
    input_type = "voc"
    output_type = "coco"

    if input_path and output_path and input_type and output_type:
        converter.convert(
            input_path=input_path,
            output_path=output_path,
            input_type=input_type,
            output_type=output_type,
        )
    else:
        print("Please pass all the required arguments")
'''

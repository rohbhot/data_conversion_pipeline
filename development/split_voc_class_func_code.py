import os
import shutil

class DataConversionPipeline:
    def __init__(self, train_dir, test_dir, val_dir, destination_directory):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir
        self.destination_directory = destination_directory

        # Define the names of the three instances
        self.instances = ['VOCdevkit_train', 'VOCdevkit_val', 'VOCdevkit_test']

    def create_instance_directory(self, instance_directory, create_image_sets=False):
        os.makedirs(instance_directory, exist_ok=True)
        if create_image_sets:
            os.makedirs(os.path.join(instance_directory, 'VOC2007', 'ImageSets', 'Main'), exist_ok=True)

    def move_files(self, source_dir, destination_dir, file_extension):
        if os.path.exists(source_dir):
            for file_name in os.listdir(source_dir):
                if file_name.endswith(file_extension):
                    shutil.move(
                        os.path.join(source_dir, file_name),
                        os.path.join(destination_dir, file_name)
                    )
        else:
            print(f"Error: Source directory '{source_dir}' does not exist.")

    def process_instances(self):
        # Process each instance
        for instance in self.instances:
            # Create the instance-specific destination directory
            instance_directory = os.path.join(self.destination_directory, instance)
            create_image_sets = instance in ['VOCdevkit_train', 'VOCdevkit_val', 'VOCdevkit_test']
            self.create_instance_directory(instance_directory, create_image_sets)

            # Move the train folder files to VOCdevkit_train
            if instance == 'VOCdevkit_train':
                train_jpeg_dir = os.path.join(instance_directory, 'VOC2007', 'JPEGImages')
                train_annotations_dir = os.path.join(instance_directory, 'VOC2007', 'Annotations')
                self.create_instance_directory(train_jpeg_dir)
                self.create_instance_directory(train_annotations_dir)
                self.move_files(self.train_dir, train_jpeg_dir, '.jpg')
                self.move_files(self.train_dir, train_annotations_dir, '.xml')

            # Move the test folder files to VOCdevkit_test
            elif instance == 'VOCdevkit_test':
                test_jpeg_dir = os.path.join(instance_directory, 'VOC2007', 'JPEGImages')
                test_annotations_dir = os.path.join(instance_directory, 'VOC2007', 'Annotations')
                self.create_instance_directory(test_jpeg_dir)
                self.create_instance_directory(test_annotations_dir)
                self.move_files(self.test_dir, test_jpeg_dir, '.jpg')
                self.move_files(self.test_dir, test_annotations_dir, '.xml')

            # Move the val folder files to VOCdevkit_val
            elif instance == 'VOCdevkit_val':
                val_jpeg_dir = os.path.join(instance_directory, 'VOC2007', 'JPEGImages')
                val_annotations_dir = os.path.join(instance_directory, 'VOC2007', 'Annotations')
                self.create_instance_directory(val_jpeg_dir)
                self.create_instance_directory(val_annotations_dir)
                self.move_files(self.val_dir, val_jpeg_dir, '.jpg')
                self.move_files(self.val_dir, val_annotations_dir, '.xml')

# Usage:
train_dir = '/mnt/2tb/General/Anand/split_images/out_split/train/'
test_dir = '/mnt/2tb/General/Anand/split_images/out_split/test/'
val_dir = '/mnt/2tb/General/Anand/split_images/out_split/val/'
destination_directory = '/mnt/2tb/General/Anand/split_images/data_conversion_pipeline'

pipeline = DataConversionPipeline(train_dir, test_dir, val_dir, destination_directory)
pipeline.process_instances()


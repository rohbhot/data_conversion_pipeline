import logging
import argparse


class DataConverter:
    def __init__(self):
        self.__class_names = {"coco": self.get_coco, "voc": self.get_voc}
        self.valid_types = list(self.__class_names.keys())
        print(self.valid_types)

    def get_coco(self):
        from .coco import COCOProcessor

        return COCOProcessor()

    def get_voc(self):
        from .voc import VOCProcessor

        return VOCProcessor()

    def convert(
        self, input_type, output_type, input_path, output_path, intermediate_path=None
    ):
        """
        Valid types: yolo, coco, kitti, voc, tfrecord, intermediate
        """
        if input_type in self.valid_types and output_type in self.valid_types:

            source_obj = self.__class_names[input_type]()
            print(self.__class_names)
            if source_obj.check_dir_structure(input_path):
                df, image_dirs = source_obj.to_intermediate(
                    input_path, write_path=intermediate_path
                )
                destination_obj = self.__class_names[output_type]()
                destination_obj.from_intermediate(df, image_dirs, output_path)

        else:
            logging.error(f"types must be one of {self.valid_types}")


if __name__ == "__main__":
    converter = DataConverter()

    parser = argparse.ArgumentParser(
        description="Convert data from one format to the other"
    )
    parser.add_argument(
        "-ip",
        "--input-path",
        help="Path to data root",
        default=False
        # required=True
    )
    parser.add_argument(
        "-op",
        "--output-path",
        help="Path to output directory",
        default=False
        # required=True
    )
    parser.add_argument(
        "-it",
        "--input-type",
        help=f"Type of input, one of {converter.valid_types}",
        default=False
        # required=True
    )
    parser.add_argument(
        "-ot",
        "--output-type",
        # help=f"Type of output, one of {[x in converter.valid_types if 'intermediate' not in x.lower()]}",
        help=f"Type of output, one of {converter.valid_types[:-1]}",
        default=False
        # required=True
    )

    args = parser.parse_args()

    if args.input_path and args.output_path and args.input_type and args.output_type:
        converter.convert(
            input_path=args.input_path,
            output_path=args.output_path,
            input_type=args.input_type,
            output_type=args.output_type,
        )
    else:
        print("Please pass all the required arguments")

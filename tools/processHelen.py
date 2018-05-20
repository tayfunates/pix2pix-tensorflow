import argparse
import os
import numpy as np
import tfimage as im

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing images")
parser.add_argument("--label_images_dir", required=True, help="path to folder containing labels inside the face")
parser.add_argument("--output_dir", required=True, help="output path")
parser.add_argument("--labels", required=True, help="output labels with comma separation. 00 and 01 are musts. e.g. 00,01,04,07")

#Resizing operation parameters
parser.add_argument("--resize", action="store_true", help="decide whether or not to resize the input and the label images")
parser.add_argument("--size", type=int, default=256, help="size to use for resize operation")

#Label parameters
parser.add_argument("--label_cut_threshold", type=int, default=128, help="threshold for converting grayscale label images to binary ones")

#Label combine settings
parser.add_argument("--combine_lips", action="store_true", help="combine lips and inner mouth to a single color if they exist")
parser.add_argument("--combine_eyebrows", action="store_true", help="combine eyebrows to a single color if they exist")
parser.add_argument("--combine_hairs", action="store_false", help="combine hair and eyebrows to a single color if they exist")

a = parser.parse_args()

output_train_directory = os.path.join(a.output_dir, "train")
output_test_directory = os.path.join(a.output_dir, "test")
output_val_directory = os.path.join(a.output_dir, "val")


def main():
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)
    if not os.path.exists(output_train_directory):
        os.makedirs(output_train_directory)
    if not os.path.exists(output_test_directory):
        os.makedirs(output_test_directory)
    if not os.path.exists(output_val_directory):
        os.makedirs(output_val_directory)


    src_paths = []
    dst_paths = []

    skipped = 0
    for src_path in im.find(a.input_dir):
        name, _ = os.path.splitext(os.path.basename(src_path))
        dst_path = os.path.join(a.output_dir, name + ".png")
        if os.path.exists(dst_path):
            skipped += 1
        else:
            src_paths.append(src_path)
            dst_paths.append(dst_path)

    print("skipping %d files that already exist" % skipped)


main()



import argparse
import os
import tempfile
import subprocess
import tensorflow as tf
import numpy as np
import tfimage as im
import threading
import time
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing images")
parser.add_argument("--label_images_dir", required=True, help="path to folder containing labels inside the face")
parser.add_argument("--output_dir", required=True, help="output path")
parser.add_argument("--labels", required=True, help="output labels with comma separation. 00 and 01 are musts. e.g. 00,01,04,07")
parser.add_argument("--workers", type=int, default=1, help="number of workers")

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



def getLabelToColorDictionary():
    colorDict = {}
    colorDict['00'] = [255.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0]
    colorDict['01'] = [0.0 / 255.0, 255.0 / 255.0, 0.0 / 255.0]
    colorDict['02'] = [0.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0]
    colorDict['03'] = [255.0 / 255.0, 102.0 / 255.0, 255.0 / 255.0]
    colorDict['04'] = [255.0 / 255.0, 255.0 / 255.0, 153.0 / 255.0]
    colorDict['05'] = [0.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0]
    colorDict['06'] = [153.0 / 255.0, 76.0 / 255.0, 0.0 / 255.0]
    colorDict['07'] = [0.0 / 255.0, 102.0 / 255.0, 102.0 / 255.0]
    colorDict['08'] = [0.0 / 255.0, 102.0 / 255.0, 255.0 / 255.0]
    colorDict['09'] = [255.0 / 255.0, 204.0 / 255.0, 204.0 / 255.0]
    colorDict['10'] = [202.0 / 255.0, 202.0 / 255.0, 202.0 / 255.0]
    colorDict['11'] = [102.0 / 255.0, 0.0 / 255.0, 204.0 / 255.0]
    return colorDict
    

def getLabelImages(label_folder, labels):
    ret = {}
    labelIds = labels.split(',')
    for lid in labelIds:
        for label_path in im.find(label_folder):
            if label_path.find('lbl'+lid) > 0: #if found the label
                ret[lid] = label_path
                break
    return ret
    
def getLabelImage(label_image_paths, color_dict, cut_threshold):
    res = None
    thresh = cut_threshold / 255.0
    for label_id, label_img_path in label_image_paths.iteritems():
        label_image = im.load(label_img_path)
        print label_img_path
        print label_image.shape
        label_image[label_image >= thresh] = 1.0
        label_image[label_image < thresh] = 0.0
        label_image = np.reshape(label_image, (label_image.shape[0], label_image.shape[1]))
        if res is None:
            res = np.empty((label_image.shape[0], label_image.shape[1], 3))
        res[label_image > 0.5, :] = color_dict[label_id]
    return res
        
complete_lock = threading.Lock()
start = None
num_complete = 0
total = 0

def complete():
    global num_complete, rate, last_complete

    with complete_lock:
        num_complete += 1
        now = time.time()
        elapsed = now - start
        rate = num_complete / elapsed
        if rate > 0:
            remaining = (total - num_complete) / rate
        else:
            remaining = 0

        print("%d/%d complete  %0.2f images/sec  %dm%ds elapsed  %dm%ds remaining" % (num_complete, total, rate, elapsed // 60, elapsed % 60, remaining // 60, remaining % 60))

        last_complete = now
    

def main():
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)
    if not os.path.exists(output_train_directory):
        os.makedirs(output_train_directory)
    if not os.path.exists(output_test_directory):
        os.makedirs(output_test_directory)
    if not os.path.exists(output_val_directory):
        os.makedirs(output_val_directory)

    splits = ['train', 'test', 'val']
    
    src_paths = []
    dst_paths = []
    
    skipped = 0
    for split in splits:
        split_folder = os.path.join(a.input_dir, split)
        for src_path in im.find(split_folder):
    
            name, _ = os.path.splitext(os.path.basename(src_path))
            dst_path = os.path.join(a.output_dir, split)
            dst_path = os.path.join(dst_path, name + ".png")
            
            if os.path.exists(dst_path):
                skipped += 1
            else:
                src_paths.append(src_path)
                dst_paths.append(dst_path)
            
    print("skipping %d files that already exist" % skipped)
    
    global total
    total = len(src_paths)
    
    print("processing %d files" % total)

    global start
    start = time.time()
    
    if a.workers == 1:
        with tf.Session() as sess:
            for src_path, dst_path in zip(src_paths, dst_paths):
            
                name, _ = os.path.splitext(os.path.basename(src_path))
            
                print 'Name: ' + name
            
                label_folder = os.path.join(a.label_images_dir, name)
            
                label_image_paths = getLabelImages(label_folder, a.labels)
            
                print label_image_paths
            
                color_dict = getLabelToColorDictionary()
            
            

                label_img = getLabelImage(label_image_paths, color_dict, a.label_cut_threshold)
                complete()
                
                im.save(label_img, dst_path)
            
                break


main()


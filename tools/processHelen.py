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
import matplotlib
import scipy.misc as sm

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing images")
parser.add_argument("--label_images_dir", required=True, help="path to folder containing labels inside the face")
parser.add_argument("--output_dir_images", required=True, help="output path")
parser.add_argument("--output_dir_labels", required=True, help="output path")
parser.add_argument("--labels", required=True, help="output labels with comma separation. 00 and 01 are musts. e.g. 00,01,04,07")
parser.add_argument("--color_map", required=True, help="Color map png")
parser.add_argument("--workers", type=int, default=1, help="number of workers")

#Resizing operation parameters
parser.add_argument("--resize", action="store_true", help="decide whether or not to resize the input and the label images")
parser.add_argument("--pad", action="store_true", help="pad instead of crop for resize operation")
parser.add_argument("--size", type=int, default=256, help="size to use for resize operation")

#Crop options
parser.add_argument("--crop", action="store_true", help="decide whether or not to crop the input and the label images")

#Label parameters
parser.add_argument("--label_cut_threshold", type=int, default=128, help="threshold for converting grayscale label images to binary ones")

a = parser.parse_args()

output_train_directory_images = os.path.join(a.output_dir_images, "train")
output_test_directory_images = os.path.join(a.output_dir_images, "test")
output_val_directory_images = os.path.join(a.output_dir_images, "val")

output_train_directory_labels = os.path.join(a.output_dir_labels, "train")
output_test_directory_labels = os.path.join(a.output_dir_labels, "test")
output_val_directory_labels = os.path.join(a.output_dir_labels, "val")

def resize(src):
    height, width, _ = src.shape
    dst = src
    if height != width:
        if a.pad:
            size = max(height, width)
            # pad to correct ratio
            oh = (size - height) // 2
            ow = (size - width) // 2
            dst = im.pad(image=dst, offset_height=oh, offset_width=ow, target_height=size, target_width=size)
        else:
            # crop to correct ratio
            size = min(height, width)
            oh = (height - size) // 2
            ow = (width - size) // 2
            dst = im.crop(image=dst, offset_height=oh, offset_width=ow, target_height=size, target_width=size)

    assert(dst.shape[0] == dst.shape[1])

    size, _, _ = dst.shape
    if size > a.size:
        dst = im.downscale(images=dst, size=[a.size, a.size])
    elif size < a.size:
        dst = im.upscale(images=dst, size=[a.size, a.size])
    return dst
    
def crop(src, cropReference):
    rows = np.any(cropReference, axis=1)
    cols = np.any(cropReference, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return src[rmin:rmax, cmin:cmax, :]


def getLabelToColorDictionary():
    colorDict = {}
    
    cmap = matplotlib.colors.ListedColormap(sm.imread(a.color_map)[0].astype(np.float32) / 255.)
    cmap = cmap(np.arange(cmap.N))
    
    #Color settings according to https://github.com/classner/generating_people
    colorDict['00'] = [255.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0]
    colorDict['01'] = cmap[11][:3]
    colorDict['02'] = cmap[11][:3]
    colorDict['03'] = cmap[11][:3]
    colorDict['04'] = cmap[20][:3]
    colorDict['05'] = cmap[21][:3]
    colorDict['06'] = cmap[19][:3]
    colorDict['07'] = cmap[18][:3]
    colorDict['08'] = cmap[18][:3]
    colorDict['09'] = cmap[18][:3]
    colorDict['10'] = [255.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0]
    return colorDict
    

def getLabelImages(label_folder):
    ret = {}
    labelIds = a.labels.split(',')
    for lid in labelIds:
        for label_path in im.find(label_folder):
            if label_path.find('lbl'+lid) > 0: #if found the label
                ret[lid] = label_path
                break
    return ret
    

def thresholdImage(img, thresh):
    img[img >= thresh] = 1.0
    img[img < thresh] = 0.0
    return img
    
    
def getLabelImage(label_image_paths, color_dict):
    res = None
    thresh = a.label_cut_threshold / 255.0
    for label_id, label_img_path in label_image_paths.iteritems():
        label_image = im.load(label_img_path)
        print label_img_path
        print label_image.shape
        
        label_image = thresholdImage(label_image, thresh)

        label_image = np.reshape(label_image, (label_image.shape[0], label_image.shape[1]))
        if res is None:
            res = np.empty((label_image.shape[0], label_image.shape[1], 3))
        res[label_image > 0.5, :] = color_dict[label_id]
    return res
    
def getCropReference(label_image_paths):
    crop_reference = im.load(label_image_paths['01'])
    thresh = a.label_cut_threshold / 255.0
    crop_reference = thresholdImage(crop_reference, thresh)
    return crop_reference
        
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
    if not os.path.exists(a.output_dir_labels):
        os.makedirs(a.output_dir_labels)
    if not os.path.exists(output_train_directory_labels):
        os.makedirs(output_train_directory_labels)
    if not os.path.exists(output_test_directory_labels):
        os.makedirs(output_test_directory_labels)
    if not os.path.exists(output_val_directory_labels):
        os.makedirs(output_val_directory_labels)
        
    processInputImages = a.resize or a.crop
    
    if not os.path.exists(a.output_dir_images) and processInputImages:
        os.makedirs(a.output_dir_images)
    if not os.path.exists(output_train_directory_images) and processInputImages:
        os.makedirs(output_train_directory_images)
    if not os.path.exists(output_test_directory_images) and processInputImages:
        os.makedirs(output_test_directory_images)
    if not os.path.exists(output_val_directory_images) and processInputImages:
        os.makedirs(output_val_directory_images)
        
    #cropped images directory

    splits = ['train', 'test', 'val']
    
    src_paths = []
    dst_paths_labels = []
    dst_paths_images = []
    
    skipped = 0
    for split in splits:
        split_folder = os.path.join(a.input_dir, split)
        for src_path in im.find(split_folder):
    
            name, _ = os.path.splitext(os.path.basename(src_path))
            dst_path_label = os.path.join(a.output_dir_labels, split)
            dst_path_label = os.path.join(dst_path_label, name + ".png")
            dst_path_image = os.path.join(a.output_dir_images, split)
            dst_path_image = os.path.join(dst_path_image, name + ".png")
            
            if os.path.exists(dst_path_label) or os.path.exists(dst_path_image):
                skipped += 1
            else:
                src_paths.append(src_path)
                dst_paths_labels.append(dst_path_label)
                dst_paths_images.append(dst_path_image)
            
    print("skipping %d files that already exist" % skipped)
    
    global total
    total = len(src_paths)
    
    print("processing %d files" % total)

    global start
    start = time.time()
    
    if a.workers == 1:
        with tf.Session() as sess:
            for src_path, dst_path_label, dst_path_image in zip(src_paths, dst_paths_labels, dst_paths_images):
            
                name, _ = os.path.splitext(os.path.basename(src_path))
            
                print 'Name: ' + name
            
                label_folder = os.path.join(a.label_images_dir, name)
            
                label_image_paths = getLabelImages(label_folder)
            
                print label_image_paths
            
                color_dict = getLabelToColorDictionary()
            

                label_img = getLabelImage(label_image_paths, color_dict)
                
                if processInputImages:
                    processedImage = im.load(src_path)
                    
                    if a.crop:
                        crop_reference = getCropReference(label_image_paths)

                        processedImage = crop(processedImage, crop_reference)
                        label_img = crop(label_img, crop_reference)
                        
                        if a.resize:
                            processedImage = resize(processedImage)
                            label_img = resize(label_img)
                        
                    im.save(processedImage, dst_path_image)
                    
                im.save(label_img, dst_path_label)
                complete()


main()


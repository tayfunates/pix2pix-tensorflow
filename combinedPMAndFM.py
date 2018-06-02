import argparse
import os
import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, './tools')
import tfimage as im

import threading
import time
import multiprocessing
import matplotlib
import scipy.misc as sm

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing images generated from portray module (PM)")
parser.add_argument("--output_dir", required=True, help="output path which will contain several subfolders")
parser.add_argument("--pix2pix", required=True, help="script for testing generating face images")
parser.add_argument("--combinescript", required=True, help="script for concatanating two image sets")
parser.add_argument("--fm_model_path", required=True, help="trained face module path")
parser.add_argument("--upscale_interpolation", required=True, help="type of upscale interpolation")
parser.add_argument("--fm_input_size", type=int, default=256, help="size to use for resize operation to input face module")
parser.add_argument("--color_map", required=True, help="color map png")
parser.add_argument("--pad", action="store_true", help="pad instead of crop for resize operation")
parser.add_argument("--preserve_aspect_ratio", action="store_true", help="pad instead of crop for resize operation")

##### TO DO Consider to resize face segmentations by using nearest neighbor interpolation

a = parser.parse_args()

output_types = ['humanseginputs', 'humangeninputs', 'croppedsegfaces', 'croppedgenfaces', 'resizedsegfaces', 'resizedgenfaces', 'combinedseggenfaces', 'generatedfaces']

complete_lock = threading.Lock()
start = None
num_complete = 0
total = 0

COLOR_EPSILON = 1e-3

def addBackground(src, skin_color, positive_colors, background_color):
    res = np.zeros(shape=(src.shape[0], src.shape[1], 3), dtype=float)
    res[:] = (background_color[0], background_color[1], background_color[2])
    res[(abs(src-skin_color)<COLOR_EPSILON).all(2)] = (skin_color[0], skin_color[1], skin_color[2])
    
    for pos_color in positive_colors:
        res[(abs(src-pos_color)<COLOR_EPSILON).all(2)] = (pos_color[0], pos_color[1], pos_color[2])
    
    return res
    
def resize(src, tfUpscaleInterpolationType):
    height, width, _ = src.shape
    dst = src
    if a.preserve_aspect_ratio:
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
                if size > a.fm_input_size:
                    dst = im.downscale(images=dst, size=[a.fm_input_size, a.fm_input_size])
                elif size < a.fm_input_size:
                    if tfUpscaleInterpolationType==tf.image.ResizeMethod.BICUBIC:
                        dst = im.upscale(images=dst, size=[a.fm_input_size, a.fm_input_size])
                    elif tfUpscaleInterpolationType==tf.image.ResizeMethod.NEAREST_NEIGHBOR:
                        dst = im.upscaleWithNearestNeighborInterpolation(images=dst, size=[a.fm_input_size, a.fm_input_size])
    else:
        if tfUpscaleInterpolationType==tf.image.ResizeMethod.BICUBIC:
            dst = im.upscale(images=dst, size=[a.fm_input_size, a.fm_input_size])
        elif tfUpscaleInterpolationType==tf.image.ResizeMethod.NEAREST_NEIGHBOR:
            dst = im.upscaleWithNearestNeighborInterpolation(images=dst, size=[a.fm_input_size, a.fm_input_size])
    
    return dst
    
def getUpscaleInterpolation():
    if a.upscale_interpolation == 'nearest':
        return tf.image.ResizeMethod.NEAREST_NEIGHBOR
    elif a.upscale_interpolation == 'bicubic':
        return tf.image.ResizeMethod.BICUBIC

def crop(src, cropReference):
    try:
        rows = np.any(cropReference, axis=1)
        cols = np.any(cropReference, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
    except IndexError:
        return src
    return src[rmin:rmax, cmin:cmax, :]
    
def getFaceCropReference(src, skin_color):
    cropReference = np.zeros(shape=(src.shape[0], src.shape[1], 3), dtype=int)
    cropReference[(abs(src-skin_color)<COLOR_EPSILON).all(2)] = (1, 1, 1)
    return cropReference

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
    
color_dict = getLabelToColorDictionary()
    
def getSkinColor():
    return color_dict['01']
    
def getBackgroundColor():
    return color_dict['00']
    
def getPositiveColors():
    ret = []
    ret.append(color_dict['04'])
    ret.append(color_dict['05'])
    ret.append(color_dict['06'])
    ret.append(color_dict['07'])
    
    return ret

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
        
def combineFaceSegFaceGen(inputDirA, inputDirB, outputDir):
    command = 'python'
    command += ' '
    command += a.combinescript
    command += ' '
    command += '--input_dir'
    command += ' '
    command += inputDirA
    command += ' '
    command += '--b_dir'
    command += ' '
    command += inputDirB
    command += ' '
    command += '--operation combine'
    command += ' '
    command += '--output_dir'
    command += ' '
    command += outputDir
    os.system(command)
        
        
def runFacePix2PixModule(inputDir, outputDir):
    command = 'python'
    command += ' '
    command += a.pix2pix
    command += ' '
    command += '--mode test'
    command += ' '
    command += '--output_dir'
    command += ' '
    command += outputDir
    command += ' '
    command += '--input_dir'
    command += ' '
    command += inputDir
    command += ' '
    command += '--checkpoint'
    command += ' '
    command += a.fm_model_path
    
    os.system(command)

def main():
    
    for output_type in output_types:
        full_subfolder_path = os.path.join(a.output_dir, output_type)
        if not os.path.exists(full_subfolder_path):
            os.makedirs(full_subfolder_path)
    
    seg_input_files = im.findContainingSubtext(a.input_dir, 'sample_inputs')
    gen_input_files = im.findContainingSubtext(a.input_dir, 'sample_outputs')
    
    output_paths = {}
    seg_src_paths = {}
    gen_src_paths = {}
    
    skipped = 0
    for seg_input_file, gen_input_file in zip(seg_input_files, gen_input_files):
        name, _ = os.path.splitext(os.path.basename(seg_input_file))
        name = name.split('_')[0]
        
        output_paths[name] = {}
        name_out_dict = {}
        output_file_exists_for_name = False
        
        for output_type in output_types:
            output_path = os.path.join(a.output_dir, output_type)
            output_path = os.path.join(output_path, name + '.png')
            
            if os.path.exists(output_path):
                output_file_exists_for_name = True
                break
             
            name_out_dict[output_type] = output_path
            
        if not output_file_exists_for_name:
            output_paths[name] = name_out_dict
            seg_src_paths[name] = seg_input_file
            gen_src_paths[name] = gen_input_file
        else:
            skipped += 1
            
    print("skipping %d files that already exist" % skipped)
    
    global total
    total = len(output_paths)
    
    print("processing %d files" % total)
    
    global start
    start = time.time()
    
    skin_color = getSkinColor()
    background_color = getBackgroundColor()
    positive_colors = getPositiveColors()
    
    upscale_interpolation = getUpscaleInterpolation()
    
    with tf.Session() as sess:
        for name, name_out_dict in output_paths.iteritems():
            print name
            print name_out_dict
            
            gen_image = im.load(gen_src_paths[name])
            seg_image = im.load(seg_src_paths[name])
            
            #Save inputs
            im.save(seg_image, name_out_dict['humanseginputs'])
            im.save(gen_image, name_out_dict['humangeninputs'])
            
            #Crop reference
            crop_reference = getFaceCropReference(seg_image, skin_color)
            
            #Cropped face segmentation
            temp_image_seg = crop(seg_image, crop_reference)
            temp_image_seg = addBackground(temp_image_seg, skin_color, positive_colors, background_color)
            im.save(temp_image_seg, name_out_dict['croppedsegfaces'])  
            #Resize face segmentation
            temp_image_seg = resize(temp_image_seg, tfUpscaleInterpolationType=upscale_interpolation)
            im.save(temp_image_seg, name_out_dict['resizedsegfaces'])
            
            #Cropped face generated
            temp_image_gen = crop(gen_image, crop_reference)
            im.save(temp_image_gen, name_out_dict['croppedgenfaces'])
            #Resize face generated
            temp_image_gen = resize(temp_image_gen, tfUpscaleInterpolationType=upscale_interpolation)
            im.save(temp_image_gen, name_out_dict['resizedgenfaces'])

            complete()
            
    #Combine resized seg and gen images to input face module
    inputDirA = os.path.join(a.output_dir, 'resizedsegfaces')
    inputDirB = os.path.join(a.output_dir, 'resizedgenfaces')
    outputDir = os.path.join(a.output_dir, 'combinedseggenfaces')
    combineFaceSegFaceGen(inputDirA, inputDirB, outputDir)
            
    #Now lets generate face images from extracted face segmentation maps
    inputDir = os.path.join(a.output_dir, 'combinedseggenfaces')
    outputDir = os.path.join(a.output_dir, 'generatedfaces')
    runFacePix2PixModule(inputDir, outputDir)    

main()

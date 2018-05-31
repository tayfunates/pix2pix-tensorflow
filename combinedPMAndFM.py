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

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing images generated from portray module (PM)")
parser.add_argument("--output_dir", required=True, help="output path which will contain several subfolders")
parser.add_argument("--fm_model_path", required=True, help="trained face module path")
parser.add_argument("--fm_input_size", type=int, default=256, help="size to use for resize operation to input face module")
parser.add_argument("--color_map", required=True, help="color map png")

##### TO DO Consider to resize face segmentations by using nearest neighbor interpolation

a = parser.parse_args()

output_types = ['humanseginputs', 'humangeninputs', 'croppedsegfaces', 'resizedsegfaces']

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
    
    for output_type in output_types:
        full_subfolder_path = os.path.join(a.output_dir, output_type)
        if not os.path.exists(full_subfolder_path):
            os.makedirs(full_subfolder_path)
    
    seg_input_files = im.findContainingSubtext(a.input_dir, 'sample_inputs')
    gen_input_files = im.findContainingSubtext(a.input_dir, 'sample_outputs')
    
    output_paths = {}
    
    skipped = 0
    for seg_input_file, gen_input_file in zip(seg_input_files, seg_input_files):
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
        else:
            skipped += 1
            
    print("skipping %d files that already exist" % skipped)
    
    global total
    total = len(output_paths)
    
    print("processing %d files" % total)
    
    global start
    start = time.time()
    
    with tf.Session() as sess:
        for name, name_out_dict in output_paths.iteritems():
            print name
            print name_out_dict
            
            complete()
           

main()

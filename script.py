"""
    This script converts the .tfrecords of the Kubric Optical Flow 'movi-f' Dataset  to MPI Sintel Directory Structure.
    Follow instructions on: https://github.com/google-research/kubric/tree/main/challenges/optical_flow to download the dataset
    or use gsutil to download the data: gsutil -m cp -r “gs://kubric-public/tfds/movi_f”

    The images are saved in .png and the optical flow maps are saved in .flo format.

    This directory structure makes it possible to write a PyTorch Dataloader for Kubric 'movi-f' split optical flow data. 
    A PyTorch Dataloader is also provided in the EzFlow Optical Flow Library: https://github.com/neu-vi/ezflow   
"""


import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from PIL import Image


def write_flow(filename, uv, v=None):
    """Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.

    Parameters
    ----------
    filename : str
        Path to file
    uv : np.ndarray
        Optical flow
    v : np.ndarray, optional
        Optional second channel
    """

    # Original code by Deqing Sun, adapted from Daniel Scharstein.
    TAG_CHAR = np.array([202021.25], np.float32)
    
    n_bands = 2

    if v is None:
        assert uv.ndim == 3
        assert uv.shape[2] == 2
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert u.shape == v.shape
    height, width = u.shape
    f = open(filename, "wb")

    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)

    # arrange into matrix form
    tmp = np.zeros((height, width * n_bands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def save_flow(flow, index, path, flow_range=None):
    index = "0"+str(index) if len(str(index)) == 1 else str(index)
    filename="frame_"+index
    
    filepath = path + "/" + filename + ".flo"
    
    minv, maxv = flow_range
    
    flow = flow.numpy()
    flow = flow / 65535 * (maxv - minv) + minv
    write_flow(filepath, flow)


def save_image(img, index, path, ext=".png"):
    
    index = "0"+str(index) if len(str(index)) == 1 else str(index)
    filename="frame_"+index+ext
    
    filepath = path + "/" + filename
    
    img = img.numpy()
    image = Image.fromarray(img)
    image.save(filepath)


def save_sample(index, sample, root_path, ext=".png"):
    os.makedirs(root_path, exist_ok=True)
    
    if len(str(index)) == 1:
        index = "000"+str(index)
    elif len(str(index)) == 2:
        index = "00"+str(index)
    elif len(str(index)) == 3:
        index = "0"+str(index)
    elif len(str(index)) == 4:
        index = str(index)
        
    dirname = "scene_" + index
    
    img_path = os.path.join(root_path, "images", dirname)
    forward_flow_path = os.path.join(root_path, "forward_flow", dirname)
    backward_flow_path = os.path.join(root_path, "backward_flow", dirname)
    
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(forward_flow_path, exist_ok=True)
    os.makedirs(backward_flow_path, exist_ok=True)
    
    video_frames = sample['video']
    forward_flow = sample['forward_flow']
    backward_flow = sample['backward_flow']
    
    forward_flow_range = sample['metadata']['forward_flow_range'].numpy()
    backward_flow_range = sample['metadata']['backward_flow_range'].numpy()
    
    for index, frame in enumerate(video_frames):
        save_image(frame,index+1, path=img_path, ext=ext)
        
        if index < len(video_frames) - 1:
            save_flow(forward_flow[index], index+1, path=forward_flow_path, flow_range=forward_flow_range)
        
        if index < len(video_frames) - 1:
            save_flow(backward_flow[index + 1], index+1, path=backward_flow_path, flow_range=backward_flow_range)
            
    print(f"{dirname} conversion completed")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert .tfrecords to Sintel Directory structure.")

    parser.add_argument(
        "--dir_path", type=str, required=True, help="Kubric dataset directory path."
    )

    parser.add_argument(
        "--save_path", type=str, required=True, help="Directory path where the dataset should be saved."
    )

    parser.add_argument(
        "--split", type=str, required=True, help="Split of the dataset. Accepted split values: 1. train 2. validation"
    )

    args = parser.parse_args()

    ds_type = "movi_f"

    args.split = args.split.lower()
    assert args.split in ['train','validation'], "Invalid split values. Accepted split values: train, validation"

    args.save_path = os.path.join(args.save_path, "training") if args.split.lower == 'train' else os.path.join(args.save_path, "validation") 

    # Tensorflow prioritizes loading on GPU by default
    # Disable loading on all GPUS
    tf.config.set_visible_devices([], 'GPU')

    try:
        ds = tfds.load(ds_type, data_dir=args.dir_path, split=args.split, shuffle_files=False)
    except:
        print(f"Kubric Dataset {ds_type} not found in location {args.dir_path}")
        sys.exit()

    for index, sample in enumerate(ds):
        save_sample(index+1, sample, root_path=args.save_path, ext=".png")

    print(f"Finished converting {args.split.lower()}. Location:{args.save_path}")

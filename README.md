# Convert Kubric Optical flow Dataset 

A script to convert .tfrecords of the Kubric Optical Flow Dataset to MPI Sintel Directory Structure. This directory structure makes it possible to write a `PyTorch Dataloader` for Kubric 'movi-f' forward and backward optical flow maps. 

A PyTorch Dataloader is also provided in the EzFlow Optical Flow Library: https://github.com/neu-vi/ezflow   
___

### Steps

- Follow instructions on: https://github.com/google-research/kubric/tree/main/challenges/optical_flow to download the dataset.

- `pip install -r requirements.txt`

- `python script.py --dir_path "path_where_dataset_is_saved" --save_path "path_where_converted_data_needs_to_be_saved" --split "train"`

Accepted split values: `train` and `validation`

The images are saved in `.png` and the optical flow maps are saved in `.flo` format.
___

### Directory Structure

If `save_path`is "kubricflow"`

    kubricflow
    ├── training
    │   ├── images
    │   │   ├── scene_0001
    │   │   │   ├── frame_01.png
    │   │   │   ├── frame_02.png
    │   │   │   ├── frame_03.png
    │   │   │   ├── frame_04.png
    │   │   │   .
    │   │   │   .
    │   │   ├── scene_0002
    │   │   ├── scene_0003
    │   │   ├── scene_0004
    │   │   .
    │   │   .
    │   │   .
    │   ├── forward_flow
    │   │   ├── scene_0001
    │   │   │   ├── frame_01.flo
    │   │   │   ├── frame_02.flo
    │   │   │   ├── frame_03.flo
    │   │   │   ├── frame_04.flo
    │   │   │   .
    │   │   │   .
    │   │   ├── scene_0002
    │   │   ├── scene_0003
    │   │   ├── scene_0004
    │   │   .
    │   │   .
    │   │   .
    │   ├── backward_flow
    │   │   ├── scene_0001
    │   │   ├── scene_0002
    │   │   ├── scene_0003
    │   │   ├── scene_0004
    │   │   .
    │   │   .
    │   │   .
    ├── validation
    │   ├── images
    │   │   ├── scene_0001
    │   │   ├── scene_0002
    │   │   .
    │   │   .
    │   ├── forward_flow
    │   │   ├── scene_0001
    │   │   ├── scene_0002
    │   │   .
    │   │   .
    │   ├── backward_flow
    │   │   ├── scene_0001
    │   │   ├── scene_0002
    │   │   .
    │   │   .




___

### References

- [Kubric](https://arxiv.org/abs/2203.03570)
- [Kubric-github](https://github.com/google-research/kubric)

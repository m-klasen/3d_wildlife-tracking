import json
from pycocotools import mask as coco_mask
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob
import cv2
import open3d as o3d
from pathlib import Path
import numpy as np
import math
from pycocotools import mask
import copy
from tqdm import tqdm

import argparse

def main(args):

    #dataset_paths = ['20201217170333','20201218080334','20201218080601','20201219164321',
    #                 '20201221080502','20201218165020','20210203175455']
    dataset_paths = sorted(os.listdir(args.root_dir))
    for ds_pth in dataset_paths:
        print("Processing video {} annotations".format(ds_pth))
        dataset_path = f'{ds_pth}'
        os.makedirs(f'{args.root_dir}/{ds_pth}/fg_out', exist_ok=True)

        with open(f"{args.root_dir}/{dataset_path}/instances_default.json", "r") as f: annots = json.load(f)
        j=0
        for _,img in enumerate(annots['images']):
            fg = np.zeros((480,848))
            for ann in annots['annotations']:
                if ann['image_id'] == img['id']:
                    if ann['area']>20:
                        rles = coco_mask.frPyObjects(ann['segmentation'], 480,848)
                        mask = np.squeeze(coco_mask.decode(rles),axis=-1)
                        fg = fg+ (mask*(ann['attributes']['track_id']+1)) if fg is not None else mask*(ann['attributes']['track_id']+1)
            try:
                cv2.imwrite(f'{args.root_dir}/{dataset_path}/fg_out/fg_{filename.split(".")[0]}.png',fg)
                j+=1
            except: pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", default="/mnt/sda/AMMOD/ammod_realsense/data/train", type=str, help="path to directory containing video clips"
    )
    args = parser.parse_args()
    main(args)
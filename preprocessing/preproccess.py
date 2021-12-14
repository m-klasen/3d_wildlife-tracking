#!/usr/bin/python
'''
    FlyingThings3D data preprocessing.
'''

import numpy as np
import os
import re
import sys
import cv2
import glob
import itertools
import load_pfm
import pickle
import argparse
import random
import multiprocessing

import warnings
warnings.filterwarnings('error')

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='/scr2/flyingthings3d/', type=str, help='input root dir')
parser.add_argument('--output_dir', default='data_processed_maxcut_35_20k_2k_8192', type=str, help='output dir')
FLAGS = parser.parse_args()

INPUT_DIR = FLAGS.input_dir
OUTPUT_DIR = FLAGS.output_dir

if not os.path.exists(OUTPUT_DIR):
    os.system('mkdir -p {}'.format(OUTPUT_DIR))

np.random.seed(0)
random.seed(0)


def bilinear_interp_val(vmap, y, x):
    '''
        bilinear interpolation on a 2D map
    '''
    h, w = vmap.shape
    x1 = int(x)
    x2 = x1 + 1
    x2 = w-1 if x2 > (w-1) else x2
    y1 = int(y)
    y2 = y1 + 1
    y2 = h-1 if y2 > (h-1) else y2
    Q11 = vmap[y1,x1]
    Q21 = vmap[y1,x2]
    Q12 = vmap[y2,x1]
    Q22 = vmap[y2,x2]
    return Q11 * (x2-x) * (y2-y) + Q21 * (x-x1) * (y2-y) + Q12 * (x2-x) * (y-y1) + Q22 * (x-x1) * (y-y1)

def get_3d_pos_xy(y_prime, x_prime, depth, focal_length=424.7448, w=720, h=280):
    '''
        depth pop up
    '''
    y = (y_prime - h / 2.) * depth / focal_length
    x = (x_prime - w / 2.) * depth / focal_length
    return [x, y, depth]

def gen_datapoint(fname_depth, fname_depth_next_frame, image, image_next_frame, n = 8192, max_cut = 50, focal_length=424.7448):

    np.random.seed(0)

    ##### generate needed data
    depth = cv2.imread(fname_depth, -1)
    depth_next_frame_np, _ = cv2.imread(fname_depth_next_frame, -1)
    rgb_np = cv2.imread(image)[:, :, ::-1] / 255.
    rgb_next_frame_np = cv2.imread(image_next_frame)[:, :, ::-1] / 255.

    ##### generate needed data
    h, w = depth.shape

    ##### point set 1 current pos
    try:
        depth_requirement = depth_np < max_cut
    except:
        return None

    satisfy_pix1 = np.column_stack(np.where(depth_requirement))
    if satisfy_pix1.shape[0] < n:
        return None
    sample_choice1 = np.random.choice(satisfy_pix1.shape[0], size=n, replace=False)
    sampled_pix1_y = satisfy_pix1[sample_choice1, 0]
    sampled_pix1_x = satisfy_pix1[sample_choice1, 1]

    current_pos1 = np.array([get_3d_pos_xy( sampled_pix1_y[i], sampled_pix1_x[i], depth_np[int(sampled_pix1_y[i]), int(sampled_pix1_x[i])] ) for i in range(n)])
    current_rgb1 = np.array([[rgb_np[h-1-int(sampled_pix1_y[i]), int(sampled_pix1_x[i]), 0], rgb_np[h-1-int(sampled_pix1_y[i]), int(sampled_pix1_x[i]), 1], rgb_np[h-1-int(sampled_pix1_y[i]), int(sampled_pix1_x[i]), 2]] for i in range(n)])
    ##### point set 1 current pos

    ##### point set 1 future pos
    ##### point set 2 current pos
    try:
        depth_requirement = depth_next_frame_np < max_cut
    except:
        return None

    satisfy_pix2 = np.column_stack(np.where(depth_next_frame_np < max_cut))
    if satisfy_pix2.shape[0] < n:
        return None
    sample_choice2 = np.random.choice(satisfy_pix2.shape[0], size=n, replace=False)
    sampled_pix2_y = satisfy_pix2[sample_choice2, 0]
    sampled_pix2_x = satisfy_pix2[sample_choice2, 1]

    current_pos2 = np.array([get_3d_pos_xy( sampled_pix2_y[i], sampled_pix2_x[i], depth_next_frame_np[int(sampled_pix2_y[i]), int(sampled_pix2_x[i])] ) for i in range(n)])
    current_rgb2 = np.array([[rgb_next_frame_np[h-1-int(sampled_pix2_y[i]), int(sampled_pix2_x[i]), 0], rgb_next_frame_np[h-1-int(sampled_pix2_y[i]), int(sampled_pix2_x[i]), 1], rgb_next_frame_np[h-1-int(sampled_pix2_y[i]), int(sampled_pix2_x[i]), 2]] for i in range(n)])

    return current_pos1, current_pos2, current_rgb1, current_rgb2

depth_list = glob.glob("20201217164720/depth_median_4/*")
color_list = glob.glob("20201217164720/infrared_crop/*")
for i in range(len(glob.glob("20201217164720/depth_median_4/*"))-1):
    depth            = depth_list[i]
    depth_next_frame = depth_list[i+1]
    image            = color_list[i]
    image_next_frame = color_list[i+1]

    d = gen_datapoint(depth, depth_next_frame, image, image_next_frame)
    np.savez_compressed('test.npz', points1=d[0], \
                                    points2=d[1], \
                                    color1=d[2], \
                                    color2=d[3])

def proc_one_scene(s, input_dir, output_dir):
    if s[-1] == '/':
        s = s[:-1]
    dis_split = s.split('/')
    train_or_test = dis_split[-4]
    ABC = dis_split[-3]
    scene_idx = dis_split[-2]
    left_right = dis_split[-1]
    for v in range(6, 15):
        fname = os.path.join(output_dir, train_or_test + '_' + ABC + '_' + scene_idx + '_' + left_right + '_' + str(v).zfill(4) + '-{}'.format(0) + '.npz')
        if os.path.exists(fname):
            continue

        fname_disparity = os.path.join(input_dir, 'disparity', train_or_test, ABC, scene_idx, left_right, str(v).zfill(4) + '.pfm')
        fname_disparity_next_frame = os.path.join(input_dir, 'disparity', train_or_test, ABC, scene_idx, left_right, str(v+1).zfill(4) + '.pfm')
        fname_image = os.path.join(input_dir, 'frames_finalpass', train_or_test, ABC, scene_idx, left_right, str(v).zfill(4) + '.png')
        fname_image_next_frame = os.path.join(input_dir, 'frames_finalpass', train_or_test, ABC, scene_idx, left_right, str(v+1).zfill(4) + '.png')
        fname_disparity_change = os.path.join(input_dir, 'disparity_change', train_or_test, ABC, scene_idx, 'into_future', left_right, str(v).zfill(4) + '.pfm')
        L_R = 'L' if left_right == 'left' else 'R'
        fname_optical_flow = os.path.join(input_dir, 'optical_flow', train_or_test, ABC, scene_idx, 'into_future', left_right, 'OpticalFlowIntoFuture_' + str(v).zfill(4) + '_' + L_R + '.pfm')

        d = gen_datapoint(fname_disparity, fname_disparity_next_frame, fname_disparity_change, fname_optical_flow, fname_image, fname_image_next_frame, focal_length=1050.)
        if d is not None:
            np.savez_compressed(fname, points1=d[0], \
                                       points2=d[1], \
                                       color1=d[2], \
                                       color2=d[3], \
                                       flow=d[4], \
                                       valid_mask1=d[5] )
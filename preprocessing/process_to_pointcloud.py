import numpy as np
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
import random
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
from tqdm import tqdm
import logging
import open3d as o3d
import cv2
import argparse

from open3d._ml3d.datasets.base_dataset import BaseDataset, BaseDatasetSplit
from open3d._ml3d.utils import make_dir, DATASET
from skimage import segmentation
from skimage.feature import peak_local_max
import scipy.ndimage as ndi
import copy
import math

from  open3d._ml3d.vis import BoundingBox3D




def crop_with_2dmask(o3dpc, mask, K=None):
    """ crop open3d point cloud with given 2d binary mask
    Args: 
        o3dpc (open3d.geometry.PointCloud): open3d point cloud
        mask (np.array): binary mask aligned with the point cloud frame shape of [H, W]
        K (np.array): intrinsic matrix of camera shape of (4x4)
        if K is not given, point cloud should be ordered
    Returns:
        o3dpc (open3d.geometry.PointCloud): filtered open3d point cloud
    """
    o3dpc = copy.deepcopy(o3dpc)
    cloud_npy = np.asarray(o3dpc.points)

    if K is None:
        mask = np.resize(mask, cloud_npy.shape[0])
        cloud_npy = cloud_npy[mask!=0]
        o3dpc = o3d.geometry.PointCloud()
        o3dpc.points = o3d.utility.Vector3dVector(cloud_npy)
    else:
        # project 3D points to 2D pixel
        cloud_npy = np.asarray(o3dpc.points)  
        x = cloud_npy[:, 0]
        y = cloud_npy[:, 1]
        z = cloud_npy[:, 2]
        px = np.uint16(x * K[0, 0]/z + K[0, 2])
        py = np.uint16(y * K[1, 1]/z + K[1, 2])
        # filter out the points out of the image
        H, W = mask.shape
        row_indices = np.logical_and(0 <= px, px < W-1)
        col_indices = np.logical_and(0 <= py, py < H-1)
        image_indices = np.logical_and(row_indices, col_indices)
        cloud_npy = cloud_npy[image_indices]
        mask_indices = mask[(py[image_indices], px[image_indices])]
        mask_indices = np.where(mask_indices != 0)[0]
        o3dpc.points = o3d.utility.Vector3dVector(cloud_npy[mask_indices])
    return o3dpc

def calc_ransac(pcl):
    # plane_model, inliers = pcl.segment_plane(distance_threshold=0.25,
    #                                          ransac_n=3000,
    #                                          num_iterations=1000)
    plane_model = [-0.00,0.72,0.40,6.96]
    [a, b, c, d] = plane_model

    # Translate plane to coordinate center
    pcl.translate((0,-d/c,0))

    # Calculate rotation angle between plane normal & z-axis
    plane_normal = tuple(plane_model[:3])
    z_axis = (0,0,1)
    def vector_angle(u, v):
        return np.arccos(np.dot(u,v) / (np.linalg.norm(u)* np.linalg.norm(v)))
    rotation_angle = vector_angle(plane_normal, z_axis)

    # Calculate rotation axis
    plane_normal_length = math.sqrt(a**2 + b**2 + c**2)
    u1 = b / plane_normal_length
    u2 = -a / plane_normal_length
    rotation_axis = (u1, u2, 0)

    # Generate axis-angle representation
    optimization_factor = 1.4
    axis_angle = tuple([x * rotation_angle * optimization_factor for x in rotation_axis])

    # Rotate point cloud
    R = pcl.get_rotation_matrix_from_axis_angle(axis_angle)
    return R

def main(args):
    #dataset_paths = ['20201221080502','20201218165020','20210203175455']#['20201217170333','20201218080334','20201218080601','20201219164321']
    dataset_paths = sorted(os.listdir(args.root_dir))
    for ds_pth in dataset_paths:
        dataset_path = f'{args.root_dir}/{ds_pth}'
        os.makedirs(f'{args.root_dir}/{ds_pth}/point_cloud', exist_ok=True)
        color_dir = str(Path(dataset_path) / "color")
        depth_dir = str(Path(dataset_path) / "depth_median_4")

        color_files = [f for f in sorted(glob.glob(color_dir + "/*left*"))]
        depth_files = [f for f in sorted(glob.glob(depth_dir + "/*"))]
        path_list   = color_files

        fx = 424.7448425292969 
        fy = 424.7448425292969
        h = 480
        ppx = 421.0166931152344
        ppy = 237.47096252441406
        w = 848
        cam = o3d.camera.PinholeCameraIntrinsic(w,h,fx,fy,ppx,ppy)
        pcl_tfm = [[1, 0, 0, 0], 
                        [0,1, 0, 0], 
                        [0, 0,1, 0], 
                        [0, 0, 0, 1]]
        extr = np.array([[1, 0., 0.,  0.], 
                        [0., -np.cos(0.3),  np.sin(0.3), 2.1], 
                        [0., -np.sin(0.3), -np.cos(0.3), 0.], 
                        [0.,            0., 0., 1.]])
        #Orient based on last frame ransac plane fitting
        color_file = color_files[-1]
        depth_file = depth_files[-1]
        depth = np.zeros((480,848),dtype=np.uint16)
        d = (cv2.imread(depth_file, -1)*1000)
        if d.shape==(280,720):
            depth[200:,:720] = d.astype(np.uint16)
        else:
            depth = d.astype(np.uint16)
        color = o3d.io.read_image(color_file)
        depth = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color,depth,depth_scale=1000,depth_trunc=25)
        pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,cam,extr)
        pcl.transform(pcl_tfm)
        R = calc_ransac(pcl)


        for idx in range(len(path_list)):
            color_file = color_files[idx]
            depth_file = depth_files[idx]

            depth = np.zeros((480,848),dtype=np.uint16)
            d = (cv2.imread(depth_file, -1)*1000)
            if d.shape==(280,720):
                depth[200:,:720] = d.astype(np.uint16)
            else:
                depth = d.astype(np.uint16)

            color = o3d.io.read_image(color_file)
            depth = o3d.geometry.Image(depth)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color,depth,depth_scale=1000,depth_trunc=100)
            pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,cam,extr)
            pcl.transform(pcl_tfm)
            #pcl.rotate(R, center=(0,0,0))
            print(f'{dataset_path}/point_cloud/{color_files[idx].split("/")[-1].replace(".jpg",".pcd")}')
            o3d.io.write_point_cloud(f'{dataset_path}/point_cloud/{color_files[idx].split("/")[-1].replace(".jpg",".pcd")}', pcl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_dir", type=str, help="path to directory containing video clips"
    )
    args = parser.parse_args()
    main(args)

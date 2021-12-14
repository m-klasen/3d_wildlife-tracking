import glob
import cv2
import os
import open3d as o3d
from pathlib import Path
import numpy as np
import math
import pycocotools.mask as pycoco_mask
import copy
import json
from tqdm import tqdm
import argparse


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
		cloud_npy = cloud_npy[mask !=0]
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


def main(args):
	dataset_paths = sorted(os.listdir(args.root_dir))
	for ds_pth in dataset_paths:
		dataset_path = f'{args.root_dir}/{ds_pth}'
		pcd_dir = str(Path(dataset_path) / "point_cloud")
		masks_dir = str(Path(dataset_path) / "fg_out")

		with open(f"{dataset_path}/instances_default.json", "r") as f: annos = json.load(f)
		pcd_files = [f for f in sorted(glob.glob(pcd_dir+"/*"))]
		masks_files = [f for f in sorted(glob.glob(masks_dir + "/*"))]
		with open(f"evaluation/label/{ds_pth}.txt","w") as save_trk_file:
			for i,(pcd_file, mask_file) in enumerate(zip(pcd_files,masks_files)):
				fn = pcd_file.split("/")[-1].replace(".pcd",".jpg")
				for img_file in annos['images']:
					if img_file['file_name']==fn:
						img_id = img_file['id']
				masks, ids = [],[]
				for img_ann in annos['annotations']:
					if img_ann['image_id']==img_id:
						rle = pycoco_mask.frPyObjects(img_ann['segmentation'], 480, 848 )
						masks.append(pycoco_mask.decode(rle).squeeze(-1))
						ids.append(img_ann['attributes']['track_id'])

				pcd = o3d.io.read_point_cloud(pcd_file)
				if masks:
					masks = np.array(masks)[:,200:,:720]
				#masks = cv2.imread(mask_file,0)[200:,:720]
				print(i, ids)
				for inst,track_id in zip(masks,ids):
					inst_pcd = crop_with_2dmask(pcd,inst)
					cl, ind = inst_pcd.remove_statistical_outlier(nb_neighbors=30,std_ratio=1.0)
					pcl_processed = inst_pcd.select_by_index(ind)
					if np.array(pcl_processed.points).shape[0] > 20:
						################################################
						# box = pcl_processed.get_oriented_bounding_box()
						# og_r = box.R.copy()
						# yaw = -np.arctan2(og_r[0,2],[og_r[1,2]])
						# r = np.linalg.inv(box.R)
						# new_r = [[np.cos(yaw),-np.sin(yaw), 0],
						# 		[np.sin(yaw), np.cos(yaw), 0],
						# 		[		  0,		   0, 1]]
						# box.rotate(r)
						# box.rotate(new_r)

						# yaw = -yaw
						# left = [np.cos(yaw), -np.sin(yaw), 0]
						# # y-axis
						# front = [np.sin(yaw), np.cos(yaw), 0]

						# # z-axis
						# up = [0, 0, 1]
						# size = np.array(box.extent)
						# center = np.array(box.get_center())
						
						################################################
						box = pcl_processed.get_axis_aligned_bounding_box()
						# x-axis
						left = [1, 0, 0]
						# y-axis
						front = [0, 1, 0]
						# z-axis
						up = [0, 0, 1]
						size = np.array(box.get_extent())
						yaw=0.
						center = np.array(box.get_center())
						#################################################
						size[1], size[2] = size[2], size[1]

						fortran_ground_truth_binary_mask = np.asfortranarray(inst)
						encoded_ground_truth = pycoco_mask.encode(fortran_ground_truth_binary_mask)
						bbox = pycoco_mask.toBbox(encoded_ground_truth)

						#(frame,tracklet_id,objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry)
						str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f\n' % (
							i,track_id,'car',0.,bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3],size[1],size[0],size[2],center[0],center[1],center[2],0.)
						save_trk_file.write(str_to_srite)
		save_trk_file.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"root_dir", type=str, help="path to directory containing video clips"
	)
	args = parser.parse_args()
	main(args)
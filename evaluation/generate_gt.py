import glob
import cv2
import open3d as o3d
from pathlib import Path
import numpy as np
import math
from pycocotools import mask
import copy
from tqdm import tqdm
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


dataset_path = '20201217164720'
pcd_dir = str(Path(dataset_path) / "point_cloud")
masks_dir = str(Path(dataset_path) / "fg_out")

pcd_files = [f for f in sorted(glob.glob(pcd_dir+"/*"))]
masks_files = [f for f in sorted(glob.glob(masks_dir + "/*"))]
with open("evaluation/label/0000.txt","w") as save_trk_file:
	for i,(pcd_file, mask_file) in enumerate(zip(pcd_files,masks_files)):
		pcd = o3d.io.read_point_cloud(pcd_file)
		masks = cv2.imread(mask_file,0)
		print(i, np.unique(masks)[1:])
		for inst in np.unique(masks)[1:]:

			inst_pcd = crop_with_2dmask(pcd,masks==inst)
			cl, ind = inst_pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
			pcl_processed = inst_pcd.select_by_index(ind)
			if np.array(pcl_processed.points).shape[0] > 20:
				pcd_bbox = pcl_processed.get_axis_aligned_bounding_box()
				center = np.array(pcd_bbox.get_center())
				size = np.array(pcd_bbox.get_extent()) # w h l
				size[1], size[2] = size[2], size[1]

				fortran_ground_truth_binary_mask = np.asfortranarray(masks==inst)
				encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
				bbox = mask.toBbox(encoded_ground_truth)

				#(frame,tracklet_id,objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry)
				str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f\n' % (
					i,inst,'car',0.,bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3],size[0],size[1],size[2],center[0],center[1],center[2],0.)
				save_trk_file.write(str_to_srite)
save_trk_file.close()
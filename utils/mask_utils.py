import os
import pycocotools.mask as pymasks
import numpy as np
import open3d as o3d
import copy

import scipy.ndimage as ndi
from pycocotools import mask as pymask
from skimage import segmentation
from skimage.feature import peak_local_max

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

def get_instance_segm_masks(cfg,json_ann,idx):
	file_anns = json_ann[idx]
	instance_masks = [];confidences = [];boxes_2d=[]

	for i, ann in enumerate(file_anns):
		box = pymasks.toBbox(ann['mask']);box[1]-=200 
		if pymasks.area(ann['mask'])>20 and box[0]<720:
			m = pymasks.decode(ann['mask'])[200:,:720]*1
			if (m>0).any():
				box = np.clip(box,0,720)
				boxes_2d.append(box)
				instance_masks = np.concatenate((instance_masks,m[None,:,:]),axis=0)\
											if len(instance_masks)>0 else m[None,:,:]
				confidences.append(ann['conf'])


	return instance_masks,boxes_2d,confidences


def watershed_fg_preds_instances(cfg,json_ann,idx) -> np.ndarray:
	file_anns = json_ann[idx]
	instance_masks = []; boxes_2d = []

	m = pymasks.decode(file_anns[0]['mask'])[200:,:720]*1
	if (m>0).any():
		distance = ndi.distance_transform_edt(m)
		coords = peak_local_max(distance, min_distance=5, footprint=np.ones((50,50)), labels=m)
		mask = np.zeros(distance.shape, dtype=bool)
		mask[tuple(coords.T)] = True
		markers, _ = ndi.label(mask)
		wsh_mask = segmentation.watershed(-distance,markers,mask=m, watershed_line=True)
		
		for m_idx in np.unique(wsh_mask)[1:]:
			if np.sum(wsh_mask==m_idx)>20:
				if len(instance_masks)>0:
					instance_masks = np.concatenate((instance_masks,((wsh_mask==m_idx)*1)[None,:,:]),axis=0).astype(np.uint8)
				else: instance_masks = (((wsh_mask==m_idx)*1)[None,:,:]).astype(np.uint8)
	for inst_mask in instance_masks:
		box = pymasks.toBbox(pymask.encode(np.asfortranarray(inst_mask)))
		boxes_2d.append(box)
	return instance_masks,boxes_2d

"""def dbscan_fg_preds_instances(cfg,pcl,json_ann,idx) -> np.ndarray:
	file_anns = json_ann[idx]
	instance_masks = []; boxes_2d = []

	m = pymasks.decode(file_anns[0]['mask'])[200:,:720]*1
	if (m>0).any():
		return instance_masks,boxes_2d
	else:
		return [], []
	for inst_mask in m:
		box = []#pymasks.toBbox(pymask.encode(np.asfortranarray(inst_mask)))
		boxes_2d.append(box)"""

def dbscan_fg_preds_instances(cfg,pcl,json_ann,idx) -> np.ndarray:
	file_anns = json_ann[idx]
	instance_masks = []; boxes_2d = []

	m = pymasks.decode(file_anns[0]['mask'])[200:,:720]*1
	if (m>0).any():
		cropped_pcl = crop_with_2dmask(pcl,m)
		cl, ind = cropped_pcl.remove_statistical_outlier(nb_neighbors=30,std_ratio=1.0)
		pcl_processed = cropped_pcl.select_by_index(ind)
		with o3d.utility.VerbosityContextManager(
			o3d.utility.VerbosityLevel.Debug) as cm:
			labels = np.array(
				pcl_processed.cluster_dbscan(eps=0.13, min_points=50, print_progress=False))
		
		np_pcl = np.zeros(201600)
		np_m   = np.zeros(len(cropped_pcl.points))
		np_m[ind] = labels+1
		np_pcl[m.reshape(-1)==1] = np_m
		np_pcl = np_pcl.reshape(280,720)
		for m_idx in np.unique(np_pcl)[1:]:
			if len(instance_masks)>0:
				instance_masks = np.concatenate((instance_masks,((np_pcl==m_idx)*1)[None,:,:]),axis=0).astype(np.uint8)
			else: instance_masks = (((np_pcl==m_idx)*1)[None,:,:]).astype(np.uint8)
	
	for inst_mask in instance_masks:
		box = pymasks.toBbox(pymask.encode(np.asfortranarray(inst_mask)))
		boxes_2d.append(box)

	return instance_masks,boxes_2d
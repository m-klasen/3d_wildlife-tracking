import open3d.ml.torch as ml3d
import numpy as np
from lindenthal import Lindenthal3D
import os



import argparse

def main(args):
	#root dir to the folder with video clips
	
	dataset_paths = sorted(os.listdir(args.root_dir)) #['20201217164720']#['20201217170333']
	"""
	Lindenthal3D Class 
	Attributes: 
	root_dir: 		-[train|valid] path to the folder in which all our prepared videos lie
	dataset_paths: 	-all videos within our root dir
	tracking_mode:  -[scene_flow|kalman] select tracking mode
	segm_mode: 		-[gt|inst_segm|watershed|dbscan] select method of 2D masks, ground truths for
					pure tracking method evaluation, watershed&dbscan is based on previous
					foreground segmentation
	gt_path: 		- Folder in which the gt 2D segmentation masks are
	images_path:	- Folder in which the video-clip images are
	depth_path:		- Folder in which the depth maps are
	fg_path:		- Path to the FG-Predction masks made by MFCN
	inst_segm_path: - Path to the Instance-Segmentation predictions by DETR
	save_eval: 		- Print out 
	"""
	for ds_pth in dataset_paths:
		dataset = Lindenthal3D(root_dir=args.root_dir, 
							dataset_path=f'{ds_pth}/',
							axis_aligned=True,       
							tracking_mode="kalman",  # ['scene_flow','kalman]
							segm_mode="inst_segm",   # ['gt', 'inst_segm', 'watershed', 'dbscan']
							gt_path="fg_out",
							images_path="color",
							depth_path="depth_median_4",
							fg_path=f"{ds_pth}/fg_predictions.json",
							inst_segm_path=f"{ds_pth}/detr_instance_predictions_conf_0.9.json",
							iou3d=0.01,
							save_eval=True)    
		if args.visualize:
			##### Evaluate 3D Tracking & Visualize with Open3D ##### 
			vis = ml3d.vis.Visualizer()
			vis.visualize_dataset(dataset, 'all', indices=range(100,200))
		else:
			##### Evaluate 3D tracking #####
			dataset.get_split('all')




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--root_dir", type=str, default=f'/mnt/sda/AMMOD/ammod_realsense/data/train', help="path to directory containing video clips", 
		
	)
	parser.add_argument(
		"--visualize", type=bool, default=False, help="visualize", 
		
	)
	args = parser.parse_args()
	main(args)

	args = parser.parse_args(["--root_dir","/mnt/sda/AMMOD/ammod_realsense/data/valid"])
	main(args)
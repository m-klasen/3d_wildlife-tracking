# Improving Wildlife Tracking with 3D Information 



To install required packages do 

```pip install -r requirements.txt ``` 

For visualization the custom Open3D-ML repository is required with 

`export OPEN3D_ML_ROOT=$ABS_PATH/Open3D-ML` to set as default path.

Required Data Format

```
 ${ROOT_DIR}
  `-- |-- 202011221080502 				# videoclip
      `-- |-- color 					# list of color images
      	  |-- | -- xxx.jpg
      	  |-- depth_median_4			# list of depth maps
      	  |-- | -- xxx.exr
      	  |-- optical_flow				# generated optical flow
      	  |-- | -- xxx.flo
      	  |-- fg_out					# ground truth instances
      `-- instances_default.json 		# ground truth annotations
      `-- intrinsics.json				# Realsense Intrinsics
      `-- fg_predictions.json 			# Foreground Predictions 
      `-- detr_instance_predictions_conf_0.9.json
      									# 2D Instance Segmentation Predictions
```

**Preprocessing Steps:** (if necessary)

1. ```python preprocess/coco_to_segm.py $ROOT_DIR``` to preprocess ground truths to  `fg_out`

2. ```python preprocess/generate_gt_oriented.py $ROOT_DIR``` to setup gt for eval in the MOT17 format under `evaluation/label`
3. ```python preprocess/opticalflow_to_scene.py $ROOT_DIR``` to precompute the optical flow expansion to scene flow
4. ```python preprocess/process_to_pointcloud.py $ROOT_DIR``` to preprocess RGB-D to pointclouds

**Example of Inference**

```
python main.py --root_dir==$PATH_TO_CLIPS
```

**Example of Evaluation of 3D Mot Metrics**

```
python evaluation/evaluate_3dmot.py gt_aabbox_kalman 3D
```

**Example of Evaluation of 2D Mot Metrics**

```
python evaluation/evaluate_3dmot.py gt_aabbox_kalman 2D
```

Results will be printed and can be found under ```results/```

**Format 3D MOT**

| Frame |    Type    |   2DBBOX (x1, y1, x2, y2)   | Score |   3D BBOX (h, w, l, x, y, z, rot_y)   | Alpha |
| ----- | :--------: | :-------------------------: | :---: | :-----------------------------------: | :---: |
| 0     | 1 (animal) | 726.4, 173.69, 917.5, 315.1 | 13.85 | 1.56, 1.58, 3.48, 2.57, 1.57, 9.72, 0 |   0   |


| Frame | Track-ID   |    Type    | Truncated  |  Occluded  | Alpha |   2DBBOX (x1, y1, x2, y2)   |   3D BBOX (h, w, l, x, y, z, rot_y)   | Score | RLE
| ----- | :--------: | :--------: | :--------: | :--------: | :---: | :-------------------------: | :-----------------------------------: | :---: | :---: |
| 0     |    0-N     | 1 (animal) |     0      |     0      |   0   | 726.4, 173.69, 917.5, 315.1 | 1.56, 1.58, 3.48, 2.57, 1.57, 9.72, 0 | [0-1] | 2D Mask Encoding |

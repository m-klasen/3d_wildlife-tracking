ROOT_DIR=/mnt/sda/AMMOD/ammod_realsense/data

python preprocessing/coco_to_segms.py --root_dir=$ROOT_DIR/valid
python preprocessing/opticalflow_to_scene.py $ROOT_DIR/valid
python preprocessing/generate_gt_oriented.py $ROOT_DIR/valid
python preprocessing/process_to_pointcloud.py $ROOT_DIR/valid
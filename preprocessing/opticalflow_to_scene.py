import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
import numpy as np
from tqdm import tqdm
import os
import argparse

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

def read_flow(fn):
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file', fp)
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * w * h)
            data2D = data.reshape((h, w, 2))
    return data2D


def main(args):
    #dataset_paths = ['20201221080502','20201218165020','20210203175455']#['20201217170333','20201218080334','20201218080601','20201219164321']
    dataset_paths = sorted(os.listdir(args.root_dir))
    for ds_pth in dataset_paths:
        dataset_path = f'{args.root_dir}/{ds_pth}'
        os.makedirs(dataset_path+"/optical_sceneflow", exist_ok=True)
        depth = sorted(glob.glob(f"{dataset_path}/depth_median_4/*"))
        flow = sorted(glob.glob(f"{dataset_path}/optical_flow_raft/*"))
        for flo, d1,d2 in tqdm(zip(flow, depth[:-1], depth[1:]), total=len(flow)):
            #print(flo,d1,d2)
            depth_i1 = cv2.imread(d1,-1)
            depth_i2 = cv2.imread(d2,-1)
            f = read_flow(flo)[200:,:720,:]
            H, W = f.shape[0], f.shape[1]
            optical_scene = np.zeros((480,848,3))


            #img_flo = flow_to_color(f)
            
            # import matplotlib.pyplot as plt
            # plt.imshow(img_flo / 255.0)
            # plt.show()
            # print(f.shape)
            # print(depth_i1.shape, depth_i1.max())
            
            for h in range(H-1):
                for w in range(W-1):
                    u, v = f[h,w]
                    idx_h, idx_w  = min(H-1,int(h+v)), min(W-1,int(w+u))
                    optical_scene[h,w,0] = u
                    optical_scene[h,w,1] = v
                    d2 = depth_i2[idx_h,idx_w]
                    d1 = depth_i1[h,w]
                    optical_scene[h,w,2] = d2 - d1
                    # if (d2 - d1) > 1.:
                    #     print("H: ",h, "W: ",w, ":::", u,v,d1,d2)
            np.save(flo.replace("optical_flow_raft","optical_sceneflow").replace(".flo",".npy"), optical_scene)
            # plt.imshow(optical_scene[:,:,2], "turbo")
            # plt.show()
            #cv2.imwrite(flo.replace(".flo",".jpg").replace("optical_flow_raft","optical_flow_viz"),img_flo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_dir", type=str, help="path to directory containing video clips"
    )
    args = parser.parse_args()
    main(args)
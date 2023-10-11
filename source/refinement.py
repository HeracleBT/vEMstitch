import cv2
import numpy as np
from Utils import loG, filter_geometry, SIFT, normalize_img, draw_matches
from collections import defaultdict
from rigid_transform import RANSAC
from elastic_transform import local_TPS
import os
from rigid_transform import rigid_transform
from scipy import linalg
from scipy.ndimage import map_coordinates


def find_overlap(im1_mask, im2_mask, H):
    box = np.array([[0, im2_mask.shape[1]-1, im2_mask.shape[1]-1, 0],
                    [0, 0,              im2_mask.shape[0]-1, im2_mask.shape[0]-1],
                    [1, 1,              1,              1]])
    box_ = linalg.solve(H, box)
    box_[0, :] = box_[0, :] / box_[2, :]
    box_[1, :] = box_[1, :] / box_[2, :]
    u_left = min(0, min(box_[0, :]))
    u_right = max(im1_mask.shape[1]-1, max(box_[0, :]))
    v_up = min(0, min(box_[1, :]))
    v_down = max(im1_mask.shape[0]-1, max(box_[1, :]))
    v_h = np.arange(v_up, v_down)
    u_w = np.arange(u_left, u_right)
    u, v = np.meshgrid(u_w, v_h)
    warped_mask1 = map_coordinates(im1_mask, [v, u])
    Z_ = H[2, 0] * u + H[2, 1] * v + H[2, 2]
    u_ = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / Z_
    v_ = (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / Z_
    warped_mask2 = map_coordinates(im2_mask, [v_, u_])
    mass = warped_mask1 + warped_mask2
    overelap_mass = np.where(mass > 1.5, 1.0, 0)
    return overelap_mass


def find_region(im1, im2, X1, X2, ok, im_mask_overlap):
    o_x_u_1 = min(X1[:, 1])
    o_x_d_1 = max(X1[:, 1])
    o_x_u_2 = min(X2[:, 1])
    o_x_d_2 = max(X2[:, 1])

    height_1 = o_x_d_1 - o_x_u_1
    height_2 = o_x_d_1 - o_x_u_1
    height = int(max(height_1, height_2, im1.shape[0] * 0.15, im2.shape[0] * 0.15))

    stride = int(o_x_d_1 - o_x_u_1)
    feature_count = defaultdict(int)
    feature_count_2 = defaultdict(int)
    x_median = int((o_x_d_1 + o_x_u_1) // 2)
    x_radius = int((o_x_d_1 - x_median) * 0.5)
    x_median_2 = int((o_x_d_2 + o_x_u_2) // 2)
    x_radius_2 = int((o_x_d_2 - x_median_2) * 0.5)

    o_y_map = np.sum(im_mask_overlap, axis=0)
    o_y_l = min(np.argwhere(o_y_map >= 1.0))[0]
    o_y_r = max(np.argwhere(o_y_map >= 1.0))[0]
    total_num = (o_y_r - o_y_l) // stride
    for y, _ in X1[ok, :]:
        n = int((y - o_y_l) // stride)
        feature_count[n] += 1
        feature_count[n - 1] += 1

    for y, _ in X2[ok, :]:
        n = int((y - o_y_l) // stride)
        feature_count_2[n] += 1
        feature_count_2[n - 1] += 1

    im1_select_key = []
    im2_select_key = []

    ratio = 0.35
    select_range = int(total_num * ratio)
    total_key = [i for i in range(total_num)]
    select_key = total_key[:select_range] + total_key[-select_range:]
    for i in select_key:
        region1 = im1[x_median - x_radius:x_median + x_radius, o_y_l + int(i * stride): o_y_l + int((i + 2) * stride)]
        region2 = im2[x_median_2 - x_radius_2:x_median_2 + x_radius_2, o_y_l + int(i * stride): o_y_l + int((i + 2) * stride)]
        if np.std(region1[region1 != 0.0]) >= 12.0 and feature_count[i] <= 3:
            im1_select_key.append(i)
        if np.std(region2[region2 != 0.0]) >= 12.0 and feature_count_2[i] <= 3:
            im2_select_key.append(i)
    if len(im1_select_key) == 0 or len(im2_select_key) == 0:
        return None, None, None
    im1_select_region = [o_y_l + int(im1_select_key[0] * stride), o_y_l + int((im1_select_key[-1] + 2) * stride)]
    im2_select_region = [o_y_l + int(im2_select_key[0] * stride), o_y_l + int((im2_select_key[-1] + 2) * stride)]
    return height, im1_select_region, im2_select_region


def fast_brief(im1, im2, im1_mask, im2_mask, X1, X2, height, im1_region, im2_region, mode):
    mask_1 = np.zeros(im1.shape)
    mask_2 = np.zeros(im2.shape)
    if height is not None:
        if mode == "d":
            mask_1[-height:, im1_region[0]: im1_region[1]] = 1.0
            mask_2[:height, im2_region[0]: im2_region[1]] = 1.0
        elif mode == "r":
            mask_1[im1_region[0]: im1_region[1], -height:] = 1.0
            mask_2[im2_region[0]: im2_region[1], :height] = 1.0
    else:
        mask_1 = im1_mask
        mask_2 = im2_mask

    new_img_1 = loG(im1) * im1_mask
    new_img_2 = loG(im2) * im2_mask
    new_img_1 = np.uint8(new_img_1 * 255)
    new_img_2 = np.uint8(new_img_2 * 255)

    fusion_lambda = 0.3
    gau_img_1 = cv2.GaussianBlur(im1, (7, 7), 0)
    gau_img_2 = cv2.GaussianBlur(im2, (7, 7), 0)
    fusion_1 = (gau_img_1 * fusion_lambda + new_img_1 * (1 - fusion_lambda)) * mask_1
    fusion_1 = np.uint8(fusion_1)
    fusion_2 = (gau_img_2 * fusion_lambda + new_img_2 * (1 - fusion_lambda)) * mask_2
    fusion_2 = np.uint8(fusion_2)

    fast = cv2.FastFeatureDetector_create(10)
    kp1 = fast.detect(fusion_1, None)
    kp2 = fast.detect(fusion_2, None)

    orb = cv2.ORB_create()
    kp1, dsp1 = orb.compute(im1, kp1)
    kp2, dsp2 = orb.compute(im2, kp2)
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=12,  # 12
                        key_size=20,  # 20
                        multi_probe_level=2)  # 2
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(dsp1, dsp2, k=2)
    good = []
    ratio = 0.6
    for k in matches:
        if len(k) == 2:
            m, n = k[0], k[1]
            if m.distance < ratio * n.distance:
                good.append([m.queryIdx, m.trainIdx])
    srcdsp = np.float32([kp1[m[0]].pt for m in good])
    tgtdsp = np.float32([kp2[m[1]].pt for m in good])

    dis = 0.0
    if mode == "d":
        dis = im1_mask.shape[0]
    elif mode == "l" or "r":
        dis = im1_mask.shape[1]
    shifting = (mode, dis)
    edge_ok = filter_geometry(srcdsp, tgtdsp, index_flag=True, shifting=shifting)

    X1 = np.vstack([X1, srcdsp[edge_ok, :]]) if X1 is not None else srcdsp[edge_ok, :]
    X2 = np.vstack([X2, tgtdsp[edge_ok, :]]) if X2 is not None else tgtdsp[edge_ok, :]

    _, ok = RANSAC(X1.copy(), X2.copy(), 2000, 0.01)

    point_num = X1.shape[0]
    centroid_1 = np.mean(X1, axis=0)
    centroid_2 = np.mean(X2, axis=0)
    X = X1 - np.tile(centroid_1, (point_num, 1))
    Y = X2 - np.tile(centroid_2, (point_num, 1))
    H = np.matmul(np.transpose(X[ok, :]), Y[ok, :])
    U, S, VT = np.linalg.svd(H)
    R = np.matmul(VT.T, U.T)
    if np.linalg.det(R) < 0:
        VT[1, :] *= -1
        R = np.matmul(VT.T, U.T)
    t = -np.matmul(R, centroid_1) + centroid_2
    H = np.zeros((3, 3))
    H[2, 2] = 1.0
    H[:2, 2] = t
    H[:2, :2] = R

    return H, ok, X1, X2


def refinement_local(im1, im2, H, X1, X2, ok, im1_mask, im2_mask, mode):
    im_mask_overlap = find_overlap(im1_mask, im2_mask, H)
    height, im1_region, im2_region = find_region(im1, im2, X1, X2, ok, im_mask_overlap)
    if height is None:
        return None, None, None
    H, ok, X1, X2 = fast_brief(im1, im2, im1_mask, im2_mask, X1, X2, height, im1_region, im2_region, mode)
    stitching_res, _, _, mass, overlap_mass = local_TPS(im1, im2, H, X1.T[:, ok], X2.T[:, ok], im1_mask, im2_mask, mode)
    return stitching_res, mass, overlap_mass



import numpy as np
import cv2
from collections import defaultdict
import os

def generate_None_list(m, n):
    a = []
    for i in range(m):
        tmp = []
        for j in range(n):
            tmp.append(None)
        a.append(tmp)
    return a


def normalize_img(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


def unique(A):
    ar, idx = np.unique(A, return_index=True, axis=1)
    return ar, idx


def appendimages(im1, im2, vertical=False):
    if vertical:
        col1 = im1.shape[1]
        col2 = im2.shape[1]
        if col1 < col2:
            im1 = np.hstack((im1, np.zeros((im1.shape[0], col2 - col1))))
        elif col1 > col2:
            im2 = np.hstack((im2, np.zeros((im2.shape[0], col1 - col2))))
        return np.concatenate((im1, im2), axis=0)
    else:
        rows1 = im1.shape[0]
        rows2 = im2.shape[0]
        if rows1 < rows2:
            im1 = np.vstack((im1, np.zeros((rows2 - rows1, im1.shape[1]))))
        elif rows1 > rows2:
            im2 = np.vstack((im2, np.zeros((rows1 - rows2, im2.shape[1]))))
        return np.concatenate((im1, im2), axis=1)


def draw_matches(im1, im2, locs1, locs2, ok, vertical=False):
    im3 = appendimages(im1, im2, vertical)
    if vertical:
        for i in range(locs1.shape[0]):
            center1 = (int(round(locs1[i, 0])), int(round(locs1[i, 1])))
            center2 = (int(round(locs2[i, 0])), int(round(locs2[i, 1]) + im1.shape[0]))
            if ok[i] == 0:
                cv2.circle(im3, center1, 3, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(im3, center2, 3, (255, 0, 0), -1, cv2.LINE_AA)
            else:
                cv2.circle(im3, center1, 3, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(im3, center2, 3, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.line(im3, center1, center2, (255, 0, 0), 1, cv2.LINE_AA)
    else:
        for i in range(locs1.shape[0]):
            center1 = (int(round(locs1[i, 0])), int(round(locs1[i, 1])))
            center2 = (int(round(locs2[i, 0] + im1.shape[1])), int(round(locs2[i, 1])))
            if ok[i] == 0:
                cv2.circle(im3, center1, 3, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(im3, center2, 3, (255, 0, 0), -1, cv2.LINE_AA)
            else:
                cv2.circle(im3, center1, 3, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(im3, center2, 3, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.line(im3, center1, center2, (255, 0, 0), 1, cv2.LINE_AA)
    return im3


# filter isolated points
def filter_isolate(src, tgt, shifting=None):
    # filter by x-axis
    row_index = np.argsort(src[:, 0])
    src_row = src[row_index, 0]
    dis = src_row[4:] - src_row[:-4]
    mean_dis = np.mean(dis) / 2
    index = []
    i = 0
    while i < src_row.shape[0]:
        if i > src_row.shape[0] - 3:
            if abs(src_row[i] - src_row[i - 2]) <= mean_dis * 3:
                index.append(i)
            i += 1
        else:
            if abs(src_row[i] - src_row[i + 2]) <= mean_dis * 3:
                index = index + [i, i + 1, i + 2]
                i = i + 3
            else:
                i += 1
    src = src[row_index, :][index, :]
    tgt = tgt[row_index, :][index, :]

    # filter by y-axis
    col_index = np.argsort(src[:, 1])
    src_col = src[col_index, 1]
    dis = src_col[4:] - src_col[:-4]
    mean_dis = np.mean(dis) / 2
    index = []
    i = 0
    while i < src.shape[0]:
        if i > src.shape[0] - 3:
            if abs(src_col[i] - src_col[i - 2]) <= mean_dis * 3:
                index.append(i)
            i += 1
        else:
            if abs(src_col[i] - src_col[i + 2]) <= mean_dis * 3:
                index = index + [i, i + 1, i + 2]
                i = i + 3
            else:
                i += 1
    return src[col_index, :][index, :], tgt[col_index, :][index, :]


# filter by corresponding
def filter_geometry(src, tgt, window_size=3, index_flag=False, shifting=None):
    new_tgt = tgt.copy()
    if shifting:
        mode, d = shifting
        if mode == "l":
            new_tgt[:, 0] = new_tgt[:, 0] - d
        elif mode == "r":
            new_tgt[:, 0] = new_tgt[:, 0] + d
        elif mode == "d":
            new_tgt[:, 1] = new_tgt[:, 1] + d
    else:
        new_tgt = tgt[:, :]

    dis = np.sqrt(np.square(src[:, 0] - new_tgt[:, 0]) + np.square(src[:, 1] - new_tgt[:, 1]))
    global_mean_dis = np.mean(dis)
    radius = window_size // 2
    index = []
    for i in range(src.shape[0]):
        if i <= radius - 1:
            dis_m = np.mean(dis[:window_size])
            if dis[i] <= dis_m * 1.5 and dis[i] <= global_mean_dis * 1.5:
                index.append(i)
        else:
            dis_m = np.mean(dis[i - radius: i + radius + 1])
            if dis[i] <= dis_m * 1.5 and dis[i] <= global_mean_dis * 1.5:
                index.append(i)
    if not index_flag:
        return src[index, :], tgt[index, :]
    else:
        return index


def rigidity_cons(x, y, x_, y_):
    flag = True
    for i in range(4):
        V = (x[(i + 1) % 4] - x[i]) * (y[(i + 2) % 4] - y[(i + 1) % 4]) - (y[(i + 1) % 4] - y[i]) * (
                    x[(i + 2) % 4] - x[(i + 1) % 4])
        V_ = (x_[(i + 1) % 4] - x_[i]) * (y_[(i + 2) % 4] - y_[(i + 1) % 4]) - (y_[(i + 1) % 4] - y_[i]) * (
                x_[(i + 2) % 4] - x_[(i + 1) % 4])
        V_s = np.sign(V)
        V_s_ = np.sign(V_)
        if V_s != V_s_:
            flag = False
            break
    return flag


def SIFT(im1, im2):
    sift = cv2.SIFT_create()
    kp1, dsp1 = sift.detectAndCompute(im1, None)  # None --> mask
    kp2, dsp2 = sift.detectAndCompute(im2, None)
    return kp1, dsp1, kp2, dsp2


def flann_match(kp1, dsp1, kp2, dsp2, ratio=0.4, im1_mask=None, im2_mask=None, shifting=None):
    """
    return DMatch (queryIdx, trainIdx, distance)
    queryIdx: index of query keypoint
    trainIdx: index of target keypoint
    distance: Euclidean distance
    """
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(dsp1, dsp2, k=2)
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            # good.append([m.queryIdx, m.trainIdx])
            if im1_mask is not None and im2_mask is not None:
                im1_x, im1_y = np.int32(np.round(kp1[m.queryIdx].pt))
                im2_x, im2_y = np.int32(np.round(kp2[m.trainIdx].pt))
                if im1_mask[im1_y][im1_x] and im2_mask[im2_y][im2_x]:
                    good.append([m.queryIdx, m.trainIdx])
            else:
                good.append([m.queryIdx, m.trainIdx])

    srcdsp = np.float32([kp1[m[0]].pt for m in good])
    tgtdsp = np.float32([kp2[m[1]].pt for m in good])

    kp_length = len(srcdsp)
    if kp_length <= 2:
        print("feature number = %d" % len(srcdsp))
        if len(srcdsp) == 1:
            return [], []
        return srcdsp, tgtdsp

    _, index = np.unique(srcdsp[:, 0], return_index=True)
    srcdsp = srcdsp[np.sort(index), :]
    tgtdsp = tgtdsp[np.sort(index), :]

    if len(srcdsp) >= 8:
        srcdsp, tgtdsp = filter_isolate(srcdsp, tgtdsp)
        tgtdsp, srcdsp = filter_isolate(tgtdsp, srcdsp)

    srcdsp, tgtdsp = filter_geometry(srcdsp, tgtdsp, shifting=shifting)
    return srcdsp, tgtdsp


def loG(img):
    img = cv2.GaussianBlur(img, (7, 7), 0)
    img = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    img = cv2.convertScaleAbs(img)
    img = normalize_img(img)
    img = np.where(img >= 0.1, img, 0)
    return img


def stitch_add_mask_linear_border(mask1, mask2, mode=None):
    height, width = mask1.shape
    x_map = np.sum(mask1, axis=1)
    y_map = np.sum(mask1, axis=0)
    y_l = min(np.argwhere(y_map >= 1.0))[0]
    y_r = max(np.argwhere(y_map >= 1.0))[0]
    x_u = min(np.argwhere(x_map >= 1.0))[0]
    x_d = max(np.argwhere(x_map >= 1.0))[0]

    mask_added = mask1 + mask2
    mask_super = np.where(mask_added > 0, 1.0, 0)
    mask_overlap = np.where(mask_added > 1.0, 1.0, 0)
    o_x_map = np.sum(mask_overlap, axis=1)
    o_y_map = np.sum(mask_overlap, axis=0)
    o_x_u = min(np.argwhere(o_x_map >= 1.0))[0]
    o_x_d = max(np.argwhere(o_x_map >= 1.0))[0]
    o_y_l = min(np.argwhere(o_y_map >= 1.0))[0]
    o_y_r = max(np.argwhere(o_y_map >= 1.0))[0]

    radius_ratio = 0.15

    x_median = (o_x_u + o_x_d) // 2
    x_radius = int((o_x_d - x_median) * radius_ratio)
    y_median = (o_y_l + o_y_r) // 2
    y_radius = int((o_y_r - y_median) * radius_ratio)

    mass_overlap_1 = np.zeros(mask_overlap.shape)
    if mode is None:
        if abs(o_x_u - x_u) <= 3:
            mass_overlap_1[x_u:o_x_d + 1, :] = np.tile(np.linspace(0, 1, o_x_d - x_u + 1).reshape(-1, 1), (1, width))
        elif abs(o_x_d - x_d) <= 3:
            mass_overlap_1[o_x_u:o_x_d + 1, :] = np.tile(np.linspace(1, 0, o_x_d - o_x_u + 1).reshape(-1, 1),
                                                         (1, width))
        elif abs(o_y_l - y_l) <= 3:
            mass_overlap_1[:, y_l:o_y_r + 1] = np.tile(np.linspace(0, 1, o_y_r - y_l + 1).reshape(1, -1), (height, 1))
        else:
            mass_overlap_1[:, o_y_l:o_y_r + 1] = np.tile(np.linspace(1, 0, o_y_r - o_y_l + 1).reshape(1, -1),
                                                         (height, 1))
    else:
        if mode == 'u':
            mass_overlap_1[x_median - x_radius:x_median + x_radius, :] = np.tile(
                np.linspace(0, 1, 2 * x_radius).reshape(-1, 1), (1, width))
            mass_overlap_1[x_median + x_radius: o_x_d, :] = 1.0
        elif mode == 'd':
            mass_overlap_1[x_median - x_radius:x_median + x_radius, :] = np.tile(
                np.linspace(1, 0, 2 * x_radius).reshape(-1, 1),
                (1, width))
            mass_overlap_1[o_x_u: x_median - x_radius, :] = 1.0

        elif mode == 'l':
            mass_overlap_1[:, y_median - y_radius:y_median + y_radius] = np.tile(
                np.linspace(0, 1, 2 * y_radius).reshape(1, -1), (height, 1))
            mass_overlap_1[:, y_median + y_radius: o_y_r] = 1.0
        else:
            mass_overlap_1[:, y_median - y_radius:y_median + y_radius] = np.tile(
                np.linspace(1, 0, 2 * y_radius).reshape(1, -1),
                (height, 1))
            mass_overlap_1[:, o_y_l: y_median - y_radius] = 1.0
    mass_overlap_1 *= mask_overlap
    mass_overlap_2 = (1 - mass_overlap_1) * mask_overlap
    mass_overlap_1 = mass_overlap_1 + mask1 - mask_overlap
    mass_overlap_2 = mass_overlap_2 + mask2 - mask_overlap
    return mass_overlap_1, mass_overlap_2, mask_super, mask_overlap


def stitch_add_mask_linear_per_border(mask1, mask2):
    height, width = mask1.shape
    mask_added = mask1 + mask2
    mask_super = np.where(mask_added > 0, 1.0, 0)
    mask_overlap = np.where(mask_added > 1.0, 1.0, 0)
    radius_ratio = 0.15

    mass_overlap_1 = np.zeros(mask_overlap.shape)
    patch_width = width // 3
    for i in range(3):
        left_w = i * patch_width
        if i < 2:
            right_w = (i + 1) * patch_width
        else:
            right_w = width
        temp_overlap = mask_overlap[:, left_w: right_w]
        temp_x_map = np.sum(temp_overlap, axis=1)
        temp_x_u = min(np.argwhere(temp_x_map >= 1.0))[0]
        temp_x_d = max(np.argwhere(temp_x_map >= 1.0))[0]
        temp_median = (temp_x_u + temp_x_d) // 2
        temp_radius = int((temp_x_d - temp_median) * radius_ratio)

        mass_overlap_1[temp_median - temp_radius:temp_median + temp_radius, left_w: right_w] = np.tile(
            np.linspace(1, 0, 2 * temp_radius).reshape(-1, 1),
            (1, right_w - left_w))
        mass_overlap_1[temp_x_u: temp_median - temp_radius, left_w: right_w] = 1.0
    mass_overlap_1 *= mask_overlap
    mass_overlap_2 = (1 - mass_overlap_1) * mask_overlap
    mass_overlap_1 = mass_overlap_1 + mask1 - mask_overlap
    mass_overlap_2 = mass_overlap_2 + mask2 - mask_overlap
    return mass_overlap_1, mass_overlap_2, mask_super, mask_overlap


def direct_stitch(im1, im2, im1_mask, im2_mask):
    im1_shape = im1.shape
    im2_shape = im2.shape

    dis_h = int((im1_shape[0] - im2_shape[0]) // 2)

    if im1_mask is None:
        im1_mask = np.ones((im1.shape[0], im1.shape[1]))

    dis_w = int(im1.shape[1] * 0.1)

    h = im1_shape[0]
    extra_w = im2_shape[1] - dis_w
    w = im1_shape[1] + extra_w
    stitching_im1_res = np.zeros((h, w))
    stitch_im1_mask = np.zeros((h, w))
    stitching_im1_res[:, :im1_shape[1]] = im1
    stitch_im1_mask[:, :im1_shape[1]] = im1_mask

    stitch_im2_res = np.zeros((h, w))
    stitch_im2_mask = np.zeros((h, w))
    stitch_im2_mask[dis_h:im2_shape[0] + dis_h, im1_shape[1] - dis_w:] = 1.0
    stitch_im2_mask = stitch_im2_mask * (1 - stitch_im1_mask)
    stitch_im2_res[dis_h:im2_shape[0] + dis_h, im1_shape[1] - dis_w:] = im2

    stitching_res = stitching_im1_res * stitch_im1_mask + stitch_im2_res * stitch_im2_mask
    mass = stitch_im1_mask + stitch_im2_mask

    return stitching_res, mass, None

from Utils import flann_match, generate_None_list, rigidity_cons
import numpy as np
import random
from scipy import linalg


def RANSAC(ps1, ps2, iter_num, min_dis):
    point_num = ps1.shape[0]

    x1 = ps1[:, 0].reshape(-1, 1)
    y1 = ps1[:, 1].reshape(-1, 1)
    x2 = ps2[:, 0].reshape(-1, 1)
    y2 = ps2[:, 1].reshape(-1, 1)

    scale = 1 / np.mean(np.vstack([x1, y1, x2, y2]))
    x1 *= scale
    y1 *= scale
    x2 *= scale
    y2 *= scale

    X = np.hstack([np.zeros((point_num, 3)), x1, y1, np.ones((point_num, 1)), -y2 * x1, -y2 * y1, -y2])
    Y = np.hstack([x1, y1, np.ones((point_num, 1)), np.zeros((point_num, 3)), -x2 * x1, -x2 * y1, -x2])

    H = generate_None_list(iter_num, 1)
    score = generate_None_list(iter_num, 1)
    ok = generate_None_list(iter_num, 1)
    A = generate_None_list(iter_num, 1)

    for it in range(iter_num):
        subset = random.sample(list(range(point_num)), 4)
        if not rigidity_cons(x1[subset, :], y1[subset, :], x2[subset, :], y2[subset, :]):
            ok[it] = False
            score[it] = 0
            continue
        A[it] = np.vstack([X[subset, :], Y[subset, :]])
        U, S, V = linalg.svd(A[it])
        h = V.T[:, 8]
        H[it] = h.reshape(3, 3)
        dis = np.dot(X, h)**2 + np.dot(Y, h)**2
        ok[it] = dis < min_dis * min_dis
        score[it] = sum(ok[it])

    score, best = max(score), np.argmax(score)
    ok = ok[best]
    A = np.vstack([X[ok, :], Y[ok, :]])
    U, S, V = linalg.svd(A, 0)
    h = V.T[:, 8]
    H = h.reshape(3, 3)
    H = np.dot(np.dot(np.array([[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1]]), H),
               np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]]))
    return H, ok


def rigid_transform(kp1, dsp1, kp2, dsp2, im1_mask, im2_mask, mode):
    dis = 0.0
    if mode == "d":
        dis = im1_mask.shape[0]
    elif mode == "l" or "r":
        dis = im1_mask.shape[1]
    shifting = (mode, dis)
    X1, X2 = flann_match(kp1, dsp1, kp2, dsp2, ratio=0.4, im1_mask=im1_mask, im2_mask=im2_mask, shifting=shifting)
    if len(X1) == 0:
        return None, None, None, None
    try:
        H, ok = RANSAC(X1.copy(), X2.copy(), 2000, 0.1)
    except Exception:
        ok = [True for _ in X1]
        return None, None, None, None
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
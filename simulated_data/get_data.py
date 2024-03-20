import cv2
import numpy as np
import os
import sys
import math
import random
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
import time

def cut_dark(img):
    x_map = np.sum(img, axis=1)
    y_map = np.sum(img, axis=0)
    y_l = min(np.argwhere(y_map >= 1.0))[0]
    y_r = max(np.argwhere(y_map >= 1.0))[0]
    x_u = min(np.argwhere(x_map >= 1.0))[0]
    x_d = max(np.argwhere(x_map >= 1.0))[0]
    # print(x_u, x_d, y_l, y_r)
    return img[x_u: x_d, y_l: y_r]

def get_inv_affine(affine):
    inv_affine = np.zeros((3,3))
    inv_affine[:2, :] = affine
    inv_affine[2, 2] = 1
    inv_affine = np.linalg.inv(inv_affine)
    inv_affine = inv_affine[:2, :]

    return inv_affine

def rigid_transform_tr(image, rotation, translation, mode='bilinear', inv=False):

    # (512,512)
    shape_size = image.shape
    H,W = image.shape

    center = np.float32(shape_size) // 2
    # Random affine
    affine = np.zeros((2, 3))
    s = np.sin(rotation * math.pi / 180.0)
    c = np.cos(rotation * math.pi / 180.0)
    affine[0, 0] = c
    affine[0, 1] = -s
    affine[0, 2] = translation[0]
    affine[1, 0] = s
    affine[1, 1] = c
    affine[1, 2] = translation[1]
    if inv:
        matrix = np.zeros((3, 3))
        matrix[:2, :] = affine
        matrix[2, 2] = 1
        matrix = np.linalg.inv(matrix)
        affine = matrix[:2, :]

    # image = cv2.warpAffine(image, warp_m[:2, :], shape_size, borderMode=cv2.BORDER_REFLECT_101)
    if mode == 'bilinear':
        image = cv2.warpAffine(image, affine, (W,H))
    if mode == 'nearest':
        image = cv2.warpAffine(image, affine, (W,H), borderMode=cv2.INTER_NEAREST)
    return image

def get_datafolder(data_path,store_path1,store_path2,store_path3):
    if not os.path.exists(store_path1):
        os.mkdir(store_path1)
    if not os.path.exists(store_path2):
        os.mkdir(store_path2)
    if not os.path.exists(store_path3):
        os.mkdir(store_path3)
    for idx in tqdm(range(1,101)):
        folder_path = store_path1 + 'C' + str(idx)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        
        path1 = data_path + 'C' + str(idx) + '.bmp'
        img = cv2.imread(path1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        path2 = folder_path + '/C' + str(idx) + '.bmp'
        cv2.imwrite(path2,img)

    for idx in tqdm(range(1,101)):
        folder_path = store_path2 + 'C' + str(idx)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        
        path1 = data_path + 'C' + str(idx) + '.bmp'
        img = cv2.imread(path1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        path2 = folder_path + '/C' + str(idx) + '.bmp'
        cv2.imwrite(path2,img)
    
    for idx in tqdm(range(1,101)):
        folder_path = store_path3 + 'C' + str(idx)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        
        path1 = data_path + 'C' + str(idx) + '.bmp'
        img = cv2.imread(path1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        path2 = folder_path + '/C' + str(idx) + '.bmp'
        cv2.imwrite(path2,img)

def make_data(store_path1,store_path2,store_path3):
    # for idx in tqdm(range(1,101)):
    for idx in tqdm(range(3,101)):
    # for idx in tqdm(range(1,3)):
        start_time = time.time()
        #tile_v1
        folder_path1 = store_path1 + 'C' + str(idx)
        img_path1 = folder_path1 + '/C' + str(idx) + '.bmp'
        img1 = cv2.imread(img_path1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        #tile_v2
        folder_path2 = store_path2 + 'C' + str(idx)
        img_path2 = folder_path2 + '/C' + str(idx) + '.bmp'
        img2 = cv2.imread(img_path2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        #tile_v3
        folder_path3 = store_path3 + 'C' + str(idx)
        img_path3 = folder_path3 + '/C' + str(idx) + '.bmp'
        img3 = cv2.imread(img_path3)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        h,w = img1.shape#3072,3072/1536,1536
        # print(h,w)
        #affine.txt
        txt_path1 = folder_path1 + '/affine.txt'
        affine_file1 = open(txt_path1,'w+')
        txt_path2 = folder_path2 + '/affine.txt'
        affine_file2 = open(txt_path2,'w+')
        txt_path3 = folder_path3 + '/affine.txt'
        affine_file3 = open(txt_path3,'w+')
        # sys.exit()
        random_state_num = np.random.randint(50, 300)
        for num in range(1,5):
            if num == 1:
                affine_matrix1 = np.zeros((2,3),dtype=float)
                fiji_path1 = folder_path1 + '/C'+ str(idx) + '_1' + '.bmp'
                mist_path1 = folder_path1 + '/C'+ str(idx) + '_1_1' + '.bmp'
                fiji_path2 = folder_path2 + '/C'+ str(idx) + '_1' + '.bmp'
                mist_path2 = folder_path2 + '/C'+ str(idx) + '_1_1' + '.bmp'
                fiji_path3 = folder_path3 + '/C'+ str(idx) + '_1' + '.bmp'
                mist_path3 = folder_path3 + '/C'+ str(idx) + '_1_1' + '.bmp'
                image1 = rigid_transform_tr(img1, rotation=0, translation=[0,0])
                image2 = rigid_transform_tr(img2, rotation=0, translation=[0,0])
                image3 = rigid_transform_tr(img3, rotation=0, translation=[0,0])
                affine_matrix1[0,0]=1
                affine_matrix1[1,1]=1
                affine_file1.write(str(affine_matrix1[0,0]))
                affine_file2.write(str(affine_matrix1[0,0]))
                affine_file3.write(str(affine_matrix1[0,0]))
                affine_file1.write(',')
                affine_file2.write(',')
                affine_file3.write(',')
                affine_file1.write(str(affine_matrix1[1,0]))
                affine_file2.write(str(affine_matrix1[1,0]))
                affine_file3.write(str(affine_matrix1[1,0]))
                affine_file1.write(',')
                affine_file2.write(',')
                affine_file3.write(',')
                affine_file1.write(str(affine_matrix1[0,1]))
                affine_file2.write(str(affine_matrix1[0,1]))
                affine_file3.write(str(affine_matrix1[0,1]))
                affine_file1.write(',')
                affine_file2.write(',')
                affine_file3.write(',')
                affine_file1.write(str(affine_matrix1[1,1]))
                affine_file2.write(str(affine_matrix1[1,1]))
                affine_file3.write(str(affine_matrix1[1,1]))
                affine_file1.write(',')
                affine_file2.write(',')
                affine_file3.write(',')
                affine_file1.write(str(affine_matrix1[0,2]))
                affine_file2.write(str(affine_matrix1[0,2]))
                affine_file3.write(str(affine_matrix1[0,2]))
                affine_file1.write(',')
                affine_file2.write(',')
                affine_file3.write(',')
                affine_file1.write(str(affine_matrix1[1,2]))
                affine_file2.write(str(affine_matrix1[1,2]))
                affine_file3.write(str(affine_matrix1[1,2]))
                affine_file1.write('\n')
                affine_file2.write('\n')
                affine_file3.write('\n')
                image1 = image1[71:1611,71:1611]#(1536+(1536-36)*0.1/2=1611)[1575,1575]
                cv2.imwrite(fiji_path1,image1)
                cv2.imwrite(mist_path1,image1)
                image2 = image2[71:1611,71:1611]#(1536+(1536-36)*0.1/2=1611)[1575,1575]
                cv2.imwrite(fiji_path2,image2)
                cv2.imwrite(mist_path2,image2)
                image3 = image3[71:1611,71:1611]#(1536+(1536-36)*0.1/2=1611)[1575,1575]
                cv2.imwrite(fiji_path3,image3)
                cv2.imwrite(mist_path3,image3)
                # sys.exit()
            elif num == 2:
                affine_matrix2 = np.zeros((2,3))
                fiji_path1 = folder_path1 + '/C'+ str(idx) + '_2' + '.bmp'
                mist_path1 = folder_path1 + '/C'+ str(idx) + '_1_2' + '.bmp'
                fiji_path2 = folder_path2 + '/C'+ str(idx) + '_2' + '.bmp'
                mist_path2 = folder_path2 + '/C'+ str(idx) + '_1_2' + '.bmp'
                fiji_path3 = folder_path3 + '/C'+ str(idx) + '_2' + '.bmp'
                mist_path3 = folder_path3 + '/C'+ str(idx) + '_1_2' + '.bmp'
                t_x = random.uniform(0.005,0.01)*w
                t_y = random.uniform(0.005,0.01)*h
                rotate = random.choice((-1, 1)) * random.uniform(0.5,1)
                image1 = rigid_transform_tr(img1, rotation=0, translation=[t_x,t_y])
                image2 = rigid_transform_tr(img2, rotation=rotate, translation=[t_x,t_y])
                image3 = rigid_transform_tr(img3, rotation=rotate, translation=[t_x,t_y])
                image3 = get_elastic_pic(image3,[[100,1500],[1500,1600]],random_state_num,folder_path3+'/2_')
                c = np.cos(rotate * math.pi / 180.0)
                s = np.sin(rotate * math.pi / 180.0)
                 
                # affine_file1.write(str(affine_matrix2[0,0]))
                affine_file1.write(str(1))
                affine_file2.write(str(affine_matrix2[0,0]))
                affine_file3.write(str(affine_matrix2[0,0]))
                affine_file1.write(',')
                affine_file2.write(',')
                affine_file3.write(',')
                # affine_file1.write(str(affine_matrix2[1,0]))
                affine_file1.write(str(0))
                affine_file2.write(str(affine_matrix2[1,0]))
                affine_file3.write(str(affine_matrix2[1,0]))
                affine_file1.write(',')
                affine_file2.write(',')
                affine_file3.write(',')
                # affine_file1.write(str(affine_matrix2[0,1]))
                affine_file1.write(str(0))
                affine_file2.write(str(affine_matrix2[0,1]))
                affine_file3.write(str(affine_matrix2[0,1]))
                affine_file1.write(',')
                affine_file2.write(',')
                affine_file3.write(',')
                # affine_file1.write(str(affine_matrix2[1,1]))
                affine_file1.write(str(1))
                affine_file2.write(str(affine_matrix2[1,1]))
                affine_file3.write(str(affine_matrix2[1,1]))
                affine_file1.write(',')
                affine_file2.write(',')
                affine_file3.write(',')
                affine_file1.write(str(affine_matrix2[0,2]))
                affine_file2.write(str(affine_matrix2[0,2]))
                affine_file3.write(str(affine_matrix2[0,2]))
                affine_file1.write(',')
                affine_file2.write(',')
                affine_file3.write(',')
                affine_file1.write(str(affine_matrix2[1,2]))
                affine_file2.write(str(affine_matrix2[1,2]))
                affine_file3.write(str(affine_matrix2[1,2]))
                affine_file1.write('\n')
                affine_file2.write('\n')
                affine_file3.write('\n')
                image1 = image1[71:1611,1461:3001]#(1536-(1536-36)*0.1/2=1461)[1575,1575]
                cv2.imwrite(fiji_path1,image1)
                cv2.imwrite(mist_path1,image1)
                image2 = image2[71:1611,1461:3001]#(1536-(1536-36)*0.1/2=1461)[1575,1575]
                cv2.imwrite(fiji_path2,image2)
                cv2.imwrite(mist_path2,image2)
                image3 = image3[71:1611,1461:3001]#(1536-(1536-36)*0.1/2=1461)[1575,1575]
                cv2.imwrite(fiji_path3,image3)
                cv2.imwrite(mist_path3,image3)
                # sys.exit()
            elif num == 3:
                affine_matrix3 = np.zeros((2,3))
                fiji_path1 = folder_path1 + '/C'+ str(idx) + '_3' + '.bmp'
                mist_path1 = folder_path1 + '/C'+ str(idx) + '_2_1' + '.bmp'
                fiji_path2 = folder_path2 + '/C'+ str(idx) + '_3' + '.bmp'
                mist_path2 = folder_path2 + '/C'+ str(idx) + '_2_1' + '.bmp'
                fiji_path3 = folder_path3 + '/C'+ str(idx) + '_3' + '.bmp'
                mist_path3 = folder_path3 + '/C'+ str(idx) + '_2_1' + '.bmp'
                t_x = random.uniform(0.005,0.01)*w
                t_y = random.uniform(0.005,0.01)*h
                rotate = random.choice((-1, 1)) * random.uniform(0.5,1)
                image1 = rigid_transform_tr(img1, rotation=0, translation=[t_x,t_y])
                image2 = rigid_transform_tr(img2, rotation=rotate, translation=[t_x,t_y])
                image3 = rigid_transform_tr(img3, rotation=rotate, translation=[t_x,t_y])
                image3 = get_elastic_pic(image3,[[1500,1600],[100,1500]],random_state_num,folder_path3+'/3_')
                # image = rigid_transform_tr(img, rotation=0, translation=[t_x,t_y])
                c = np.cos(rotate * math.pi / 180.0)
                s = np.sin(rotate * math.pi / 180.0)
                affine_matrix3[0,0]=c
                affine_matrix3[0,1]=-s
                affine_matrix3[1,0]=s
                affine_matrix3[1,1]=c
                affine_matrix3[0,2]=t_x
                affine_matrix3[1,2]=t_y
                affine_file1.write(str(1))
                affine_file2.write(str(affine_matrix3[0,0]))
                affine_file3.write(str(affine_matrix3[0,0]))
                affine_file1.write(',')
                affine_file2.write(',')
                affine_file3.write(',')
                affine_file1.write(str(0))
                affine_file2.write(str(affine_matrix3[1,0]))
                affine_file3.write(str(affine_matrix3[1,0]))
                affine_file1.write(',')
                affine_file2.write(',')
                affine_file3.write(',')
                affine_file1.write(str(0))
                affine_file2.write(str(affine_matrix3[0,1]))
                affine_file3.write(str(affine_matrix3[0,1]))
                affine_file1.write(',')
                affine_file2.write(',')
                affine_file3.write(',')
                affine_file1.write(str(1))
                affine_file2.write(str(affine_matrix3[1,1]))
                affine_file3.write(str(affine_matrix3[1,1]))
                affine_file1.write(',')
                affine_file2.write(',')
                affine_file3.write(',')
                affine_file1.write(str(affine_matrix3[0,2]))
                affine_file2.write(str(affine_matrix3[0,2]))
                affine_file3.write(str(affine_matrix3[0,2]))
                affine_file1.write(',')
                affine_file2.write(',')
                affine_file3.write(',')
                affine_file1.write(str(affine_matrix3[1,2]))
                affine_file2.write(str(affine_matrix3[1,2]))
                affine_file3.write(str(affine_matrix3[1,2]))
                affine_file1.write('\n')
                affine_file2.write('\n')
                affine_file3.write('\n')
                image1 = image1[1461:3001,71:1611]#(1536-(1536-36)*0.1/2=1461)[1575,1575]
                cv2.imwrite(fiji_path1,image1)
                cv2.imwrite(mist_path1,image1)
                image2 = image2[1461:3001,71:1611]#(1536-(1536-36)*0.1/2=1461)[1575,1575]
                cv2.imwrite(fiji_path2,image2)
                cv2.imwrite(mist_path2,image2)
                image3 = image3[1461:3001,71:1611]#(1536-(1536-36)*0.1/2=1461)[1575,1575]
                cv2.imwrite(fiji_path3,image3)
                cv2.imwrite(mist_path3,image3)
                # sys.exit()
            elif num == 4:
                affine_matrix4 = np.zeros((2,3))
                fiji_path1 = folder_path1 + '/C'+ str(idx) + '_4' + '.bmp'
                mist_path1 = folder_path1 + '/C'+ str(idx) + '_2_2' + '.bmp'
                fiji_path2 = folder_path2 + '/C'+ str(idx) + '_4' + '.bmp'
                mist_path2 = folder_path2 + '/C'+ str(idx) + '_2_2' + '.bmp'
                fiji_path3 = folder_path3 + '/C'+ str(idx) + '_4' + '.bmp'
                mist_path3 = folder_path3 + '/C'+ str(idx) + '_2_2' + '.bmp'
                t_x = random.uniform(0.005,0.01)*w
                t_y = random.uniform(0.005,0.01)*h
                rotate = random.choice((-1, 1)) * random.uniform(0.5,1)
                image1 = rigid_transform_tr(img1, rotation=0, translation=[t_x,t_y])
                image2 = rigid_transform_tr(img2, rotation=rotate, translation=[t_x,t_y])
                image3 = rigid_transform_tr(img3, rotation=rotate, translation=[t_x,t_y])
                c = np.cos(rotate * math.pi / 180.0)
                s = np.sin(rotate * math.pi / 180.0)
                affine_matrix4[0,0]=c
                affine_matrix4[0,1]=-s
                affine_matrix4[1,0]=s
                affine_matrix4[1,1]=c
                affine_matrix4[0,2]=t_x
                affine_matrix4[1,2]=t_y
                affine_file1.write(str(1))
                affine_file2.write(str(affine_matrix4[0,0]))
                affine_file3.write(str(affine_matrix4[0,0]))
                affine_file1.write(',')
                affine_file2.write(',')
                affine_file3.write(',')
                affine_file1.write(str(0))
                affine_file2.write(str(affine_matrix4[1,0]))
                affine_file3.write(str(affine_matrix4[1,0]))
                affine_file1.write(',')
                affine_file2.write(',')
                affine_file3.write(',')
                affine_file1.write(str(0))
                affine_file2.write(str(affine_matrix4[0,1]))
                affine_file3.write(str(affine_matrix4[0,1]))
                affine_file1.write(',')
                affine_file2.write(',')
                affine_file3.write(',')
                affine_file1.write(str(1))
                affine_file2.write(str(affine_matrix4[1,1]))
                affine_file3.write(str(affine_matrix4[1,1]))
                affine_file1.write(',')
                affine_file2.write(',')
                affine_file3.write(',')
                affine_file1.write(str(affine_matrix4[0,2]))
                affine_file2.write(str(affine_matrix4[0,2]))
                affine_file3.write(str(affine_matrix4[0,2]))
                affine_file1.write(',')
                affine_file2.write(',')
                affine_file3.write(',')
                affine_file1.write(str(affine_matrix4[1,2]))
                affine_file2.write(str(affine_matrix4[1,2]))
                affine_file3.write(str(affine_matrix4[1,2]))
                affine_file1.write('\n')
                affine_file2.write('\n')
                affine_file3.write('\n')
                image1 = image1[1461:3001,1461:3001]#(1536-(1536-36)*0.1/2=1461)[1575,1575]
                cv2.imwrite(fiji_path1,image1)
                cv2.imwrite(mist_path1,image1)
                image2 = image2[1461:3001,1461:3001]#(1536-(1536-36)*0.1/2=1461)[1575,1575]
                cv2.imwrite(fiji_path2,image2)
                cv2.imwrite(mist_path2,image2)
                image3 = image3[1461:3001,1461:3001]#(1536-(1536-36)*0.1/2=1461)[1575,1575]
                cv2.imwrite(fiji_path3,image3)
                cv2.imwrite(mist_path3,image3)
                # sys.exit()
        affine_file1.close() 
        affine_file2.close() 
        affine_file3.close()   
        restore_img(idx,store_path1,store_path2,store_path3)
        print(time.time()-start_time)

def overlap_mask_elastic_transform(image, mask, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx_ran_deform = (random_state.rand(*shape) * 2 - 1) * mask
    res_mask = 1 - mask
    dx_ran_res = ((random_state.rand(*shape) * 2 - 1) * 0.01) * res_mask
    dx_ran = dx_ran_deform + dx_ran_res
    dx_ran[:, :, 1] = dx_ran[:, :, 0]
    dy_ran_deform = (random_state.rand(*shape) * 2 - 1) * mask
    dy_ran_res = ((random_state.rand(*shape) * 2 - 1) * 0.01) * res_mask
    dy_ran = dy_ran_deform + dy_ran_res
    dy_ran[:, :, 1] = dy_ran[:, :, 0]
    dx = gaussian_filter(dx_ran, sigma) * alpha
    dy = gaussian_filter(dy_ran, sigma) * alpha
    dz = np.zeros_like(dx)
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), np.stack([dx[..., 0], dy[..., 0]], axis=0)

def get_elastic_pic(transformed_total,area,random_state_num,path):
    #area is 2*2,[[h_start,h_end],[w_start,w_end]]
    deformed_mask = np.zeros(transformed_total.shape)
    for i in range(0, 3072, 80):
        deformed_mask[i:i + 5, :] = 1.0
        deformed_mask[:, i:i + 5] = 1.0

    im_merge = np.concatenate((transformed_total[..., None], deformed_mask[..., None]), axis=2)
    transformed_overlap = np.zeros(transformed_total.shape)
    transformed_overlap[area[0][0]:area[0][1],area[1][0]:area[1][1]] = 1
    im_trans_mask = np.concatenate((transformed_overlap[..., None], transformed_overlap[..., None]), axis=2)
    count = 0
    while count < 4:
        im_merge_t, indices = overlap_mask_elastic_transform(im_merge, im_trans_mask, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08,random_state=np.random.RandomState(random_state_num))
        im_t = im_merge_t[..., 0]
        im_mask_t = im_merge_t[..., 1]
        im_merge = im_merge_t
        count += 1
    im_t = im_merge_t[..., 0]#im_merge_t[..., 0]
    im_mask_t = im_merge_t[..., 1]
    cv2.imwrite(path + 'im_mask_t1.bmp',im_mask_t*255)
    return im_t

def restore_img(idx,store_path1,store_path2,store_path3):
    # for idx in tqdm(range(1,101)):
    for num in tqdm(range(1,4)):
        if num == 1:
            store_path = store_path1
        elif num == 2:
            store_path = store_path2
        elif num == 3:
            store_path = store_path3
        folder_path = store_path + 'C' + str(idx)
        img_path = folder_path + '/C' + str(idx) + '.bmp'
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h,w = img.shape
    #restore img
        affine_path = folder_path + '/affine.txt'
        affine=open(affine_path,'r')
        affine_matrix = affine.readlines()
        # print(affine_matrix)

        #pic1
        mask1 = np.zeros((h,w))
        image1 = np.zeros((h,w))
        img1_path = folder_path + '/C'+ str(idx) + '_1' + '.bmp'
        img1 = cv2.imread(img1_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        mask1[71:1611,71:1611]=1
        image1[71:1611,71:1611]=img1
        # [36:1611,1461:3036]
        affine_matrix1 = affine_matrix[0]
        affine1 = np.zeros((2,3))
        affine1[0,0] = affine_matrix1.split(',')[0]
        affine1[1,0] = affine_matrix1.split(',')[1]
        affine1[0,1] = affine_matrix1.split(',')[2]
        affine1[1,1] = affine_matrix1.split(',')[3]
        affine1[0,2] = affine_matrix1.split(',')[4]
        affine1[1,2] = affine_matrix1.split(',')[5]
        # print(affine1)
        inv_affine1 = get_inv_affine(affine1)
        pic1 = cv2.warpAffine(image1,inv_affine1,(h,w))
        Mask1 = cv2.warpAffine(mask1,inv_affine1,(h,w))
        # Mask1 = np.where(pic1 > 0, 1.0, 0)
        Mask1 = np.where(Mask1 >= 1.0, 1.0, 0)
        pic1=pic1*Mask1

        #pic2
        mask2 = np.zeros((h,w))
        image2 = np.zeros((h,w))
        img2_path = folder_path + '/C'+ str(idx) + '_2' + '.bmp'
        img2 = cv2.imread(img2_path)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        mask2[71:1611,1461:3001]=1
        image2[71:1611,1461:3001]=img2
        # [36:1611,1461:3036]
        affine_matrix2 = affine_matrix[1]
        affine2 = np.zeros((2,3))
        affine2[0,0] = affine_matrix2.split(',')[0]
        affine2[1,0] = affine_matrix2.split(',')[1]
        affine2[0,1] = affine_matrix2.split(',')[2]
        affine2[1,1] = affine_matrix2.split(',')[3]
        affine2[0,2] = affine_matrix2.split(',')[4]
        affine2[1,2] = affine_matrix2.split(',')[5]
        # print(affine2)
        inv_affine2 = get_inv_affine(affine2)
        pic2 = cv2.warpAffine(image2,inv_affine2,(h,w))
        Mask2 = cv2.warpAffine(mask2,inv_affine2,(h,w))
        # Mask2 = np.where(pic2 > 0, 1.0, 0)
        Mask2 = np.where(Mask2 >= 1.0, 1.0, 0)
        pic2=pic2*Mask2

        #pic3
        mask3 = np.zeros((h,w))
        image3 = np.zeros((h,w))
        img3_path = folder_path + '/C'+ str(idx) + '_3' + '.bmp'
        img3 = cv2.imread(img3_path)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        mask3[1461:3001,71:1611]=1
        image3[1461:3001,71:1611]=img3
        # [36:1611,1461:3036]
        affine_matrix3 = affine_matrix[2]
        affine3 = np.zeros((2,3))
        affine3[0,0] = affine_matrix3.split(',')[0]
        affine3[1,0] = affine_matrix3.split(',')[1]
        affine3[0,1] = affine_matrix3.split(',')[2]
        affine3[1,1] = affine_matrix3.split(',')[3]
        affine3[0,2] = affine_matrix3.split(',')[4]
        affine3[1,2] = affine_matrix3.split(',')[5]
        # print(affine3)
        inv_affine3 = get_inv_affine(affine3)
        pic3 = cv2.warpAffine(image3,inv_affine3,(h,w))
        Mask3 = cv2.warpAffine(mask3,inv_affine3,(h,w))
        Mask3 = np.where(Mask3 >= 1.0, 1.0, 0)
        pic3=pic3*Mask3

        #pic4
        mask4 = np.zeros((h,w))
        image4 = np.zeros((h,w))
        img4_path = folder_path + '/C'+ str(idx) + '_4' + '.bmp'
        img4 = cv2.imread(img4_path)
        img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        mask4[1461:3001,1461:3001]=1
        image4[1461:3001,1461:3001]=img4
        # [36:1611,1461:3036]
        affine_matrix4 = affine_matrix[3]
        affine4 = np.zeros((2,3))
        affine4[0,0] = affine_matrix4.split(',')[0]
        affine4[1,0] = affine_matrix4.split(',')[1]
        affine4[0,1] = affine_matrix4.split(',')[2]
        affine4[1,1] = affine_matrix4.split(',')[3]
        affine4[0,2] = affine_matrix4.split(',')[4]
        affine4[1,2] = affine_matrix4.split(',')[5]
        # print(affine4)
        inv_affine4 = get_inv_affine(affine4)
        pic4 = cv2.warpAffine(image4,inv_affine4,(h,w))
        Mask4 = cv2.warpAffine(mask4,inv_affine4,(h,w))
        # Mask4 = np.where(Mask4 > 0, 1.0, 0)
        Mask4 = np.where(Mask4 >= 1.0, 1.0, 0)
        pic4 = pic4*Mask4
        # pic44 = pic4*Mask4+pic3*Mask3
        # restore4_path = folder_path + '/C'+ str(idx) + '_restore44' + '.bmp'
        # cv2.imwrite(restore4_path,pic44)

        #start restore
        mask1to2 = Mask1 + Mask2
        mask1to2_ones = np.where(mask1to2 > 1.5, 1.0, 0)
        mask1to2_half = np.where(mask1to2 > 1.5, 0.5, 0)
        pic12 = pic1*(Mask1-mask1to2_half)+pic2*(Mask2-mask1to2_half)
        # pic12 = pic1*(Mask1-mask1to2_ones)+pic2*(Mask2-mask1to2_ones)+(pic1+pic2)*mask1to2_ones/2.0

        mask3to4 = Mask3 + Mask4
        mask3to4_ones = np.where(mask3to4 > 1.5, 1.0, 0)
        mask3to4_half = np.where(mask3to4 > 1.5, 0.5, 0)
        pic34 = pic3*(Mask3-mask3to4_half)+pic4*(Mask4-mask3to4_half)
        # pic34 = pic3*(Mask3-mask3to4_ones)+pic4*(Mask4-mask3to4_ones)+(pic3+pic4)*mask3to4_ones/2.0

        mask12 = Mask1 + Mask2 - mask1to2_ones
        mask34 = Mask3 + Mask4 - mask3to4_ones
        mask12to34 = mask12 + mask34
        mask12to34_half = np.where(mask12to34 > 1.5, 0.5, 0)
        mask12to34_ones = np.where(mask12to34 > 1.5, 1.0, 0)
        # pic = pic12*(mask12-mask12to34_ones)+pic34*(mask34 - mask12to34_ones)+(pic12+pic34)*mask12to34_ones/2.0
        pic = pic12*(mask12-mask12to34_half)+pic34*(mask34 - mask12to34_half)
        restore_path = folder_path + '/C'+ str(idx) + '_restore' + '.bmp'
        cv2.imwrite(restore_path,pic)

if __name__ == '__main__':
    data_path = '~/stitch_data/'
    store_path1 = '~/stitch_data/simulation1/'
    store_path2 = '~/stitch_data/simulation2/'
    store_path3 = '~/stitch_data/simulation3/'
    make_data(store_path1,store_path2,store_path3)

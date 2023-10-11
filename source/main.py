from stitching import three_stitching, two_stitching
import os
from tqdm import tqdm
import numpy as np
import cv2
from argparse import ArgumentParser

parser = ArgumentParser(description="Autostitch")
parser.add_argument('--input_path', type=str, default="", help='input file path')
parser.add_argument('--store_path', type=str, default="", help='store res path')
parser.add_argument('--pattern', type=int, default=3, help='two or three')
parser.add_argument('--refine', action='store_true', default=False, help='refine or not')

args = parser.parse_args()
pattern = args.pattern
refine_flag = args.refine

data_path = args.input_path
store_path = args.store_path

image_list = []
for root, dirs, files in os.walk(data_path):
    for file in files:
        first, _, _ = file.split(".")[0].split("_")
        image_list.append(first)
image_list = list(set(image_list))
image_list.sort()

if not os.path.exists(store_path):
    os.mkdir(store_path)

for top_num in tqdm(image_list):
    if pattern == 3:
        try:
            three_stitching(data_path, store_path, top_num, refine_flag=refine_flag)
        except Exception:
            final_res = np.zeros((1000, 1000))
            cv2.imwrite(os.path.join(store_path, "".join([str(top_num), "-res", ".bmp"])), final_res)
    elif pattern == 2:
        try:
            two_stitching(data_path, store_path, top_num, refine_flag=refine_flag)
        except Exception:
            final_res = np.zeros((1000, 1000))
            cv2.imwrite(os.path.join(store_path, "".join([str(top_num), "-res", ".bmp"])), final_res)



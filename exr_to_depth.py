# coding: utf-8


import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt
import OpenEXR as exr
import Imath
from tqdm import tqdm
import argparse
import os

Ma = 0.900000000000
mi = 100.00000000000

parser = argparse.ArgumentParser(description='exr to depth program')
parser.add_argument('--input_img_directory', type=str, default='datasets/exr', help='input directory')
parser.add_argument('--output_img_directory', type=str, default='datasets/train', help='output directory')

args = parser.parse_args()

input_dir_name = args.input_img_directory
output_dir_name = args.output_img_directory

# 画像のファイル名を読み込み
input_list = os.listdir(input_dir_name)

# 順番を0からにする
input_list.sort()

# 保存用ディレクトリの作成
os.makedirs(output_dir_name)

for i in tqdm(range(len(input_list))):
        # if os.path.exists(home_path+path+"/def_cut_noi/"+ str(i).zfill(6) + ".png") == True:
    # exrfile = exr.InputFile('datasets/data/exr/'+str(i).zfill(4)+'.exr')
    exrfile = exr.InputFile(os.path.join(input_dir_name, input_list[i]))
    # exrfile = exr.InputFile('60to0_touei/exr/'+str(i-1)+'.exr')
    #exrfile = exr.InputFile("0001.exr")
    header = exrfile.header()

    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
        
    channelData = dict()
        
        # convert all channels in the image to numpy arrays
    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)
        
        channelData[c] = C
        
    colorChannels = ['R', 'G', 'B', 'A'] if 'A' in header['channels'] else ['R', 'G', 'B']
    img = np.concatenate([channelData[c][...,np.newaxis] for c in colorChannels], axis=2)
        
    """    # linear to standard RGB
    img[..., :3] = np.where(img[..., :3] <= 0.0031308,
                            12.92 * img[..., :3],
                            1.055 * np.power(img[..., :3], 1 / 2.4) - 0.055)
    """
    #print(img.shape)

    if mi > np.min(img):
        mi = np.min(img)
    img_temp = np.where(img >=100000.0 , 0 , img)
    print(np.max(img_temp))
    if np.max(img_temp) < 100000.0:
        if Ma < np.max(img_temp):
            Ma  = np.max(img_temp)

print("Max: "'{:.9f}'. format(Ma))
print("min: "'{:.9f}'. format(mi))
    
for i in tqdm(range(len(input_list))):
    # exrfile = exr.InputFile('datasets/data/exr/'+str(i).zfill(4)+'.exr')
    exrfile = exr.InputFile(os.path.join(input_dir_name, input_list[i]))
    # exrfile = exr.InputFile('60to0_touei/exr/'+str(i-1)+'.exr')
    #exrfile = exr.InputFile("0001.exr")
    header = exrfile.header()

    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
        
    channelData = dict()
        
        # convert all channels in the image to numpy arrays
    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)
        
        channelData[c] = C
        
    colorChannels = ['R', 'G', 'B', 'A'] if 'A' in header['channels'] else ['R', 'G', 'B']
    img = np.concatenate([channelData[c][...,np.newaxis] for c in colorChannels], axis=2)
    nor = (img - mi) / (Ma - mi)

    img2 = nor * 255

    cv2.imwrite(output_dir_name + '/' + 'img'+str(i+1).zfill(4)+'.png', img2)

#print(img)
#print("Max: ". format(np.max(img)))
#print("min: ". format(np.min(img)))
print("Max: "'{:.9f}'. format(Ma))
print("min: "'{:.9f}'. format(mi))
#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/11/16 17:31
# @Author   : Mayanmei
# @FileName : data_combination.py
# ============================================

import os
import cv2
import numpy as np
import json
from PIL import Image, ImageDraw
from data_generator.color_processor import format_convert,nparray2base64
import logging
logger = logging.getLogger(__name__)

'''
    此项目做数据合成，目的是生成样本+样本增强，例如：多张印章合成到一张大图上
'''

def init_log():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_files(data_path, exts=['jpg', 'png', 'jpeg', 'JPG']):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    for parent, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    return files


def get_json(input_path, output_path):
    files = os.listdir(input_path)
    for file in files:
        print(file)
        name, ext = os.path.splitext(file)
        image_path = os.path.join(input_path, file)
        image = cv2.imread(image_path)
        if image is not None:
            h, w, _ = image.shape
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
            print(len(contours))
            cnt = contours[0]
            hull = cv2.convexHull(cnt)
            hull = np.array(hull)
            point = format_convert(hull)
            cv2.polylines(image, [hull], True, (0, 255, 0), 3)

            class_points = {
                "label": "stamp",
                "points": point,
                "group_id": " ",
                "shape_type": "polygon",
                "flags": {}
            }

            prediction = {"version": "3.16.7",
                          "flags": {},
                          'shapes': class_points,
                          "imagePath": file,
                          "imageData": nparray2base64(image),
                          "imageHeight": h,
                          "imageWidth": w
                          }
            prediction_json_path = os.path.join(input_path + name + ".json")
            with open(prediction_json_path, "w", encoding='utf-8') as g:
                json.dump(prediction, g, indent=2, sort_keys=True, ensure_ascii=False)

            cv2.imwrite(os.path.join(output_path + file),image)



if __name__ == "__main__":
    init_log()
    input_path = "data/enhance/input/"
    output_path = "data/enhance/debug/"
    get_json(input_path, output_path)

    # input = "/Users/yanmeima/Desktop/generator/"
    # resize = "data/enhance/resize/"
    # files = os.listdir(input)
    # for file in files:
    #     img_path = os.path.join(input,file)
    #     img = cv2.imread(img_path, -1)
    #     if img is not None:
    #         h,w = img.shape[:2]
    #         img = cv2.resize(img,(int(w/3),int(h/3)), interpolation=cv2.INTER_AREA)
    #         cv2.imwrite(os.path.join(resize,file[:-4] + "_2" + ".png"), img)


#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import os
import cv2
import json
import numpy as np
import PIL.Image as Image
import random
from data_generator.color_processor import nparray2base64
import logging
logger = logging.getLogger(__name__)

POSSIBILITY_PERSPECTIVE = 0.8  # 需要被做透视的概率
POSSIBILITY_RESIZE = 0.7  # 需要被做压缩的概率
# 图片缩放最小比例
RESIZE_MIN = 0.3
# 图片缩放最大比例
RESIZE_MAX = 1.1

image_size = 330  # 每张小图片的大小
image_colnum = 3  # 合并成一张图后，一行有几个小图
number = 15  # 随机取图片的个数
W = 1750  # 背景图片的缩放大小
H = 1500  # 背景图片的缩放大小
w_interval = 60  # 两两小图的行间距
h_interval = 30  # 两两小图的列间距

# 随机接受概率
def _random_accept(accept_possibility):
    return np.random.choice([True, False], p=[accept_possibility, 1 - accept_possibility])

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


def random_perspective(img, boxes):
    '''
    随机透射
    :param img:
    :param boxes:
    :return:
    '''
    if not _random_accept(POSSIBILITY_PERSPECTIVE):
        logger.debug("不随机透射")
        return img, boxes
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGRA)
    # #透射
    # TODO 随机生成坐标 计算输出坐标
    # 输入、输出图像上相应的四个点
    h, w = img.shape[:2]
    # 四点透视
    pts1 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
    # TODO 任意固定两个点
    # TODO 往哪儿透射，设定一个幅度 可以各边各20% 的幅度吧
    percent = 0.2
    w_percent = w * percent
    h_percent = h * percent

    # TODO 四点分别几个点位置偏可以 有概率的选择，否则维持原点

    x1 = np.random.randint(0, w_percent)
    y1 = np.random.randint(0, h_percent)
    x2 = np.random.randint(0, w_percent)
    y2 = h - np.random.randint(1, h_percent)
    x3 = w - np.random.randint(1, w_percent)
    y3 = h - np.random.randint(1, h_percent)
    x4 = w - np.random.randint(1, w_percent)
    y4 = np.random.randint(0, h_percent)

    pts2 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    logger.info("随机透射：%r,%r", pts1, pts2)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

    dst = Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGRA2RGBA))
    # new_boxes = cv2.perspectiveTransform(np.array(boxes), M)
    if boxes is not None and len(boxes) > 0:
        temp = np.array(boxes, dtype="float32")
        temp = np.array([temp])
        box_new = cv2.perspectiveTransform(temp, M)
        box_new = box_new.astype(np.int)
        box_list = box_new.tolist()

    return dst, box_list[0]


def random_resize(img, boxes):
    '''
    随机压缩
    :param img:
    :param boxes:
    :return:
    '''
    if not _random_accept(POSSIBILITY_RESIZE):
        logger.debug("不随机压缩")
        return img, boxes

    ratio = random.uniform(RESIZE_MIN, RESIZE_MAX)
    w,h = img.size
    new_w, new_h = int(ratio * w), int(ratio * h)
    logger.info("图片随机缩放比例：%r", ratio)

    img = img.resize((int(ratio * w),int(ratio * h)), Image.ANTIALIAS)
    boxes = np.array(boxes) * ratio
    return img, boxes


def resize_by_width(infile, image_size):
    """按照宽度进行所需比例缩放"""
    im = Image.open(infile).convert("RGBA")
    # # todo: 透明背景转白色
    # sp = im.size
    # width = sp[0]
    # height = sp[1]
    # print(sp)
    # for yh in range(height):
    #     for xw in range(width):
    #         dot = (xw, yh)
    #         color_d = im.getpixel(dot)
    #         if (color_d[3] == 0):
    #             color_d = (255, 255, 255, 255)
    #             im.putpixel(dot, color_d)

    (x, y) = im.size
    lv = round(x / image_size, 2) + 0.01
    x_s = int(x // lv)
    y_s = int(y // lv)
    #print("x_s", x_s, y_s)
    out = im.resize((x_s, y_s), Image.ANTIALIAS)

    json_path = os.path.join(infile[:-4] + ".json")
    with open(json_path, 'r', encoding="utf-8") as f:
        json_data = json.load(f)
        shapes = json_data['shapes']
        pts = shapes['points']
        new_pts = []
        for point in pts:
            p_x = point[0] // lv
            p_y = point[1] // lv
            box = [p_x, p_y]
            new_pts.append(box)
        box = np.array(new_pts)

    return out, box


def get_new_img_xy(infile, image_size):
    """返回一个图片的宽、高像素"""
    im = Image.open(infile)
    (x, y) = im.size
    lv = round(x / image_size, 2) + 0.01
    x_s = x // lv
    y_s = y // lv
    # print("x_s", x_s, y_s)
    # out = im.resize((x_s, y_s), Image.ANTIALIAS)
    return x_s, y_s


# 定义图像拼接函数
def image_compose(to_image, image_colnum, image_size, image_rownum, image_names, image_save_path, x_new, y_new):
    #to_image = Image.new('RGB', (image_colnum * x_new, image_rownum * y_new), "white")  # 创建一个新图

    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    total_num = 0
    json_result = []
    for y in range(1, image_rownum + 1):
        for x in range(1, image_colnum + 1):
            from_image, box = resize_by_width(image_names[image_colnum * (y - 1) + x - 1], image_size)
            # from_image = Image.open(image_names[image_colnum * (y - 1) + x - 1]).resize((image_size,image_size ), Image.ANTIALIAS)

            # 随机透射 RGB 进去
            from_image, new_box = random_perspective(from_image, box)
            # 随机压缩
            from_image, new_box = random_resize(from_image, new_box)
            print(type(new_box))
            print(new_box)

            # 60,20可以调节相邻两张小图间隔
            x_begin = (x - 1) * (x_new + 170)
            y_begin = (y - 1) * (y_new + 25)
            to_image.paste(from_image, (x_begin, y_begin), from_image)

            # todo：贴图后章的新坐标
            latest_box = []
            for point in new_box:
                p_x = x_begin + point[0]
                p_y = y_begin + point[1]
                if p_x > W:
                    p_x = W
                if p_y > H:
                    p_y = H
                new_point = [p_x, p_y]
                latest_box.append(new_point)
            class_points = {
                "label": "stamp",
                "points": latest_box,
                "group_id": " ",
                "shape_type": "polygon",
                "flags": {}
            }
            json_result.append(class_points)

            total_num += 1
            if total_num == len(image_names):
                break

    return to_image, json_result


def merge_images(to_image, image_fullpath_list,image_save_path,image_size,image_colnum):
    # image_rownum = 4  # 图片间隔，也就是合并成一张图后，一共有几行
    image_rownum_yu = len(image_fullpath_list) % image_colnum
    if image_rownum_yu == 0:
        image_rownum = len(image_fullpath_list) // image_colnum
    else:
        image_rownum = len(image_fullpath_list) // image_colnum + 1

    x_list = []
    y_list = []
    for img_file in image_fullpath_list:
        img_x, img_y = get_new_img_xy(img_file, image_size)
        x_list.append(img_x)
        y_list.append(img_y)

    print("x_list", sorted(x_list))
    print("y_list", sorted(y_list))
    x_new = int(x_list[len(x_list) // 5 * 4])
    y_new = int(x_list[len(y_list) // 5 * 4])
    to_image,json_result = image_compose(to_image, image_colnum, image_size, image_rownum, image_fullpath_list, image_save_path, x_new, y_new)

    save_file(to_image, json_result, image_save_path)


def save_file(to_image,json_result,image_save_path):
    h,w = to_image.size
    to_image.save(image_save_path)
    img_pil2cv = cv2.cvtColor(np.array(to_image), cv2.COLOR_RGB2BGR)
    prediction = {"version": "3.16.7",
                  "flags": {},
                  'shapes': json_result,
                  "imagePath": image_save_path,
                  "imageData": nparray2base64(img_pil2cv),
                  "imageHeight": h,
                  "imageWidth": w
                  }
    prediction_json_path = os.path.join(image_save_path[:-4] + ".json")
    with open(prediction_json_path, "w", encoding='utf-8') as g:
        json.dump(prediction, g, indent=2, sort_keys=True, ensure_ascii=False)




if __name__ == '__main__':

    input_image_path = 'data/enhance/input/'
    output_image_path = 'data/enhance/output/'
    bg_image_path = 'data/enhance/bj/'

    num_gen = 10  # 合成几张大图

    image_fullpath_list = get_files(input_image_path)
    print("image_fullpath_list", len(image_fullpath_list), image_fullpath_list)

    bg_image_fullpath_list = get_files(bg_image_path)
    print("bg_image_fullpath_list", len(bg_image_fullpath_list), bg_image_fullpath_list)

    for i in range(5, num_gen):
        image_fullpath_list = random.sample(image_fullpath_list, number)
        bg_image_fullpath = random.sample(bg_image_fullpath_list, 1)
        print("bg_image_fullpath", bg_image_fullpath)
        to_image = Image.open(bg_image_fullpath[0])
        to_image = to_image.resize((W, H), Image.ANTIALIAS)

        image_save_path = os.path.join(output_image_path, str(i) + ".jpg")

        merge_images(to_image, image_fullpath_list, image_save_path, image_size, image_colnum)


    # ceshi
    # img_bgr = cv2.imread(r'data/enhance/input/公章.png',-1)
    # b, g, r = cv2.split(img_bgr)  # 分离三个颜色通道
    # img_rgb = cv2.merge([r, g, b])  # 融合三个颜色通道生成新图片
    # #img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(r'data/enhance/output/公章rgb1.png',img_rgb)

    # img = cv2.imread(r'data/enhance/input/公章.png', -1)
    # cv2.imwrite(r'data/enhance/output/公章rgb1_t.png', img)
    #
    # image = Image.open(r'data/enhance/input/公章.png')
    # # image = image.convert('RGBA')
    # # sp = image.size
    # # width = sp[0]
    # # height = sp[1]
    # # print(sp)
    # # for yh in range(height):
    # #     for xw in range(width):
    # #         dot = (xw, yh)
    # #         color_d = image.getpixel(dot)
    # #         if (color_d[3] == 0):
    # #             color_d = (255, 255, 255, 255)
    # #             image.putpixel(dot, color_d)
    # to_image = Image.open("data/bg/3400px4679px (2).jpg")
    # to_image.paste(image,(100,100),image)
    # to_image.save(r'data/enhance/output/公章1_t.png')
    #
    # to_image = Image.new('RGB', (1200, 1200), "white")  # 创建一个新图
    # to_image.save(r'data/enhance/output/背景.png')
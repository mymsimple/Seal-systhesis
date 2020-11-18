# -*- coding: utf-8 -*-
import sys
import cv2
import os
import base64
import json
import numpy as np
import logging
logger = logging.getLogger(__name__)

'''
    印章打标，提取印章，参考：https://blog.csdn.net/wsp_1138886114/article/details/82858380
    opencv形态变换：https://www.jianshu.com/p/dcecaf62da71
'''
# 颜色范围1
# COLOR = {
#     "RED": [[(0, 43, 46), (10, 255, 255)], [(156, 43, 46), (180, 255, 255)]],
#     "YELLOW": [(26, 43, 46), (34, 255, 255)],
#     "BLUE": [(100, 43, 46), (124, 255, 255)]
# }

np.set_printoptions(threshold=np.inf)

def stamp_extract(image, hue_image):
    low_range = np.array([156, 43, 46])
    high_range = np.array([180, 255, 255])
    mask = cv2.inRange(hue_image, low_range, high_range)

    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    #mask = cv2.erode(mask, element) # 腐蚀效果不好
    mask = cv2.dilate(mask, element) # 膨胀
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element) # 闭运算效果不好
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element) # 开运算效果不好

    index1 = mask == 255
    img = np.zeros(image.shape, np.uint8)
    img[:, :] = (255,255,255)
    img[index1] = image[index1]

    # 检测物体轮廓，参考：https://blog.csdn.net/laobai1015/article/details/76400725
    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(image, contours, -1, (0, 0, 255), 3)

    print(len(contours))
    result = []
    all_area = []
    for i in range(0,len(contours)-1):
        print(i)
        cnt = contours[i]
        hull = cv2.convexHull(cnt)
        hull = np.array(hull)
        point = format_convert(hull)
        rect = cv2.minAreaRect(np.array(hull))  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        w = rect[1][0]
        h = rect[1][1]
        area = GetAreaOfPolyGonbyVector(hull)
        # all_area.append(area)
        # all_area.sort(reverse=True)

        # 过滤小的多余的轮廓
        if area < 5000:
            continue
        else:
            cv2.polylines(image, [hull], True, (0, 255, 0), 3)
            class_points = {
                "label": "stamp",
                "points": point,
                "group_id": " ",
                "shape_type": "polygon",
                "flags": {}
            }
            result.append(class_points)

    #print(all_area)
    return img, image, mask, result


# 求凸包及多边形面积，参考：https://blog.csdn.net/cymy001/article/details/81366033
def GetAreaOfPolyGonbyVector(points):
    # 基于向量叉乘计算多边形面积
    area = 0
    if len(points) < 3: return 0
        #raise Exception("error")

    for i in range(0,len(points)-1):
        p1 = points[i]
        p2 = points[i + 1]
        triArea = (p1[0][0]*p2[0][1] - p2[0][0]*p1[0][1])/2
        area += triArea

    fn = (points[-1][0][0]*points[0][0][1]-points[0][0][0]*points[-1][0][1])/2
    return abs(area+fn)


def format_convert(approx):
    pos = approx.tolist()
    points = []
    for p in pos:
        x1 = np.float(p[0][0])
        y1 = np.float(p[0][1])
        points.append([x1, y1])
    return points


def nparray2base64(img_data):
    """
        nparray格式的图片转为base64（cv2直接读出来的就是）
    :param img_data:
    :return:
    """
    _, d = cv2.imencode('.jpg', img_data)
    return str(base64.b64encode(d), 'utf-8')


def save_file(base64_img, image, hue_image, extract_dir, counter_dir, name, f):
    h, w, _ = image.shape
    extract_image, image, mask, result = stamp_extract(image, hue_image)
    prediction = {"version": "3.16.7",
                  "flags": {},
                  'shapes': result,
                  "imagePath": f,
                  "imageData": base64_img,
                  "imageHeight": h,
                  "imageWidth": w
                  }
    prediction_json_path = os.path.join(counter_dir + name + ".json")
    with open(prediction_json_path, "w", encoding='utf-8') as g:
        json.dump(prediction, g, indent=2, sort_keys=True, ensure_ascii=False)

    cv2.imwrite(os.path.join(extract_dir, name + ".jpg"), extract_image)
    cv2.imwrite(os.path.join(counter_dir, name + ".jpg"), image)


# 单张测试
# if __name__ == '__main__':
#     image = cv2.imread("/Users/yanmeima/Desktop/yinzhang/test.jpg")
#     hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     extract_image, image, mask, result = stamp_extract(image, hue_image)
#     cv2.imwrite('/Users/yanmeima/Desktop/yinzhang/extract.jpg', extract_image)
#     cv2.imwrite('/Users/yanmeima/Desktop/yinzhang/mask.jpg', mask)
#     cv2.imwrite('/Users/yanmeima/Desktop/yinzhang/test_line.jpg', image)

if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG,
                        handlers=[logging.StreamHandler()])

    image_dir = "/Users/yanmeima/Desktop/yinzhang/images/"
    extract_dir = "/Users/yanmeima/Desktop/yinzhang/extract/"
    counter_dir = "/Users/yanmeima/Desktop/yinzhang/test/"

    if not os.path.exists(extract_dir): os.mkdir(extract_dir)
    if not os.path.exists(counter_dir): os.mkdir(counter_dir)

    files = os.listdir(image_dir)
    for f in files:
        print("处理图片：", f)
        name, ext = os.path.splitext(f)
        image = cv2.imread(os.path.join(image_dir, f))
        base64_img = nparray2base64(image)
        if image is None: continue
        hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        save_file(base64_img, image, hue_image, extract_dir, counter_dir, name, f)

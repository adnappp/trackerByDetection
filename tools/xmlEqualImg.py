
# -*- coding:utf-8 -*-
import os
import time
import random
import numpy
xmlfilepath = '/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/test/Annotations'
imagefilepath = '/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/test/JPEGImages'
total_xml = os.listdir(xmlfilepath)

total_image = os.listdir(imagefilepath)

num=len(total_xml)
image_num = len(total_image)
list_xml=range(num)
list_image=range(image_num)

xml=[]
image=[]
def deal():
    '''这个函数获取xml和图像的名称，不包含后缀名'''
    for i  in list_xml:
        #name=total_xml[i][:-4] #
        #改进为
        name = total_xml[i].split(".")[0]
        xml.append(name)
    for j in list_image:
        name=total_image[j].split(".")[0] #同xml，防止出现.jpeg或者jpg两种不同的格式
        image.append(name)


def equal():
    #获取xml和image相同部分，去除不一样的情况
    intersection = [x for x in xml if x in set(image)]
    print(len(intersection))
    #对xml去除不一致的地方
    for i in total_xml:
        filePath = os.path.split(i)  # 分割出目录与文件
        fileMsg = os.path.splitext(filePath[1])  # 分割出文件与文件扩展名
        #print(filePath[1])
        fileExt = fileMsg[1]  # 取出后缀名(列表切片操作)
        fileName = fileMsg[0]
        if fileName not in intersection:
            os.remove(os.path.join(xmlfilepath,filePath[1]))
    #对image去除不一致的地方
    for j in total_image:
        filePath = os.path.split(j)  # 分割出目录与文件
        fileMsg = os.path.splitext(filePath[1])  # 分割出文件与文件扩展名
        fileExt = fileMsg[1]  # 取出后缀名(列表切片操作)
        fileName = fileMsg[0]
        if fileName not in intersection:
            os.remove(os.path.join(imagefilepath,filePath[1]))

if __name__ == "__main__":
    deal()
    equal()
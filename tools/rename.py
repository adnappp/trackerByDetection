# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 18:04:24 2017

@author: wuhui
用来对图片进行重命名，并初始化目录结构
"""

import cv2
import os
from xml.etree.ElementTree import ElementTree
import  sys,os
import struct

def test(xmlfile):
    '''这个函数只是测试功能，实际中没有用到'''
    tree = ElementTree()
    tree.parse(xmlfile)
    all_objects = tree.getroot().getchildren()

    for object in all_objects:
        print object
        if object.tag == "filename":
            print object.text

def modifiedXml(xmlfile):
    '''这个函数是用来修改xml文件内部的filename的部分，因为在rename函数之后，每个xml文件命名是00000x的形式
       所以也对xml内部filename这个部分进行了修改
    '''
    tree = ElementTree()
    tree.parse(xmlfile)
    all_objects = tree.getroot().getchildren()

    for object in all_objects:
        if object.tag == "filename": #找到xml内部filename这个object
            #其中os.path.splitext(os.path.split(xmlfile)[1])[0] + ".jpg"就是00000X.jpg的结果
            object.text = os.path.splitext(os.path.split(xmlfile)[1])[0] + ".jpg" #找到就对其 进行修改
            print object.text
        #以下的if语句是之前因为只有一类进行的修改，当时以防不一致，所以为了保持一致加上，现在这部分要去掉
        '''
        if object.tag == "object":
            object.find("name").text = "1"
        '''
    tree.write(xmlfile,encoding="utf-8")

def rename(path):
    '''对图片和xml文件进行重命名，都命名成000000x的形式'''
    '''
    输入:
       path:为图片和xml所在的文件夹
    '''
    xmls = os.listdir(path + "/Annotations") #xml文件夹
    print xmls
    #imgs = os.listdir(path + "/JPEGImages")
    #print imgs
    cnt = 1
    prename = "000000"
    for xml in xmls: #遍历所有xml文件
        tree = ElementTree()
        tree.parse(os.path.join(path + "/Annotations",xml))
        print os.path.join(path + "/Annotations",xml)
        all_objects = tree.getroot().getchildren()
        for object in all_objects:
            #找到xml文件中对应的图片名字，保证xml和图片对应。便于之后修改
            if object.tag == "filename": 
                img = object.text 
        #新的图片的名字
        imgnewName = path + "/JPEGImages/" + prename[0:len(prename) - len(str(cnt))] + str(cnt) + ".jpg"
        print imgnewName
        print path+"/JPEGImages/"+img
        os.rename(path +"/JPEGImages/" + img, imgnewName)
        #modified the xml file
        xmlnewName = path + "/Annotations/" + prename[0:len(prename) - len(str(cnt))] + str(cnt) + ".xml"
        print xmlnewName
        print path + "/Annotations/" + xml
        os.rename(path + "/Annotations/" + xml, xmlnewName)
        #修改xml中对应的filename的图片名称为新的名称
        modifiedXml(xmlnewName)
        cnt += 1
    print cnt
    print 'done!'

if __name__ == '__main__':
    rename("/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/test")
    #modifiedXml("/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/test/Annotations/000001.xml")



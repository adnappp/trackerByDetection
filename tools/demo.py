#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
'''
# CLASSES = ('__background__',
#            '1')
CLASSES = ['__background__']
id_file = open('/home/panda/my-tf-faster-rcnn-simple/ids.txt', 'r')
for line in id_file.readlines():
    line = line.strip()
    CLASSES.append(line)
CLASSES = tuple(CLASSES)

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_100000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def get_box(dets,thresh=0.5):
    inds = np.where(dets[:, -1] >= thresh)[0]
    #because per class only has one cow
    if len(inds)>0:
        inds = np.where(dets[:,-1]==np.max(dets[:,-1]))[0]
    else:
        return None
    boxes = dets[inds,:4]
    '''
    for i in inds:
        box = dets[i,:4]
        if boxes is None:
            boxes = box
        else:
            boxes = np.array((boxes,box))
    '''
    return boxes

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    # print(class_name)
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    bbox_account = 0
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        bbox_account = bbox_account + 1
        # '''
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 4)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        text = '{:s}\{:.3f}'.format(class_name, score)
        cv2.putText(im, text, (int(bbox[0]), int(bbox[1] - 2)), font, 2, (0, 0, 255), 1)
        print(text + "," + (str(bbox[0])) + "," + str(bbox[1]) + "," + str(bbox[2]) + "," + str(bbox[3]))
        # '''
        '''
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    #plt.draw()
    '''
    print("the nubmer of object is:", bbox_account)


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        # print(cls)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
    return im

def demo2(sess, net, im):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    #im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        # print(cls)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
    return im
def demo3(sess,net,im):
    scores, boxes = im_detect(sess, net, im)
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    result = None
    classes = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        #print('a')
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        # print(cls)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        boxess = get_box(dets,CONF_THRESH)
        if boxess is not None:
            classes.append(cls)
            if result is None:
                result = boxess
            else:
                result = np.vstack((result,boxess))

    return result,classes

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args
def tar_detection(img):
    demonet = 'vgg16'
    dataset = 'pascal_voc'
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                           NETS[demonet][0])
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    net = vgg16()
    net.create_architecture("TEST", len(CLASSES),
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    targets,ids = demo3(sess, net, img)
    return targets,ids

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()


    # model path
    #demonet = args.demo_net
    demonet = 'vgg16'

    print("demonet is %s" % demonet)
    dataset = args.dataset
    dataset = 'pascal_voc'
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                           NETS[demonet][0])
    print("tfmodel is %s" % tfmodel)

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net = vgg16()
    net.create_architecture("TEST", len(CLASSES),
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    
    print('Loaded network {:s}'.format(tfmodel))
    '''
    cap = cv2.VideoCapture('saveVideo.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps =25
    videoWriter = cv2.VideoWriter('result2.mp4',fourcc,fps,(640,360))

    while(cap.isOpened()): 
       ret, frame = cap.read() 
       im = demo2(sess,net,frame)
       result = cv2.resize(im,(640,360))
       videoWriter.write(result)
       cv2.imshow("a",im)
       k = cv2.waitKey(20) 
       if (k & 0xff == ord('q')): 
          break
 
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()
    '''
    im_names = [  # "/home/chenxingli/dengtaAI/dataset/testimages/input/443048211133329708.jpg",
        # "/home/chenxingli/dengtaAI/dataset/testimages/input/119286569232544007.jpg",
        # "/home/chenxingli/dengtaAI/dataset/testimages/input/721687407986871341.jpg",
        "/home/panda/my-tf-faster-rcnn-simple/first.jpg"
    ]

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        im = demo(sess, net, im_name)
        # print("/home/chenxingli/dengtaAI/dataset/testimages/output/"+im_name)
        cv2.imshow('test',im)
       # cv2.waitKey(0)
        cv2.imwrite("result.jpg",im)
        #cv2.imwrite("E:/code/my-tf-faster-rcnn-simple/data/test/r1.jpg", im)
    plt.show()  # /home/dengta/faster-rcnn/my-tf-faster-rcnn-simple/data/test/test_result.jpg

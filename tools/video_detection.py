from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from imutils.video import VideoStream
# from imutils.video import FPS
import numpy as np
import argparse
# import imutils
import time
import cv2

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
from object_detection.utils import ops as utils_ops
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
'''
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
'''
CLASSES = ('__background__',
           '1')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_10000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}
#VIDEO ={'video': ('/home/dengta/Vedios/20180517_IMG_5645.MOV',)}
video_path = "/home/dengta/Videos/20180517_IMG_5645.MOV"

class LabelWithIOU:
    def __init__(self):
        self._record = np.zeros((300000, 4))
        self._name_idx = 0
        self._thresh = 0.2
        self._names = {0: "N2018P676402", 1: "N2018P424150",
                       2: "N2018P750476", 3: "N2018P431157",
                       4: "N2018P687413", 5: "N2018P432158",
                       6: "N2018P482208", 7: "N2018P434160",
                       8: "N2018P753479", 9: "N2018P795521",
                       10: "N2018P818544", 11: "N2018P919645",
                       12: "N2018P810536", 13: "N2018P674400",
                       14: "N2018P672398"}
        self._seq_name = {}
        self._count = 0

    def _iou(self, bb):
        ixmin = np.maximum(self._record[:, 0], bb[0])
        iymin = np.maximum(self._record[:, 1], bb[1])
        ixmax = np.minimum(self._record[:, 2], bb[2])
        iymax = np.minimum(self._record[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (self._record[:, 2] - self._record[:, 0] + 1.) *
               (self._record[:, 3] - self._record[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
        return ovmax, jmax

    def get_name(self, bb):
        iou, seq = self._iou(bb)
        # print(iou, idx)
        if iou > self._thresh:
            name = self._seq_name[seq]
        else:
            name = self._names[self._name_idx % 15]
            self._name_idx += 1
        self._record[self._count] = np.array(bb)
        self._seq_name[self._count] = name
        self._count += 1
        print(self._count)
        return name

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    mask = im
    im = im[:, :, (2, 1, 0)]
    #print(im.shape())
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    #mask = im#.transpose((1, 2, 0)).astype(np.uint8).copy()  #np.zeros_like(im, dtype=np.uint8).copy()
    label = LabelWithIOU()
    label.__init__()

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        #'''
        cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX #cv2.cv2.InitFont(cv2.FONT_HERSHEY_SIMPLEX,1,1,0,3,8)
        text = label.get_name(bbox)
        cv2.putText(mask, text, (int(bbox[0]), int(bbox[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        #'''
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
    plt.draw()
    '''
    return mask

def demo(sess, net, im):
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
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        img = vis_detections(im, cls, dets, thresh=CONF_THRESH)
    return img

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

def video_detection():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    print("demonet is %s" %demonet)
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])
    print("tfmodel is %s"%tfmodel)


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 2,
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    #initialize the vedio stream
    #initialize the FPS counter
    print("[INFO] starting video stream....")
    vs = cv2.VideoCapture(video_path)
    fps = 5
    if not vs.isOpened():
        print("Error opening video stream or file")
        exit(1)
    ## some videowriter props
    sz = (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #output
    '''
    fourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("/root/chenxingli/my-tf-faster-rcnn-simple/data/test/test_result.avi", fourcc,
                                  fps,(1280,720))  # 1280, 720
    '''

    # loop over the frames from the video stream
    ret, frame = vs.read()
    print('shape : ', frame.shape)
    # output
    #cv2.VideoWriter_fourcc(*'mp4v')
    #cv2.VideoWriter_fourcc(*'avc1')
    #cv2.VideoWriter_fourcc(*'MJPG')
    fourcc  = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #/root/chenxingli/my-tf-faster-rcnn-simple/data/test/test_result.avi
    videoWriter = cv2.VideoWriter("/home/dengta/Videos/20180517_IMG_5645.mp4", fourcc,
                                  fps, sz, True)  # (frame.shape[0],frame.shape[1])
    while True:
        #grap the frame from the threaded video stream and resize it
        #to have a maximum width of 400 pixels
        if frame is None:
            break
        #frame = imutils.resize(frame, width=400)

        # grab the frame dimensions and convert it to a blob
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        print('Demo for data/demo/{}'.format(frame))
        img = demo(sess, net, frame)
        #img = cv2.resize(img, (1920, 1080))
        videoWriter.write(img)
        ret, frame = vs.read()
        print('shape : ', type(frame))
        #cv2.imshow("Frame",img)



    cv2.destroyAllWindows()
    videoWriter.release()
    vs.release()


if __name__ == '__main__':
    video_detection()






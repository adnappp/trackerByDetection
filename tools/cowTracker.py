import numpy as np
import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import time
import argparse
import cv2
from demo import tar_detection
from muti_tracker import muti_tracker
import pandas as pd
import datetime
from cStringIO import StringIO
import PIL
#from sort import Sort
#from detector import GroundTruthDetections

def main():
   # tracker = Sort(use_dlib=True)
    total_time = 0.0
    total_frames = 0
    CameraId = 0
    plt.ion()
    fig = plt.figure()
   #video save
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps =25
    videoWriter = cv2.VideoWriter('trackResult.mp4',fourcc,fps,(640,360))
    colours = np.random.rand(32, 3)
    test_mov='saveVideo.avi'
    cap =cv2.VideoCapture(test_mov)
    ret, frame1 = cap.read()
   #use target detection to get per cows position and id
    targets,ids = tar_detection(frame1)
    print(targets)
   #initialize trackers
    cowtracker = muti_tracker()
    cowtracker.begin(targets, frame1,ids)
   #log_path
    log_file = 'tools/log.csv'
   #begin track
    i=0
    while(True):
        i=i+1
        targets=[]
        ids=[]
        ret ,frame = cap.read()
        #period detection
        if i<=15:
            targets, ids = tar_detection(frame)
            results = cowtracker.update2(targets, ids, frame)
            print('555555')
        else:
            if i%50==0 :
                targets, ids = tar_detection(frame)
            #print "detect"
        #detection result
            results = cowtracker.update2(targets,ids,frame)

        #visualization
        axl = fig.add_subplot(111, aspect="equal")
        axl.imshow(frame)

        if i%3 ==0:
            log = pd.read_csv(log_file)

            rowNum = log.index.size
            colNum = log.columns.size
            log.loc[rowNum+1] = ['' for j in range(colNum)]
            #time
            now_time = datetime.datetime.now()
            time1_str = datetime.datetime.strftime(now_time, '%Y-%m-%d %H:%M:%S')
            log['Time'].loc[rowNum+1]=time1_str
            for d in results:
                d = d.astype(np.int32)
                cowid=d[4]

                data = str(d[0] + d[2] / 2) + ',' + str(d[1] + d[3] / 2) + ',' + str(CameraId)
                log[str(cowid)].loc[rowNum+1]=data
            log.to_csv(log_file,index=0)


        for d in results:
            d = d.astype(np.int32)
            cowid = d[4]
            #print(d)

            axl.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3,ec=colours[cowid % 32, :]))
            axl.set_adjustable('box-forced')
        plt.axis('off')
        fig.canvas.flush_events()
        plt.draw()
        fig.tight_layout()
        #video save
        fig.savefig("temp.jpg",dpi=200)
        tempIm = cv2.imread("temp.jpg")
        #print(tempIm)
        resultIm = cv2.resize(tempIm,(640,360))
        videoWriter.write(resultIm)
        axl.cla()
    cap.release()
    videoWriter.release()
main()

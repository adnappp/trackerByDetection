import cv2
video_path = "saveVideo.avi"
cap = cv2.VideoCapture(video_path)
'''
frame_num = cap.get(7)
print frame_num
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
i=0
flag = int(frame_num*0.7)
success,frame1 = cap.read()
sz = frame1.shape
print sz
fps =25
videoWriter = cv2.VideoWriter('saveVideo.avi',fourcc,fps,(sz[1],sz[0]))
'''
i=0

while(i<1):
    i=i+1
    success,frame = cap.read()
    cv2.imwrite('first.jpg',frame)

#videoWriter.release()
cap.release()


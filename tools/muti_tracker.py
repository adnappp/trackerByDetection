from correlation_tracker import CorrelationTracker
import numpy as np
from data_association import associate_detections_to_trackers

min_dist = 15
max_dist = 100
class muti_tracker:

    def __init__(self):
        self.trackers = []
        self.frame_count = 0
    def update2(self,dets,detids,img=None):
        if self.trackers !=[]:
            for tracker in self.trackers:
                tracker.do_tracke(img)
        if dets!=[]:
            to_del=[]
            for i ,det1 in enumerate(dets[:-1]):
                for j in range(i+1,len(dets)):
                    if self.cal_dist(det1,dets[j])<min_dist:
                        to_del.append(i)
            print(dets)
            for t in reversed(to_del):
                np.delete(dets,t,0)

        if dets !=[] and self.trackers!=[]:
            #
            i=0
            tempdic = {}
            for i,detid in enumerate(detids):
                tempdic[detid]=[dets[i],i]
            to_del =[]
            while i<len(self.trackers):
                tid = self.trackers[i].get_id()
                if tid in tempdic :
                    temp = tempdic[tid]
                    flag =0
                    tdel = 0
                    for j,tracker in enumerate(self.trackers):
                        tpos = tracker.get_state()
                        if j!=i and self.cal_dist(tpos,temp[0])<min_dist:
                            flag =1
                            tdel =temp[1]
                    if flag ==1:
                        to_del.append(tdel)
                    else:
                        self.trackers.pop(i)
                        i=i-1
                i=i+1
            for t in reversed(to_del):
                np.delete(dets, t, 0)
            #
            to_del2=[]
            for j,det in enumerate(dets):
                for tracker in self.trackers:
                    if self.cal_dist(det,tracker.get_state())<min_dist and detids[j] != tracker.get_id():
                        to_del2.append(j)
            for t in reversed(to_del2):
                np.delete(dets, t, 0)

        self.add(dets,img,detids)
        #to_del=[]
        #for i , trk1 in enumerate(self.trackers[:-1]):
         #   trk1.get_state()
          #  for j in range(i+1,len(self.trackers)):

        trks = np.zeros((len(self.trackers), 5))
        for t , trk in enumerate(trks):
            pos = self.trackers[t].get_state()
            cowid = self.trackers[t].get_id()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cowid]
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        return trks
    def cal_dist(self,a,b):
       # a,b = a.astype(np.int),b.astype(np.int)
        ca = np.array([int(a[0])+int(a[2])/2,int(a[1])+int(a[3])/2])
        cb = np.array([int(b[0])+int(b[2])/2,int(b[1])+int(b[3])/2])
        return  np.linalg.norm(ca - cb)




    def update(self, dets, detids,img=None):
        """
        Params:
          dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1

        if dets !=[]:
            del self.trackers[:]
            self.begin(dets,img,detids)
            trks = np.zeros(((len(dets)),5))
            for i ,trk in enumerate(trks):
                det = dets[i]
                trk[:]=[long(det[0]),long(det[1]),long(det[2]),long(det[3]),detids[i]]
            trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
            return trks


        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos,cowid = self.trackers[t].do_tracke(img)  # for kal!
            # print(pos)
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cowid]
            if (np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        #dets
        '''
        if dets != []:
            newdets=range(len(detids))
            newdets=np.array(newdets)
            dels = []
            for i,tracker in enumerate(self.trackers):
                tempid = tracker.get_id()
                changeflag=False
                for j,detid in enumerate(detids):
                    if detid==tempid:
                        dpos=dets[j]
                        tracker.update(dets,img)
                        changeflag=True
                        #find new dets
                        np.delete(newdets, np.where(newdets == j)[0])

                if changeflag==False:
                    dels.append(i)
            if dels != []:
                for t in reversed(dels):
                    self.trackers.pop(t)
            if len(newdets) != 0:
                for ids in newdets:
                    trk =  CorrelationTracker(dets[ids], img,detids[ids])
                    self.trackers.append(trk)





        
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if dets == []:
                trk.update([], img)
            d = trk.get_state()
            if ((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))
        '''
        return trks
    def begin(self,dets,img,cowids):
        self.trackers=[]
     #   print(dets)
        for i,det in enumerate(dets):
            trk = CorrelationTracker(det, img,cowids[i])
            self.trackers.append(trk)
    def add(self,dets,img,cowids):
        for i,det in enumerate(dets):
            trk = CorrelationTracker(det, img,cowids[i])
            self.trackers.append(trk)





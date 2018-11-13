"""
@author: Mahmoud I.Zidan
"""

from dlib import correlation_tracker, rectangle

'''Appearance Model'''
class CorrelationTracker:

  def __init__(self,bbox,img,cowid):
    print(bbox[1])
   # print(img)
    self.tracker = correlation_tracker()
    
    self.tracker.start_track(img,rectangle(long(bbox[0]),long(bbox[1]),long(bbox[2]),long(bbox[3])))
    self.cowid=cowid

    self.confidence = 0. # measures how confident the tracker is! (a.k.a. correlation score)


  def do_tracke(self,img):
    self.confidence = self.tracker.update(img)

    return self.get_state(),self.cowid

  def update(self,bbox,img):
    '''
    self.time_since_update = 0
    self.hits += 1
    self.hit_streak += 1
'''
    '''re-start the tracker with detected positions (it detector was active)'''
    if bbox != []:
      self.tracker.start_track(img, rectangle(long(bbox[0]), long(bbox[1]), long(bbox[2]), long(bbox[3])))
    '''
    Note: another approach is to re-start the tracker only when the correlation score fall below some threshold
    i.e.: if bbox !=[] and self.confidence < 10.
    but this will reduce the algo. ability to track objects through longer periods of occlusions.
    '''

  def get_state(self):
    pos = self.tracker.get_position()
    return [pos.left(), pos.top(),pos.right(),pos.bottom()]
  def get_id(self):
    return self.cowid
 # def update_pos(self,pos):
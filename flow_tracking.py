import numpy as np
from scipy.optimize import linear_sum_assignment
from evaluation.bbox_util import convert_3dbox_to_8corner, iou3d
class ClassPropertyDescriptor(object):

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self

def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.25):   
	"""
	Assigns detections to tracked object (both represented as bounding boxes)
	detections:  N x 8 x 3
	trackers:	M x 8 x 3
	Returns 3 lists of matches, unmatched_detections and unmatched_trackers
	"""
	if (len(trackers)==0): 
		return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 8, 3), dtype=int)	
	iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

	for d, det in enumerate(detections):
		for t, trk in enumerate(trackers):
			iou_matrix[d, t] = iou3d(det, trk)[0]			 # det: 8 x 3, trk: 8 x 3
	# matched_indices = linear_assignment(-iou_matrix)	  # hougarian algorithm, compatible to linear_assignment in sklearn.utils

	row_ind, col_ind = linear_sum_assignment(-iou_matrix)	  
	matched_indices = np.stack((row_ind, col_ind), axis=1)

	unmatched_detections = []
	for d, det in enumerate(detections):
		if (d not in matched_indices[:, 0]): unmatched_detections.append(d)
	unmatched_trackers = []
	for t, trk in enumerate(trackers):
		if (t not in matched_indices[:, 1]): unmatched_trackers.append(t)

	#filter out matched with low IOU
	matches = []
	for m in matched_indices:
		if (iou_matrix[m[0], m[1]] < iou_threshold):
			unmatched_detections.append(m[0])
			unmatched_trackers.append(m[1])
		else: matches.append(m.reshape(1, 2))
	if (len(matches) == 0): 
		matches = np.empty((0, 2),dtype=int)
	else: matches = np.concatenate(matches, axis=0)

	return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class BoxTracker(object):
	"""
	This class represents the internel state of individual tracked objects observed as bbox.
	"""
	count = 0
	@classproperty
	def id(cls): return cls.count

	@id.setter
	def id(cls, value):
		cls.count = value

	def __init__(self, bbox3D,info):
		"""
		Initialises a tracker using initial bounding box.
		"""

		self.time_since_update = 0
		self.id = BoxTracker.count
		BoxTracker.count += 1
		self.history = []
		self.det_hist = []
		self.hits = 1           # number of total hits including the first detection
		self.hit_streak = 1     # number of continuing hit considering the first detection
		self.first_continuing_hit = 1
		self.still_first = True
		self.age = 0        # other info associated
		self.history.append(bbox3D)
		self.det_hist.append(bbox3D)
		self.info = info

	def update(self, bbox3D,info): 
		""" 
		Updates the state vector with observed bbox.
		"""
		self.time_since_update = 0
		self.hits += 1
		self.hit_streak += 1          # number of continuing hit
		if self.still_first:
			self.first_continuing_hit += 1      # number of continuing hit in the fist time
		self.det_hist.append(bbox3D)
		self.info = info


	def predict(self, preds):       
		self.age += 1
		if (self.time_since_update > 0):
			self.hit_streak = 0
			self.still_first = False
		self.time_since_update += 1
		self.history.append(preds)
		return self.history[-1]

	def get_state(self):
		return self.history[-1]

class F_Tracker(object):
	def __init__(self, max_age=2, min_hits=3, iou=0.25):
		self.max_age = max_age
		self.min_hits = min_hits
		self.trackers = []
		self.frame_count = 0
		# self.reorder = [3, 4, 5, 6, 2, 1, 0]
		# self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
		# x y z w h l theta
		self.reorder = [0, 1, 2, 6, 5, 3, 4] # [x,y,z,theta,l,w,h]
		self.reorder_back = [0, 1, 2, 5, 6, 4, 3] # [x,y,z,w,h,l,theta]
		self.iou = iou
		BoxTracker.id = 0


	def update(self, dets_all, preds,prev_bboxes):
		dets, info = dets_all['dets'], dets_all['info']
		dets = dets[:, self.reorder]
		preds = preds[:, self.reorder]
		prev_bboxes = prev_bboxes[:, self.reorder]
		self.frame_count += 1


		trks = np.zeros((len(self.trackers), 7))		 # N x 7 , # get predicted locations from existing trackers.
		to_del = []
		ret = []
		for t, trk in enumerate(trks):
			idx = np.unique(np.where(prev_bboxes[:,:3]==self.trackers[t].det_hist[-1][:3])[0])
			if len(idx)!=0:
				#pos = self.trackers[t].predict().reshape((-1, 1))
				pos = self.trackers[t].predict(preds[idx][0]).reshape((-1, 1))
				trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
				if (np.all(pos[:3]==[-1.,-1.,-1.])): 
					to_del.append(t)
			else: to_del.append(t)   
			
		#trks = np.ma.compress_rows(np.ma.masked_invalid(trks))   
		for t in reversed(to_del): 
			trks = np.delete(trks,t, axis=0)
			self.trackers.pop(t)

		
		dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]
		if len(dets_8corner) > 0: dets_8corner = np.stack(dets_8corner, axis=0)
		else: dets_8corner = []
		trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks]
		if len(trks_8corner) > 0: trks_8corner = np.stack(trks_8corner, axis=0)
		matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner,iou_threshold=self.iou)


		# update matched trackers with assigned detections
		for t, trk in enumerate(self.trackers):
			if t not in unmatched_trks:
				d = matched[np.where(matched[:, 1] == t)[0], 0]	 # a list of index
				trk.update(dets[d,:][0],info[d,:][0])

		# create and initialise new trackers for unmatched detections
		for i in unmatched_dets:		# a scalar of index
			trk = BoxTracker(dets[i],info[i])  
			self.trackers.append(trk)
		i = len(self.trackers)
		for trk in reversed(self.trackers):
			d = trk.get_state()		  # bbox location
			d = d[self.reorder_back]			# change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]

			if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
				ret.append(np.concatenate((d, [trk.id + 1], trk.info)).reshape(1, -1)) # +1 as MOT benchmark requires positive
			i -= 1

			# remove dead tracklet
			if (trk.time_since_update >= self.max_age): 
				self.trackers.pop(i)
		if (len(ret) > 0): return np.concatenate(ret)			# h,w,l,x,y,z,theta, ID, other info, confidence
		return np.empty((0, 15))	
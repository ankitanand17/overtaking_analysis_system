"""
sort.py - Minimal SORT tracker implementation (Kalman + IOU matching)
Author: compact version for Smart Overtaking System

This file implements the Simple Online and Realtime Tracking (SORT) algorithm.
It models object trajectories in 2D image coordinates using Kalman filters and
associates new bounding box detections to existing tracks using the Hungarian
matching logic (approximated here using IoU matrix thresholding).
"""

import numpy as np
from numpy.linalg import inv
import math
from collections import OrderedDict

def iou(bb_test, bb_gt):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.

    Args:
        bb_test (list/tuple): Bounding box [x1, y1, x2, y2].
        bb_gt (list/tuple): Bounding box [x1, y1, x2, y2].

    Returns:
        float: Intersection over Union overlap score between 0.0 and 1.0.
    """
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    
    # Calculate width and height of the intersection box
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    inter = w * h
    
    # Calculate individual box areas
    area1 = max(0., (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]))
    area2 = max(0., (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]))
    
    # Calculate union area
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union

class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed 
    as bounding boxes. Uses a Kalman filter to model constant velocity motion.
    """
    count = 0
    def __init__(self, bbox):
        """
        Initializes a new tracker state using the initial bounding box detection.

        Args:
            bbox (list): [x1, y1, x2, y2] coordinates.
        """
        x1,y1,x2,y2 = bbox
        
        # Convert bounding box coordinates to state variables:
        # cx, cy: Center point coordinates
        # s: Scale (area of bounding box)
        # r: Aspect ratio (width / height)
        cx = (x1+x2)/2.0
        cy = (y1+y2)/2.0
        w = max(1.0, x2-x1)
        h = max(1.0, y2-y1)
        s = w * h
        r = w / (h + 1e-6)
        
        # State vector x: [cx, cy, s, r, cx_dot, cy_dot, s_dot]^T
        # Constant velocity model is used for center coordinates and area.
        self._x = np.array([cx, cy, s, r, 0., 0., 0.]).reshape((7,1))
        
        # State Covariance matrix P: initial uncertainties
        self._P = np.eye(7) * 10.0
        
        # State Transition matrix F
        self._F = np.eye(7)
        dt = 1.0
        for i,j in ((0,4),(1,5),(2,6)):
            self._F[i,j] = dt
            
        # Measurement matrix H (we only measure [cx, cy, s, r])
        self._H = np.zeros((4,7))
        self._H[0,0] = 1.0; self._H[1,1] = 1.0; self._H[2,2] = 1.0; self._H[3,3] = 1.0
        
        # Measurement Covariance matrix R
        self._R = np.eye(4) * 1.0
        
        # Process Noise Covariance matrix Q
        self._Q = np.eye(7) * 0.01
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def predict(self):
        """
        Advances the Kalman filter state vector x and covariance matrix P 
        based on the constant velocity motion model transition.

        Returns:
            list: Predicted bounding box state estimate [x1, y1, x2, y2].
        """
        # x_k = F * x_{k-1}
        self._x = np.dot(self._F, self._x)
        # P_k = F * P_{k-1} * F^T + Q
        self._P = np.dot(self._F, np.dot(self._P, self._F.T)) + self._Q
        
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._x.copy())
        return self.get_state()

    def update(self, bbox):
        """
        Updates the Kalman filter state vector x and covariance matrix P 
        using the newly observed bounding box measurements.

        Args:
            bbox (list): [x1, y1, x2, y2] new bounding box measurements.
        """
        x1,y1,x2,y2 = bbox
        cx = (x1+x2)/2.0
        cy = (y1+y2)/2.0
        w = max(1.0, x2-x1)
        h = max(1.0, y2-y1)
        s = w * h
        r = w / (h + 1e-6)
        
        # Measurement vector z
        z = np.array([cx, cy, s, r]).reshape((4,1))
        
        # Innovation covariance and Kalman gain calculation
        yk = z - np.dot(self._H, self._x)
        S = np.dot(self._H, np.dot(self._P, self._H.T)) + self._R
        K = np.dot(self._P, np.dot(self._H.T, inv(S)))
        
        # Updated state vector x
        self._x = self._x + np.dot(K, yk)
        
        # Updated state covariance matrix P
        I = np.eye(self._F.shape[0])
        self._P = np.dot(I - np.dot(K, self._H), self._P)
        
        # Reset trackers metrics
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

    def get_state(self):
        """
        Translates the current Kalman filter state estimation vector [cx, cy, s, r] 
        back into bounding box coordinates.

        Returns:
            list: Bounding box coordinates [x1, y1, x2, y2].
        """
        cx = float(self._x[0]); cy = float(self._x[1])
        s = float(self._x[2]); r = float(self._x[3])
        
        # w = sqrt(s * r), h = s / w
        w = math.sqrt(abs(s * r))
        h = max(1e-3, s / (w + 1e-6))
        
        x1 = cx - w/2.0; y1 = cy - h/2.0; x2 = cx + w/2.0; y2 = cy + h/2.0
        return [x1, y1, x2, y2]

class Sort:
    """
    SORT (Simple Online and Realtime Tracking) algorithm coordinator class.
    Manages active target trackers, updates prediction, and handles matching logic.
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initializes the SORT tracker manager.

        Args:
            max_age (int): Max consecutive frames to keep dead tracks alive before deletion.
            min_hits (int): Minimum detections count before confirming track as target.
            iou_threshold (float): Minimum IoU overlap required to match detector boxes to tracks.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        """
        Performs SORT updates on every frame.

        Args:
            dets (np.ndarray): Nx5 array representing detections [x1, y1, x2, y2, score].

        Returns:
            np.ndarray: Mx5 array of verified active tracks [x1, y1, x2, y2, track_id].
        """
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        ret = []
        
        # 1. Obtain predictions from all active trackers
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t, :4] = pos
            trks[t, 4] = trk.id
            
        # If no detections exist in this frame, clean up aged trackers
        if dets.shape[0] == 0:
            i = len(self.trackers)
            for trk in reversed(self.trackers):
                if trk.time_since_update > self.max_age:
                    self.trackers.pop(i-1)
                i -= 1
            return np.empty((0,5))
            
        # 2. Compute IoU matrix between detections and existing tracker predictions
        iou_matrix = np.zeros((dets.shape[0], len(self.trackers)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(self.trackers):
                iou_matrix[d, t] = iou(det[:4], trk.get_state())
                
        # 3. Associate detections to trackers based on maximum IoU overlap
        matched_indices = []
        if iou_matrix.size > 0:
            for d in range(iou_matrix.shape[0]):
                t = np.argmax(iou_matrix[d])
                if iou_matrix[d, t] >= self.iou_threshold:
                    matched_indices.append((d, t))
                    
        # Filter matching conflicts to ensure unique 1-to-1 association
        unmatched_dets = set(range(dets.shape[0]))
        unmatched_trks = set(range(len(self.trackers)))
        matches = []
        for d,t in matched_indices:
            if d in unmatched_dets and t in unmatched_trks:
                matches.append((d,t))
                unmatched_dets.remove(d); unmatched_trks.remove(t)
                
        # 4. Update matched trackers with new detection measurements
        for (d,t) in matches:
            self.trackers[t].update(dets[d,:4])
            
        # 5. Create new trackers for unmatched detections
        for d in list(unmatched_dets):
            trk = KalmanBoxTracker(dets[d,:4])
            self.trackers.append(trk)
            
        # 6. Delete aged-out trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i-1)
            i -= 1
            
        # 7. Collect and return output coordinates for active tracks
        for trk in self.trackers:
            if (trk.hit_streak >= self.min_hits) or (self.frame_count <= self.min_hits):
                bbox = trk.get_state()
                ret.append([bbox[0], bbox[1], bbox[2], bbox[3], trk.id])
                
        if len(ret) == 0:
            return np.empty((0,5))
        return np.asarray(ret)

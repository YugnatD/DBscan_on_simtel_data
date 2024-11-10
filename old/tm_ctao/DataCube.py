import os
import sys
import time
import numpy as np



import tm_ctao.HECS as HECS
import tm_ctao.Frame as Frame


class DataCube:
  def __init__(self, frame_list: list):
    self._frame_list = frame_list
    self._num_frames = len(frame_list)
    self._num_r = frame_list[0].get_num_r()
    self._num_c = frame_list[0].get_num_c()
    self._num_a = frame_list[0].get_num_a()
  
  def copy(self):
    frame_list = [frame.copy() for frame in self._frame_list]
    return DataCube(frame_list)
  
  def get_frame(self, frame_num: int):
    return self._frame_list[frame_num]
  
  def get_num_frames(self):
    return self._num_frames
  
  def get_num_r(self):
    return self._num_r
  
  def get_num_c(self):
    return self._num_c
  
  def get_value(self, frame_num: int, a: int, r: int, c: int):
    return self._frame_list[frame_num].get_value_arc(a, r, c)
  
  def set_value(self, frame_num: int, a: int, r: int, c: int, value):
    self._frame_list[frame_num].set_value_arc(a, r, c, value)

  def plot(self):
    for frame in self._frame_list:
      frame.plot()
  
  def get_points_arc_above_threshold(self, threshold: float):
    points = []
    for frame_index in range(self._num_frames):
      for a in range(self._num_a):
        for r in range(self._num_r):
          for c in range(self._num_c):
            if self.get_value(frame_index, a, r, c) > threshold:
              points.append((a, r, c, frame_index))
    return points
  
  def dbscan_convolve(self, eps_time: int, eps_xy: int, min_samples: int):
    # create a copy of the data cube
    data_cube_out = self.copy()
    for a in range(self._num_a):
      for r in range(self._num_r):
        for c in range(self._num_c):
          # get the neighbors of the pixel to compute the sum
          neighbors = self._frame_list[0].get_list_neighbors_epsilon(a, r, c, eps_xy)
          for frame_index in range(0, self._num_frames):
            sum = 0
            # get the epsilon time frame
            for frame_index_time in range(0, eps_time):
              frame_to_compute = frame_index - frame_index_time
              if frame_to_compute < 0:
                frame_to_compute = 0
              # pass over the neighbors and compute the sum
              for neighbor in neighbors:
                sum += self._frame_list[frame_to_compute].get_value_arc(neighbor[0], neighbor[1], neighbor[2])
            # if the sum is lower than the min_samples, set the pixel to 0
            if sum < min_samples:
              data_cube_out.set_value(frame_index, a, r, c, 0)
                
    return data_cube_out
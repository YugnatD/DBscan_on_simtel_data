import numpy as np
import math
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize


import tm_ctao.HECS as HECS

class Frame(HECS.Hecs):
    def __init__(self, num_r: int, num_c: int, arc_points_cam: list):
        # Initialize parent Hecs class
        super().__init__(num_r, num_c)
        self._arc_points_cam = arc_points_cam
        # set all pixel in arc_points_cam to 1
        # for arc_point in self._arc_points_cam:
        #     self.set_value_arc(arc_point[0], arc_point[1], arc_point[2], 1)
    
    def load_points(self, points):
        # check that the number of points is the same as the number of arc points
        if len(points) != len(self._arc_points_cam):
            print("Error: Number of points does not match number of arc points")
            return
        for i in range(len(points)):
            self.set_value_arc(self._arc_points_cam[i][0], self._arc_points_cam[i][1], self._arc_points_cam[i][2], points[i])

    # check if the pixel is valid
    # or if the pixel is a padding pixel
    def is_pixel_valid(self, a, r, c):
        # check if a, r, c are in the array
        if r < 0 or c < 0:
            return False
        for arc_point in self._arc_points_cam:
            if arc_point[0] == a and arc_point[1] == r and arc_point[2] == c:
                return True
        return False
    
    def copy(self):
        frame = Frame(self.get_num_r(), self.get_num_c(), self._arc_points_cam)
        for a in range(self.get_num_a()):
            for r in range(self.get_num_r()):
                for c in range(self.get_num_c()):
                    frame.set_value_arc(a, r, c, self.get_value_arc(a, r, c))
        return frame
    
    def plot(self):
      min_val = 0.0
      max_val = 1.0
      size_hexagon = 0.55
      norm = Normalize(vmin=min_val, vmax=max_val)
      patches = []
      colors = []
      for a in range(2):
        for r in range(self._a.shape[1]):
            for c in range(self._a.shape[2]):
                # check if the pixel is valid
                if not self.is_pixel_valid(a, r, c):
                    continue
                val = self.get_value_arc(a, r, c)
                center_x, center_y = self.arc_to_xy(a, r, c)
                hexagon_vertices = self.hexagon((center_x, center_y), size_hexagon)
                patches.append(Polygon(hexagon_vertices))
                norm_val = norm(val)
                color = plt.cm.viridis(norm_val)
                colors.append(color)
      
      fig, ax = plt.subplots()
      collection = PatchCollection(patches, match_original=True)
      collection.set_facecolor(colors)  # Set colors directly
      # show the colorbar
      # fig.colorbar(collection, ax=ax)
      ax.add_collection(collection)
      sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
      sm.set_array([min_val, max_val])
      cbar = fig.colorbar(sm, ax=ax)
      width = self._num_c
      height = self._num_r
      ax.set_xlim(-size_hexagon, width * size_hexagon * 1.83)
      ax.set_ylim(-size_hexagon, height * size_hexagon * 2 * 1.87)
      # ax.set_ylim(-size_hexagon, width * size_hexagon * 1.88)
      ax.set_aspect('equal')
      plt.gca().invert_yaxis()
      plt.show()
      plt.close(fig)
    





  

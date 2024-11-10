import numpy as np
import math
import os

from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

class NeighbourHECS:
  TOP_LEFT = 0
  TOP_RIGHT = 1
  LEFT = 2
  RIGHT = 3
  BOTTOM_LEFT = 4
  BOTTOM_RIGHT = 5

class Hecs:
    def __init__(self, num_r: int, num_c: int):
        self._num_a = 2 
        self._num_r = num_r
        self._num_c = num_c
        
        # Initialize arrays with zeros
        self._a = np.zeros((self._num_a, self._num_r, self._num_c), dtype=float)

        # Populate hexagon list and setup positions based on arc_to_xy method
        for a in range(self._num_a):
            for r in range(self._num_r):
                for c in range(self._num_c):
                    self._a[a][r][c] = 0.0
    
    def copy(self):
        hecs = Hecs(self._num_r, self._num_c)
        for a in range(self._num_a):
            for r in range(self._num_r):
                for c in range(self._num_c):
                    hecs.set_value_arc(a, r, c, self.get_value_arc(a, r, c))
        return hecs

    def get_num_a(self):
        return self._num_a
    
    def get_num_r(self):
        return self._num_r
    
    def get_num_c(self):
        return self._num_c
    
    def set_value_arc(self, a, r, c, value):
      # check if a, r, c are in the array
      if r < 0 or c < 0:
        return
      if r >= self._a.shape[1] or c >= self._a.shape[2]:
        return
      # if a < 0 or a >= self._a.shape[0]:
      #   return
      self._a[a][r][c] = value
    
    def get_value_arc(self, a, r, c):
      # check if a, r, c are in the array
      if r < 0 or c < 0:
        return 0.0
      if r >= self._a.shape[1] or c >= self._a.shape[2]:
        return 0.0
      return self._a[a][r][c]
  
    def xy_to_arc(self, x, y, shifted=True, eps=1e-3):
      # add a small offset to avoid rounding errors
      x += eps
      y += eps
      if shifted:
        a = (int(y / (np.sqrt(3)/2)) % 2)
        r = int(y / (np.sqrt(3)))
        if a == 0:
          c = int(x - 0.5)
        else: # a == 1
          c = int(x)
      else:
        a = int(y / (np.sqrt(3)/2) % 2)
        r = int(y / (np.sqrt(3)))
        if a == 0:
          c = int(x)
        else: # a == 1
          c = int(x - 0.5)
      return (a, r, c)
  
    def arc_to_xy(self, a, r, c, shifted=True):
      if shifted:
        x = ((1-a) / 2) + c
      else:
        x = (a / 2) + c  
      y = np.sqrt(3) * (a * 0.5 + r)
      return (x, y)
  
    def get_neighbors_arc(self, a, r, c, neighbour: NeighbourHECS, shifted=True):
      if shifted:
        if neighbour == NeighbourHECS.TOP_LEFT:
          return (1-a, r-(1-a), c-a)
        elif neighbour == NeighbourHECS.TOP_RIGHT:
          return (1-a, r-(1-a), c+(1-a))
        elif neighbour == NeighbourHECS.LEFT:
          return (a, r, c-1)
        elif neighbour == NeighbourHECS.RIGHT:
          return (a, r, c+1)
        elif neighbour == NeighbourHECS.BOTTOM_LEFT:
          return (1-a, r+a, c-a)
        elif neighbour == NeighbourHECS.BOTTOM_RIGHT:
          return (1-a, r+a, c+(1-a))
        else:
          return None
      else:
        if neighbour == NeighbourHECS.TOP_LEFT:
          return (1-a, r-(1-a), c-(1-a))
        elif neighbour == NeighbourHECS.TOP_RIGHT:
          return (1-a, r-(1-a), c+a)
        elif neighbour == NeighbourHECS.LEFT:
          return (a, r, c-1)
        elif neighbour == NeighbourHECS.RIGHT:
          return (a, r, c+1)
        elif neighbour == NeighbourHECS.BOTTOM_LEFT:
          return (1-a, r+a, c-(1-a))
        elif neighbour == NeighbourHECS.BOTTOM_RIGHT:
          return (1-a, r+a, c+a)
        else:
          return None
    
    def get_list_of_neighbors(self, a, r, c, shifted=True):
      neighbours = []
      neighbours.append(self.get_neighbors_arc(a, r, c, NeighbourHECS.TOP_LEFT, shifted=shifted))
      neighbours.append(self.get_neighbors_arc(a, r, c, NeighbourHECS.TOP_RIGHT, shifted=shifted))
      neighbours.append(self.get_neighbors_arc(a, r, c, NeighbourHECS.LEFT, shifted=shifted))
      neighbours.append((a, r, c)) # the pixel itself
      neighbours.append(self.get_neighbors_arc(a, r, c, NeighbourHECS.RIGHT, shifted=shifted))
      neighbours.append(self.get_neighbors_arc(a, r, c, NeighbourHECS.BOTTOM_LEFT, shifted=shifted))
      neighbours.append(self.get_neighbors_arc(a, r, c, NeighbourHECS.BOTTOM_RIGHT, shifted=shifted))
      return neighbours
    

    def get_list_neighbors_epsilon(self, a: int, r: int, c: int, epsilon: int, shifted: bool = True) -> List[Tuple[int, int, int]]:
        neighbours = []
        
        # Define the top-left starting point
        a_top_left = (a + epsilon) % 2
        r_top_left = r - (epsilon // 2) - int(not bool(a))
        c_top_left = c - (epsilon // 2) - a

        # Initialize the temporary variables for traversing
        a_temp, r_temp, c_temp = a_top_left, r_top_left, c_top_left

        # Traverse the hexagon's upper side
        for i_r in range(epsilon + 1):
            for i_c in range(epsilon + i_r + 1):
                a_temp, r_temp, c_temp = self.get_neighbors_arc(a_temp, r_temp, c_temp, NeighbourHECS.RIGHT, shifted)
                neighbours.append((a_temp, r_temp, c_temp))
            a_temp, r_temp, c_temp = a_top_left, r_top_left, c_top_left
            a_temp, r_temp, c_temp = self.get_neighbors_arc(a_temp, r_temp, c_temp, NeighbourHECS.BOTTOM_LEFT, shifted)
            a_top_left, r_top_left, c_top_left = a_temp, r_temp, c_temp

        # Reset position to top left
        a_top_left, r_top_left, c_top_left = self.get_neighbors_arc(a_top_left, r_top_left, c_top_left, NeighbourHECS.RIGHT, shifted)
        a_temp, r_temp, c_temp = a_top_left, r_top_left, c_top_left

        # Traverse the hexagon's lower side
        for i_r in range(epsilon, 0, -1):
            for i_c in range(epsilon + i_r, 0, -1):
                a_temp, r_temp, c_temp = self.get_neighbors_arc(a_temp, r_temp, c_temp, NeighbourHECS.RIGHT, shifted)
                neighbours.append((a_temp, r_temp, c_temp))
            a_temp, r_temp, c_temp = a_top_left, r_top_left, c_top_left
            a_temp, r_temp, c_temp = self.get_neighbors_arc(a_temp, r_temp, c_temp, NeighbourHECS.BOTTOM_RIGHT, shifted)
            a_top_left, r_top_left, c_top_left = a_temp, r_temp, c_temp

        return neighbours

    def hexagon(self, center, size):
      """Generate the vertices of a hexagon centered at `center` with a given `size`."""
      angle = np.linspace(0, 2 * np.pi, 7) + np.pi / 6 # rotate the hexagon by 90 degrees
      # angle = np.linspace(0, 2 * np.pi, 7) # no rotation
      x_hexagon = center[0] + size * np.cos(angle)
      y_hexagon = center[1] + size * np.sin(angle)
      return np.vstack([x_hexagon, y_hexagon]).T
    
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
    
    

  

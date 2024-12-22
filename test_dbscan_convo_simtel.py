from eventio import SimTelFile
import pandas as pd
import numpy as np
import pickle as pkl
import sys
import time
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
#
from astropy.table import Table
from collections import defaultdict
from ctapipe.io.astropy_helpers import write_table
#
from tables import open_file

import csv
# Full Python Module
# from tm_ctao import Frame
# from tm_ctao import HECS
# from tm_ctao import DataCube

import tm_ctao_cpp
from tm_ctao_cpp import Frame
from tm_ctao_cpp import Hecs
from tm_ctao_cpp import Datacube

from libGammaFile import GammaFile


simtelIn = "scratch/simtel_data/gamma_diffuse/data/corsika_run1.simtel.gz"

# Read the simtel file
data = GammaFile(simtelIn)

arc_points_shrink, arc_points = data.load_arc_points_from_csv("camera_file/CTA_LST_Pixels_info_shrink.csv")
arc_points_full = data.load_arc_point_from_csv_full("camera_file/CTA_LST_Pixels_info.csv")
# neighbors_xy = data.load_epsilon_neighbors_from_csv(_CONVOLVE_KERNEL_SIZE_XY)
max_r_iso = max([r for (a, r, c) in arc_points_shrink]) + 1
max_c_iso = max([c for (a, r, c) in arc_points_shrink]) + 1
max_r_full = max([r for (a, r, c) in arc_points]) + 1
max_c_full = max([c for (a, r, c) in arc_points]) + 1


for i in range(1000):
    event = data.get_event(lst_id=1, event_number=i)

    
    # print(event)
    n_pe = event["n_pe"]
    n_pixels = event["n_pixels"]
    x_mean = event["L1_trigger_info"]["x_mean"]
    y_mean = event["L1_trigger_info"]["y_mean"]
    t_mean = event["L1_trigger_info"]["t_mean"]
    if n_pe < 200:
        continue
    iso_0 = event["X_iso"] # 1141x75 datacube
    full_0 = event["X_full"] # 7639x75 datacube
    wf0 = event["wf"] # full 7987x75 datacube
    wf0 = wf0 / np.max(wf0)

    print("Event: ", i)
    print("n_pe: ", n_pe)
    print("n_pixels: ", n_pixels)
    print("x_mean: ", x_mean)
    print("y_mean: ", y_mean)
    print("t_mean: ", t_mean)

    datacube_cpp_full = Datacube(max_r_full, max_c_full, wf0, arc_points_full)
    datacube_cpp_full.refresh_plot()
    datacube_cpp_full.plot(600, 600, 10, 35)
    input("Press Enter to close...")
    datacube_cpp_full.close_plot()

    data_cube_cpp = Datacube(max_r_iso, max_c_iso, iso_0, arc_points_shrink)
    data_cube_cpp.refresh_plot()
    data_cube_cpp.plot(600, 600, 10, 35)
    input("Press Enter to close...")
    data_cube_cpp.close_plot()

    # convolve the datacube
    neighbors_xy = data.load_epsilon_neighbors_from_csv(1)
    data_out = tm_ctao_cpp.dbscan_convolve_mt(iso_0, neighbors_xy, 3, 8, 8)
    data_cube = Datacube(max_r_iso, max_c_iso, data_out, arc_points_shrink)
    data_cube.refresh_plot()
    data_cube.plot(600, 600, 10, 35)
    input("Press Enter to close...")
    data_cube.close_plot()

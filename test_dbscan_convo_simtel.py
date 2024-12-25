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
import libGammaFile


simtelIn = "scratch/simtel_data/gamma_diffuse/data/corsika_run1.simtel.gz"

# Read the simtel file
data = GammaFile(simtelIn)

arc_points_shrink, arc_points, xy_points = data.load_arc_points_from_csv("camera_file/CTA_LST_Pixels_info_shrink.csv")
arc_points_full, xy_points_full = data.load_arc_point_from_csv_full("camera_file/CTA_LST_Pixels_info.csv")
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
    if n_pe < 250:
        continue
    iso_0 = event["X_iso"] # 1141x75 datacube
    full_0 = event["X_full"] # 7639x75 datacube
    wf0 = event["wf"] # full 7987x75 datacube
    wf0 = wf0 / np.max(wf0)
    L3_digitalsum = event["L3_digitalsum"]
    L3_digitalsum_all = event["L3_digitalsum_all"]

    print("Event INFO")
    print("Event: ", i)
    print("n_pe: ", n_pe)
    print("n_pixels: ", n_pixels)
    print("x_mean: ", x_mean)
    print("y_mean: ", y_mean)
    print("t_mean: ", t_mean)

    # apply DBSCAN
    X = data.pixel_mapping_extended_all[L3_digitalsum_all>libGammaFile._DBSCAN_digitalsum_threshold]
    channel_list_cut=data.L3_trigger_DBSCAN_pixel_cluster_list_all_extended[L3_digitalsum_all>libGammaFile._DBSCAN_digitalsum_threshold]
    # print the time column
    # print("X: ", X[:,2])
    X=X*[[1, 1, libGammaFile._time_norm]] 
    #
    dbscan = DBSCAN( eps = libGammaFile._DBSCAN_eps, min_samples = libGammaFile._DBSCAN_min_samples)
    clusters_dbscan = dbscan.fit_predict(X)
    clustersID_dbscan = np.unique(clusters_dbscan)

    if (len(clustersID_dbscan) > 1) :
        clustersID_dbscan = clustersID_dbscan[clustersID_dbscan>-1]
        clustersIDmax = np.argmax([len(clusters_dbscan[clusters_dbscan==clID]) for clID in clustersID_dbscan])
        #
        n_clusters_dbscan = len(clustersID_dbscan)
        n_points_dbscan = len(clusters_dbscan[clusters_dbscan==clustersIDmax])
        #
        x_mean_dbscan = np.mean(X[clusters_dbscan==clustersIDmax][:,0])
        y_mean_dbscan = np.mean(X[clusters_dbscan==clustersIDmax][:,1])
        t_mean_dbscan = np.mean(X[clusters_dbscan==clustersIDmax][:,2]) / libGammaFile._time_norm
        print("n_clusters_dbscan: ", n_clusters_dbscan)
        print("n_points_dbscan: ", n_points_dbscan)
        print("x_mean_dbscan: ", x_mean_dbscan)
        print("y_mean_dbscan: ", y_mean_dbscan)
        print("t_mean_dbscan: ", t_mean_dbscan)
    else:
        print("n_clusters_dbscan: ", 0)
        print("n_points_dbscan: ", 0)
        print("x_mean_dbscan: ", 0)
        print("y_mean_dbscan: ", 0)
        print("t_mean_dbscan: ", 0)
    
    # apply convolve
    neighbors_xy = data.load_epsilon_neighbors_from_csv(1)
    data_out = tm_ctao_cpp.dbscan_convolve_mt(iso_0, neighbors_xy, 3, 8, 8)
    data_cube_out = Datacube(max_r_iso, max_c_iso, data_out, arc_points_shrink)
    # temporal_values = (data.pixel_mapping_extended[:, :, 2] * libGammaFile._time_norm_isolated)[0]
    # temporal value if a list of flot from 0 to datacube.shape[2]
    temporal_values = np.arange(0, iso_0.shape[1])
    points_converted = data_cube_out.get_points_xyt_above_threshold_centered(0.9, temporal_values, arc_points_shrink, xy_points)

    if (len(points_converted) > 0):
        clustersID_convo = [0] # id of all the clusters that exist
        clustersIDmax_convo = 0 # id of the cluster with the most points
        n_clusters_convo = 1
        n_points_convo = len(points_converted)
        x_mean_convo = np.mean([point[0] for point in points_converted]) / 100.0 # convert to meters
        y_mean_convo = np.mean([point[1] for point in points_converted]) / 100.0 # convert to meters
        t_mean_convo = np.mean([point[2] for point in points_converted])
    else:
        n_clusters_convo = 0
        n_points_convo = 0
        x_mean_convo = 0
        y_mean_convo = 0
        t_mean_convo = 0
    print("n_clusters_convo: ", n_clusters_convo)
    print("n_points_convo: ", n_points_convo)
    print("x_mean_convo: ", x_mean_convo)
    print("y_mean_convo: ", y_mean_convo)
    print("t_mean_convo: ", t_mean_convo)


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

    
    data_cube_out.refresh_plot()
    data_cube_out.plot(600, 600, 10, 35)
    input("Press Enter to close...")
    data_cube_out.close_plot()

    

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
# sf = SimTelFile(simtelIn)
# wf=np.array([], dtype=np.uint16)

# Define constants corresponding to the parameters
simtelIn = "scratch/simtel_data/gamma_diffuse/data/corsika_run1.simtel.gz"
dl1In = "scratch/ctapipe_data/gamma_diffuse/data/gamma_diffuse_run1.dl1.h5"
# outpkl = "scratch/simtel_data/gamma_diffuse/npe/corsika_run1.npe.pkl"
# outcsv = "scratch/simtel_data/gamma_diffuse/npe/corsika_run1.npe.csv"
# outh5 = "scratch/simtel_data/gamma_diffuse/npe/corsika_run1.npe.h5"

pixel_mapping_csv = "pixel_mapping.csv"
isolated_flower_seed_super_flower_csv = "isolated_flower_seed_super_flower.list"
isolated_flower_seed_flower_csv = "isolated_flower_seed_flower.list"
all_seed_flower_csv = "all_seed_flower.list"

_n_max_noise_events=3000
_npe_noise=20
_n_of_time_sample=75
_time_of_one_sample_s=(_n_of_time_sample*1000/1024.0*1.0e-9)
_event_modulo=100
#
#
#
_time_norm_isolated=0.09
_DBSCAN_eps_isolated = 0.11
_DBSCAN_digitalsum_threshold_isolated = 2165
_DBSCAN_min_samples_isolated = 3
#
#
#
_time_norm=0.06
_DBSCAN_eps = 0.11
_DBSCAN_digitalsum_threshold = 2160
_DBSCAN_min_samples = 50
#
#

class GammaFile:
  def __init__(self, simtelIn):
    self.simtelIn = simtelIn
    self.sf = SimTelFile(self.simtelIn)
    self.wf = np.array([], dtype=np.uint16)
    self.pixel_mapping = np.genfromtxt(pixel_mapping_csv)
    self.isolated_flower_seed_flower = np.genfromtxt(isolated_flower_seed_flower_csv,dtype=int) 
    self.isolated_flower_seed_super_flower = np.genfromtxt(isolated_flower_seed_super_flower_csv,dtype=int)
    self.all_seed_flower = np.genfromtxt(all_seed_flower_csv,dtype=int)
    self.L1_trigger_pixel_cluster_list = self.isolated_flower_seed_super_flower
    self.L3_trigger_DBSCAN_pixel_cluster_list = self.isolated_flower_seed_flower
    self.L3_trigger_DBSCAN_pixel_cluster_list_all = self.all_seed_flower
    # self.arc_points_shrink, self.arc_points = self.load_arc_points_from_csv("camera_file/CTA_LST_Pixels_info_shrink.csv")
    # self.neighbors_xy = self.load_epsilon_neighbors_from_csv(_CONVOLVE_KERNEL_SIZE_XY)
    self.pixel_mapping_extended=self.extend_pixel_mapping( pixel_mapping=self.pixel_mapping, channel_list=self.L3_trigger_DBSCAN_pixel_cluster_list, number_of_wf_time_samples=_n_of_time_sample)
    self.pixel_mapping_extended_all=self.extend_pixel_mapping( pixel_mapping=self.pixel_mapping, channel_list=self.L3_trigger_DBSCAN_pixel_cluster_list_all, number_of_wf_time_samples=_n_of_time_sample)
    #
    self.L3_trigger_DBSCAN_pixel_cluster_list_extended=self.extend_channel_list( channel_list=self.L3_trigger_DBSCAN_pixel_cluster_list,
                                                                       number_of_wf_time_samples=_n_of_time_sample)
    self.L3_trigger_DBSCAN_pixel_cluster_list_all_extended=self.extend_channel_list( channel_list=self.L3_trigger_DBSCAN_pixel_cluster_list_all,
                                                                           number_of_wf_time_samples=_n_of_time_sample)
    # self.event_iso_list = []
    # self.event_full_list = []
    # for ev in self.sf:
    #   LSTID=ev['telescope_events'].keys()
    #   wf_list = []
    #   n_pe_per_tel_list=[]
    #   LSTID_list=[]
    #   L1_trigger_info_LST1=None
    #   L1_trigger_info_LST2=None
    #   L1_trigger_info_LST3=None
    #   L1_trigger_info_LST4=None
    #   DBSCAN_clusters_info_LST1=None
    #   DBSCAN_clusters_info_LST2=None
    #   DBSCAN_clusters_info_LST3=None
    #   DBSCAN_clusters_info_LST4=None
    #   DBSCAN_clusters_info_isolated_LST1=None
    #   DBSCAN_clusters_info_isolated_LST2=None
    #   DBSCAN_clusters_info_isolated_LST3=None
    #   DBSCAN_clusters_info_isolated_LST4=None
    #   #
    #   mask_cl_LST1=None
    #   mask_cl_LST2=None
    #   mask_cl_LST3=None
    #   mask_cl_LST4=None
    #   #
    #   #
    #   ev_time=[0,0,0,0]
    #   nphotons=[0,0,0,0]
    #   n_pe=[0,0,0,0]
    #   n_pixels=[0,0,0,0]

    #   for i in LSTID:
    #     print("i: ", i)
    #     wf_list.append(ev['telescope_events'][i]['adc_samples'][0])
    #     n_pe_per_tel_list.append(int(ev['photoelectrons'][i-1]['n_pe']))
    #     LSTID_list.append(int(i-1))
    #     #
    #     #
    #     ev_time[(i-1)] = float(ev['telescope_events'][i]['header']['readout_time'])
    #     nphotons[(i-1)]=int(len(ev['photons'][(i-1)]))
    #     n_pe[(i-1)]=int(ev['photoelectrons'][(i-1)]['n_pe'])
    #     n_pixels[(i-1)]=int(ev['photoelectrons'][(i-1)]['n_pixels']-np.sum(ev['photoelectrons'][(i-1)]['photoelectrons']==0))

    #   for i in np.arange(0,len(n_pe)) :
    #     if (n_pe[i] == 0 and i == 0) :
    #         L1_trigger_info_LST1 = self.def_L1_trigger_info()
    #         DBSCAN_clusters_info_isolated_LST1 = self.def_clusters_info()
    #         DBSCAN_clusters_info_LST1 = self.def_clusters_info()
    #         mask_cl_LST1=np.zeros(self.pixel_mapping.shape[0],dtype=int)
    #     elif (n_pe[i] == 0 and i == 1) :
    #         L1_trigger_info_LST2 = self.def_L1_trigger_info()
    #         DBSCAN_clusters_info_isolated_LST2 = self.def_clusters_info()
    #         DBSCAN_clusters_info_LST2 = self.def_clusters_info()
    #         mask_cl_LST2=np.zeros(self.pixel_mapping.shape[0],dtype=int)
    #     elif (n_pe[i] == 0 and i == 2) :
    #         L1_trigger_info_LST3 = self.def_L1_trigger_info()
    #         DBSCAN_clusters_info_isolated_LST3 = self.def_clusters_info()
    #         DBSCAN_clusters_info_LST3 = self.def_clusters_info()
    #         mask_cl_LST3=np.zeros(self.pixel_mapping.shape[0],dtype=int)
    #     elif (n_pe[i] == 0 and i == 3) :
    #         L1_trigger_info_LST4 = self.def_L1_trigger_info()
    #         DBSCAN_clusters_info_isolated_LST4 = self.def_clusters_info()
    #         DBSCAN_clusters_info_LST4 = self.def_clusters_info()                
    #         mask_cl_LST4=np.zeros(self.pixel_mapping.shape[0],dtype=int)
    #   for wf, npe, lst_id in zip( wf_list, n_pe_per_tel_list, LSTID_list) :
    #     L1_digitalsum = self.digital_sum(wf=wf, digi_sum_channel_list=self.L1_trigger_pixel_cluster_list)
    #     L3_digitalsum = self.digital_sum(wf=wf, digi_sum_channel_list=self.L3_trigger_DBSCAN_pixel_cluster_list)
    #     L3_digitalsum_all = self.digital_sum(wf=wf, digi_sum_channel_list=self.L3_trigger_DBSCAN_pixel_cluster_list_all)
    #     L1_trigger_info = self.get_L1_trigger_info(digitalsum=L1_digitalsum, pixel_mapping=self.pixel_mapping, digi_sum_channel_list=self.L1_trigger_pixel_cluster_list)
    #     X_iso = L3_digitalsum>_DBSCAN_digitalsum_threshold_isolated
    #     X_iso = X_iso.astype(float)
    #     X_full = L3_digitalsum_all>_DBSCAN_digitalsum_threshold
    #     X_full = X_full.astype(float)
    

  # def get_event(self, event_id):
  #   return self.event_iso_list[event_id], self.event_full_list[event_id]
  
  # def get_num_events(self):
  #    return len(self.event_iso_list)
  def get_event(self, lst_id, event_number):
    with SimTelFile(self.simtelIn) as sf:
        for i, ev in enumerate(sf):
            LSTID=ev['telescope_events'].keys()
            if i != event_number:
                continue

            if lst_id not in LSTID:
                print("lst_id not in ev['telescope_events']")
                return None

            # Extract waveform and metadata for the specified LST ID
            wf = ev['telescope_events'][lst_id]['adc_samples'][0]
            n_pe = int(ev['photoelectrons'][lst_id - 1]['n_pe'])
            ev_time = float(ev['telescope_events'][lst_id]['header']['readout_time'])
            n_pixels = int(ev['photoelectrons'][lst_id - 1]['n_pixels'] - np.sum(ev['photoelectrons'][lst_id - 1]['photoelectrons'] == 0))

            L1_digitalsum = self.digital_sum(wf, self.L1_trigger_pixel_cluster_list)
            L3_digitalsum = self.digital_sum(wf, self.L3_trigger_DBSCAN_pixel_cluster_list)
            L3_digitalsum_all = self.digital_sum(wf, self.L3_trigger_DBSCAN_pixel_cluster_list_all)

            L1_trigger_info = self.get_L1_trigger_info(L1_digitalsum, self.pixel_mapping, self.L1_trigger_pixel_cluster_list)

            X_iso = L3_digitalsum > _DBSCAN_digitalsum_threshold_isolated
            X_iso = X_iso.astype(float)
            X_full = L3_digitalsum_all > _DBSCAN_digitalsum_threshold
            X_full = X_full.astype(float)
            return {
                "wf": wf,
                "n_pe": n_pe,
                "ev_time": ev_time,
                "n_pixels": n_pixels,
                "L1_trigger_info": L1_trigger_info,
                "L3_digitalsum": L3_digitalsum,
                "L3_digitalsum_all": L3_digitalsum_all,
                "X_iso": X_iso,
                "X_full": X_full
            }

    return None

  def def_L1_trigger_info(self, max_digi_sum=0,
                         x_mean=-999.0, y_mean=-999.0, t_mean=-999.0,
                         channelID=-999, timeID=-999):
    L1_trigger_info={'max_digi_sum':max_digi_sum,
                     'x_mean':x_mean,
                     'y_mean':y_mean,
                     't_mean':t_mean,
                     'channelID':channelID,
                     'timeID':timeID}
    #
    return L1_trigger_info

  def get_L1_trigger_info(self, digitalsum, pixel_mapping, digi_sum_channel_list):
    max_digi_sum=np.max(digitalsum)
    row, col = np.unravel_index(np.argmax(digitalsum), digitalsum.shape)
    channelID=digi_sum_channel_list[row,0]
    timeID=col
    x_mean=pixel_mapping[channelID,0]
    y_mean=pixel_mapping[channelID,1]
    t_mean=col
    return self.def_L1_trigger_info( max_digi_sum=max_digi_sum,
                                x_mean=x_mean, y_mean=y_mean, t_mean=t_mean,
                                channelID=channelID, timeID=timeID)

  def def_clusters_info(self, n_digitalsum_points=0, n_clusters=0, n_points=0,
                       x_mean=-999.0, y_mean=-999.0, t_mean=-999.0,
                       channelID=-999, timeID=-999):
    #
    clusters_info={'n_digitalsum_points':n_digitalsum_points,
                   'n_clusters':n_clusters,
                   'n_points':n_points,
                   'x_mean':x_mean,
                   'y_mean':y_mean,
                   't_mean':t_mean,
                   'channelID':channelID,
                   'timeID':timeID}
    #
    return clusters_info

  def digital_sum(self, wf, digi_sum_channel_list):
    digital_sum_result = np.array([np.sum(wf[digi_sum_channel_list[i]],axis=0) for i in np.arange(0,len(digi_sum_channel_list))])
    return digital_sum_result
  
  def extend_pixel_mapping(self, pixel_mapping, channel_list, number_of_wf_time_samples):
    pixel_mapping_extended=pixel_mapping[channel_list[:,0]].copy()
    pixel_mapping_extended=pixel_mapping_extended[:,:-1]
    pixel_mapping_extended=np.expand_dims(pixel_mapping_extended,axis=2)
    pixel_mapping_extended=np.swapaxes(pixel_mapping_extended, 1, 2)
    pixel_mapping_extended=np.concatenate(([pixel_mapping_extended for i in np.arange(0,number_of_wf_time_samples)]), axis=1)
    pixt=np.array([i for i in np.arange(0,number_of_wf_time_samples)]).reshape(1,number_of_wf_time_samples)
    pixt=np.concatenate(([pixt for i in np.arange(0,channel_list.shape[0])]), axis=0)
    pixt=np.expand_dims(pixt,axis=2)
    pixel_mapping_extended=np.concatenate((pixel_mapping_extended,pixt), axis=2)
    return pixel_mapping_extended
  
  def extend_channel_list(self, channel_list, number_of_wf_time_samples):
    channel_list_extended=channel_list.copy()
    # print("channel_list_extended.shape ", channel_list_extended.shape)
    channel_list_extended=np.expand_dims(channel_list_extended,axis=2)
    channel_list_extended=np.swapaxes(channel_list_extended, 1, 2)
    channel_list_extended=np.concatenate(([channel_list_extended for i in np.arange(0,number_of_wf_time_samples)]), axis=1)
    # print("channel_list_extended.shape ", channel_list_extended.shape)
    return channel_list_extended

  def load_epsilon_neighbors_from_csv(self,eps):
    csv_file = 'camera_file/CTA_LST_Pixels_info_epsilon_' + str(eps) + '.csv'
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        # skip the header
        next(reader)
        neighbors = [list(map(int, row)) for row in reader]
    return neighbors
  
  def load_arc_points_from_csv(sefl, filename):
    arc_points_shrink = []
    arc_points = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            a = int(row['full_a'])
            r = int(row['full_r'])
            c = int(row['full_c'])
            arc_points.append((a, r, c))
            a = int(row['shrinked_a'])
            r = int(row['shrinked_r'])
            c = int(row['shrinked_c'])
            arc_points_shrink.append((a, r, c))
    return arc_points_shrink, arc_points
  
#   pixel_number,pixel_type,x_position,y_position,module_number,board_number,channel_number,board_id_number,pixel_on,relative_qe,a,r,c
# 0,1,0.0,0.0,0,0,0,0x0,1,1.0,0,29,48
# 1,1,2.43,0.0,0,1,0,0x0,1,1.0,0,29,49
# 2,1,1.215,2.104,0,2,0,0x0,1,1.0,1,28,49
  def load_arc_point_from_csv_full(self, filename):
    arc_points = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            a = int(row['a'])
            r = int(row['r'])
            c = int(row['c'])
            arc_points.append((a, r, c))
    return arc_points
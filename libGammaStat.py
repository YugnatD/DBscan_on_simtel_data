import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from scipy.interpolate import splprep, splev
from scipy.interpolate import griddata
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

class GammaStat:
  def __init__(self, df_conv, df_dbscan, eps_xy, eps_t, min_samples):
    self.df_conv = df_conv
    self.df_dbscan = df_dbscan
    self.eps_xy = eps_xy
    self.eps_t = eps_t
    self.min_samples = min_samples
    # check that the two len are the same
    assert len(df_conv) == len(df_dbscan)
    self.df_conv_20pe = df_conv[df_conv['n_pe_LST1'] == 20]
    self.df_dbscan_20pe = df_dbscan[df_dbscan['n_pe_LST1'] == 20]
    # check that the two len are the same
    assert len(self.df_conv_20pe) == len(self.df_dbscan_20pe)
    # check that the energy between the two df are the same and zero
    mean_energy_diff = np.mean(df_dbscan['energy'].values-df_conv['energy'].values)
    assert mean_energy_diff < 1e-10
    self.df_dbscan_trg_l1=df_dbscan[df_dbscan['L1_max_digi_sum_LST1']>14963]
    self.df_dbscan_trg_l2=self.df_dbscan_trg_l1[self.df_dbscan_trg_l1['L3_iso_n_points_LST1']>7]
    self.df_dbscan_trg_l1_20pe=self.df_dbscan_20pe[self.df_dbscan_20pe['L1_max_digi_sum_LST1']>14963]
    self.df_dbscan_trg_l2_20pe=self.df_dbscan_trg_l1_20pe[self.df_dbscan_trg_l1_20pe['L3_iso_n_points_LST1']>7]
    self.df_conv_trg=self.df_conv[self.df_conv['L3_iso_n_points_LST1']>0]
    self.df_conv_not_trg=self.df_conv[self.df_conv['L3_iso_n_points_LST1']==0]

    # self.df_dbscan_trg = self.df_dbscan[self.df_dbscan['L3_iso_n_points_LST1']>7]
    self.df_dbscan_trg_l2_only=self.df_dbscan[self.df_dbscan['L3_iso_n_points_LST1']>7]
    # self.df_dbscan_trg_cl_only=self.df_dbscan[self.df_dbscan['L3_cl_n_points_LST1']>39]
    # self.df_dbscan_trg_l2_only_20pe=self.df_dbscan_20pe[self.df_dbscan_20pe['L3_iso_n_points_LST1']>7]
    # self.df_dbscan_trg_cl_only_20pe=self.df_dbscan_20pe[self.df_dbscan_20pe['L3_cl_n_points_LST1']>39]
    self.df_dbscan_trg = self.df_dbscan_trg_l2_only

    y_true = df_conv['n_pe_LST1'] >= 50
    y_pred = df_conv['L3_iso_n_points_LST1']>0
    self.cm_conv = confusion_matrix(y_true, y_pred)
    self.acc = (self.cm_conv[0,0]+self.cm_conv[1,1])/np.sum(self.cm_conv)
    self.eff = self.cm_conv[1,1]/np.sum(self.cm_conv[1,:])
    self.pur = self.cm_conv[1,1]/np.sum(self.cm_conv[:,1])
    self.f1 = 2*self.cm_conv[1,1]/(np.sum(self.cm_conv[:,1])+np.sum(self.cm_conv[1,:]))

    # compute the confusion matrix for the dbscan
    y_true_dbscan = df_dbscan['n_pe_LST1'] >= 50
    y_pred_dbscan = self.df_dbscan['L3_iso_n_points_LST1']>7
    self.cm_dbscan = confusion_matrix(y_true_dbscan, y_pred_dbscan)
    self.acc_dbscan = (self.cm_dbscan[0,0]+self.cm_dbscan[1,1])/np.sum(self.cm_dbscan)
    self.eff_dbscan = self.cm_dbscan[1,1]/np.sum(self.cm_dbscan[1,:])
    self.pur_dbscan = self.cm_dbscan[1,1]/np.sum(self.cm_dbscan[:,1])
    self.f1_dbscan = 2*self.cm_dbscan[1,1]/(np.sum(self.cm_dbscan[:,1])+np.sum(self.cm_dbscan[1,:]))

  # def plotNumPointsTriggered(self, filename=None):
  #   plt.hist(self.df_conv['L3_iso_n_points_LST1'].values, bins=np.linspace(0.0, 100, num=100), edgecolor='black', alpha=1.0, label='L3_iso_n_points_LST1')
  #   plt.hist(self.df_conv_trg['L3_iso_n_points_LST1'].values, bins=np.linspace(0.0, 100, num=100), edgecolor='black', alpha=0.3, label='L3_iso_n_points_LST1 Triggered')
  #   plt.hist(self.df_dbscan_trg['L3_iso_n_points_LST1'].values, bins=np.linspace(0.0, 100, num=100), edgecolor='black', alpha=0.3, label='DBSCAN L3_iso_n_points_LST1 Triggered')
  #   plt.grid(True)
  #   plt.yscale('log')
  #   plt.title('number of points in cluster' + f' eps_xy={self.eps_xy} eps_t={self.eps_t} min_samples={self.min_samples}')
  #   plt.xlabel('number of points in cluster')
  #   plt.ylabel('number of events')
  #   plt.legend()
  #   try:
  #     if filename is not None:
  #       plt.savefig(filename)
  #     else:
  #       plt.show()
  #   except:
  #     print('Error in plotNumPointsTriggered')
  #   plt.close()

  def plotNumPointsTriggered(self, filename=None):
    # Create a figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)

    # First subplot: L3_iso_n_points_LST1 and conv
    axes[0].hist(self.df_conv['L3_iso_n_points_LST1'].values, bins=np.linspace(0.0, 100, num=100), 
                 edgecolor='black', alpha=1.0, label='L3_iso_n_points_LST1')
    axes[0].hist(self.df_conv_trg['L3_iso_n_points_LST1'].values, bins=np.linspace(0.0, 100, num=100), 
                 edgecolor='black', alpha=0.3, label='L3_iso_n_points_LST1 Triggered')
    axes[0].grid(True)
    axes[0].set_yscale('log')
    axes[0].set_title('Number of Points in Cluster (Conv)')
    axes[0].set_xlabel('Number of Points in Cluster')
    axes[0].set_ylabel('Number of Events')
    axes[0].legend()

    # Second subplot: L3_iso_n_points_LST1 and dbscan
    axes[1].hist(self.df_conv['L3_iso_n_points_LST1'].values, bins=np.linspace(0.0, 100, num=100), 
                 edgecolor='black', alpha=1.0, label='L3_iso_n_points_LST1')
    axes[1].hist(self.df_dbscan_trg['L3_iso_n_points_LST1'].values, bins=np.linspace(0.0, 100, num=100), 
                 edgecolor='black', alpha=0.3, label='DBSCAN L3_iso_n_points_LST1 Triggered')
    axes[1].grid(True)
    axes[1].set_yscale('log')
    axes[1].set_title('Number of Points in Cluster (DBSCAN)')
    axes[1].set_xlabel('Number of Points in Cluster')
    axes[1].set_ylabel('Number of Events')
    axes[1].legend()

    # Save or show the figure
    try:
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
    except Exception as e:
        print(f'Error in saving/plotting: {e}')
    plt.close(fig)

  
  def plotTimeDistribution(self, filename=None):
    plt.hist((self.df_conv['L3_iso_t_mean_LST1'].values*0.09)/1.5, bins=np.linspace(0.0, 5.0, num=100), edgecolor='black', alpha=1.0, label='CONVO L3_iso_t_mean_LST1')
    plt.hist(self.df_dbscan['L3_iso_t_mean_LST1'].values/1.5, bins=np.linspace(0.0, 5.0, num=100), edgecolor='black', alpha=0.25, label='DBSCAN L3_iso_t_mean_LST1')
    plt.title('L3_iso_t_mean_LST1' + f' eps_xy={self.eps_xy} eps_t={self.eps_t} min_samples={self.min_samples}')
    plt.grid(True)
    plt.xlabel('L3_iso_t_mean_LST1, ns')
    plt.ylabel('Number of events')
    plt.yscale('log')
    plt.legend()
    if filename is not None:
      plt.savefig(filename)
    else:
      plt.show()
    plt.close()
  
  # plt L3_iso_x_mean_LST1
  def plotXDistribution(self, filename=None):
    plt.hist(self.df_conv['L3_iso_x_mean_LST1'].values, bins=np.linspace(-1.0, 1.0, num=100), edgecolor='black', alpha=1.0, label='CONVO L3_iso_x_mean_LST1')
    plt.hist(self.df_dbscan['L3_iso_x_mean_LST1'].values, bins=np.linspace(-1.0, 1.0, num=100), edgecolor='black', alpha=0.25, label='DBSCAN L3_iso_x_mean_LST1')
    plt.title('L3_iso_x_mean_LST1' + f' eps_xy={self.eps_xy} eps_t={self.eps_t} min_samples={self.min_samples}')
    plt.grid(True)
    plt.xlabel('L3_iso_x_mean_LST1')
    plt.ylabel('Number of events')
    plt.yscale('log')
    plt.legend()
    if filename is not None:
      plt.savefig(filename)
    else:
      plt.show()
    plt.close()
  
  # plt L3_iso_y_mean_LST1
  def plotYDistribution(self, filename=None):
    plt.hist(self.df_conv['L3_iso_y_mean_LST1'].values, bins=np.linspace(-1.0, 1.0, num=100), edgecolor='black', alpha=1.0, label='CONVO L3_iso_y_mean_LST1')
    plt.hist(self.df_dbscan['L3_iso_y_mean_LST1'].values, bins=np.linspace(-1.0, 1.0, num=100), edgecolor='black', alpha=0.25, label='DBSCAN L3_iso_y_mean_LST1')
    plt.title('L3_iso_y_mean_LST1' + f' eps_xy={self.eps_xy} eps_t={self.eps_t} min_samples={self.min_samples}')
    plt.grid(True)
    plt.xlabel('L3_iso_y_mean_LST1')
    plt.ylabel('Number of events')
    plt.yscale('log')
    plt.legend()
    if filename is not None:
      plt.savefig(filename)
    else:
      plt.show()
    plt.close()
    
  # def plotNPEDistribution(self, filename=None):
  #   plt.hist(self.df_conv['n_pe_LST1'].values, bins=np.linspace(0.0, 2000, num=100), edgecolor='black', alpha=1.0, label='all events conv')
  #   plt.hist(self.df_conv_trg['n_pe_LST1'].values, bins=np.linspace(0.0, 2000, num=100), edgecolor='black', alpha=1.0, label='triggered conv')
  #   plt.hist(self.df_dbscan_trg['n_pe_LST1'].values, bins=np.linspace(0.0, 2000, num=100), edgecolor='black', alpha=0.5, label='triggered dbscan')
  #   plt.grid(True)
  #   plt.yscale('log')  # Set y-axis to logarithmic scale
  #   plt.title('number of photoelectrons' + f' eps_xy={self.eps_xy} eps_t={self.eps_t} min_samples={self.min_samples}')
  #   plt.xlabel('number of photoelectrons')
  #   plt.ylabel('number of events')
  #   plt.legend()
  #   if filename is not None:
  #     plt.savefig(filename)
  #   else:
  #     plt.show()
  #   plt.close()
  
  def plotNPEDistribution(self, filename=None):
    # Create a figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)

    # First subplot: All events and triggered events (conv)
    axes[0].hist(self.df_conv['n_pe_LST1'].values, bins=np.linspace(0.0, 2000, num=100), 
                 edgecolor='black', alpha=1.0, label='All Events Conv')
    axes[0].hist(self.df_conv_trg['n_pe_LST1'].values, bins=np.linspace(0.0, 2000, num=100), 
                 edgecolor='black', alpha=1.0, label='Triggered Conv')
    axes[0].grid(True)
    axes[0].set_yscale('log')
    axes[0].set_title('Number of Photoelectrons (Conv)')
    axes[0].set_xlabel('Number of Photoelectrons')
    axes[0].set_ylabel('Number of Events')
    axes[0].legend()

    # Second subplot: All events and triggered events (dbscan)
    axes[1].hist(self.df_conv['n_pe_LST1'].values, bins=np.linspace(0.0, 2000, num=100), 
                 edgecolor='black', alpha=1.0, label='All Events Conv')
    axes[1].hist(self.df_dbscan_trg['n_pe_LST1'].values, bins=np.linspace(0.0, 2000, num=100), 
                 edgecolor='black', alpha=0.5, label='Triggered DBSCAN')
    axes[1].grid(True)
    axes[1].set_yscale('log')
    axes[1].set_title('Number of Photoelectrons (DBSCAN)')
    axes[1].set_xlabel('Number of Photoelectrons')
    axes[1].set_ylabel('Number of Events')
    axes[1].legend()

    # Save or show the figure
    try:
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
    except Exception as e:
        print(f'Error in saving/plotting: {e}')
    plt.close(fig)


  # def plotNPEDistributionRange(self, filename=None, range=(0.0, 2000)):
  #   # get the npe for the conv trigger between the range
  #   npe_conv = self.df_conv['n_pe_LST1'].values
  #   npe_conv = npe_conv[(npe_conv >= range[0]) & (npe_conv <= range[1])]
  #   # get the npe for the conv not trigger between the range
  #   npe_conv_trg = self.df_conv_trg['n_pe_LST1'].values
  #   npe_conv_trg = npe_conv_trg[(npe_conv_trg >= range[0]) & (npe_conv_trg <= range[1])]
  #   # get the npe for the dbscan between the range
  #   npe_dbscan_trg = self.df_dbscan_trg['n_pe_LST1'].values
  #   npe_dbscan_trg = npe_dbscan_trg[(npe_dbscan_trg >= range[0]) & (npe_dbscan_trg <= range[1])]
  #   # plot the histogram
  #   plt.hist(npe_conv, bins=np.linspace(range[0], range[1], num=100), edgecolor='black', alpha=1.0, label='all events conv')
  #   plt.hist(npe_conv_trg, bins=np.linspace(range[0], range[1], num=100), edgecolor='black', alpha=1.0, label='triggered conv')
  #   plt.hist(npe_dbscan_trg, bins=np.linspace(range[0], range[1], num=100), edgecolor='black', alpha=0.5, label='triggered dbscan')
  #   plt.grid(True)
  #   plt.yscale('log')  # Set y-axis to logarithmic scale
  #   plt.title('number of photoelectrons' + f' eps_xy={self.eps_xy} eps_t={self.eps_t} min_samples={self.min_samples}')
  #   plt.xlabel('number of photoelectrons')
  #   plt.ylabel('number of events')
  #   plt.legend()
  #   if filename is not None:
  #     plt.savefig(filename)
  #   else:
  #     plt.show()
  #   plt.close()
  

  def plotNPEDistributionRange(self, filename=None, range=(0.0, 2000)):
    # Filter data within the specified range
    npe_conv = self.df_conv['n_pe_LST1'].values
    npe_conv = npe_conv[(npe_conv >= range[0]) & (npe_conv <= range[1])]

    npe_conv_trg = self.df_conv_trg['n_pe_LST1'].values
    npe_conv_trg = npe_conv_trg[(npe_conv_trg >= range[0]) & (npe_conv_trg <= range[1])]

    npe_dbscan_trg = self.df_dbscan_trg['n_pe_LST1'].values
    npe_dbscan_trg = npe_dbscan_trg[(npe_dbscan_trg >= range[0]) & (npe_dbscan_trg <= range[1])]

    # Create a figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)

    # First subplot: All events and triggered events (conv)
    axes[0].hist(npe_conv, bins=np.linspace(range[0], range[1], num=100), 
                 edgecolor='black', alpha=1.0, label='All Events Conv')
    axes[0].hist(npe_conv_trg, bins=np.linspace(range[0], range[1], num=100), 
                 edgecolor='black', alpha=1.0, label='Triggered Conv')
    axes[0].grid(True)
    axes[0].set_yscale('log')
    axes[0].set_title('Number of Photoelectrons (Conv)')
    axes[0].set_xlabel('Number of Photoelectrons')
    axes[0].set_ylabel('Number of Events')
    axes[0].legend()

    # Second subplot: All events and triggered events (dbscan)
    axes[1].hist(npe_conv, bins=np.linspace(range[0], range[1], num=100), 
                 edgecolor='black', alpha=1.0, label='All Events Conv')
    axes[1].hist(npe_dbscan_trg, bins=np.linspace(range[0], range[1], num=100), 
                 edgecolor='black', alpha=0.5, label='Triggered DBSCAN')
    axes[1].grid(True)
    axes[1].set_yscale('log')
    axes[1].set_title('Number of Photoelectrons (DBSCAN)')
    axes[1].set_xlabel('Number of Photoelectrons')
    axes[1].set_ylabel('Number of Events')
    axes[1].legend()

    # Save or show the figure
    try:
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
    except Exception as e:
        print(f'Error in saving/plotting: {e}')
    plt.close(fig)


  # def plotEnergyDistribution(self, filename=None):
  #   hist_energy_conv=plt.hist(self.df_conv['energy'].values, bins=np.logspace(np.log10(0.005),np.log10(50),50), edgecolor='black', alpha=1.0, label='CONVO')
  #   hist_energy_conv_trg=plt.hist(self.df_conv_trg['energy'].values, bins=np.logspace(np.log10(0.005),np.log10(50),50), edgecolor='black', alpha=1.0, label='CONVO TRG')
  #   hist_energy_dbscan_trg=plt.hist(self.df_dbscan_trg['energy'].values, bins=np.logspace(np.log10(0.005),np.log10(50),50), edgecolor='black', alpha=0.5, label='DBSCAN TRG')
  #   plt.grid(True)
  #   plt.xscale('log')
  #   plt.yscale('log')
  #   plt.xlabel('energy')
  #   plt.legend()
  #   plt.ylabel('number of events')
  #   plt.title('Energy distribution' + f' eps_xy={self.eps_xy} eps_t={self.eps_t} min_samples={self.min_samples}')
  #   if filename is not None:
  #     plt.savefig(filename)
  #   else:
  #     plt.show()
  #   plt.close()

  def plotEnergyDistribution(self, filename=None):
    # Create a figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)

    # Energy bins for histograms
    energy_bins = np.logspace(np.log10(0.005), np.log10(50), 50)

    # First subplot: CONVO and CONVO TRG
    axes[0].hist(self.df_conv['energy'].values, bins=energy_bins, 
                 edgecolor='black', alpha=1.0, label='CONVO')
    axes[0].hist(self.df_conv_trg['energy'].values, bins=energy_bins, 
                 edgecolor='black', alpha=1.0, label='CONVO TRG')
    axes[0].grid(True)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Energy')
    axes[0].set_ylabel('Number of Events')
    axes[0].set_title('Energy Distribution (CONVO)')
    axes[0].legend()

    # Second subplot: CONVO and DBSCAN TRG
    axes[1].hist(self.df_conv['energy'].values, bins=energy_bins, 
                 edgecolor='black', alpha=1.0, label='CONVO')
    axes[1].hist(self.df_dbscan_trg['energy'].values, bins=energy_bins, 
                 edgecolor='black', alpha=0.5, label='DBSCAN TRG')
    axes[1].grid(True)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Energy')
    axes[1].set_ylabel('Number of Events')
    axes[1].set_title('Energy Distribution (DBSCAN)')
    axes[1].legend()

    # Save or show the figure
    try:
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
    except Exception as e:
        print(f'Error in saving/plotting: {e}')
    plt.close(fig)

  
  def plotConfusionMatrixConvo(self, filename=None):
    # add a legend
    plt.text(0.5, 0.5, f'Accuracy: {self.acc:.2f}\nEfficiency: {self.eff:.2f}\nPurity: {self.pur:.2f}\nF1: {self.f1:.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes )

    # Display confusion matrix
    # 0 = Not triggered, 1 = Triggered
    disp = ConfusionMatrixDisplay(confusion_matrix=self.cm_conv, display_labels=["Not Triggered", "Triggered"])
    # disp.plot(cmap="viridis")
    disp.plot(include_values=True, cmap='viridis', ax=plt.gca())
    plt.title('Confusion matrix' + f' eps_xy={self.eps_xy} eps_t={self.eps_t} min_samples={self.min_samples}')
    if filename is not None:
      plt.savefig(filename)
    else:
      plt.show()
    plt.close()
  
  #####################################################################################################################################
  # plotting for the dbscan
  
  def plotConfusionMatrixDBSCAN(self, filename=None):
    # add a legend
    plt.text(0.5, 0.5, f'Accuracy: {self.acc_dbscan:.2f}\nEfficiency: {self.eff_dbscan:.2f}\nPurity: {self.pur_dbscan:.2f}\nF1: {self.f1_dbscan:.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes )
    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=self.cm_dbscan, display_labels=["Not Triggered", "Triggered"])
    # disp.plot(cmap="viridis")
    disp.plot(include_values=True, cmap='viridis', ax=plt.gca())
    plt.title('Confusion matrix DBSCAN')
    if filename is not None:
      plt.savefig(filename)
    else:
      plt.show()
    plt.close()
  

# Define a plane fitting function: Z = a*X + b*Y + c
def plane(XY, a, b, c):
    x, y = XY
    return a * x + b * y + c

# out of the class function
# make a graph with a list
# def plotAccuracy(stat_list, filename=None):
#   eps_xy = [stat.eps_xy for stat in stat_list]
#   eps_t = [stat.eps_t for stat in stat_list]
#   min_samples = [stat.min_samples for stat in stat_list]
#   accuracy = [stat.acc for stat in stat_list]  # Assuming `acc` is computed in GammaStat

#   # 3D Scatter Plot
#   fig = plt.figure(figsize=(10, 6))
#   ax = fig.add_subplot(111, projection='3d')
#   scatter = ax.scatter(eps_xy, eps_t, min_samples, c=accuracy, cmap='viridis', s=100)
#   ax.set_xlabel('Epsilon XY')
#   ax.set_ylabel('Epsilon T')
#   ax.set_zlabel('Min Samples')
#   # set grid to be on range for each axis
#   ax.set_xticks(eps_xy)
#   ax.set_yticks(eps_t)
#   ax.set_zticks(min_samples)
#   fig.colorbar(scatter, label='Accuracy')
#   plt.title("3D Scatter Plot of GammaStat Parameters")
#   if filename is not None:
#     plt.savefig(filename)
#   else:
#     plt.show()

def plotAccuracy(stat_list, filename=None):
    eps_xy = [stat.eps_xy for stat in stat_list]
    eps_t = [stat.eps_t for stat in stat_list]
    min_samples = [stat.min_samples for stat in stat_list]
    accuracy = [stat.acc for stat in stat_list]  # Assuming `acc` is computed in GammaStat

    # Find the index of the maximum accuracy
    max_acc_index = accuracy.index(max(accuracy))
    best_eps_xy = eps_xy[max_acc_index]
    best_eps_t = eps_t[max_acc_index]
    best_min_samples = min_samples[max_acc_index]
    best_accuracy = accuracy[max_acc_index]

    # 3D Scatter Plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(eps_xy, eps_t, min_samples, c=accuracy, cmap='viridis', s=100)
    ax.set_xlabel('Epsilon XY')
    ax.set_ylabel('Epsilon T')
    ax.set_zlabel('Min Samples')

    # Add annotation for the point with the best accuracy
    ax.text(best_eps_xy, best_eps_t, best_min_samples, 
            f'Best: {best_accuracy:.2f}', 
            color='red', fontsize=10)

    # Set grid to be on range for each axis
    ax.set_xticks(sorted(set(eps_xy)))
    ax.set_yticks(sorted(set(eps_t)))
    ax.set_zticks(sorted(set(min_samples)))

    fig.colorbar(scatter, label='Accuracy')
    plt.title("F(eps_xy, eps_t, min_samples) = Accuracy")
    
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
  
  # plot the efficiency
def plotEfficiency(stat_list, filename=None):
    eps_xy = [stat.eps_xy for stat in stat_list]
    eps_t = [stat.eps_t for stat in stat_list]
    min_samples = [stat.min_samples for stat in stat_list]
    efficiency = [stat.eff for stat in stat_list]  # Assuming `eff` is computed in GammaStat

    # Find the index of the maximum accuracy
    max_acc_index = efficiency.index(max(efficiency))
    best_eps_xy = eps_xy[max_acc_index]
    best_eps_t = eps_t[max_acc_index]
    best_min_samples = min_samples[max_acc_index]
    best_efficiency = efficiency[max_acc_index]

    # 3D Scatter Plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(eps_xy, eps_t, min_samples, c=efficiency, cmap='viridis', s=100)
    ax.set_xlabel('Epsilon XY')
    ax.set_ylabel('Epsilon T')
    ax.set_zlabel('Min Samples')

    # Add annotation for the point with the best accuracy
    ax.text(best_eps_xy, best_eps_t, best_min_samples, 
            f'Best: {best_efficiency:.2f}', 
            color='red', fontsize=10)

    # Set grid to be on range for each axis
    ax.set_xticks(sorted(set(eps_xy)))
    ax.set_yticks(sorted(set(eps_t)))
    ax.set_zticks(sorted(set(min_samples)))

    fig.colorbar(scatter, label='Accuracy')
    plt.title("F(eps_xy, eps_t, min_samples) = Accuracy")
    
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def plotAccuracyPlane(stat_list, filename=None):
    # Extract data from GammaStat objects
    eps_xy = [stat.eps_xy for stat in stat_list]
    eps_t = [stat.eps_t for stat in stat_list]
    min_samples = [stat.min_samples for stat in stat_list]
    accuracy = [stat.acc for stat in stat_list]  # Assuming `acc` is computed in GammaStat

    # Identify the points with the highest accuracy for each (eps_xy, eps_t)
    unique_pairs = list(set(zip(eps_xy, eps_t)))
    max_accuracy_points = []
    for pair in unique_pairs:
        xy, t = pair
        indices = [i for i, (x, y) in enumerate(zip(eps_xy, eps_t)) if x == xy and y == t]
        max_idx = max(indices, key=lambda i: accuracy[i])
        max_accuracy_points.append((eps_xy[max_idx], eps_t[max_idx], min_samples[max_idx], accuracy[max_idx]))

    # Extract the maximum accuracy points for the surface
    surface_eps_xy, surface_eps_t, surface_min_samples, surface_accuracy = zip(*max_accuracy_points)

    # Find the point with the overall maximum accuracy
    overall_max_idx = np.argmax(surface_accuracy)
    best_point = (surface_eps_xy[overall_max_idx], surface_eps_t[overall_max_idx], surface_min_samples[overall_max_idx])
    best_accuracy = surface_accuracy[overall_max_idx]

    # Create a grid for surface interpolation
    x_grid, y_grid = np.meshgrid(
        np.linspace(min(surface_eps_xy), max(surface_eps_xy), 50),
        np.linspace(min(surface_eps_t), max(surface_eps_t), 50)
    )

    # Interpolate the surface
    z_grid = griddata(
        (surface_eps_xy, surface_eps_t),
        surface_min_samples,
        (x_grid, y_grid),
        method='cubic'
    )

    # 3D Scatter Plot with the Surface
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of all points
    scatter = ax.scatter(eps_xy, eps_t, min_samples, c=accuracy, cmap='viridis', s=100, label='Data Points')
    ax.set_xlabel('Epsilon XY')
    ax.set_ylabel('Epsilon T')
    ax.set_zlabel('Min Samples')

    # Overlay the 3D surface
    surface = ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', alpha=0.6, edgecolor='none')

    # Annotate the point with the best accuracy
    ax.text3D(
        best_point[0], best_point[1], best_point[2],
        f"Best\nAcc: {best_accuracy:.2f}",
        fontsize=10, color='red', horizontalalignment='center'
    )

    # Set grid ticks to match unique values
    ax.set_xticks(sorted(set(eps_xy)))
    ax.set_yticks(sorted(set(eps_t)))
    ax.set_zticks(sorted(set(min_samples)))
    cbar = fig.colorbar(scatter, label='Accuracy')
    plt.title("3D Scatter Plot of GammaStat Parameters with Interpolated Surface")

    # Add a custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Data Points'),
        Patch(facecolor='blue', edgecolor='none', alpha=0.6, label='Interpolated Surface'),
    ]
    ax.legend(handles=legend_elements)

    # Save or display the plot
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
  
  # plot accuracy in function of min_samples (x = min_samples, y = accuracy) for each pair of eps_xy and eps_t in a 2d plot
def plotAccuracyMinSamples(stat_list, eps_xy, eps_t, filename=None):
    # Filter the list of GammaStat objects by eps_xy and eps_t
    filtered_stats = [stat for stat in stat_list if stat.eps_xy == eps_xy and stat.eps_t == eps_t]

    # Extract data from the filtered GammaStat objects
    min_samples = [stat.min_samples for stat in filtered_stats]
    accuracy = [stat.acc for stat in filtered_stats]

    # sort the data based on the order of min_samples
    min_samples, accuracy = zip(*sorted(zip(min_samples, accuracy)))

    # get he best accuracy
    best_accuracy = max(accuracy)
    best_min_samples = min_samples[accuracy.index(best_accuracy)]

    plt.figure(figsize=(10, 6))
    plt.plot(min_samples, accuracy, marker='o', linestyle='-', color='blue')
    plt.xlabel('Min Samples')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs. Min Samples for eps_xy={eps_xy}, eps_t={eps_t}')
    plt.legend([f'Best Accuracy: {best_accuracy:.2f} at Min Samples: {best_min_samples}'])
    plt.grid(True)
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

    

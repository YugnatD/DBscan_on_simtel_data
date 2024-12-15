import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class GammaStat:
  def __init__(self, df_conv, df_dbscan):
    self.df_conv = df_conv
    self.df_dbscan = df_dbscan
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
    #
    #
    self.df_dbscan_trg_l1_20pe=self.df_dbscan_20pe[self.df_dbscan_20pe['L1_max_digi_sum_LST1']>14963]
    self.df_dbscan_trg_l2_20pe=self.df_dbscan_trg_l1_20pe[self.df_dbscan_trg_l1_20pe['L3_iso_n_points_LST1']>7]
    self.df_conv_trg=self.df_conv[self.df_conv['L3_iso_n_points_LST1']>0]
    self.df_conv_trg_20pe=self.df_conv_20pe[self.df_conv_20pe['L3_iso_n_points_LST1']>0]
    self.df_dbscan_trg_l2_only=self.df_dbscan[self.df_dbscan['L3_iso_n_points_LST1']>7]
    self.df_dbscan_trg_cl_only=self.df_dbscan[self.df_dbscan['L3_cl_n_points_LST1']>39]
    self.df_dbscan_trg_l2_only_20pe=self.df_dbscan_20pe[self.df_dbscan_20pe['L3_iso_n_points_LST1']>7]
    self.df_dbscan_trg_cl_only_20pe=self.df_dbscan_20pe[self.df_dbscan_20pe['L3_cl_n_points_LST1']>39]

  def plotNumPointsTriggered(self, filename=None):
    plt.hist(self.df_conv['L3_iso_n_points_LST1'].values, bins=np.linspace(0.0, 100, num=100), edgecolor='black', alpha=1.0, label='L3_iso_n_points_LST1')
    plt.hist(self.df_conv_trg['L3_iso_n_points_LST1'].values, bins=np.linspace(0.0, 100, num=100), edgecolor='black', alpha=0.3, label='L3_iso_n_points_LST1 Triggered')
    plt.grid(True)
    plt.yscale('log')
    plt.title('number of points in cluster')
    plt.xlabel('number of points in cluster')
    plt.ylabel('number of events')
    plt.legend()
    try:
      if filename is not None:
        plt.savefig(filename)
      else:
        plt.show()
    except:
      print('Error in plotNumPointsTriggered')
    plt.close()
  
  def plotTimeDistribution(self, filename=None):
    plt.hist(self.df_conv['L3_iso_t_mean_LST1'].values/1.5, bins=np.linspace(0.0, 5.0, num=100), edgecolor='black', alpha=1.0, label='CONVO L3_iso_t_mean_LST1')
    plt.hist(self.df_dbscan['L3_iso_t_mean_LST1'].values/1.5, bins=np.linspace(0.0, 5.0, num=100), edgecolor='black', alpha=0.25, label='DBSCAN L3_iso_t_mean_LST1')
    plt.title('L3_iso_t_mean_LST1')
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
    
  def plotNPEDistribution(self, filename=None):
    plt.hist(self.df_conv['n_pe_LST1'].values, bins=np.linspace(0.0, 2000, num=100), edgecolor='black', alpha=1.0, label='all events conv')
    plt.hist(self.df_conv_trg['n_pe_LST1'].values, bins=np.linspace(0.0, 2000, num=100), edgecolor='black', alpha=1.0, label='triggered conv')
    plt.grid(True)
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.title('number of photoelectrons')
    plt.xlabel('number of photoelectrons')
    plt.ylabel('number of events')
    plt.legend()
    if filename is not None:
      plt.savefig(filename)
    else:
      plt.show()
    plt.close()
  
  def plotEnergyDistribution(self, filename=None):
    hist_energy_conv=plt.hist(self.df_conv['energy'].values, bins=np.logspace(np.log10(0.005),np.log10(50),50), edgecolor='black', alpha=1.0, label='CONVO')
    hist_energy_conv_trg=plt.hist(self.df_conv_trg['energy'].values, bins=np.logspace(np.log10(0.005),np.log10(50),50), edgecolor='black', alpha=1.0, label='CONVO TRG')
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('energy')
    plt.legend()
    plt.ylabel('number of events')
    plt.title('Energy distribution')
    if filename is not None:
      plt.savefig(filename)
    else:
      plt.show()
    plt.close()
  
  def plotConfusionMatrix(self, filename=None):
    true_labels = (self.df_dbscan['L3_iso_n_points_LST1'] > 7).astype(int)  # Example ground truth
    predicted_labels = (self.df_conv['L3_iso_n_points_LST1'] > 7).astype(int)  # Example predictions
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # compute the accuracy
    acc = np.sum(np.logical_and(true_labels, predicted_labels)) / len(true_labels)

    # compute the precision
    eff = np.sum(np.logical_and(true_labels, predicted_labels)) / np.sum(true_labels)

    # compute the purity
    pur = np.sum(np.logical_and(true_labels, predicted_labels)) / np.sum(predicted_labels)

    # compute the F1 score
    f1 = 2 * eff * pur / (eff + pur)

    # add a legend
    plt.text(0.5, 0.5, f'Accuracy: {acc:.2f}\nEfficiency: {eff:.2f}\nPurity: {pur:.2f}\nF1: {f1:.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes )

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
    # disp.plot(cmap="viridis")
    disp.plot(include_values=True, cmap='viridis', ax=plt.gca())
    plt.title('Confusion matrix')
    if filename is not None:
      plt.savefig(filename)
    else:
      plt.show()
    plt.close()

    

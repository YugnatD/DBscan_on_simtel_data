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
    self.df_conv_trg_20pe=self.df_conv_20pe[self.df_conv_20pe['L3_iso_n_points_LST1']>0]
    self.df_dbscan_trg_l2_only=self.df_dbscan[self.df_dbscan['L3_iso_n_points_LST1']>7]
    self.df_dbscan_trg_cl_only=self.df_dbscan[self.df_dbscan['L3_cl_n_points_LST1']>39]
    self.df_dbscan_trg_l2_only_20pe=self.df_dbscan_20pe[self.df_dbscan_20pe['L3_iso_n_points_LST1']>7]
    self.df_dbscan_trg_cl_only_20pe=self.df_dbscan_20pe[self.df_dbscan_20pe['L3_cl_n_points_LST1']>39]
    true_labels = (self.df_dbscan['L3_iso_n_points_LST1'] > 7).astype(int) 
    predicted_labels = (self.df_conv['L3_iso_n_points_LST1'] > 7).astype(int)
    
    # print(self.df_conv['n_pixels_LST1'])
    self.cm = confusion_matrix(true_labels, predicted_labels)
    self.acc = (self.cm[0,0]+self.cm[1,1])/np.sum(self.cm)
    self.eff = self.cm[1,1]/np.sum(self.cm[1,:])
    self.pur = self.cm[1,1]/np.sum(self.cm[:,1])
    self.f1 = 2*self.cm[1,1]/(np.sum(self.cm[:,1])+np.sum(self.cm[1,:]))

  def plotNumPointsTriggered(self, filename=None):
    plt.hist(self.df_conv['L3_iso_n_points_LST1'].values, bins=np.linspace(0.0, 100, num=100), edgecolor='black', alpha=1.0, label='L3_iso_n_points_LST1')
    plt.hist(self.df_conv_trg['L3_iso_n_points_LST1'].values, bins=np.linspace(0.0, 100, num=100), edgecolor='black', alpha=0.3, label='L3_iso_n_points_LST1 Triggered')
    plt.grid(True)
    plt.yscale('log')
    plt.title('number of points in cluster' + f' eps_xy={self.eps_xy} eps_t={self.eps_t} min_samples={self.min_samples}')
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
    
  def plotNPEDistribution(self, filename=None):
    plt.hist(self.df_conv['n_pe_LST1'].values, bins=np.linspace(0.0, 2000, num=100), edgecolor='black', alpha=1.0, label='all events conv')
    plt.hist(self.df_conv_trg['n_pe_LST1'].values, bins=np.linspace(0.0, 2000, num=100), edgecolor='black', alpha=1.0, label='triggered conv')
    plt.grid(True)
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.title('number of photoelectrons' + f' eps_xy={self.eps_xy} eps_t={self.eps_t} min_samples={self.min_samples}')
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
    plt.title('Energy distribution' + f' eps_xy={self.eps_xy} eps_t={self.eps_t} min_samples={self.min_samples}')
    if filename is not None:
      plt.savefig(filename)
    else:
      plt.show()
    plt.close()
  
  def plotConfusionMatrix(self, filename=None):
    # add a legend
    plt.text(0.5, 0.5, f'Accuracy: {self.acc:.2f}\nEfficiency: {self.eff:.2f}\nPurity: {self.pur:.2f}\nF1: {self.f1:.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes )

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=self.cm, display_labels=["Class 0", "Class 1"])
    # disp.plot(cmap="viridis")
    disp.plot(include_values=True, cmap='viridis', ax=plt.gca())
    plt.title('Confusion matrix' + f' eps_xy={self.eps_xy} eps_t={self.eps_t} min_samples={self.min_samples}')
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


    

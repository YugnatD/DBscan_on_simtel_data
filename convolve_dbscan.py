import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
import matplotlib
from matplotlib import dates
import matplotlib.pyplot as plt
import time
from pathlib import Path
#from scipy.optimize import curve_fit
from tables import open_file
import libGammaStat as gs

# matplotlib.use('TkAgg')  # Use a GUI backend like TkAgg

# get a list of folder in the result folder
folder_list = [f for f in Path('result').iterdir() if f.is_dir()]
# remove the image folder from the list
folder_list = [f for f in folder_list if f.parts[1] != 'image']
# pass over each folder and open it, while putting it into a dictionary using the second folder as key
df_dbscan = pd.read_csv('DBSCAN_gamma_corsika_run1.npe.csv')

# create a folder to store the results
Path('result/image').mkdir(parents=True, exist_ok=True)

stat_dict = {}
for folder in folder_list:
    folder_name = folder.parts[1]
    df_dbscan_conv = pd.read_csv(folder / 'corsika_run1.npe.csv')
    stat_dict[folder_name] = gs.GammaStat(df_dbscan_conv, df_dbscan)
    # create the image of the result for each folder
    stat_dict[folder_name].plotNumPointsTriggered('result/image/' + folder_name + '_num_points.png')
    stat_dict[folder_name].plotTimeDistribution('result/image/' + folder_name + '_time_distribution.png')
    stat_dict[folder_name].plotNPEDistribution('result/image/' + folder_name + '_npe_distribution.png')
    stat_dict[folder_name].plotEnergyDistribution('result/image/' + folder_name + '_energy_distribution.png')
    stat_dict[folder_name].plotConfusionMatrix('result/image/' + folder_name + '_confusion_matrix.png')

# then create a markdown file with the list of images, organize them by type of plot
with open('result/image/index.md', 'w') as f:
    f.write('# Image list\n\n')
    f.write('## Number of points in cluster\n\n')
    for folder in folder_list:
        folder_name = folder.parts[1]
        # write the parameter for the image
        f.write(f'{folder_name}\n\n')
        f.write(f'![{folder_name}](./{folder_name}_num_points.png)\n\n')
    f.write('## Time distribution\n\n')
    for folder in folder_list:
        folder_name = folder.parts[1]
        f.write(f'{folder_name}\n\n')
        f.write(f'![{folder_name}](./{folder_name}_time_distribution.png)\n\n')
    f.write('## NPE distribution\n\n')
    for folder in folder_list:
        folder_name = folder.parts[1]
        f.write(f'{folder_name}\n\n')
        f.write(f'![{folder_name}](./{folder_name}_npe_distribution.png)\n\n')
    f.write('## Energy distribution\n\n')
    for folder in folder_list:
        folder_name = folder.parts[1]
        f.write(f'{folder_name}\n\n')
        f.write(f'![{folder_name}](./{folder_name}_energy_distribution.png)\n\n')
    f.write('## Confusion matrix\n\n')
    for folder in folder_list:
        folder_name = folder.parts[1]
        f.write(f'{folder_name}\n\n')
        f.write(f'![{folder_name}](./{folder_name}_confusion_matrix.png)\n\n')
print('Done')

import math
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.colors as colors


gps_pos_file = "1:30.json"
gzb_gps_info = {}
with open(gps_pos_file, "r") as fs:
    gzb_gps_info = json.load(fs)

gzb_true_pos = np.array(gzb_gps_info['true_pos'])
gzb_gps_pos = np.array(gzb_gps_info['gps_pos'])
vis_sat = np.array(gzb_gps_info['vis_sat'])
dop = np.array(gzb_gps_info['dop'])
laser_scan = np.array(gzb_gps_info['laser_scan'])

#Calculating the size of the heat map based on the maximum and minimum coordinate values in ENU coordinate system
scale = 5

#Computing the minimum value of 
size_x_min = (int)((np.abs(gzb_true_pos[:,0].min()))/scale)
size_y_min = (int)((np.abs(gzb_true_pos[:,1].min()))/scale)
size_x = (int)((np.abs(gzb_true_pos[:,0].max()))/scale + size_x_min)
size_y = (int)((np.abs(gzb_true_pos[:,1].max()))/scale + size_y_min)
heat_map = np.zeros((size_y+1,size_x+1))
sat_map = np.zeros((size_y+1,size_x+1))
dop_map = np.zeros((size_y+1,size_x+1))
obs_map = np.zeros((size_y+1,size_x+1))

for i in range(gzb_true_pos.shape[0]):
    y_grid = (int)(gzb_true_pos[i,1]/scale + size_y_min)
    x_grid = (int)(gzb_true_pos[i,0]/scale + size_x_min)
    error = np.linalg.norm(gzb_true_pos[i,:] - gzb_gps_pos[i,:])
    obs_map[y_grid,x_grid] = laser_scan[i]<np.inf 
    sat_map[y_grid,x_grid] = vis_sat[i]
    if obs_map[y_grid,x_grid]:
        heat_map[y_grid,x_grid] = 50
        sat_map[y_grid,x_grid] = 0
        dop_map[y_grid,x_grid] = 50
    else:
        if not math.isnan(error):
            if error > 30:
                error = 30
            heat_map[y_grid,x_grid] = error
            dop_map[y_grid,x_grid] = dop[i,3]
        else:
            heat_map[y_grid,x_grid] = 40
            dop_map[y_grid,x_grid] = 40
        
from matplotlib.pyplot import cm
import matplotlib.colors
cmap1 = colors.LinearSegmentedColormap.from_list("", ["green","yellow","orange","red", "darkred"])
cmap2 = colors.LinearSegmentedColormap.from_list("", ["darkblue","blue","lightblue","white"])

map_ = [heat_map, sat_map, dop_map]
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
cmaps = [cmap1, cmap2, cmap1]
clim = [[0,40],[0,8],[0,40]]
labels = ['GNSS Error Map', 'Num of Visible Satellite', 'HDOP']
for i in range(3):
    ax = axs[i]
    pcm = ax.pcolormesh(map_[i],
                        cmap=cmaps[i],vmin=clim[i][0], vmax=clim[i][1])
    fig.colorbar(pcm, ax=ax)
    ax.axis('off')
    ax.set_title(labels[i])

plt.tight_layout()
plt.show()

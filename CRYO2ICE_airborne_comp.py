# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:28:38 2024

@author: rmfha
"""

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.dates as mdates
import cartopy.feature as cfeature
import cartopy
from matplotlib import colors
import scipy.signal as scipy
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from sklearn.linear_model import LinearRegression
from scipy.interpolate import griddata
import numpy as np
from datetime import datetime, timedelta
import copy
from math import radians, cos, sin, asin, sqrt
import os
import numpy as np
import proplot as pplt
import h5py
import pandas as pd
import netCDF4
import sys
# import dates
from matplotlib.colors import LogNorm
from scipy import signal
import cartopy.crs as ccrs

sys.path.append(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison')

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,rgb2hex
cmap_check = plt.cm.get_cmap('RdYlBu_r')
cmap_qual = [rgb2hex(cmap_check((0/8))),rgb2hex(cmap_check((1/8))),rgb2hex(cmap_check((2/8))), rgb2hex(cmap_check((4.5/8))),
            rgb2hex(cmap_check((5/8))), rgb2hex(cmap_check(6/8)), rgb2hex(cmap_check(7/8)), rgb2hex(cmap_check(8/8))]
cmap_qual2 = LinearSegmentedColormap.from_list('list', cmap_qual, N = len(cmap_qual))
cmap_use = plt.cm.get_cmap('RdYlBu_r', 7) 

#%% Functions

def numpy_nan_mean(a):
    
    return np.nan if np.all(a!=a) else np.nanmean(a)

def CRYO2ICE_smooth_data(df1,df2, var1='snow_depth', dist_req=3500,var_out='output', lat1='lat', lon1='lon', lat2='lat', lon2='lon'):
    
    # df1, lat1, lon2: dataset to use for search 
    # df2, lat2, lon1: reference dataset 
    
    from sklearn.neighbors import BallTree
    import numpy as np
    import pandas as pd

    query_lats = df2[[lat2]].to_numpy()
    query_lons = df2[[lon2]].to_numpy()

    tree = BallTree(np.deg2rad(df1[[lat1, lon1]].values),  metric='haversine')

    dist_in_metres = dist_req
    earth_radius_in_metres = 6371*1000
    radius = dist_in_metres/earth_radius_in_metres

    is_within, distances = tree.query_radius(np.deg2rad(np.c_[query_lats, query_lons]), r=radius, count_only=False, return_distance=True) 
    distances_in_metres = distances*earth_radius_in_metres

    w_mean_data_df1 = np.empty(len(query_lats))
    k = 0
    for i in is_within:
        if len(i)>5:
            data_df1 = df1[var1].iloc[i]
            
            w_mean_data_df1_comp = np.ma.average(data_df1, weights=(1/distances_in_metres[k]))
                
        else:
            w_mean_data_df1_comp = np.nan
    
        
        w_mean_data_df1[k] = w_mean_data_df1_comp
        k +=1
    df2[var_out] = w_mean_data_df1
    df2[var_out][~df2['snow_depth'].notna()]=np.nan
    return df2


#%% Read data


df_CRYO2ICE_CryoTEMPO = pd.read_hdf(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data\CRYO2ICE_CryoTEMPO_CS_OFFL_SIR_TDP_SI_ANTARC_20221213T201353_20221213T202031_28_04332_C001_CASSIS_AMSR2.h5', index_col=None, header=0)

df_airborne = pd.read_csv(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\{}_roughness.csv'.format('df_airborne'))

#%% Compute along-track distance
# Haversine function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3  # Earth radius in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# Extract the first point
first_lat = df_airborne.loc[0, 'latitude']
first_lon = df_airborne.loc[0, 'longitude']

# Calculate the haversine distance for each point in the DataFrame
df_airborne['haversine_distance'] = df_airborne.apply(
    lambda row: haversine(first_lat, first_lon, row['latitude'], row['longitude']),
    axis=1
)


#%% Bin airborne data into 1-km segments
import numpy as np
import pandas as pd

# Assume df_airborne is already defined and loaded with your data
j = 1e3  # in metres 
max_distance = np.max(df_airborne['haversine_distance'])
bins = np.arange(0, max_distance + j, j)

# Create bins
df_airborne['bin'] = pd.cut(df_airborne['haversine_distance'], bins=bins, labels=bins[:-1] + j / 2)

# Filter each "hs" column with the requirement that values must be between -0.05 and 1.5 m
hs_columns = [
    'hs_ALS-Ku_MAX', 'hs_ALS-Ka_MAX', 'hs_ALS-C/S_MAX',
    'hs_ALS-Ku_TFMRA50', 'hs_ALS-Ka_TFMRA50', 'hs_PEAK', 'hs_CWT'
]

for col in hs_columns:
    df_airborne[col] = df_airborne[col].where((df_airborne[col] >= -0.05) & (df_airborne[col] <= 1.5))


# Define a function to apply np.nanmean
def nanmean(series):
    return np.nanmean(series)

# Group by bins and calculate the nanmean for each group
grouped = df_airborne.groupby('bin').agg({
    'latitude': nanmean,
    'longitude': nanmean,
    'hs_ALS-Ku_MAX': nanmean,
    'hs_ALS-Ka_MAX': nanmean,
    'hs_ALS-C/S_MAX': nanmean,
    'hs_ALS-Ku_TFMRA50': nanmean,
    'hs_ALS-Ka_TFMRA50': nanmean,
    'hs_PEAK': nanmean,
    'hs_CWT': nanmean
}).reset_index()

# Rename the bin column to along_track_dist_in_metres
grouped.rename(columns={'bin': 'along_track_dist_in_metres'}, inplace=True)

df_airborne_1km = grouped

print(df_airborne_1km)


#%% Bin to CRYO2ICE
df_airborne = df_airborne_1km
dist = 3.5e3
var_hs = 'hs_ALS-Ku_MAX'
print('CRYO2ICE search using airborne data: {} ....'.format(var_hs))
df_check = df_airborne[(df_airborne[var_hs].notna()) & (df_airborne[var_hs]<1.5) & (df_airborne[var_hs]>-0.05)]
df_CRYO2ICE_CryoTEMPO = CRYO2ICE_smooth_data(df_check, df_CRYO2ICE_CryoTEMPO, var1 = var_hs, lat1='latitude', lon1='longitude', dist_req=dist, var_out = var_hs)

var_hs = 'hs_ALS-Ka_MAX'
print('CRYO2ICE search using airborne data: {} ....'.format(var_hs))
df_check = df_airborne[(df_airborne[var_hs].notna()) & (df_airborne[var_hs]<1.5) & (df_airborne[var_hs]>-0.05)]
df_CRYO2ICE_CryoTEMPO = CRYO2ICE_smooth_data(df_check, df_CRYO2ICE_CryoTEMPO, var1 = var_hs, lat1='latitude', lon1='longitude', dist_req=dist, var_out = var_hs)

var_hs = 'hs_ALS-C/S_MAX'
print('CRYO2ICE search using airborne data: {} ....'.format(var_hs))
df_check = df_airborne[(df_airborne[var_hs].notna()) & (df_airborne[var_hs]<1.5) & (df_airborne[var_hs]>-0.05)]
df_CRYO2ICE_CryoTEMPO = CRYO2ICE_smooth_data(df_check, df_CRYO2ICE_CryoTEMPO, var1 = var_hs, lat1='latitude', lon1='longitude', dist_req=dist, var_out = var_hs)

var_hs = 'hs_ALS-Ku_TFMRA50'
print('CRYO2ICE search using airborne data: {} ....'.format(var_hs))
df_check = df_airborne[(df_airborne[var_hs].notna()) & (df_airborne[var_hs]<1.5) & (df_airborne[var_hs]>-0.05)]
df_CRYO2ICE_CryoTEMPO = CRYO2ICE_smooth_data(df_check, df_CRYO2ICE_CryoTEMPO, var1 = var_hs, lat1='latitude', lon1='longitude', dist_req=dist, var_out = var_hs)

var_hs = 'hs_ALS-Ka_TFMRA50'
print('CRYO2ICE search using airborne data: {} ....'.format(var_hs))
df_check = df_airborne[(df_airborne[var_hs].notna()) & (df_airborne[var_hs]<1.5) & (df_airborne[var_hs]>-0.05)]
df_CRYO2ICE_CryoTEMPO = CRYO2ICE_smooth_data(df_check, df_CRYO2ICE_CryoTEMPO, var1 = var_hs, lat1='latitude', lon1='longitude', dist_req=dist, var_out = var_hs)

var_hs = 'hs_PEAK'
print('CRYO2ICE search using airborne data: {} ....'.format(var_hs))
df_check = df_airborne[(df_airborne[var_hs].notna()) & (df_airborne[var_hs]<1.5) & (df_airborne[var_hs]>-0.05)]
df_CRYO2ICE_CryoTEMPO = CRYO2ICE_smooth_data(df_check, df_CRYO2ICE_CryoTEMPO, var1 = var_hs, lat1='latitude', lon1='longitude', dist_req=dist, var_out = var_hs)

var_hs = 'hs_CWT'
print('CRYO2ICE search using airborne data: {} ....'.format(var_hs))
df_check = df_airborne[(df_airborne[var_hs].notna()) & (df_airborne[var_hs]<1.5) & (df_airborne[var_hs]>-0.05)]
df_CRYO2ICE_CryoTEMPO = CRYO2ICE_smooth_data(df_check, df_CRYO2ICE_CryoTEMPO, var1 = var_hs, lat1='latitude', lon1='longitude', dist_req=dist, var_out = var_hs)


#%% Plotting
def fit_linear(val1, val2):
    from scipy import stats
    # Fit the model to the data
    val1 = np.array(val1)
    val2 = np.array(val2)
    mask = np.isfinite(val1) & np.isfinite(val2)
    res = stats.linregress(val1[mask], val2[mask])
    
    from sklearn.metrics import mean_squared_error

    rmsd = mean_squared_error(val1[mask], val2[mask], squared=False)

    #rmsd = np.sqrt(((((val1[mask] - val2[mask])** 2))*3).mean())
    
    return res, rmsd
#fig, ax = pplt.subplots([[1]], axwidth=2.5, axheight=2.5, sharex=False, sharey=False, )


fig, axs = pplt.subplots([[1, 1, 1],
                          [2, 3, 4], 
                          [5, 6, 7]], share=0,
                         axwidth=7, axheight=2, sharex=False, sharey=False)
fig.patch.set_facecolor('white')

df_CRYO2ICE_CryoTEMPO_check = df_CRYO2ICE_CryoTEMPO[df_CRYO2ICE_CryoTEMPO['hs_PEAK'].notna()].reset_index()

ax = axs[0]
s1 = 1
leg1=ax.plot(df_CRYO2ICE_CryoTEMPO_check['lat'], df_CRYO2ICE_CryoTEMPO_check['snow_depth'], label='CRYO2ICE{}: {:.2f} $\pm$ {:.2f} m'.format('$_{CryoTEMPO}$', np.nanmean(df_CRYO2ICE_CryoTEMPO_check['snow_depth']), np.nanstd(df_CRYO2ICE_CryoTEMPO_check['snow_depth'])), linewidth=2, markersize=s1, linestyle='-', marker='o', zorder=10, c='k')

leg2=ax.scatter(df_CRYO2ICE_CryoTEMPO_check['lat'], df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ku_MAX'], label='CRYO2ICE{}: {:.2f} $\pm$ {:.2f} m'.format('$_{h_{s, ALS-Ku_{MAX}}}$', np.nanmean(df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ku_MAX']), np.nanstd(df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ku_MAX'])), linewidth=0.5, markersize=s1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[0])
leg3=ax.scatter(df_CRYO2ICE_CryoTEMPO_check['lat'], df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ka_MAX'], label='CRYO2ICE{}: {:.2f} $\pm$ {:.2f} m'.format('$_{h_{s, ALS-Ka_{MAX}}}$', np.nanmean(df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ka_MAX']), np.nanstd(df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ka_MAX'])), linewidth=0.5, markersize=s1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[1])
leg4=ax.plot(df_CRYO2ICE_CryoTEMPO_check['lat'], df_CRYO2ICE_CryoTEMPO_check['hs_ALS-C/S_MAX'], label='CRYO2ICE{}: {:.2f} $\pm$ {:.2f} m'.format('$_{h_{s, ALS-C/S_{MAX}}}$', np.nanmean(df_CRYO2ICE_CryoTEMPO_check['hs_ALS-C/S_MAX']), np.nanstd(df_CRYO2ICE_CryoTEMPO_check['hs_ALS-C/S_MAX'])), linewidth=0.5, markersize=s1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[2])

leg5=ax.scatter(df_CRYO2ICE_CryoTEMPO_check['lat'], df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ku_TFMRA50'], label='CRYO2ICE{}: {:.2f} $\pm$ {:.2f} m'.format('$_{h_{s, ALS-Ku_{TFMRA50}}}$', np.nanmean(df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ku_TFMRA50']), np.nanstd(df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ku_TFMRA50'])), linewidth=0.5, markersize=s1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[3])
leg6=ax.scatter(df_CRYO2ICE_CryoTEMPO_check['lat'], df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ka_TFMRA50'], label='CRYO2ICE{}: {:.2f} $\pm$ {:.2f} m'.format('$_{h_{s, ALS-Ka_{TFMRA50}}}$', np.nanmean(df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ka_TFMRA50']), np.nanstd(df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ka_TFMRA50'])), linewidth=0.5, markersize=s1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[4])

leg7=ax.plot(df_CRYO2ICE_CryoTEMPO_check['lat'], df_CRYO2ICE_CryoTEMPO_check['hs_PEAK'], label='CRYO2ICE{}: {:.2f} $\pm$ {:.2f} m'.format('$_{h_{s, PEAK}}$', np.nanmean(df_CRYO2ICE_CryoTEMPO_check['hs_PEAK']), np.nanstd(df_CRYO2ICE_CryoTEMPO_check['hs_PEAK'])), linewidth=0.5, markersize=s1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[5])
leg8=ax.plot(df_CRYO2ICE_CryoTEMPO_check['lat'], df_CRYO2ICE_CryoTEMPO_check['hs_CWT'], label='CRYO2ICE{}: {:.2f} $\pm$ {:.2f} m'.format('$_{h_{s, CWT}}$', np.nanmean(df_CRYO2ICE_CryoTEMPO_check['hs_CWT']), np.nanstd(df_CRYO2ICE_CryoTEMPO_check['hs_CWT'])), linewidth=0.5, markersize=s1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[6])

ax.legend([leg1, leg2, leg3, leg4, leg5, leg6, leg7, leg8],loc='b', markersize=5, order='F', ncols=3, frameon=False) 
ax.format(ylabel='snow depth, h$_s$ (m)',xlabel='latitude (degrees N)', lefttitle='Airborne observations binned to {:.0f} km-segments before CRYO2ICE smoothing'.format(j/1e3))

xlim1, ylim1 = (0, 1), (0, 1)
x = np.arange(0, 1, 0.1)
s1 = 10
ax = axs[1]
val1, val2 = 'snow_depth', 'hs_ALS-Ku_MAX'
ax.scatter(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2], s=s1, c=cmap_qual[0], edgecolor='k', linewidth=0.5)
ax.format(xlim=xlim1, ylim=ylim1, xlabel='', lefttitle='h$_{s, {ALS}-Ku_{MAX}}$', ylabel='airborne snow depth (m)')
ax.plot(x, x, c='k', zorder=-10, linewidth=0.5)

res, rmsd = fit_linear(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2])
corr = df_CRYO2ICE_CryoTEMPO_check[val1].corr(df_CRYO2ICE_CryoTEMPO_check[val2])
vals = np.arange(-10, 30, 0.5)
plot_show = ax.plot(vals, res.intercept+vals*res.slope,c=cmap_qual[0], linewidth=0.5, label='Bias: {:.2f} m\nPearsons correlation: {:.2f}\nIntercept: {:.2f} m\nSlope: {:.2f}\nRMSD: {:.2f} m'.format(np.nanmean(df_CRYO2ICE_CryoTEMPO_check[val1]-df_CRYO2ICE_CryoTEMPO_check[val2]),corr,res.intercept, res.slope, rmsd), markersize=0, facecolor=cmap_qual[0])
ax.legend(plot_show, loc='ul', handlelength=0, frameon=False)

ax = axs[2]
val1, val2 = 'snow_depth', 'hs_ALS-Ka_MAX'
ax.scatter(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2], s=s1, c=cmap_qual[1], edgecolor='k', linewidth=0.5)
ax.format(xlim=xlim1, ylim=ylim1, xlabel='', lefttitle='h$_{s, {ALS}-Ka_{MAX}}$', ylabel='')
ax.plot(x, x, c='k', zorder=-10, linewidth=0.5)

res, rmsd = fit_linear(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2])
corr = df_CRYO2ICE_CryoTEMPO_check[val1].corr(df_CRYO2ICE_CryoTEMPO_check[val2])
vals = np.arange(-10, 30, 0.5)
plot_show = ax.plot(vals, res.intercept+vals*res.slope,c=cmap_qual[1], linewidth=0.5, label='Bias: {:.2f} m\nPearsons correlation: {:.2f}\nIntercept: {:.2f} m\nSlope: {:.2f}\nRMSD: {:.2f} m'.format(np.nanmean(df_CRYO2ICE_CryoTEMPO_check[val1]-df_CRYO2ICE_CryoTEMPO_check[val2]),corr,res.intercept, res.slope, rmsd), markersize=0, facecolor=cmap_qual[0])
ax.legend(plot_show, loc='ul', handlelength=0, frameon=False)

ax = axs[3]
val1, val2 = 'snow_depth', 'hs_ALS-C/S_MAX'
ax.scatter(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2], s=s1, c=cmap_qual[2], edgecolor='k', linewidth=0.5)
ax.format(xlim=xlim1, ylim=ylim1, xlabel='', lefttitle='h$_{s, {ALS}-C/S_{MAX}}$', ylabel='')
ax.plot(x, x, c='k', zorder=-10, linewidth=0.5)

res, rmsd = fit_linear(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2])
corr = df_CRYO2ICE_CryoTEMPO_check[val1].corr(df_CRYO2ICE_CryoTEMPO_check[val2])
vals = np.arange(-10, 30, 0.5)
plot_show = ax.plot(vals, res.intercept+vals*res.slope,c=cmap_qual[2], linewidth=0.5, label='Bias: {:.2f} m\nPearsons correlation: {:.2f}\nIntercept: {:.2f} m\nSlope: {:.2f}\nRMSD: {:.2f} m'.format(np.nanmean(df_CRYO2ICE_CryoTEMPO_check[val1]-df_CRYO2ICE_CryoTEMPO_check[val2]),corr,res.intercept, res.slope, rmsd), markersize=0, facecolor=cmap_qual[0])
ax.legend(plot_show, loc='ul', handlelength=0, frameon=False)

ax = axs[4]
val1, val2 = 'snow_depth', 'hs_ALS-Ku_TFMRA50'
plot1=ax.scatter(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2], s=s1, c=cmap_qual[3], edgecolor='k', linewidth=0.5, marker='^')
#ax.format(xlim=xlim1, ylim=ylim1, xlabel='', lefttitle='h$_{s, {ALS}-Ku_{TFMRA50}}$', ylabel='airborne snow depth (m)')
ax.plot(x, x, c='k', zorder=-10, linewidth=0.5)

res, rmsd = fit_linear(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2])
corr = df_CRYO2ICE_CryoTEMPO_check[val1].corr(df_CRYO2ICE_CryoTEMPO_check[val2])
vals = np.arange(-10, 30, 0.5)
ax.plot(vals, res.intercept+vals*res.slope,c=cmap_qual[3], linewidth=0.5, markersize=0, facecolor=cmap_qual[3])
plot_show ='{}, Bias: {:.2f} m\nPearsons correlation: {:.2f}\nIntercept: {:.2f} m\nSlope: {:.2f}\nRMSD: {:.2f} m'.format(r'$\mathbf{Ku}$-$\mathbf{band}$',np.nanmean(df_CRYO2ICE_CryoTEMPO_check[val1]-df_CRYO2ICE_CryoTEMPO_check[val2]),corr,res.intercept, res.slope, rmsd)

val1, val2 = 'snow_depth', 'hs_ALS-Ka_TFMRA50'
plot2=ax.scatter(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2], s=s1, c=cmap_qual[4], edgecolor='grey', linewidth=0.5)
ax.format(xlim=xlim1, ylim=ylim1, xlabel='', lefttitle='h$_{s, {ALS}-Ku/Ka_{TFMRA50}}$', ylabel='airborne snow depth (m)')
ax.plot(x, x, c='k', zorder=-10, linewidth=0.5)

res, rmsd = fit_linear(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2])
corr = df_CRYO2ICE_CryoTEMPO_check[val1].corr(df_CRYO2ICE_CryoTEMPO_check[val2])
vals = np.arange(-10, 30, 0.5)
ax.plot(vals, res.intercept+vals*res.slope,c=cmap_qual[4], linewidth=0.5, markersize=0, facecolor=cmap_qual[0])
plot_show2 ='{}, Bias: {:.2f} m\nPearsons correlation: {:.2f}\nIntercept: {:.2f} m\nSlope: {:.2f}\nRMSD: {:.2f} m'.format(r'$\mathbf{Ka}$-$\mathbf{band}$',np.nanmean(df_CRYO2ICE_CryoTEMPO_check[val1]-df_CRYO2ICE_CryoTEMPO_check[val2]),corr,res.intercept, res.slope, rmsd)
ax.legend([plot1, plot2],[plot_show, plot_show2], loc='ul', handlelength=1, frameon=False, markersize=50, ncols=1)

ax = axs[5]
val1, val2 = 'snow_depth', 'hs_PEAK'
ax.scatter(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2], s=s1, c=cmap_qual[5], edgecolor='k', linewidth=0.5)
ax.format(xlim=xlim1, ylim=ylim1, xlabel='CRYO2ICE CryoTEMPO{} snow depth (m)'.format('$_{smooth}$'), lefttitle='h$_{s, PEAK}$', ylabel='')
ax.plot(x, x, c='k', zorder=-10, linewidth=0.5)

res, rmsd = fit_linear(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2])
corr = df_CRYO2ICE_CryoTEMPO_check[val1].corr(df_CRYO2ICE_CryoTEMPO_check[val2])
vals = np.arange(-10, 30, 0.5)
plot_show = ax.plot(vals, res.intercept+vals*res.slope,c=cmap_qual[5], linewidth=0.5, label='Bias: {:.2f} m\nPearsons correlation: {:.2f}\nIntercept: {:.2f} m\nSlope: {:.2f}\nRMSD: {:.2f} m'.format(np.nanmean(df_CRYO2ICE_CryoTEMPO_check[val1]-df_CRYO2ICE_CryoTEMPO_check[val2]),corr,res.intercept, res.slope, rmsd), markersize=0, facecolor=cmap_qual[0])
ax.legend(plot_show, loc='ul', handlelength=0, frameon=False)


ax = axs[6]
val1, val2 = 'snow_depth', 'hs_CWT'
ax.scatter(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2], s=s1, c=cmap_qual[6], edgecolor='k', linewidth=0.5)
ax.format(xlim=xlim1, ylim=ylim1, xlabel='', lefttitle='h$_{s, CWT}$', ylabel='')
ax.plot(x, x, c='k', zorder=-10, linewidth=0.5)

res, rmsd = fit_linear(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2])
corr = df_CRYO2ICE_CryoTEMPO_check[val1].corr(df_CRYO2ICE_CryoTEMPO_check[val2])
vals = np.arange(-10, 30, 0.5)
plot_show = ax.plot(vals, res.intercept+vals*res.slope,c=cmap_qual[6], linewidth=0.5, label='Bias: {:.2f} m\nPearsons correlation: {:.2f}\nIntercept: {:.2f} m\nSlope: {:.2f}\nRMSD: {:.2f} m'.format(np.nanmean(df_CRYO2ICE_CryoTEMPO_check[val1]-df_CRYO2ICE_CryoTEMPO_check[val2]),corr,res.intercept, res.slope, rmsd), markersize=0, facecolor=cmap_qual[0])
ax.legend(plot_show, loc='ul', handlelength=0, frameon=False)
fig.format(abc='(a)', abcloc='l')
fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\Figure_comp_CRYO2ICE_air_along_orbit_1kmairborne.png', dpi=300)

print(len(df_CRYO2ICE_CryoTEMPO_check))
#%% Compute 25 km-segments

# Extract the first point
first_lat = df_CRYO2ICE_CryoTEMPO.loc[0, 'lat']
first_lon = df_CRYO2ICE_CryoTEMPO.loc[0, 'lon']

# Calculate the haversine distance for each point in the DataFrame
df_CRYO2ICE_CryoTEMPO['haversine_distance'] = df_CRYO2ICE_CryoTEMPO.apply(
    lambda row: haversine(first_lat, first_lon, row['lat'], row['lon']),
    axis=1
)


# Assume df_airborne is already defined and loaded with your data
j = 25e3  # in metres (40 meters)
max_distance = np.max(df_CRYO2ICE_CryoTEMPO['haversine_distance'])
bins = np.arange(0, max_distance + j, j)

# Create bins
df_CRYO2ICE_CryoTEMPO['bin'] = pd.cut(df_CRYO2ICE_CryoTEMPO['haversine_distance'], bins=bins, labels=bins[:-1] + j / 2)

# Filter each "hs" column with the requirement that values must be between -0.05 and 1.5 m
hs_columns = [
    'hs_ALS-Ku_MAX', 'hs_ALS-Ka_MAX', 'hs_ALS-C/S_MAX',
    'hs_ALS-Ku_TFMRA50', 'hs_ALS-Ka_TFMRA50', 'hs_PEAK', 'hs_CWT',
]

for col in hs_columns:
    df_CRYO2ICE_CryoTEMPO[col] = df_CRYO2ICE_CryoTEMPO[col].where((df_CRYO2ICE_CryoTEMPO[col] >= -0.05) & (df_CRYO2ICE_CryoTEMPO[col] <= 1.5))


# Define a function to apply np.nanmean
def nanmean(series):
    return np.nanmean(series)

# Group by bins and calculate the nanmean for each group
grouped = df_CRYO2ICE_CryoTEMPO.groupby('bin').agg({
    'lat': nanmean,
    'lon': nanmean,
    'hs_ALS-Ku_MAX': nanmean,
    'hs_ALS-Ka_MAX': nanmean,
    'hs_ALS-C/S_MAX': nanmean,
    'hs_ALS-Ku_TFMRA50': nanmean,
    'hs_ALS-Ka_TFMRA50': nanmean,
    'hs_PEAK': nanmean,
    'hs_CWT': nanmean,
    'snow_depth':nanmean
}).reset_index()

# Rename the bin column to along_track_dist_in_metres
grouped.rename(columns={'bin': 'along_track_dist_in_metres'}, inplace=True)

df_CRYO2ICE_CryoTEMPO = grouped


#%% Plotting 25-km segments
#fig, ax = pplt.subplots([[1]], axwidth=2.5, axheight=2.5, sharex=False, sharey=False, )


fig, axs = pplt.subplots([[1, 1, 1],
                          [2, 3, 4], 
                          [5, 6, 7]], share=0,
                         axwidth=7, axheight=2, sharex=False, sharey=False)
fig.patch.set_facecolor('white')

df_CRYO2ICE_CryoTEMPO_check = df_CRYO2ICE_CryoTEMPO[df_CRYO2ICE_CryoTEMPO['hs_PEAK'].notna()].reset_index()

ax = axs[0]
s1 = 10
leg1=ax.plot(df_CRYO2ICE_CryoTEMPO_check['lat'], df_CRYO2ICE_CryoTEMPO_check['snow_depth'], label='CRYO2ICE{}: {:.2f} $\pm$ {:.2f} m'.format('$_{CryoTEMPO}$', np.nanmean(df_CRYO2ICE_CryoTEMPO_check['snow_depth']), np.nanstd(df_CRYO2ICE_CryoTEMPO_check['snow_depth'])), linewidth=2, markersize=s1, linestyle='-', marker='o', zorder=10, c='k')

leg2=ax.scatter(df_CRYO2ICE_CryoTEMPO_check['lat'], df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ku_MAX'], label='CRYO2ICE{}: {:.2f} $\pm$ {:.2f} m'.format('$_{h_{s, ALS-Ku_{MAX}}}$', np.nanmean(df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ku_MAX']), np.nanstd(df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ku_MAX'])), linewidth=0.5, markersize=s1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[0])
leg3=ax.scatter(df_CRYO2ICE_CryoTEMPO_check['lat'], df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ka_MAX'], label='CRYO2ICE{}: {:.2f} $\pm$ {:.2f} m'.format('$_{h_{s, ALS-Ka_{MAX}}}$', np.nanmean(df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ka_MAX']), np.nanstd(df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ka_MAX'])), linewidth=0.5, markersize=s1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[1])
leg4=ax.plot(df_CRYO2ICE_CryoTEMPO_check['lat'], df_CRYO2ICE_CryoTEMPO_check['hs_ALS-C/S_MAX'], label='CRYO2ICE{}: {:.2f} $\pm$ {:.2f} m'.format('$_{h_{s, ALS-C/S_{MAX}}}$', np.nanmean(df_CRYO2ICE_CryoTEMPO_check['hs_ALS-C/S_MAX']), np.nanstd(df_CRYO2ICE_CryoTEMPO_check['hs_ALS-C/S_MAX'])), linewidth=0.5, markersize=s1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[2])

leg5=ax.scatter(df_CRYO2ICE_CryoTEMPO_check['lat'], df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ku_TFMRA50'], label='CRYO2ICE{}: {:.2f} $\pm$ {:.2f} m'.format('$_{h_{s, ALS-Ku_{TFMRA50}}}$', np.nanmean(df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ku_TFMRA50']), np.nanstd(df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ku_TFMRA50'])), linewidth=0.5, markersize=s1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[3])
leg6=ax.scatter(df_CRYO2ICE_CryoTEMPO_check['lat'], df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ka_TFMRA50'], label='CRYO2ICE{}: {:.2f} $\pm$ {:.2f} m'.format('$_{h_{s, ALS-Ka_{TFMRA50}}}$', np.nanmean(df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ka_TFMRA50']), np.nanstd(df_CRYO2ICE_CryoTEMPO_check['hs_ALS-Ka_TFMRA50'])), linewidth=0.5, markersize=s1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[4])

leg7=ax.plot(df_CRYO2ICE_CryoTEMPO_check['lat'], df_CRYO2ICE_CryoTEMPO_check['hs_PEAK'], label='CRYO2ICE{}: {:.2f} $\pm$ {:.2f} m'.format('$_{h_{s, PEAK}}$', np.nanmean(df_CRYO2ICE_CryoTEMPO_check['hs_PEAK']), np.nanstd(df_CRYO2ICE_CryoTEMPO_check['hs_PEAK'])), linewidth=0.5, markersize=s1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[5])
leg8=ax.plot(df_CRYO2ICE_CryoTEMPO_check['lat'], df_CRYO2ICE_CryoTEMPO_check['hs_CWT'], label='CRYO2ICE{}: {:.2f} $\pm$ {:.2f} m'.format('$_{h_{s, CWT}}$', np.nanmean(df_CRYO2ICE_CryoTEMPO_check['hs_CWT']), np.nanstd(df_CRYO2ICE_CryoTEMPO_check['hs_CWT'])), linewidth=0.5, markersize=s1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[6])

ax.legend([leg1, leg2, leg3, leg4, leg5, leg6, leg7, leg8],loc='b', markersize=5, order='F', ncols=3, frameon=False) 
ax.format(ylabel='snow depth, h$_s$ (m)',xlabel='latitude (degrees N)', lefttitle='Airborne observations binned to 1 km-segments, CRYO2ICE identification (7-km along-track), lastly 25-km smoothing')

xlim1, ylim1 = (0, 1), (0, 1)
x = np.arange(0, 1, 0.1)
s1 = 10
ax = axs[1]
val1, val2 = 'snow_depth', 'hs_ALS-Ku_MAX'
ax.scatter(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2], s=s1, c=cmap_qual[0], edgecolor='k', linewidth=0.5)
ax.format(xlim=xlim1, ylim=ylim1, xlabel='', lefttitle='h$_{s, {ALS}-Ku_{MAX}}$', ylabel='airborne snow depth (m)')
ax.plot(x, x, c='k', zorder=-10, linewidth=0.5)

res, rmsd = fit_linear(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2])
corr = df_CRYO2ICE_CryoTEMPO_check[val1].corr(df_CRYO2ICE_CryoTEMPO_check[val2])
vals = np.arange(-10, 30, 0.5)
plot_show = ax.plot(vals, res.intercept+vals*res.slope,c=cmap_qual[0], linewidth=0.5, label='Bias: {:.2f} m\nPearsons correlation: {:.2f}\nIntercept: {:.2f} m\nSlope: {:.2f}\nRMSD: {:.2f} m'.format(np.nanmean(df_CRYO2ICE_CryoTEMPO_check[val1]-df_CRYO2ICE_CryoTEMPO_check[val2]),corr,res.intercept, res.slope, rmsd), markersize=0, facecolor=cmap_qual[0])
ax.legend(plot_show, loc='ul', handlelength=0, frameon=False)

ax = axs[2]
val1, val2 = 'snow_depth', 'hs_ALS-Ka_MAX'
ax.scatter(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2], s=s1, c=cmap_qual[1], edgecolor='k', linewidth=0.5)
ax.format(xlim=xlim1, ylim=ylim1, xlabel='', lefttitle='h$_{s, {ALS}-Ka_{MAX}}$', ylabel='')
ax.plot(x, x, c='k', zorder=-10, linewidth=0.5)

res, rmsd = fit_linear(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2])
corr = df_CRYO2ICE_CryoTEMPO_check[val1].corr(df_CRYO2ICE_CryoTEMPO_check[val2])
vals = np.arange(-10, 30, 0.5)
plot_show = ax.plot(vals, res.intercept+vals*res.slope,c=cmap_qual[1], linewidth=0.5, label='Bias: {:.2f} m\nPearsons correlation: {:.2f}\nIntercept: {:.2f} m\nSlope: {:.2f}\nRMSD: {:.2f} m'.format(np.nanmean(df_CRYO2ICE_CryoTEMPO_check[val1]-df_CRYO2ICE_CryoTEMPO_check[val2]),corr,res.intercept, res.slope, rmsd), markersize=0, facecolor=cmap_qual[0])
ax.legend(plot_show, loc='ul', handlelength=0, frameon=False)

ax = axs[3]
val1, val2 = 'snow_depth', 'hs_ALS-C/S_MAX'
ax.scatter(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2], s=s1, c=cmap_qual[2], edgecolor='k', linewidth=0.5)
ax.format(xlim=xlim1, ylim=ylim1, xlabel='', lefttitle='h$_{s, {ALS}-C/S_{MAX}}$', ylabel='')
ax.plot(x, x, c='k', zorder=-10, linewidth=0.5)

res, rmsd = fit_linear(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2])
corr = df_CRYO2ICE_CryoTEMPO_check[val1].corr(df_CRYO2ICE_CryoTEMPO_check[val2])
vals = np.arange(-10, 30, 0.5)
plot_show = ax.plot(vals, res.intercept+vals*res.slope,c=cmap_qual[2], linewidth=0.5, label='Bias: {:.2f} m\nPearsons correlation: {:.2f}\nIntercept: {:.2f} m\nSlope: {:.2f}\nRMSD: {:.2f} m'.format(np.nanmean(df_CRYO2ICE_CryoTEMPO_check[val1]-df_CRYO2ICE_CryoTEMPO_check[val2]),corr,res.intercept, res.slope, rmsd), markersize=0, facecolor=cmap_qual[0])
ax.legend(plot_show, loc='ul', handlelength=0, frameon=False)

ax = axs[4]
val1, val2 = 'snow_depth', 'hs_ALS-Ku_TFMRA50'
plot1=ax.scatter(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2], s=s1, c=cmap_qual[3], edgecolor='k', linewidth=0.5, marker='^')
#ax.format(xlim=xlim1, ylim=ylim1, xlabel='', lefttitle='h$_{s, {ALS}-Ku_{TFMRA50}}$', ylabel='airborne snow depth (m)')
ax.plot(x, x, c='k', zorder=-10, linewidth=0.5)

res, rmsd = fit_linear(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2])
corr = df_CRYO2ICE_CryoTEMPO_check[val1].corr(df_CRYO2ICE_CryoTEMPO_check[val2])
vals = np.arange(-10, 30, 0.5)
ax.plot(vals, res.intercept+vals*res.slope,c=cmap_qual[3], linewidth=0.5, markersize=0, facecolor=cmap_qual[3])
plot_show ='{}, Bias: {:.2f} m\nPearsons correlation: {:.2f}\nIntercept: {:.2f} m\nSlope: {:.2f}\nRMSD: {:.2f} m'.format(r'$\mathbf{Ku}$-$\mathbf{band}$',np.nanmean(df_CRYO2ICE_CryoTEMPO_check[val1]-df_CRYO2ICE_CryoTEMPO_check[val2]),corr,res.intercept, res.slope, rmsd)

val1, val2 = 'snow_depth', 'hs_ALS-Ka_TFMRA50'
plot2=ax.scatter(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2], s=s1, c=cmap_qual[4], edgecolor='grey', linewidth=0.5)
ax.format(xlim=xlim1, ylim=ylim1, xlabel='', lefttitle='h$_{s, {ALS}-Ku/Ka_{TFMRA50}}$', ylabel='airborne snow depth (m)')
ax.plot(x, x, c='k', zorder=-10, linewidth=0.5)

res, rmsd = fit_linear(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2])
corr = df_CRYO2ICE_CryoTEMPO_check[val1].corr(df_CRYO2ICE_CryoTEMPO_check[val2])
vals = np.arange(-10, 30, 0.5)
ax.plot(vals, res.intercept+vals*res.slope,c=cmap_qual[4], linewidth=0.5, markersize=0, facecolor=cmap_qual[0])
plot_show2 ='{}, Bias: {:.2f} m\nPearsons correlation: {:.2f}\nIntercept: {:.2f} m\nSlope: {:.2f}\nRMSD: {:.2f} m'.format(r'$\mathbf{Ka}$-$\mathbf{band}$',np.nanmean(df_CRYO2ICE_CryoTEMPO_check[val1]-df_CRYO2ICE_CryoTEMPO_check[val2]),corr,res.intercept, res.slope, rmsd)
ax.legend([plot1, plot2],[plot_show, plot_show2], loc='ul', handlelength=1, frameon=False, markersize=50, ncols=1)

ax = axs[5]
val1, val2 = 'snow_depth', 'hs_PEAK'
ax.scatter(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2], s=s1, c=cmap_qual[5], edgecolor='k', linewidth=0.5)
ax.format(xlim=xlim1, ylim=ylim1, xlabel='CRYO2ICE CryoTEMPO{} snow depth (m)'.format('$_{smooth}$'), lefttitle='h$_{s, PEAK}$', ylabel='')
ax.plot(x, x, c='k', zorder=-10, linewidth=0.5)

res, rmsd = fit_linear(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2])
corr = df_CRYO2ICE_CryoTEMPO_check[val1].corr(df_CRYO2ICE_CryoTEMPO_check[val2])
vals = np.arange(-10, 30, 0.5)
plot_show = ax.plot(vals, res.intercept+vals*res.slope,c=cmap_qual[5], linewidth=0.5, label='Bias: {:.2f} m\nPearsons correlation: {:.2f}\nIntercept: {:.2f} m\nSlope: {:.2f}\nRMSD: {:.2f} m'.format(np.nanmean(df_CRYO2ICE_CryoTEMPO_check[val1]-df_CRYO2ICE_CryoTEMPO_check[val2]),corr,res.intercept, res.slope, rmsd), markersize=0, facecolor=cmap_qual[0])
ax.legend(plot_show, loc='ul', handlelength=0, frameon=False)


ax = axs[6]
val1, val2 = 'snow_depth', 'hs_CWT'
ax.scatter(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2], s=s1, c=cmap_qual[6], edgecolor='k', linewidth=0.5)
ax.format(xlim=xlim1, ylim=ylim1, xlabel='', lefttitle='h$_{s, CWT}$', ylabel='')
ax.plot(x, x, c='k', zorder=-10, linewidth=0.5)

res, rmsd = fit_linear(df_CRYO2ICE_CryoTEMPO_check[val1], df_CRYO2ICE_CryoTEMPO_check[val2])
corr = df_CRYO2ICE_CryoTEMPO_check[val1].corr(df_CRYO2ICE_CryoTEMPO_check[val2])
vals = np.arange(-10, 30, 0.5)
plot_show = ax.plot(vals, res.intercept+vals*res.slope,c=cmap_qual[6], linewidth=0.5, label='Bias: {:.2f} m\nPearsons correlation: {:.2f}\nIntercept: {:.2f} m\nSlope: {:.2f}\nRMSD: {:.2f} m'.format(np.nanmean(df_CRYO2ICE_CryoTEMPO_check[val1]-df_CRYO2ICE_CryoTEMPO_check[val2]),corr,res.intercept, res.slope, rmsd), markersize=0, facecolor=cmap_qual[0])
ax.legend(plot_show, loc='ul', handlelength=0, frameon=False)
fig.format(abc='(a)', abcloc='l')
fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\Figure_comp_CRYO2ICE_air_along_orbit_25km.png', dpi=300)

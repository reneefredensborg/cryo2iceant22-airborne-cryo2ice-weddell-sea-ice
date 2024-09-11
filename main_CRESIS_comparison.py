# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:16:19 2023

@author: rmfha
"""

#%% Initialisation
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
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,rgb2hex
cmap_check = plt.cm.get_cmap('RdYlBu_r')
cmap_qual = [rgb2hex(cmap_check((0/8))),rgb2hex(cmap_check((1/8))),rgb2hex(cmap_check((2/8))), rgb2hex(cmap_check((4.5/8))),
            rgb2hex(cmap_check((5/8))), rgb2hex(cmap_check(6/8)), rgb2hex(cmap_check(7/8)), rgb2hex(cmap_check(8/8))]
cmap_qual2 = LinearSegmentedColormap.from_list('list', cmap_qual, N = len(cmap_qual))
cmap_use = plt.cm.get_cmap('RdYlBu_r', 7) 

#%% Functions


def peak_centroid(wf_kuband, time_conv):
    # N_obs = 600
    num_rec = len(wf_kuband)
    a_s_peaks, s_i_centroids, s_i_centroids_t, a_s_peaks_t = np.zeros(
        num_rec), np.zeros(num_rec), np.zeros(num_rec), np.zeros(num_rec)
    a_s_peaks_dd, s_i_dd, s_i_dd_t, a_s_peaks_dd_t = np.zeros(
        num_rec), np.zeros(num_rec), np.zeros(num_rec), np.zeros(num_rec)
    for N_obs in np.arange(0, num_rec):
        print('Waveform: {}/{}'.format(N_obs, num_rec))
        # noise=np.nanmean(wf_kuband[N_obs][0:500])
        try:
            # s_i_centroids[N_obs] = np.nansum(wf_kuband[N_obs]*np.arange(0, len(wf_kuband[0])))/np.nansum(wf_kuband[N_obs])
            # s_i_centroids_t[N_obs]=np.nansum(wf_kuband[N_obs]*time_conv[N_obs])/np.nansum(wf_kuband[N_obs])
            # a_s_peaks[N_obs]=np.nanargmax(wf_kuband[N_obs][:])
            # a_s_peaks_t[N_obs]=time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])]
            idx_val = np.arange(np.nanargmax(
                wf_kuband[N_obs][:])-100, np.nanargmax(wf_kuband[N_obs][:])+156)

            s_i_centroids[N_obs] = np.nansum(wf_kuband[N_obs][idx_val]*np.arange(
                0, len(wf_kuband[0][idx_val])))/np.nansum(wf_kuband[N_obs][idx_val])+idx_val[0]
            s_i_centroids_t[N_obs] = np.nansum(
                wf_kuband[N_obs][idx_val]*time_conv[N_obs][idx_val])/np.nansum(wf_kuband[N_obs][idx_val])
            a_s_peaks[N_obs] = np.nanargmax(
                wf_kuband[N_obs][idx_val])+idx_val[0]
            a_s_peaks_t[N_obs] = time_conv[N_obs][np.nanargmax(
                wf_kuband[N_obs][idx_val])+idx_val[0]]

            peaks = scipy.find_peaks(
                wf_kuband[N_obs], prominence=np.nanmax(wf_kuband[N_obs])*0.5)
            a_s_peaks_dd[N_obs] = np.nanargmax(wf_kuband[N_obs][:])
            s_i_dd[N_obs] = peaks[0][-1]
            a_s_peaks_dd_t[N_obs] = time_conv[N_obs][np.nanargmax(
                wf_kuband[N_obs][:])]
            s_i_dd_t[N_obs] = time_conv[N_obs][peaks[0][-1]]
        except:
            print('Peaks not identified. NaN-initialization.')
            s_i_centroids[N_obs] = np.nan
            a_s_peaks[N_obs] = np.nan
            a_s_peaks_dd[N_obs] = np.nan
            s_i_dd[N_obs] = np.nan

    idx_check = s_i_centroids < a_s_peaks
    a_s_peaks[idx_check] = s_i_centroids[idx_check]
    a_s_peaks_t[idx_check] = s_i_centroids_t[idx_check]

    return a_s_peaks, a_s_peaks_t, s_i_centroids, s_i_centroids_t, a_s_peaks_dd, s_i_dd, a_s_peaks_dd_t, s_i_dd_t

def ppk(pwr_waveform, n_bins=10):
    ppk = []
    for i in pwr_waveform:
        try:
            idx_val = np.arange(np.nanargmax(i)-100, np.nanargmax(i)+156)
            wf_i = i[idx_val]
            noise = np.nanmean(wf_i[10:19])
            wf_i = wf_i[wf_i > noise]
            
            #pp_first = np.nanmax(wf_i)/wf_i
            #pp = np.nansum(pp_first)
            pp = np.nanmax(wf_i)/np.nanmean(wf_i)
            ppk = np.append(ppk, pp)
        except:
            ppk = np.append(ppk, np.nan)
    return ppk

def ppk_left_right(pwr_waveform):
    ppk_r, ppk_l = [], []
    for i in pwr_waveform:
        try:
            idx_val = np.arange(np.nanargmax(i)-100, np.nanargmax(i)+156)
            wf_i = i[idx_val]
            pp_max = np.nanmax(wf_i)
            idx_max = np.argmax(wf_i)
            
            pp_left = 3*(pp_max/np.sum(wf_i[idx_max-3:idx_max-1]))
            pp_right = 3*(pp_max/np.sum(wf_i[idx_max+1:idx_max+3]))
            ppk_l = np.append(ppk_l,pp_left)
            ppk_r = np.append(ppk_r,pp_right)
        except:
            ppk_l = np.append(ppk_l,np.nan)
            ppk_r = np.append(ppk_r,np.nan)

        ppk_l[np.isinf(ppk_l)]=np.nan
        ppk_r[np.isinf(ppk_r)]=np.nan
        
    return ppk_l, ppk_r


def max_power(pwr_waveform):
    power_true_comb = []
    for i in pwr_waveform: 
        try: 
            idx_val = np.arange(np.nanargmax(i)-100, np.nanargmax(i)+156)
            wf_i = i[idx_val]
            max_power_val = np.nanmax(wf_i)
            power_true_comb = np.append(power_true_comb, max_power_val)
        except:
            power_true_comb = np.append(power_true_comb, np.nan)
    return power_true_comb


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometres. Use 3956 for miles. Determines return value units.
    r = 6371
    return c * r

def load_data2(path, fp, lon_var, lat_var, fb_var):
    nc_file = netCDF4.Dataset(path + '/' + fp, 'r')
    lon = np.array(nc_file.variables[lon_var][:], dtype=np.float32)
    lat = np.array(nc_file.variables[lat_var][:], dtype=np.float32)
    fb = np.array(nc_file.variables[fb_var][:], dtype=np.float32)

    return lon, lat, fb


def prep_data(path, fp):
    lon, lat, MSS = load_data2(path, fp, 'lon', 'lat', 'mss')

    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    lat_T, lon_T, MSS_T = lat_mesh.flatten(), lon_mesh.flatten(), MSS.flatten()
    df_DTU21 = pd.DataFrame({'lat': lat_T, 'lon': lon_T, 'MSS': MSS_T})
#   df_DTU21 = df_DTU21[(df_DTU21['lat']<-60)].reset_index(drop=True)
    del lon, lat, MSS, lon_mesh, lat_mesh, lat_T, lon_T, MSS_T

    return df_DTU21


def grid_data(lat_ori, lon_ori, var_ori, ref_lat, ref_lon, method_name):
    grid_var = griddata((lon_ori.flatten(), lat_ori.flatten()),
                        var_ori.flatten(), (ref_lon, ref_lat), method=method_name)

    return grid_var


def identify_DTU21MSS_along_CRYO2ICE_track(data_check, df_grid_DTU21):
    from sklearn.neighbors import BallTree
    import numpy as np

    # Only data within 825 m of CryoSat-2 footprint (find approximate area of coincident data)
    query_lats = data_check[['lat']].to_numpy()
    query_lons = data_check[['lon']].to_numpy()

    tree = BallTree(np.deg2rad(
        df_grid_DTU21[['lat', 'lon']].values), leaf_size=15, metric='haversine')

    distances, indices = tree.query(
        np.deg2rad(np.c_[query_lats, query_lons]), k=1)

    MSS = []
    for i in indices:
        MSS = np.append(MSS, df_grid_DTU21['MSS'][int(i)])

#    data_check['MSS_DTU21'] = MSS

    return MSS

def identify_roughness_along_orbit(data_find, data_check, param_search, lat1='lat', lon1='lon', lat2='lat', lon2='lon', search_type='NN', dist_req=False):
     from sklearn.neighbors import BallTree
     import numpy as np

     # data_find = dataframe to be collocated with (output size)
     # data_check = dataframe to be checked for to find the correct output (NN or radius search)
     # Only data within 825 m of CryoSat-2 footprint (find approximate area of coincident data)

     query_lats = data_find[[lat1]].to_numpy()
     query_lons = data_find[[lon1]].to_numpy()

     tree = BallTree(np.deg2rad(
         data_check[[lat2, lon2]].values),  metric='haversine')

     if search_type == 'NN':
         distances, indices = tree.query(
             np.deg2rad(np.c_[query_lats, query_lons]), k=1)
         new_param = []
         for i in indices:
             new_param = np.append(new_param, data_check[param_search][int(i)])
     elif search_type == 'RADIUS':
         dist_in_metres = dist_req
         earth_radius_in_metres = 6371*1000
         radius = dist_in_metres/earth_radius_in_metres

         is_within, distances = tree.query_radius(np.deg2rad(
             np.c_[query_lats, query_lons]), r=radius, count_only=False, return_distance=True)
         distances_in_metres = distances*earth_radius_in_metres

         new_param, new_lat_loc, new_lon_loc = np.empty(
             len(query_lats)), np.empty(len(query_lats)), np.empty(len(query_lats))
         k = 0
         for i in is_within:
             array_rel = data_check[param_search].iloc[i]
             array_rel_lat = data_check[lat2].iloc[i]
             array_rel_lon = data_check[lon2].iloc[i]
             if array_rel.size > 0:
                 mean_new_param_comp = np.nanstd(array_rel)
                 mean_lat_comp = np.nanmean(array_rel_lat)
                 mean_lon_comp = np.nanmean(array_rel_lon)
             else:
                 # Handle the case when the array is empty
                 mean_new_param_comp = np.nan
                 mean_lat_comp = np.nan
                 mean_lon_comp = np.nan
                 print(
                     'Observation point {}/{}: No points within search requirements.'.format(k, len(query_lats)))

             new_param[k] = mean_new_param_comp
             new_lat_loc[k] = mean_lat_comp
             new_lon_loc[k] = mean_lon_comp
             k = k+1
     else:
         print('Search type not correct. Input either "NN" or "RADIUS"')

 #    data_check['MSS_DTU21'] = MSS

     return new_param, new_lat_loc, new_lon_loc


def identify_along_orbit(data_find, data_check, param_search, lat1='lat', lon1='lon', lat2='lat', lon2='lon', search_type='NN', dist_req=False):
    from sklearn.neighbors import BallTree
    import numpy as np

    # data_find = dataframe to be collocated with (output size)
    # data_check = dataframe to be checked for to find the correct output (NN or radius search)
    # Only data within 825 m of CryoSat-2 footprint (find approximate area of coincident data)

    query_lats = data_find[[lat1]].to_numpy()
    query_lons = data_find[[lon1]].to_numpy()

    tree = BallTree(np.deg2rad(
        data_check[[lat2, lon2]].values),  metric='haversine')

    if search_type == 'NN':
        distances, indices = tree.query(
            np.deg2rad(np.c_[query_lats, query_lons]), k=1)
        new_param = []
        for i in indices:
            new_param = np.append(new_param, data_check[param_search][int(i)])
    elif search_type == 'RADIUS':
        dist_in_metres = dist_req
        earth_radius_in_metres = 6371*1000
        radius = dist_in_metres/earth_radius_in_metres

        is_within, distances = tree.query_radius(np.deg2rad(
            np.c_[query_lats, query_lons]), r=radius, count_only=False, return_distance=True)
        distances_in_metres = distances*earth_radius_in_metres

        new_param, new_lat_loc, new_lon_loc = np.empty(
            len(query_lats)), np.empty(len(query_lats)), np.empty(len(query_lats))
        k = 0
        for i in is_within:
            array_rel = data_check[param_search].iloc[i]
            array_rel_lat = data_check[lat2].iloc[i]
            array_rel_lon = data_check[lon2].iloc[i]
            if array_rel.size > 0:
                mean_new_param_comp = np.nanmean(array_rel)
                mean_lat_comp = np.nanmean(array_rel_lat)
                mean_lon_comp = np.nanmean(array_rel_lon)
            else:
                # Handle the case when the array is empty
                mean_new_param_comp = np.nan
                mean_lat_comp = np.nan
                mean_lon_comp = np.nan
                print(
                    'Observation point {}/{}: No points within search requirements.'.format(k, len(query_lats)))

            new_param[k] = mean_new_param_comp
            new_lat_loc[k] = mean_lat_comp
            new_lon_loc[k] = mean_lon_comp
            k = k+1
    else:
        print('Search type not correct. Input either "NN" or "RADIUS"')

#    data_check['MSS_DTU21'] = MSS

    return new_param, new_lat_loc, new_lon_loc


c = 300e6

#%% Figure plotting functions


def plot_waveforms_examples(n_extra, axs, title_spec, ALS_along_radar_track):
    N_obs = val_x*n1+n_extra
    ax = axs

    fn = 'kuband_20221213_02_74_232_002'
    ds = netCDF4.Dataset(
        r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
    wf_kuband = ds.variables['waveform'][:]
    range_40 = ds.variables['range_tfmra40'][:]
    range_50 = ds.variables['range_tfmra50'][:]
    range_80 = ds.variables['range_tfmra80'][:]

    # range_40 = ds.variables['retracking_gate_tfmra40'][:]
    # range_50 = ds.variables['retracking_gate_tfmra50'][:]
    # range_80 = ds.variables['retracking_gate_tfmra80'][:]

    time = ds.variables['two_way_travel_time'][:]
    time_conv = (time*c/2)+offset_ku
    # wf_kuband = 10 * np.log10(wf_kuband)

    idx = np.arange(np.nanargmax(
        wf_kuband[N_obs][:])-100, np.nanargmax(wf_kuband[N_obs][:])+156)
    # noise=np.nanmean(wf_kuband[N_obs][0:500])
    centroid = np.nansum(
        wf_kuband[N_obs][idx]*time_conv[N_obs][idx])/np.nansum(wf_kuband[N_obs][idx])
    # peaks=scipy.find_peaks(wf_kuband[0], height=noise+noise*10, width=2)
    peaks = scipy.find_peaks(wf_kuband[N_obs][idx], prominence=np.nanmax(
        wf_kuband[N_obs][idx])*0.2, distance=25)

    leg1 = ax.axhline(y=(range_40[N_obs]*c/2)+offset_ku,
                      color='b', linestyle='-', linewidth=1, label='TFMRA40')
    leg2 = ax.axhline(y=(range_50[N_obs]*c/2)+offset_ku,
                      color='red', linestyle='--', linewidth=1, label='TFMRA50')
    leg3 = ax.axhline(y=(range_80[N_obs]*c/2)+offset_ku,
                      color='dodgerblue', linestyle='--', linewidth=1, label='TFMRA80')

    leg4 = ax.axhline(y=centroid, color='orange', linestyle='-', linewidth=1)
    # ax.scatter(wf_kuband[N_obs][peaks[0][:]+idx[0]],time_conv[N_obs][peaks[0][:]+idx[0]], marker='*', s=10, zorder=2)
    leg5 = ax.axhline(y=time_conv[N_obs][np.nanargmax(
        wf_kuband[N_obs][:])], color='magenta', linestyle='--', linewidth=1)
    ax.plot(wf_kuband[N_obs-5:N_obs+5][:, idx],
            time_conv[N_obs-5:N_obs+5][:, idx], zorder=0, c='lightgrey')
    leg6 = ax.plot(wf_kuband[N_obs][idx], time_conv[N_obs]
                   [idx], zorder=0, c='k', linewidth=0.5)
    # ax.scatter(wf_kuband[N_obs][np.nanargmax(wf_kuband[N_obs][:])], time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])],marker='*', s=10, zorder=2)
    # ax.format(ultitle='$\delta$range={:.2f} m'.format((centroid-time_conv[N_obs][peaks[0][0]+idx[0]])))
    ax.format(abc="(a)", abcloc='l', ylabel='range (m)', xlim=(np.nanmin(wf_kuband[N_obs])+0.05*np.nanmean(wf_kuband[N_obs]), np.nanmax(
        wf_kuband[N_obs])-0.05*np.nanmean(wf_kuband[N_obs])), ylim=(time_conv[N_obs][idx[-1]], time_conv[N_obs][idx[0]]), ultitle='Ku-band'.format(fn[0:2]))

    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        range_40[N_obs]*c/2)+offset_ku, color='b', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        range_50[N_obs]*c/2)+offset_ku, color='r', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        range_80[N_obs]*c/2)+offset_ku, color='dodgerblue', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(
        wf_kuband[N_obs][idx]), y=centroid, color='orange', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=time_conv[N_obs][np.nanargmax(
        wf_kuband[N_obs][:])], color='magenta', marker=4, s=30, label='')

    elevation = ds.variables['elevation'][:]
    leg0 = ax.axhline(y=elevation[N_obs] -
                      ALS_along_radar_track[N_obs], c='green')

    # ax.format(ltitle=r'$h_{}$ = {:.2f} m'.format('{R, ALS}', elevation[N_obs]-ALS_along_radar_track[N_obs]))

    handle0 = r'$h_{}$ = {:.2f} m'.format(
        'R', elevation[N_obs]-ALS_along_radar_track[N_obs])
    handle1 = r'$h_{}$ = {:.2f} m'.format('R', (range_40[N_obs]*c/2)+offset_ku)
    handle2 = r'$h_{}$ = {:.2f} m'.format('R', (range_50[N_obs]*c/2)+offset_ku)
    handle3 = r'$h_{}$ = {:.2f} m'.format('R', (range_80[N_obs]*c/2)+offset_ku)
    handle4 = r'$h_{}$ = {:.2f} m'.format('R', centroid)
    handle5 = r'$h_{}$ = {:.2f} m'.format(
        'R', time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])])
    handle6 = r'$h_{}$ = {:.2f} m'.format(
        '{s,MaxC}', (centroid-time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])])/params["n_snow"])
    frame = ax.legend([leg0, leg1, leg2, leg3, leg5, leg4, leg6], [handle0, handle1, handle2, handle3, handle5, handle4, handle6],
                      loc='lr', handlelength=0, markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, prop={'size': 7})

    ax = axs.panel_axes('r', width=1, space=0)

    fn = 'kaband_20221213_02_74_232_002'
    ds = netCDF4.Dataset(
        r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
    wf_kuband = ds.variables['waveform'][:]
    time = ds.variables['two_way_travel_time'][:]
    range_40 = ds.variables['range_tfmra40'][:]
    range_50 = ds.variables['range_tfmra50'][:]
    range_80 = ds.variables['range_tfmra80'][:]
    time_conv = (time*c/2)+offset_ka+offset_Ku_Ka
    # wf_kuband = 10 * np.log10(wf_kuband)

    idx = np.arange(np.nanargmax(
        wf_kuband[N_obs][:])-100, np.nanargmax(wf_kuband[N_obs][:])+156)
    # noise=np.nanmean(wf_kuband[N_obs][0:500])
    centroid = np.nansum(
        wf_kuband[N_obs][idx]*time_conv[N_obs][idx])/np.nansum(wf_kuband[N_obs][idx])
    # peaks=scipy.find_peaks(wf_kuband[0], height=noise+noise*10, width=2)
    peaks = scipy.find_peaks(wf_kuband[N_obs][idx], prominence=np.nanmax(
        wf_kuband[N_obs][idx])*0.2, distance=25)

    leg1 = ax.axhline(y=(range_40[N_obs]*c/2)+offset_ka+offset_Ku_Ka,
                      color='b', linestyle='-', linewidth=1, label='TFMRA40')
    leg2 = ax.axhline(y=(range_50[N_obs]*c/2)+offset_ka+offset_Ku_Ka,
                      color='red', linestyle='--', linewidth=1, label='TFMRA50')
    leg3 = ax.axhline(y=(range_80[N_obs]*c/2)+offset_ka+offset_Ku_Ka,
                      color='dodgerblue', linestyle='--', linewidth=1, label='TFMRA80')

    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        range_40[N_obs]*c/2)+offset_ka+offset_Ku_Ka, color='b', marker=4, s=30, label='')

    leg4 = ax.axhline(y=centroid+offset_Ku_Ka, color='orange',
                      linestyle='-', linewidth=1)
    # ax.scatter(wf_kuband[N_obs][peaks[0][:]+idx[0]],time_conv[N_obs][peaks[0][:]+idx[0]], marker='*', s=10, zorder=2)
    leg5 = ax.axhline(y=time_conv[N_obs][np.nanargmax(
        wf_kuband[N_obs][:])]+offset_Ku_Ka, color='brown', linestyle='--', linewidth=1)
    # ax.scatter(wf_kuband[N_obs][np.nanargmax(wf_kuband[N_obs][:])], time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])],marker='*', s=10, zorder=2)
    # ax.format(ultitle='$\delta$range={:.2f} m'.format((centroid-time_conv[N_obs][peaks[0][0]+idx[0]])))
    ax.plot(wf_kuband[N_obs-5:N_obs+5][:, idx], time_conv[N_obs -
            5:N_obs+5][:, idx]+offset_Ku_Ka, zorder=0, c='lightgrey')
    leg6 = ax.plot(wf_kuband[N_obs][idx], time_conv[N_obs]
                   [idx]+offset_Ku_Ka, zorder=0, c='k', linewidth=0.5)

    ax.format(xlabel='power (W)', xlim=(np.nanmin(wf_kuband[N_obs])+0.05*np.nanmean(wf_kuband[N_obs]), np.nanmax(wf_kuband[N_obs])-0.05*np.nanmean(
        wf_kuband[N_obs])), ylim=(time_conv[N_obs][idx[-1]], time_conv[N_obs][idx[0]]), ultitle='Ka-band'.format(fn[0:2]))
    ax.xaxis.labelpad = 15

    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        range_40[N_obs]*c/2)+offset_ka+offset_Ku_Ka, color='b', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        range_50[N_obs]*c/2)+offset_ka+offset_Ku_Ka, color='r', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        range_80[N_obs]*c/2)+offset_ka+offset_Ku_Ka, color='dodgerblue', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=centroid +
               offset_Ku_Ka, color='orange', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=time_conv[N_obs][np.nanargmax(
        wf_kuband[N_obs][:])]+offset_Ku_Ka, color='brown', marker=4, s=30, label='')

    elevation = ds.variables['elevation'][:]
    leg0 = ax.axhline(
        y=elevation[N_obs]-ALS_along_radar_track[N_obs]+offset_Ku_Ka, c='green')

    handle0 = r'$h_{}$ = {:.2f} m'.format(
        'R', elevation[N_obs]-ALS_along_radar_track[N_obs]+offset_Ku_Ka)
    handle1 = r'$h_{}$ = {:.2f} m'.format(
        'R', (range_40[N_obs]*c/2)+offset_ka+offset_Ku_Ka)
    handle2 = r'$h_{}$ = {:.2f} m'.format(
        'R', (range_50[N_obs]*c/2)+offset_ka+offset_Ku_Ka)
    handle3 = r'$h_{}$ = {:.2f} m'.format(
        'R', (range_80[N_obs]*c/2)+offset_ka+offset_Ku_Ka)
    handle4 = r'$h_{}$ = {:.2f} m'.format('R', centroid+offset_Ku_Ka)
    handle5 = r'$h_{}$ = {:.2f} m'.format(
        'R', time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])]+offset_Ku_Ka)
    handle6 = r'$h_{}$ = {:.2f} m'.format(
        '{s,MaxC}', (centroid-time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])])/params["n_snow"])
    frame = ax.legend([leg0, leg1, leg2, leg3, leg5, leg4, leg6], [handle0, handle1, handle2, handle3, handle5, handle4, handle6],
                      loc='lr', handlelength=0, markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, prop={'size': 7})

    ax = axs.panel_axes('r', width=1, space=0)
    ax.patch.set_facecolor('lightgrey')
    fn = 'snow_20221213_02_74_232_002'
    ds = netCDF4.Dataset(
        r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
    wf_kuband = ds.variables['waveform'][:]
    time = ds.variables['two_way_travel_time'][:]
    range_40 = ds.variables['range_tfmra40'][:]
    range_50 = ds.variables['range_tfmra50'][:]
    range_80 = ds.variables['range_tfmra80'][:]

    as_cwt = ds.variables['range_air_snow_cwt_TN'][:]
    si_cwt = ds.variables['range_snow_ice_cwt_TN'][:]
    as_peak = ds.variables['range_air_snow_peakiness'][:]
    si_peak = ds.variables['range_snow_ice_peakiness'][:]
    time_conv = (time*c/2)+offset_snow+offset_Ku_snow
    # wf_kuband = 10 * np.log10(wf_kuband)
    idx = np.arange(np.nanargmax(
        wf_kuband[N_obs][:])-100, np.nanargmax(wf_kuband[N_obs][:])+156)

    # ax.axhline(y=(range_40[N_obs]*c/2)+offset_snow, color='green', linestyle='-', linewidth=1, label='TFMRA40')
    # ax.axhline(y=(range_50[N_obs]*c/2)+offset_snow, color='yellow', linestyle='--', linewidth=1, label='TFMRA50')
    # ax.axhline(y=(range_80[N_obs]*c/2)+offset_snow, color='magenta', linestyle='--', linewidth=1, label='TFMRA80')

    leg1 = ax.axhline(y=(as_cwt[N_obs]*c/2)+offset_snow+offset_Ku_snow,
                      color='firebrick', linestyle='-', linewidth=1, label='TFMRA50')
    leg2 = ax.axhline(y=(si_cwt[N_obs]*c/2)+offset_snow+offset_Ku_snow,
                      color='red', linestyle='--', linewidth=1, label='TFMRA80')

    leg3 = ax.axhline(y=(as_peak[N_obs]*c/2)+offset_snow+offset_Ku_snow,
                      color='blue', linestyle='-', linewidth=1, label='TFMRA50')
    leg4 = ax.axhline(y=(si_peak[N_obs]*c/2)+offset_snow+offset_Ku_snow,
                      color='dodgerblue', linestyle='--', linewidth=1, label='TFMRA80')

    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        as_cwt[N_obs]*c/2)+offset_snow+offset_Ku_snow, color='firebrick', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        si_cwt[N_obs]*c/2)+offset_snow+offset_Ku_snow, color='r', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        as_peak[N_obs]*c/2)+offset_snow+offset_Ku_snow, color='blue', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        si_peak[N_obs]*c/2)+offset_snow+offset_Ku_snow, color='dodgerblue', marker=4, s=30, label='')

    elevation = ds.variables['elevation'][:]
    leg0 = ax.axhline(
        y=elevation[N_obs]-ALS_along_radar_track[N_obs]+offset_Ku_snow, c='green')

    ax.plot(wf_kuband[N_obs-5:N_obs+5][:, idx], time_conv[N_obs -
            5:N_obs+5][:, idx]+offset_Ku_snow, zorder=0, c='grey')
    leg5 = ax.plot(wf_kuband[N_obs][idx], time_conv[N_obs]
                   [idx]+offset_Ku_snow, zorder=0, c='k', linewidth=0.5)
    # ax.scatter(wf_kuband[N_obs][np.nanargmax(wf_kuband[N_obs][:])], time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])],marker='*', s=10, zorder=2)
    # ax.format(ultitle='$\delta$range={:.2f} m'.format((centroid-time_conv[N_obs][peaks[0][0]+idx[0]])))
    ax.format(xlim=(np.nanmin(wf_kuband[N_obs])+0.05*np.nanmean(wf_kuband[N_obs]), np.nanmax(
        wf_kuband[N_obs])-0.05*np.nanmean(wf_kuband[N_obs])), ultitle='C/S-band'.format(fn[0:2]))
    # %,title='$\delta$range={:.2f} m\nsd = {:.2f} m'.format((centroid-time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])]), ((centroid-time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])])/params["n_snow"]))

    handle0 = r'$h_{}$ = {:.2f} m'.format(
        'R', elevation[N_obs]-ALS_along_radar_track[N_obs]+offset_Ku_snow)
    handle1 = r'$h_{}$ = {:.2f} m'.format(
        'R', (as_cwt[N_obs]*c/2)+offset_snow+offset_Ku_snow)
    handle2 = r'$h_{}$ = {:.2f} m'.format(
        'R', (si_cwt[N_obs]*c/2)+offset_snow+offset_Ku_snow)
    handle3 = r'$h_{}$ = {:.2f} m'.format(
        'R', (as_peak[N_obs]*c/2)+offset_snow+offset_Ku_snow)
    handle4 = r'$h_{}$ = {:.2f} m'.format(
        'R', (si_peak[N_obs]*c/2)+offset_snow+offset_Ku_snow)
    handle5 = r'$h_{}$ = {:.2f} m'.format(
        '{s,CWT}', ((si_cwt[N_obs]*c/2)-(as_cwt[N_obs]*c/2))/params["n_snow"])
    handle6 = r'$h_{}$ = {:.2f} m'.format(
        '{s,peak}', ((si_peak[N_obs]*c/2)-(as_peak[N_obs]*c/2))/params["n_snow"])
    frame = ax.legend([leg0, leg1, leg2, leg3, leg4, leg5, leg5], [handle0, handle1, handle2, handle3, handle4, handle5, handle6],
                      loc='lr', handlelength=0, markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, prop={'size': 7})

    return idx


def plot_waveforms_examples_right(n_extra, axs, title_spec, ALS_along_radar_track):
    N_obs = val_x*n1+n_extra
    ax = axs

    ax.patch.set_facecolor('lightgrey')
    fn = 'snow_20221213_02_74_232_002'
    ds = netCDF4.Dataset(
        r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
    wf_kuband = ds.variables['waveform'][:]
    time = ds.variables['two_way_travel_time'][:]
    range_40 = ds.variables['range_tfmra40'][:]
    range_50 = ds.variables['range_tfmra50'][:]
    range_80 = ds.variables['range_tfmra80'][:]

    as_cwt = ds.variables['range_air_snow_cwt_TN'][:]
    si_cwt = ds.variables['range_snow_ice_cwt_TN'][:]
    as_peak = ds.variables['range_air_snow_peakiness'][:]
    si_peak = ds.variables['range_snow_ice_peakiness'][:]
    time_conv = (time*c/2)+offset_snow+offset_Ku_snow
    # wf_kuband = 10 * np.log10(wf_kuband)
    idx = np.arange(np.nanargmax(
        wf_kuband[N_obs][:])-100, np.nanargmax(wf_kuband[N_obs][:])+156)

    # ax.axhline(y=(range_40[N_obs]*c/2)+offset_snow, color='green', linestyle='-', linewidth=1, label='TFMRA40')
    # ax.axhline(y=(range_50[N_obs]*c/2)+offset_snow, color='yellow', linestyle='--', linewidth=1, label='TFMRA50')
    # ax.axhline(y=(range_80[N_obs]*c/2)+offset_snow, color='magenta', linestyle='--', linewidth=1, label='TFMRA80')

    leg1 = ax.axhline(y=(as_cwt[N_obs]*c/2)+offset_snow+offset_Ku_snow,
                      color='firebrick', linestyle='-', linewidth=1, label='TFMRA50')
    leg2 = ax.axhline(y=(si_cwt[N_obs]*c/2)+offset_snow+offset_Ku_snow,
                      color='red', linestyle='--', linewidth=1, label='TFMRA80')

    leg3 = ax.axhline(y=(as_peak[N_obs]*c/2)+offset_snow+offset_Ku_snow,
                      color='blue', linestyle='-', linewidth=1, label='TFMRA50')
    leg4 = ax.axhline(y=(si_peak[N_obs]*c/2)+offset_snow+offset_Ku_snow,
                      color='dodgerblue', linestyle='--', linewidth=1, label='TFMRA80')

    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        as_cwt[N_obs]*c/2)+offset_snow+offset_Ku_snow, color='firebrick', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        si_cwt[N_obs]*c/2)+offset_snow+offset_Ku_snow, color='r', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        as_peak[N_obs]*c/2)+offset_snow+offset_Ku_snow, color='blue', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        si_peak[N_obs]*c/2)+offset_snow+offset_Ku_snow, color='dodgerblue', marker=4, s=30, label='')

    elevation = ds.variables['elevation'][:]
    leg0 = ax.axhline(
        y=elevation[N_obs]-ALS_along_radar_track[N_obs]+offset_Ku_snow, c='green')

    ax.plot(wf_kuband[N_obs-5:N_obs+5][:, idx], time_conv[N_obs -
            5:N_obs+5][:, idx]+offset_Ku_snow, zorder=0, c='grey')
    leg5 = ax.plot(wf_kuband[N_obs][idx], time_conv[N_obs]
                   [idx]+offset_Ku_snow, zorder=0, c='k', linewidth=0.5)
    # ax.scatter(wf_kuband[N_obs][np.nanargmax(wf_kuband[N_obs][:])], time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])],marker='*', s=10, zorder=2)
    # ax.format(ultitle='$\delta$range={:.2f} m'.format((centroid-time_conv[N_obs][peaks[0][0]+idx[0]])))
    ax.format(xlim=(np.nanmin(wf_kuband[N_obs])+0.05*np.nanmean(wf_kuband[N_obs]), np.nanmax(
        wf_kuband[N_obs])-0.05*np.nanmean(wf_kuband[N_obs])), ultitle='C/S-band'.format(fn[0:2]))
    # %,title='$\delta$range={:.2f} m\nsd = {:.2f} m'.format((centroid-time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])]), ((centroid-time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])])/params["n_snow"]))

    handle0 = r'$h_{}$ = {:.2f} m'.format(
        'R', elevation[N_obs]-ALS_along_radar_track[N_obs]+offset_Ku_snow)
    handle1 = r'$h_{}$ = {:.2f} m'.format(
        'R', (as_cwt[N_obs]*c/2)+offset_snow+offset_Ku_snow)
    handle2 = r'$h_{}$ = {:.2f} m'.format(
        'R', (si_cwt[N_obs]*c/2)+offset_snow+offset_Ku_snow)
    handle3 = r'$h_{}$ = {:.2f} m'.format(
        'R', (as_peak[N_obs]*c/2)+offset_snow+offset_Ku_snow)
    handle4 = r'$h_{}$ = {:.2f} m'.format(
        'R', (si_peak[N_obs]*c/2)+offset_snow+offset_Ku_snow)
    handle5 = r'$h_{}$ = {:.2f} m'.format(
        '{s,CWT}', ((si_cwt[N_obs]*c/2)-(as_cwt[N_obs]*c/2))/params["n_snow"])
    handle6 = r'$h_{}$ = {:.2f} m'.format(
        '{s,peak}', ((si_peak[N_obs]*c/2)-(as_peak[N_obs]*c/2))/params["n_snow"])
    frame = ax.legend([leg0, leg1, leg2, leg3, leg4, leg5, leg5], [handle0, handle1, handle2, handle3, handle4, handle5, handle6],
                      loc='lr', handlelength=0, markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, prop={'size': 7})

    # ax.format(ltitle=r'$h_{}$ = {:.2f} m'.format('{R,ALS}', elevation[N_obs]-ALS_along_radar_track[N_obs]))

    ax = axs.panel_axes('l', width=1, space=0)

    fn = 'kaband_20221213_02_74_232_002'
    ds = netCDF4.Dataset(
        r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
    wf_kuband = ds.variables['waveform'][:]
    time = ds.variables['two_way_travel_time'][:]
    range_40 = ds.variables['range_tfmra40'][:]
    range_50 = ds.variables['range_tfmra50'][:]
    range_80 = ds.variables['range_tfmra80'][:]
    time_conv = (time*c/2)+offset_ka+offset_Ku_Ka
    # wf_kuband = 10 * np.log10(wf_kuband)

    idx = np.arange(np.nanargmax(
        wf_kuband[N_obs][:])-100, np.nanargmax(wf_kuband[N_obs][:])+156)
    # noise=np.nanmean(wf_kuband[N_obs][0:500])
    centroid = np.nansum(
        wf_kuband[N_obs][idx]*time_conv[N_obs][idx])/np.nansum(wf_kuband[N_obs][idx])
    # peaks=scipy.find_peaks(wf_kuband[0], height=noise+noise*10, width=2)
    peaks = scipy.find_peaks(wf_kuband[N_obs][idx], prominence=np.nanmax(
        wf_kuband[N_obs][idx])*0.2, distance=25)

    leg1 = ax.axhline(y=(range_40[N_obs]*c/2)+offset_ka+offset_Ku_Ka,
                      color='b', linestyle='-', linewidth=1, label='TFMRA40')
    leg2 = ax.axhline(y=(range_50[N_obs]*c/2)+offset_ka+offset_Ku_Ka,
                      color='red', linestyle='--', linewidth=1, label='TFMRA50')
    leg3 = ax.axhline(y=(range_80[N_obs]*c/2)+offset_ka+offset_Ku_Ka,
                      color='dodgerblue', linestyle='--', linewidth=1, label='TFMRA80')

    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        range_40[N_obs]*c/2)+offset_ka+offset_Ku_Ka, color='b', marker=4, s=30, label='')

    leg4 = ax.axhline(y=centroid+offset_Ku_Ka, color='orange',
                      linestyle='-', linewidth=1)
    # ax.scatter(wf_kuband[N_obs][peaks[0][:]+idx[0]],time_conv[N_obs][peaks[0][:]+idx[0]], marker='*', s=10, zorder=2)
    leg5 = ax.axhline(y=time_conv[N_obs][np.nanargmax(
        wf_kuband[N_obs][:])]+offset_Ku_Ka, color='magenta', linestyle='--', linewidth=1)
    # ax.scatter(wf_kuband[N_obs][np.nanargmax(wf_kuband[N_obs][:])], time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])],marker='*', s=10, zorder=2)
    # ax.format(ultitle='$\delta$range={:.2f} m'.format((centroid-time_conv[N_obs][peaks[0][0]+idx[0]])))
    ax.plot(wf_kuband[N_obs-5:N_obs+5][:, idx], time_conv[N_obs -
            5:N_obs+5][:, idx]+offset_Ku_Ka, zorder=0, c='lightgrey')
    leg6 = ax.plot(wf_kuband[N_obs][idx], time_conv[N_obs]
                   [idx]+offset_Ku_Ka, zorder=0, c='k', linewidth=0.5)

    ax.format(xlabel='power (W)', xlim=(np.nanmin(wf_kuband[N_obs])+0.05*np.nanmean(wf_kuband[N_obs]), np.nanmax(wf_kuband[N_obs])-0.05*np.nanmean(
        wf_kuband[N_obs])), ylim=(time_conv[N_obs][idx[-1]], time_conv[N_obs][idx[0]]), ultitle='Ka-band'.format(fn[0:2]))
    ax.xaxis.labelpad = 15

    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        range_40[N_obs]*c/2)+offset_ka+offset_Ku_Ka, color='b', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        range_50[N_obs]*c/2)+offset_ka+offset_Ku_Ka, color='r', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        range_80[N_obs]*c/2)+offset_ka+offset_Ku_Ka, color='dodgerblue', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=centroid +
               offset_Ku_Ka, color='orange', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=time_conv[N_obs][np.nanargmax(
        wf_kuband[N_obs][:])]+offset_Ku_Ka, color='magenta', marker=4, s=30, label='')

    elevation = ds.variables['elevation'][:]
    leg0 = ax.axhline(
        y=elevation[N_obs]-ALS_along_radar_track[N_obs]+offset_Ku_Ka, c='green')

    handle0 = r'$h_{}$ = {:.2f} m'.format(
        'R', elevation[N_obs]-ALS_along_radar_track[N_obs]+offset_Ku_Ka)
    handle1 = r'$h_{}$ = {:.2f} m'.format(
        'R', (range_40[N_obs]*c/2)+offset_ka+offset_Ku_Ka)
    handle2 = r'$h_{}$ = {:.2f} m'.format(
        'R', (range_50[N_obs]*c/2)+offset_ka+offset_Ku_Ka)
    handle3 = r'$h_{}$ = {:.2f} m'.format(
        'R', (range_80[N_obs]*c/2)+offset_ka+offset_Ku_Ka)
    handle4 = r'$h_{}$ = {:.2f} m'.format('R', centroid+offset_Ku_Ka)
    handle5 = r'$h_{}$ = {:.2f} m'.format(
        'R', time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])]+offset_Ku_Ka)
    handle6 = r'$h_{}$ = {:.2f} m'.format(
        '{s,MaxC}', (centroid-time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])])/params["n_snow"])
    frame = ax.legend([leg0, leg1, leg2, leg3, leg5, leg4, leg6], [handle0, handle1, handle2, handle3, handle5, handle4, handle6],
                      loc='lr', handlelength=0, markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, prop={'size': 7})

    ax = axs.panel_axes('l', width=1, space=0)
    fn = 'kuband_20221213_02_74_232_002'
    ds = netCDF4.Dataset(
        r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
    wf_kuband = ds.variables['waveform'][:]
    range_40 = ds.variables['range_tfmra40'][:]
    range_50 = ds.variables['range_tfmra50'][:]
    range_80 = ds.variables['range_tfmra80'][:]

    # range_40 = ds.variables['retracking_gate_tfmra40'][:]
    # range_50 = ds.variables['retracking_gate_tfmra50'][:]
    # range_80 = ds.variables['retracking_gate_tfmra80'][:]

    time = ds.variables['two_way_travel_time'][:]
    time_conv = (time*c/2)+offset_ku
    # wf_kuband = 10 * np.log10(wf_kuband)

    idx = np.arange(np.nanargmax(
        wf_kuband[N_obs][:])-100, np.nanargmax(wf_kuband[N_obs][:])+156)
    # noise=np.nanmean(wf_kuband[N_obs][0:500])
    centroid = np.nansum(
        wf_kuband[N_obs][idx]*time_conv[N_obs][idx])/np.nansum(wf_kuband[N_obs][idx])
    # peaks=scipy.find_peaks(wf_kuband[0], height=noise+noise*10, width=2)
    peaks = scipy.find_peaks(wf_kuband[N_obs][idx], prominence=np.nanmax(
        wf_kuband[N_obs][idx])*0.2, distance=25)

    leg1 = ax.axhline(y=(range_40[N_obs]*c/2)+offset_ku,
                      color='b', linestyle='-', linewidth=1, label='TFMRA40')
    leg2 = ax.axhline(y=(range_50[N_obs]*c/2)+offset_ku,
                      color='red', linestyle='--', linewidth=1, label='TFMRA50')
    leg3 = ax.axhline(y=(range_80[N_obs]*c/2)+offset_ku,
                      color='dodgerblue', linestyle='--', linewidth=1, label='TFMRA80')

    leg4 = ax.axhline(y=centroid, color='orange', linestyle='-', linewidth=1)
    # ax.scatter(wf_kuband[N_obs][peaks[0][:]+idx[0]],time_conv[N_obs][peaks[0][:]+idx[0]], marker='*', s=10, zorder=2)
    leg5 = ax.axhline(y=time_conv[N_obs][np.nanargmax(
        wf_kuband[N_obs][:])], color='brown', linestyle='--', linewidth=1)
    ax.plot(wf_kuband[N_obs-5:N_obs+5][:, idx],
            time_conv[N_obs-5:N_obs+5][:, idx], zorder=0, c='lightgrey')
    leg6 = ax.plot(wf_kuband[N_obs][idx], time_conv[N_obs]
                   [idx], zorder=0, c='k', linewidth=0.5)
    # ax.scatter(wf_kuband[N_obs][np.nanargmax(wf_kuband[N_obs][:])], time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])],marker='*', s=10, zorder=2)
    # ax.format(ultitle='$\delta$range={:.2f} m'.format((centroid-time_conv[N_obs][peaks[0][0]+idx[0]])))
    ax.format(abc="(a)", abcloc='l', ylabel='range (m)', xlim=(np.nanmin(wf_kuband[N_obs])+0.05*np.nanmean(wf_kuband[N_obs]), np.nanmax(
        wf_kuband[N_obs])-0.05*np.nanmean(wf_kuband[N_obs])), ylim=(time_conv[N_obs][idx[-1]], time_conv[N_obs][idx[0]]), ultitle='Ku-band'.format(fn[0:2]))

    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        range_40[N_obs]*c/2)+offset_ku, color='b', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        range_50[N_obs]*c/2)+offset_ku, color='r', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        range_80[N_obs]*c/2)+offset_ku, color='dodgerblue', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(
        wf_kuband[N_obs][idx]), y=centroid, color='orange', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=time_conv[N_obs][np.nanargmax(
        wf_kuband[N_obs][:])], color='brown', marker=4, s=30, label='')

    elevation = ds.variables['elevation'][:]
    leg0 = ax.axhline(y=elevation[N_obs] -
                      ALS_along_radar_track[N_obs], c='green')

    handle0 = r'$h_{}$ = {:.2f} m'.format(
        'R', elevation[N_obs]-ALS_along_radar_track[N_obs])
    handle1 = r'$h_{}$ = {:.2f} m'.format('R', (range_40[N_obs]*c/2)+offset_ku)
    handle2 = r'$h_{}$ = {:.2f} m'.format('R', (range_50[N_obs]*c/2)+offset_ku)
    handle3 = r'$h_{}$ = {:.2f} m'.format('R', (range_80[N_obs]*c/2)+offset_ku)
    handle4 = r'$h_{}$ = {:.2f} m'.format('R', centroid)
    handle5 = r'$h_{}$ = {:.2f} m'.format(
        'R', time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])])
    handle6 = r'$h_{}$ = {:.2f} m'.format(
        '{s,MaxC}', (centroid-time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])])/params["n_snow"])
    frame = ax.legend([leg0, leg1, leg2, leg3, leg5, leg4, leg6], [handle0, handle1, handle2, handle3, handle5, handle4, handle6],
                      loc='lr', handlelength=0, markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, prop={'size': 7})

    return idx

    return idx


#%%% Compute radar specifics , Ku

import math

val = 'snow' ### 'ku', 'ka', 'snow' 

if val == 'ku':
    freq_c = 15e9
    beta = 19
elif val == 'ka':
    freq_c = 35e9
    beta = 19
elif val == 'snow':
    freq_c = 5e9
    beta = 30

H = 300 # m 
lambda_c = c/(freq_c) # GHz
#lambda_c = c/3500e6
L = np.sqrt((H*lambda_c)/2) # MAXIMUM

print('Compute statistics for frequency = {:E} Hz, altitude = {} m'.format(freq_c, H))
print('L (maximum) = {:.2f} m'.format(L))

PRF = 3.125e3 # pulse-repitition-freq
v = 330/3.6 # convert from k/m to m/s
n_sum = 16 # sum of averages
L = (n_sum*v)/PRF
print('L (unfocused synthetic aperture length) = {:.2f} m'.format(L))
sigma_along = H * np.tan ( np.arcsin ( lambda_c / ( 2 * L)))
print('Along-track SAR footprint = {:.2f} m, with 5-boxcar smoothing = {:.2f} m'.format(sigma_along, sigma_along/5))

T = 0
B = 6e9
k_t = 1.5

sigma_fresnel = np.sqrt(2*(H+(T/np.sqrt(2.35)))*lambda_c)
print('Cross-track Fresnel = {:.2f} m'.format(sigma_fresnel))

sigma_y_ku = 2 * np.sqrt ( ((H + ( T / np.sqrt(3.15) ) ) * c * k_t)/B)
print('Cross-track with windowing = {:.2f} m'.format(sigma_y_ku))

sigma_y_pulse = 2 * np.sqrt((k_t * c* H)/B)
print('Cross-track with pulse_limited = {:.2f} m'.format(sigma_y_pulse))

sigma_cross = 2 * H * np.tan(np.radians(beta)/2)
print('Cross-track with footprint as function of range = {:.2f} m'.format(sigma_cross))


beta_ku = lambda_ku / (2*L)
beta_ku
r_at = 2 * H * np.tan ( beta_ku / 2)
r_at


sigma = 2 * H * np.tan ( beta_ku / 2)
sigma

lambda_ka = c/(np.median([32, 38])*1e+9)
L_ka = np.sqrt((H*lambda_ka)/2)
L_ka

H = 500 # m 

lambda_snow =c/(np.median(3.5)*1e+9)
L_snow = np.sqrt((H*lambda_snow)/2)
L_snow

sigma_snow = H * np.tan ( np.arcsin ( lambda_snow / ( 2 * L_snow)))
sigma_snow

# %% Leverams correct from inches to m

conv_inch_m = 0.0254

leverarms_Ka_TX = np.array([-112.692, 27.280, 108.929])*conv_inch_m
leverarms_Ka_RX = np.array([-116.137, 15.980, 108.929])*conv_inch_m
leverarms_Ka = np.mean([leverarms_Ka_TX, leverarms_Ka_RX], axis=0)

leverarms_Ku_TX = np.array([-116.015, 29.380, 106.836])*conv_inch_m
leverarms_Ku_RX = np.array([-113.252, 13.880, 106.836])*conv_inch_m
leverarms_Ku = np.mean([leverarms_Ku_TX, leverarms_Ku_RX], axis=0)

leverarms_SC_TX = np.array([-102.964, 14.843, 106.002])*conv_inch_m
leverarms_SC_RX = np.array([-125.981, 28.621, 106.002])*conv_inch_m
leverarms_SC = np.mean([leverarms_SC_TX, leverarms_SC_RX], axis=0)
# print(np.around(leverarms_SC_RX,2))

offset_Ku_snow = leverarms_SC_TX[2]-leverarms_Ku_TX[2]
offset_Ku_Ka = leverarms_Ka_TX[2]-leverarms_Ku_TX[2]


#%% Radar load data

fn = 'snow_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
elevation_air_snow_ellipsoid_cwt_TN = ds.variables['elevation_air_snow_ellipsoid_cwt_TN'][:]
elevation_snow_ice_ellipsoid_cwt_TN = ds.variables['elevation_snow_ice_ellipsoid_cwt_TN'][:]
elevation_air_snow_ellipsoid_peakiness = ds.variables['elevation_air_snow_ellipsoid_peakiness'][:]
elevation_snow_ice_ellipsoid_peakiness = ds.variables['elevation_snow_iceellipsoid_peakiness'][:]
latitude = ds.variables['lat'][:]
longitude = ds.variables['lon'][:]
flight_alt_snow = ds.variables['elevation']

fn = 'kuband_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
elevation_ellipsoid_tfmra50 = ds.variables['elevation_ellipsoid_tfmra50'][:]
latitude_comb = ds.variables['lat'][:]
longitude_comb = ds.variables['lon'][:]
gpstime = ds.variables['gps_time'][:]
flight_alt_ku = ds.variables['elevation']
elevation_air_snow_ellipsoid_cwt_TN_ku = ds.variables['elevation_air_snow_ellipsoid_cwt_TN'][:]
elevation_snow_ice_ellipsoid_cwt_TN_ku = ds.variables['elevation_snow_ice_ellipsoid_cwt_TN'][:]
elevation_air_snow_ellipsoid_peakiness_ku = ds.variables['elevation_air_snow_ellipsoid_peakiness'][:]
elevation_snow_ice_ellipsoid_peakiness_ku = ds.variables['elevation_snow_iceellipsoid_peakiness'][:]

fn = 'kaband_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
elevation_ellipsoid_tfmra50_ka = ds.variables['elevation_ellipsoid_tfmra50'][:]
latitude_comb_ka = ds.variables['lat'][:]
longitude_comb_ka = ds.variables['lon'][:]
gpstime_ka = ds.variables['gps_time'][:]
flight_alt_ka = ds.variables['elevation']
elevation_air_snow_ellipsoid_cwt_TN_ka = ds.variables['elevation_air_snow_ellipsoid_cwt_TN'][:]
elevation_snow_ice_ellipsoid_cwt_TN_ka = ds.variables['elevation_snow_ice_ellipsoid_cwt_TN'][:]
elevation_air_snow_ellipsoid_peakiness_ka = ds.variables['elevation_air_snow_ellipsoid_peakiness'][:]
elevation_snow_ice_ellipsoid_peakiness_ka = ds.variables['elevation_snow_iceellipsoid_peakiness'][:]
# %% Load swath ALS data

fn = r'C:\Users\rmfha\OneDrive - Danmarks Tekniske Universitet\DEFIANT2022\Data\SCANNER\347\Raw\347_183301_1x1.scn'
data_ALS_scn1 = pd.read_csv(fn, header=None, delim_whitespace=True)
fn = r'C:\Users\rmfha\OneDrive - Danmarks Tekniske Universitet\DEFIANT2022\Data\SCANNER\347\Raw\347_194156_1x1.scn'
data_ALS_scn2 = pd.read_csv(fn, header=None, delim_whitespace=True)
fn = r'C:\Users\rmfha\OneDrive - Danmarks Tekniske Universitet\DEFIANT2022\Data\SCANNER\347\Raw\347_203917_1x1.scn'
data_ALS_scn3 = pd.read_csv(fn, header=None, delim_whitespace=True)

frames = [data_ALS_scn1, data_ALS_scn2, data_ALS_scn3]

data_ALS_scn = pd.concat(frames).reset_index()

#%% Compute roughness along scan! --- OPEN IF NEED TO COMPUTE (took +48 hours) ---> DO NOT RUN UNLESS ABSOLUTELY NECESSARY!
import ALS_func_CRYO2ICE_ant as ALS_func


df_ALS_scan_v1 = df_ALS_scan = pd.DataFrame(
    {'latitude': data_ALS_scn[1], 'longitude': data_ALS_scn[2], 'elevation': data_ALS_scn[3], 'scan_number':data_ALS_scn[5]})
df_ALS_scan_v1 = df_ALS_scan_v1.sort_index()
df_ALS_scan_v1_roughness, df_ALS_scan_v1_anomaly = ALS_func.compute_roughness(df_ALS_scan_v1)

df_ALS_scan_v1_roughness.to_csv(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\{}_roughness.csv'.format('df_ALS_scan_v1'))
df_ALS_scan_v1_anomaly.to_csv(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\{}_anomaly.csv'.format('df_ALS_scan_v1'))
df_ALS_scan_v1.to_csv(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\{}.csv'.format('df_ALS_scan_v1'))

#%%%
#%%% READ ALS ROUGHNESS

#%% READ ROUGHNESS DATA 
#%% READ ALS ROGUHNESS
df_ALS_scan_v1_roughness = pd.read_csv(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\{}_roughness.csv'.format('df_ALS_scan_v1'))
df_ALS_scan_v1_anomaly = pd.read_csv(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\{}_anomaly.csv'.format('df_ALS_scan_v1'))
df_ALS_scan_v1 = pd.read_csv(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\{}.csv'.format('df_ALS_scan_v1'))


#%%% Extract nadir-profiles along ALS track

# elevation profile
df_ALS_scan = pd.DataFrame(
    {'lat': data_ALS_scn[1], 'lon': data_ALS_scn[2], 'elev': data_ALS_scn[3]}).reset_index()
df_Ku = pd.DataFrame({'lat': latitude_comb, 'lon': longitude_comb})
ALS_along_radar_track, lat_ALS_along_radar_track, lon_ALS_along_radar_track = identify_along_orbit(
    df_Ku, df_ALS_scan, 'elev', search_type='RADIUS', dist_req=2.5)
ALS_along_radar_track[np.abs(
    ALS_along_radar_track-elevation_ellipsoid_tfmra50) > 3] = np.nan

# scan line number
df_ALS_scan = pd.DataFrame(
    {'lat': data_ALS_scn[1], 'lon': data_ALS_scn[2], 'elev': data_ALS_scn[3], 'scan_nr':data_ALS_scn[5]}).reset_index()
df_Ku = pd.DataFrame({'lat': latitude_comb, 'lon': longitude_comb})
ALS_scan_number_along_radar_track, lat_ALS_along_radar_track, lon_ALS_along_radar_track = identify_along_orbit(
    df_Ku, df_ALS_scan, 'scan_nr', search_type='RADIUS', dist_req=2.5)

# sigma_5 m 
df_ALS_scan = pd.DataFrame(
    {'lat': data_ALS_scn[1], 'lon': data_ALS_scn[2], 'elev': data_ALS_scn[3], 'scan_nr':data_ALS_scn[5]}).reset_index()
df_Ku = pd.DataFrame({'lat': latitude_comb, 'lon': longitude_comb})
ALS_roughness_along_radar_track, lat_ALS_along_radar_track, lon_ALS_along_radar_track = identify_roughness_along_orbit(
    df_Ku, df_ALS_scan, 'elev', search_type='RADIUS', dist_req=2.5)


# %% Compute centroids per waveform
fn = 'kuband_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
wf_kuband = ds.variables['waveform'][:]
time = ds.variables['two_way_travel_time'][:]
time_conv = time*c/2
a_s_peaks, a_s_peaks_t, s_i_centroids, s_i_centroids_t, a_s_peaks_dd, s_i_dd, a_s_peaks_dd_t, s_i_dd_t = peak_centroid(
    wf_kuband, time_conv)

fn = 'kaband_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
wf_kuband = ds.variables['waveform'][:]
time = ds.variables['two_way_travel_time'][:]
time_conv = time*c/2

a_s_peaks_ka, a_s_peaks_t_ka, s_i_centroids_ka, s_i_centroids_t_ka, a_s_peaks_dd_ka, s_i_dd_ka, a_s_peaks_dd_t_ka, s_i_dd_t_ka = peak_centroid(
    wf_kuband, time_conv)

fn = 'snow_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
wf_kuband = ds.variables['waveform'][:]
time = ds.variables['two_way_travel_time'][:]
time_conv = time*c/2

a_s_peaks_snow, a_s_peaks_t_snow, s_i_centroids_snow, s_i_centroids_t_snow, a_s_peaks_dd_snow, s_i_dd_snow, a_s_peaks_dd_t_snow, s_i_dd_t_snow = peak_centroid(
    wf_kuband, time_conv)


# %% Compute calibration offset in manually detected leads 
num_rec = 8
min_val, max_val = np.zeros(num_rec), np.zeros(num_rec)
offset_ku_list, offset_ka_list, offset_snow_list = np.zeros(
    num_rec), np.zeros(num_rec), np.zeros(num_rec)
min_val[0], max_val[0] = -65.1901, -65.1908
# min_val[1], max_val[1] = -68.4021, -68.4034
min_val[2], max_val[2] = -68.4051, -68.4059
min_val[3], max_val[3] = -68.4087, -68.4094
min_val[4], max_val[4] = -69.3859, -69.3875
min_val[5], max_val[5] = -69.3892, -69.3907
# min_val[6], max_val[6] = -70.7162, -70.717
min_val[6], max_val[6] = -70.7583, -70.7589
min_val[7], max_val[7] = -69.9682, -69.9696

lead_lat_loc, lead_lon_loc = np.zeros(num_rec), np.zeros(num_rec)
N_list = np.zeros(num_rec)
offset_ku_list_max, offset_ka_list_max, offset_snow_list_max = np.zeros(
    num_rec), np.zeros(num_rec), np.zeros(num_rec)

for i in np.arange(0, num_rec):
    idx_lat_Ku = np.where(
        (latitude_comb < min_val[i]) & (latitude_comb > max_val[i]))
    idx_lat_Ka = np.where((latitude_comb_ka < min_val[i]) & (
        latitude_comb_ka > max_val[i]))
    idx_lat_ALS_nadir = np.where((lat_ALS_along_radar_track < min_val[i]) & (
        lat_ALS_along_radar_track > max_val[i]))

    lead_lat_loc[i] = np.nanmean(lat_ALS_along_radar_track[idx_lat_ALS_nadir])
    lead_lon_loc[i] = np.nanmean(lon_ALS_along_radar_track[idx_lat_ALS_nadir])
    N_list[i] = len(idx_lat_Ku[0])
    offset_ku_list[i] = np.nanmean(np.nanmean(
        elevation_ellipsoid_tfmra50[idx_lat_Ku[0]])-np.nanmean(ALS_along_radar_track[idx_lat_ALS_nadir[0]]))
    offset_ka_list[i] = np.nanmean(np.nanmean(
        elevation_ellipsoid_tfmra50_ka[idx_lat_Ka[0]]) - np.nanmean(ALS_along_radar_track[idx_lat_ALS_nadir[0]]))
    offset_snow_list[i] = np.nanmean(np.nanmean(
        elevation_air_snow_ellipsoid_peakiness[idx_lat_Ka[0]]) - np.nanmean(ALS_along_radar_track[idx_lat_ALS_nadir[0]]))
    
    offset_ku_list_max[i] = np.nanmean(np.nanmean(
        (flight_alt_ku-a_s_peaks_t)[idx_lat_Ku[0]])-np.nanmean(ALS_along_radar_track[idx_lat_ALS_nadir[0]]))
    offset_ka_list_max[i] = np.nanmean(np.nanmean(
        (flight_alt_ka-a_s_peaks_t_ka)[idx_lat_Ka[0]]) - np.nanmean(ALS_along_radar_track[idx_lat_ALS_nadir[0]]))
    offset_snow_list_max[i] = np.nanmean(np.nanmean(
        (flight_alt_snow-a_s_peaks_t_snow)[idx_lat_Ka[0]]) - np.nanmean(ALS_along_radar_track[idx_lat_ALS_nadir[0]]))


offset_ku = np.nanmean(offset_ku_list)
offset_ka = np.nanmean(offset_ka_list)
offset_snow = np.nanmean(offset_snow_list)

offset_ku_max = np.nanmean(offset_ku_list_max)
offset_ka_max = np.nanmean(offset_ka_list_max)
offset_snow_max = np.nanmean(offset_snow_list_max)

# %% Extract time along track from ALS scan data

df_ALS_scan_time = pd.DataFrame(
    {'lat': data_ALS_scn[1], 'lon': data_ALS_scn[2], 'time': data_ALS_scn[0]}).reset_index()
df_Ku = pd.DataFrame({'lat': latitude_comb, 'lon': longitude_comb})
ALS_time_along_radar_track, lat_ALS_along_radar_track, lon_ALS_along_radar_track = identify_along_orbit(
    df_Ku, df_ALS_scan_time, 'time', search_type='RADIUS', dist_req=5)


n = 300
n1 = 8
val_x = 1000

ALS_time_along_radar_track_filt = ALS_time_along_radar_track[val_x*n1:val_x*(
    n1+1)]
print('Start time: {}'.format(ALS_time_along_radar_track_filt[0]))
print('Stop time: {}'.format(ALS_time_along_radar_track_filt[-1]))

#%%

along_track_dist = np.zeros(len(latitude_comb))
for i in np.arange(0, len(latitude_comb)):
    along_track_dist[i] = haversine(longitude_comb[0], latitude_comb[0],longitude_comb[i], latitude_comb[i])


# %% computing PPK! CHECK IF THIS IS OK? NOT RELEVANT ANYMORE


def ppk(pwr_waveform, n_bins=10):
    ppk = []
    for i in pwr_waveform:
        try:
            noise = np.nanmean(i[0:n_bins])
            i = i[i > noise]

            pp = np.nanmax(i)/np.nanmean(i)
            ppk = np.append(ppk, pp)
        except:
            ppk = np.append(ppk, np.nan)
    return ppk


fn = 'kuband_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
wf_kuband = ds.variables['waveform'][:]

fn = 'kaband_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
wf_kaband = ds.variables['waveform'][:]


wf_kuband_ppk = ppk(wf_kuband, 100)
wf_kaband_ppk = ppk(wf_kaband, 100)





# %% Snow parameters

params = {
    'snow_density': 0.3,  # check?
}  # cwt TN algorithm
params['n_snow_CWT'] = np.sqrt((1 + 0.51 * params['snow_density']) ** 3)
params['n_snow'] = (1+0.51*params['snow_density'])**(1.5)

# %%

# divnorm=colors.TwoSlopeNorm(vmin=-60., vcenter=0., vmax=40)

fn = r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data\347_air1_10Hz_data.txt'

dtypes = {'SeqNum': int, 'Latitude': float, 'Longitude': float, 'H-Ell': float, 'Pitch': float,
          'Roll': float, 'Heading': float, 'SDHeight': float, 'CorrTime': float, 'UTCTime': str, 'UTCDate': str}
data = pd.read_csv(fn, delim_whitespace=True)
data = data[1:-1]
data = data.dropna()

file = h5py.File(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data/AMSR_U2_L3_SeaIce12km_B04_20221213.he5', 'r')
file.keys()
lon_AMSR2 = np.array(file['HDFEOS/GRIDS/SpPolarGrid12km/lon'])
lat_AMSR2 = np.array(file['HDFEOS/GRIDS/SpPolarGrid12km/lat'])
snow_depth = np.array(
    file['HDFEOS/GRIDS/SpPolarGrid12km/Data Fields/SI_12km_SH_SNOWDEPTH_5DAY'])
sea_ice_concentration = np.array(
    file['HDFEOS/GRIDS/SpPolarGrid12km/Data Fields/SI_12km_SH_ICECON_DAY'])
file.close()

missing_snow_depth = np.where(snow_depth != 110, np.nan, snow_depth)
land_snow_depth = np.where(snow_depth != 120, np.nan, snow_depth)
OW_snow_depth = np.where(snow_depth != 130, np.nan, snow_depth)
MYI_snow_depth = np.where(snow_depth != 140, np.nan, snow_depth)
variability_snow_depth = np.where(snow_depth != 150, np.nan, snow_depth)
melt_snow_depth = np.where(snow_depth != 160, np.nan, snow_depth)
snow_depth_only = np.where((snow_depth > 100) & (
    snow_depth != 140), np.nan, snow_depth)
snow_depth_only = np.where(snow_depth_only == 140, -9999, snow_depth_only)


fn = r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data\20221213_CASSIS.Depth'
data_CASSIS = pd.read_csv(fn, delim_whitespace=True, header=None)
# data = data[1:-1]
data_CASSIS[data_CASSIS[4]==0]=np.nan

data_CRYO2ICE = pd.read_hdf(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data\CRYO2ICE_CryoTEMPO_CS_OFFL_SIR_TDP_SI_ANTARC_20221213T201353_20221213T202031_28_04332_C001_CASSIS_AMSR2.h5', index_col=None, header=0)
data_CRYO2ICE=data_CRYO2ICE[data_CRYO2ICE['snow_depth'].notna()]

# %% Compute ERA5 area val. for Figure 1

fn = 'ERA5_daily_2m_air_temp_Dec2022_v2'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data\ERA5'+'/'+fn+'.nc', 'r')  # open CS2 data

ERA5_lat = ds.variables['lat'][:]
ERA5_lon = ds.variables['lon'][:]
ERA5_2m_temp_dec22 = ds.variables['t2m'][:]
ERA5_time = ds.variables['time'][:]

fn = 'ERA5_daily_total_precipitation_Dec2022_v2'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data\ERA5'+'/'+fn+'.nc', 'r')  # open CS2 data
ERA5_total_prep_dec22 = ds.variables['tp'][:]

fn = 'ERA5_daily_snowfall_Dec2022_v2'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data\ERA5'+'/'+fn+'.nc', 'r')  # open CS2 data
ERA5_snowfall_dec22 = ds.variables['sf'][:]

fn = 'ERA5_daily_2m_air_temp_Nov2022_v2'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data\ERA5'+'/'+fn+'.nc', 'r')  # open CS2 data
ERA5_2m_temp_nov22 = ds.variables['t2m'][:]
ERA5_time = ds.variables['time'][:]

fn = 'ERA5_daily_total_precipitation_Nov2022_v2'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data\ERA5'+'/'+fn+'.nc', 'r')  # open CS2 data
ERA5_total_prep_nov22 = ds.variables['sf'][:]

fn = 'ERA5_daily_totalprecipitation_Nov2022_v2'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data\ERA5'+'/'+fn+'.nc', 'r')  # open CS2 data
ERA5_snowfall_nov22 = ds.variables['tp'][:]

ERA5_lon_mesh2, ERA5_lat_mesh2 = np.meshgrid(ERA5_lon, ERA5_lat)
ERA_lon_flat, ERA_lat_flat = ERA5_lon_mesh2.flatten(), ERA5_lat_mesh2.flatten()
idx_lat = np.where((ERA_lat_flat > -65) | (ERA_lat_flat < -72))
idx_lon = np.where((ERA_lon_flat > -48) | (ERA_lon_flat < -60))


var_in = copy.deepcopy(ERA5_2m_temp_dec22)
m, n, r = np.shape(var_in)
ERA5_2m_temp_dec22_T = var_in.reshape(m, n*r)
ERA5_2m_temp_dec22_T[:, idx_lat] = np.nan
ERA5_2m_temp_dec22_T[:, idx_lon] = np.nan
ERA5_2m_temp_dec22_T_mean = np.nanmean(ERA5_2m_temp_dec22_T, axis=1)
ERA5_2m_temp_dec22_T_max = np.nanmax(ERA5_2m_temp_dec22_T, axis=1)
ERA5_2m_temp_dec22_T_min = np.nanmin(ERA5_2m_temp_dec22_T, axis=1)

var_in = copy.deepcopy(ERA5_2m_temp_nov22)
m, n, r = np.shape(var_in)
ERA5_2m_temp_nov22_T = var_in.reshape(m, n*r)
ERA5_2m_temp_nov22_T[:, idx_lat] = np.nan
ERA5_2m_temp_nov22_T[:, idx_lon] = np.nan
ERA5_2m_temp_nov22_T_mean = np.nanmean(ERA5_2m_temp_nov22_T, axis=1)
ERA5_2m_temp_nov22_T_max = np.nanmax(ERA5_2m_temp_nov22_T, axis=1)
ERA5_2m_temp_nov22_T_min = np.nanmin(ERA5_2m_temp_nov22_T, axis=1)

var_in = copy.deepcopy(ERA5_total_prep_dec22)
m, n, r = np.shape(var_in)
ERA5_total_prep_dec22_T = var_in.reshape(m, n*r)
ERA5_total_prep_dec22_T[:, idx_lat] = np.nan
ERA5_total_prep_dec22_T[:, idx_lon] = np.nan
ERA5_total_prep_dec22_T_mean = np.nanmean(ERA5_total_prep_dec22_T, axis=1)
ERA5_total_prep_dec22_T_max = np.nanmax(ERA5_total_prep_dec22_T, axis=1)
ERA5_total_prep_dec22_T_min = np.nanmin(ERA5_total_prep_dec22_T, axis=1)

var_in = copy.deepcopy(ERA5_total_prep_nov22)
m, n, r = np.shape(var_in)
ERA5_total_prep_nov22_T = var_in.reshape(m, n*r)
ERA5_total_prep_nov22_T[:, idx_lat] = np.nan
ERA5_total_prep_nov22_T[:, idx_lon] = np.nan
ERA5_total_prep_nov22_T_mean = np.nanmean(ERA5_total_prep_nov22_T, axis=1)
ERA5_total_prep_nov22_T_max = np.nanmax(ERA5_total_prep_nov22_T, axis=1)
ERA5_total_prep_nov22_T_min = np.nanmin(ERA5_total_prep_nov22_T, axis=1)

var_in = copy.deepcopy(ERA5_snowfall_dec22)
m, n, r = np.shape(var_in)
ERA5_snowfall_dec22_T = var_in.reshape(m, n*r)
ERA5_snowfall_dec22_T[:, idx_lat] = np.nan
ERA5_snowfall_dec22_T[:, idx_lon] = np.nan
ERA5_snowfall_dec22_T_mean = np.nanmean(ERA5_snowfall_dec22_T, axis=1)
ERA5_snowfall_dec22_T_max = np.nanmax(ERA5_snowfall_dec22_T, axis=1)
ERA5_snowfall_dec22_T_min = np.nanmin(ERA5_snowfall_dec22_T, axis=1)

var_in = copy.deepcopy(ERA5_snowfall_nov22)
m, n, r = np.shape(var_in)
ERA5_snowfall_nov22_T = var_in.reshape(m, n*r)
ERA5_snowfall_nov22_T[:, idx_lat] = np.nan
ERA5_snowfall_nov22_T[:, idx_lon] = np.nan
ERA5_snowfall_nov22_T_mean = np.nanmean(ERA5_snowfall_nov22_T, axis=1)
ERA5_snowfall_nov22_T_max = np.nanmax(ERA5_snowfall_nov22_T, axis=1)
ERA5_snowfall_nov22_T_min = np.nanmin(ERA5_snowfall_nov22_T, axis=1)

ERA5_2m_temp_comb_mean = np.append(
    ERA5_2m_temp_nov22_T_mean, ERA5_2m_temp_dec22_T_mean)
ERA5_snowfall_comb_mean = np.append(
    ERA5_snowfall_nov22_T_mean, ERA5_snowfall_dec22_T_mean)
ERA5_total_prep_comb_mean = np.append(
    ERA5_total_prep_nov22_T_mean, ERA5_total_prep_dec22_T_mean)

ERA5_2m_temp_comb_min = np.append(
    ERA5_2m_temp_nov22_T_min, ERA5_2m_temp_dec22_T_min)
ERA5_snowfall_comb_min = np.append(
    ERA5_snowfall_nov22_T_min, ERA5_snowfall_dec22_T_min)
ERA5_total_prep_comb_min = np.append(
    ERA5_total_prep_nov22_T_min, ERA5_total_prep_dec22_T_min)

ERA5_2m_temp_comb_max = np.append(
    ERA5_2m_temp_nov22_T_max, ERA5_2m_temp_dec22_T_max)
ERA5_snowfall_comb_max = np.append(
    ERA5_snowfall_nov22_T_max, ERA5_snowfall_dec22_T_max)
ERA5_total_prep_comb_max = np.append(
    ERA5_total_prep_nov22_T_max, ERA5_total_prep_dec22_T_max)


#%% NEW FIGURES  %% Figure 1 - THE CRYOSPHERE 


fig, ax = pplt.subplots([1], share=0, axwidth=3, axheight=4.5, proj={1: 'laea'}, proj_kw={'lat_0': -70, 'lon_0': -45})
fig.patch.set_facecolor('white')
resol = '10m'
land = cartopy.feature.NaturalEarthFeature('physical', 'land',
                                           scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
cmap_check = plt.get_cmap('_rdylbu_r_copy') # old cmap = ylorrd
axs = ax[0]
axs.add_feature(cfeature.LAND, facecolor='lightgrey')
axs.coastlines(resolution=resol, color='k')

proj_in = ccrs.PlateCarree()

# cm = x=axs.pcolormesh(lon_AMSR2, lat_AMSR2,snow_depth_only,  cmap='blues', zorder=1, vmin=0, vmax=50)
cm = axs.scatter(lon_AMSR2.flatten(), lat_AMSR2.flatten(), c=snow_depth_only.flatten(),  cmap='_rdylbu_r_copy', zorder=1, vmin=0, vmax=50, s=5, extend='max')
#cm = axs.scatter(data_CASSIS[3], data_CASSIS[2], c=data_CASSIS[4]/10, cmap='_rdylbu_r_copy', zorder=1, s=3, vmin=0, vmax=50, extend='max')
axs.colorbar(cm, label='13 December 2022 CASSIS or AMSR2 snow depth, h$_s$ (cm)', loc='b')


leg2 = axs.scatter(np.array(data['Longitude']), np.array(
    data['Latitude']), c=cmap_qual[6], s=0.1, marker="o", zorder=3, label='Airborne')
leg3 = axs.scatter(longitude_comb[val_x*n1:val_x*(n1+1)],
                   latitude_comb[val_x*n1:val_x*(n1+1)], c=cmap_qual[0], s=5, zorder=15, label='Subset')
start1=axs.scatter(longitude_comb[0], latitude_comb[0], c=cmap_qual[5],
            s=50, marker='X', edgecolor='k', label='', zorder=11)
stop1=axs.scatter(longitude_comb[-1], latitude_comb[-1], c=cmap_qual[6],
            s=50, marker='X', edgecolor='k', label='', zorder=11)

df = data_CRYO2ICE[data_CRYO2ICE['snow_depth'].notna()].reset_index()
leg1 = axs.scatter(np.array(df['lon']), np.array(
    df['lat']), c='grey', s=5, zorder=10, label='CRYO2ICE (CryoTEMPO)')
start2=im = axs.scatter(np.array(df['lon'])[0], np.array(
    df['lat'])[0], c='w', s=50, marker='X', edgecolor='k', zorder=11)
stop2=im = axs.scatter(np.array(df['lon'])[-1], np.array(
    df['lat'])[-1], c='grey', s=50, marker='X', edgecolor='k', zorder=11)
leg4 = axs.scatter(lead_lon_loc, lead_lat_loc, c=cmap_qual[4], edgecolor='k', s=100, marker='*', zorder=12, label='lead locations' )


area_lon = [-60, -48, -48, -60, -60]
area_lat = [-65, -65, -72, -72, -65]
axs.plot(area_lon, area_lat, c='k', ls='-', linewidth=0.5)


axs.format(lonlim=(-65, -45), latlim=(-60, -75), labels=True, lrtitle='AMSR2 h$_s$')
# axs.format(boundinglat=-60)
#axs.legend([leg3, leg1, leg2, leg4], ['Subset example', 'CRYO2ICE',
#           'CRYO2ICEANT22', 'Manually-detected leads'], loc='ul', ncols=1, markersize=15, scatterpoints=3,)

from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt

axs.legend([(start1, stop1), (start2, stop2), leg3, leg1, leg2, leg4], ['Start/Stop CRYO2ICEANT22', 'Start/Stop CRYO2ICE{}'.format('$_{CryoTEMPO}$'), 'Subset example', 'CRYO2ICE{}'.format('$_{CryoTEMPO}$'),
           'CRYO2ICEANT22', 'Manually-detected leads'], loc='ul', ncols=1,  numpoints=1, markersize=50, handler_map={tuple: HandlerTuple(ndivide=None)})

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins = inset_axes(axs, width="35%", height="20%", loc="lower left", 
                   axes_class=cartopy.mpl.geoaxes.GeoAxes, 
                   axes_kwargs=dict(map_projection=cartopy.crs.SouthPolarStereo()))

axins.add_feature(cfeature.LAND, facecolor='lightgrey')
axins.coastlines(resolution='50m', color='k', linewidth=0.5)
axins.set_extent([0, 360, -90, -50], crs=ccrs.PlateCarree())

axins.scatter(data_CASSIS[3], data_CASSIS[2], c=data_CASSIS[4]/10,
                     cmap='_rdylbu_r_copy', zorder=1, s=0.5, vmin=0, vmax=50, transform=cartopy.crs.PlateCarree())

area_lon = [-80, -40, -40, -80, -80]
area_lat = [-58, -58, -75, -75, -58]
axins.plot(area_lon, area_lat, c='k', ls='-', linewidth=0.5, transform=cartopy.crs.PlateCarree())


fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\Figure1_new.png', dpi=300)

#%% Figure 2 NEW - THE CRYOSPHERE


fig, ax = pplt.subplots([[1, 1, 1], 
                         [2, 3, 4],
                         [2, 3, 4]], share=0, axwidth=8, axheight=1.2, proj={2: 'laea', 3: 'laea', 4: 'laea'}, proj_kw={'lat_0': -70, 'lon_0': -45})
fig.patch.set_facecolor('white')
resol = '10m'
land = cartopy.feature.NaturalEarthFeature('physical', 'land',
                                           scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])



axs = ax[0]
range_val = np.arange(0, 61)
range_time = pd.date_range(start="2022-11-01", end="2022-12-31")
ERA5_filt_comb = pd.DataFrame({'date': range_time.strftime('%Y%m%d'),
                               '2mt_celsius': ERA5_2m_temp_comb_mean-273.15,
                               'sf': ERA5_snowfall_comb_mean*1000,
                               'tp': ERA5_total_prep_comb_mean*1000,
                               'min_2mt_celsius': ERA5_2m_temp_comb_min-273.15,
                               'min_sf': ERA5_snowfall_comb_min*1000,
                               'min_tp': ERA5_total_prep_comb_min*1000,
                               'max_2mt_celsius': ERA5_2m_temp_comb_max-273.15,
                               'max_sf': ERA5_snowfall_comb_max*1000,
                               'max_tp': ERA5_total_prep_comb_max*1000})
ERA5_filt_comb['datetime'] = pd.to_datetime(range_time, format='%Y-%m-%d')

idx = np.where(ERA5_filt_comb['datetime']==pd.to_datetime("2022-12-13", format='%Y-%m-%d'))
ERA5_idx = ERA5_filt_comb[0:idx[0][0]]
ERA5_idx['sd_320']=ERA5_idx['max_sf']/320

sd_acc_ERA5_total = np.sum(ERA5_idx['sd_320'])

leg1 = axs.plot(ERA5_filt_comb['datetime'], ERA5_filt_comb['2mt_celsius'],
                c='k', linestyle='-', linewidth=1, label='2 m air temperature (2mTa)')
leg2 = axs.fill_between(ERA5_filt_comb['datetime'], ERA5_filt_comb['min_2mt_celsius'], ERA5_filt_comb['max_2mt_celsius'],
                        c='k', linestyle='-', linewidth=1, label='Minimum-maximum 2mTa', alpha=0.2)
axs.format(xlabel='', ylim=(-20, 15), ylabel='2m air temperature ($^\circ$C)',
           ultitle='Daily averaged-ERA5 estimates in area of interest (box in b-d)')
axs.axvline(pd.to_datetime("2022-12-13", format='%Y-%m-%d'), c='r')
axs.text(pd.to_datetime("2022-12-14", format='%Y-%m-%d'),
         11, 'CRYO2ICEANT22', c='r')


ax1 = axs.twinx()
leg3 = ax1.plot(ERA5_filt_comb['datetime'], ERA5_filt_comb['tp'],
                c=cmap_qual[2], linestyle='-', linewidth=1, label='Total precipitation (tp)')
leg4 = ax1.fill_between(ERA5_filt_comb['datetime'], ERA5_filt_comb['min_tp'], ERA5_filt_comb['max_tp'],
                        c=cmap_qual[2], linestyle='-', linewidth=1, label='Minimum-maximum tp', alpha=0.2)
leg5 = ax1.plot(ERA5_filt_comb['datetime'], ERA5_filt_comb['sf'],
                c=cmap_qual[6], linestyle='--', linewidth=1, label='Snowfall (sf)')
leg6 = ax1.fill_between(ERA5_filt_comb['datetime'], ERA5_filt_comb['min_sf'], ERA5_filt_comb['max_sf'],
                        c=cmap_qual[6], linestyle='-', linewidth=1, label='Minimum-maximum sf', alpha=0.2)
ax1.format(ylabel='precipitation\n (mm of water equivalent)', ylim=(0, 1.2))
ax1.legend([leg1, leg3, leg5, leg2, leg4, leg6], loc='t', ncols=3, pad=-0.5, order='C')

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Date format
# Set tick locator for sampling every 5 days
locator = mdates.DayLocator(interval=5)  # Sampling every 5 days
ax1.xaxis.set_major_locator(locator)
# Rotate the tick labels for better readability
ax1.xaxis.set_tick_params(rotation=45)



axs = ax[1]
axs.add_feature(cfeature.LAND, facecolor='lightgrey')
axs.coastlines(resolution=resol, color='k')
axs.format(lonlim=(-65, -45), latlim=(-60, -75), ultitle='13 December 2022')

ERA5_lon_mesh, ERA5_lat_mesh = np.meshgrid(ERA5_lon, ERA5_lat)
# cm = axs.scatter(ERA5_lon_mesh.flatten(), ERA5_lat_mesh.flatten(),c=(ERA5_2m_temp_dec22_mean.flatten()-ERA5_2m_temp_dec22[13, :, :].flatten()),  cmap='RdBu_r', zorder=1, vmin=-10, vmax=10, s=3, extend='both')
# axs.colorbar(cm, label='ERA5 2m air temperature anomaly ($^\circ$C)', loc='r')
cm = axs.scatter(ERA5_lon_mesh.flatten(), ERA5_lat_mesh.flatten(), c=ERA5_2m_temp_dec22[13, :, :].flatten()-273.15,  cmap='RdBu_r', zorder=1, vmin=-10, vmax=10, s=10, extend='both')
axs.colorbar(cm, label='ERA5 2m air temperature ($^\circ$C)',
             loc='b', rotation=45)

axs.scatter(np.array(data['Longitude']), np.array(
    data['Latitude']), c='red', s=0.1, marker="o", label='')

area_lon = [-60, -48, -48, -60, -60]
area_lat = [-65, -65, -72, -72, -65]
axs.plot(area_lon, area_lat, c='k', ls='-', linewidth=0.5)


axs = ax[2]
axs.add_feature(cfeature.LAND, facecolor='lightgrey')
axs.coastlines(resolution=resol, color='k')
axs.format(lonlim=(-65, -45), latlim=(-60, -75),
           ultitle='December 2022 vs.\n13 December 2022')

cm = axs.scatter(ERA5_lon_mesh.flatten(), ERA5_lat_mesh.flatten(), c=(np.nanmean(ERA5_total_prep_dec22[:, :, :], axis=0) *
                 1000-ERA5_total_prep_dec22[13, :, :]*1000).flatten(),  cmap='RdBu_r', zorder=1, s=10, extend='both')
axs.colorbar(
    cm, label='ERA5 total precipitation anomaly\n(mm of water equivalent)', loc='b', rotation=45)

axs.scatter(np.array(data['Longitude']), np.array(
    data['Latitude']), c='red', s=0.1, marker="o", label='')

area_lon = [-60, -48, -48, -60, -60]
area_lat = [-65, -65, -72, -72, -65]
axs.plot(area_lon, area_lat, c='k', ls='-', linewidth=0.5)

axs = ax[3]
axs.add_feature(cfeature.LAND, facecolor='lightgrey')
axs.coastlines(resolution=resol, color='k')
axs.format(lonlim=(-65, -45), latlim=(-60, -75), ultitle='13 December 2022')

cm = axs.scatter(ERA5_lon_mesh.flatten(), ERA5_lat_mesh.flatten(), c=(
    ERA5_snowfall_dec22[13, :, :]*1000-ERA5_total_prep_dec22[13, :, :]*1000).flatten(),  cmap='RdBu_r', zorder=1, s=10, extend='both', vmin=-0.0003*1000)

axs.colorbar(
    cm, label='ERA5 snowfall-total precipitation\n(mm of water equivalent)', loc='b', rotation=45)

axs.scatter(np.array(data['Longitude']), np.array(
    data['Latitude']), c='red', s=0.1, marker="o", label='')

area_lon = [-60, -48, -48, -60, -60]
area_lat = [-65, -65, -72, -72, -65]
axs.plot(area_lon, area_lat, c='k', ls='-', linewidth=0.5)

fig.format(abc='(a)', abcloc='ul')

fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\Figure2_new.png', dpi=300)




#%% COMPUTE TIME

fn = 'kuband_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
time = ds.variables['gps_time']

from cftime import num2date, DatetimeNoLeap
t_unit = "seconds since 1970-01-01T00:00:00";
# Convert time values to dates
tvalue = num2date(time, units=t_unit, calendar='gregorian')
str_time = [i.strftime("%Y-%m-%dT%H:%M:%S") for i in tvalue] # to display dates as string
#str_time = [str(i) for i in tvalue]


#%% Figure 2 - ECHOGRAMS AND LIDAR 

c = 300e6

fig, axs = pplt.subplots([[1],
                          [2]], axwidth=7, axheight=1, sharex=False, sharey=False)
fig.patch.set_facecolor('white')
# n = 300
# n1=8
# val_x = 300

## PAPER 
n = 300
n1 = 70
# n1=8
val_x = 1000


n0_extra = 880
#idx = plot_waveforms_examples(n0_extra, axs[0], '', ALS_along_radar_track)

n1_extra = 180
#idx = plot_waveforms_examples_right(    n1_extra, axs[1], '(b)', ALS_along_radar_track)


ax = axs[0]
fn = 'kuband_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
wf_kuband = ds.variables['waveform'][val_x*n1:val_x*(n1+1)]
lat_extra = ds.variables['lat'][val_x*n1:val_x*(n1+1)]
wf_wb = 10 * np.log10(wf_kuband)
# cb_data = ax.imshow(wf_wb.transpose(), cmap='magma_r', aspect="auto", vmin=-120, vmax=-40,zorder=1, extend='both')
cb_data = ax.imshow(wf_wb.transpose(), cmap='greys', aspect="auto", vmin=-120, vmax=-40, extent=[np.max(
    latitude_comb[val_x*n1:val_x*(n1+1)]), np.min(latitude_comb[val_x*n1:val_x*(n1+1)]), len(wf_kuband[0]), 0], zorder=1, extend='both')
ax.set_yticks([600, 700, 800])

ax.axvline(lat_extra[n1_extra], marker=11,
           markersize=6, c='k', zorder=15, linewidth=0.5, linestyle='--')


elevation = ds.variables['elevation'][:]
time = ds.variables['two_way_travel_time'][:]
time_conv = (time*c/2)+offset_ku
ALS_Ku= elevation-ALS_along_radar_track
ALS_retrack_gate_Ku = np.ones(len(ALS_Ku))*np.nan
for i in np.arange(0, len(ALS_Ku)):
    ALS_retrack_gate_Ku[i] = np.argmin(np.abs(time_conv[i]-ALS_Ku[i]))
ALS_retrack_gate_Ku[ALS_retrack_gate_Ku==0]=np.nan
ax.plot(lat_extra, ALS_retrack_gate_Ku[val_x * n1:val_x*(n1+1)], c='green', linewidth=0.5)

rtck_gate_kuband = ds.variables['retracking_gate_tfmra40'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_kuband, c=cmap_qual[0], linewidth=0.5)
rtck_gate_kuband = ds.variables['retracking_gate_tfmra50'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_kuband, c=cmap_qual[6], linewidth=0.5)
rtck_gate_kuband = ds.variables['retracking_gate_tfmra80'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_kuband, c=cmap_qual[1], linewidth=0.5)
ax.plot(lat_extra, a_s_peaks[val_x*n1:val_x*(n1+1)], c='brown', linewidth=0.5)
#ax.plot(lat_extra, s_i_centroids[val_x*n1:val_x*(n1+1)], c='orange', linewidth=0.5)

'''
rtck_gate_as = ds.variables['retracking_gate_air_snow_peakiness'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_as, c=cmap_qual[0], linewidth=0.5, linestyle='-')
rtck_gate_is = ds.variables['retracking_gate_snow_ice_peakiness'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_is, c=cmap_qual[1], linewidth=0.5, linestyle='-')

rtck_gate_as = ds.variables['retracking_gate_air_snow_cwt_TN'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_as, c=cmap_qual[6], linewidth=0.5)
rtck_gate_is = ds.variables['retracking_gate_snow_ice_snow_cwt_TN'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_is, c=cmap_qual[5], linewidth=0.5)
'''
ax.arrow(-69.97, 780, -0.012, 0, facecolor='k', zorder=15, head_length = 0.001, head_width = 20)
ax.text(-69.97, 830, 'snow-ice interface', color='k', zorder=15)


txt = ['Figure 3']
x = [lat_extra[n1_extra+5]]
for i, txt in enumerate(txt):
    ax.annotate(txt, (x[i], 580), c='k')


ax.format(ultitle='Ku', ylim=(900, 500), xlabel='', ylabel='')


ax = axs[0].panel_axes('b', width=1, space=0)
fn = 'kaband_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
wf_kuband = ds.variables['waveform'][val_x*n1:val_x*(n1+1)]
lat_extra = ds.variables['lat'][val_x*n1:val_x*(n1+1)]
time = ds.variables['two_way_travel_time'][:]
#range_res = np.abs((time[0][0] - time[0][1]) * c / 2)

wf_wb = 10 * np.log10(wf_kuband)
# cb_data = ax.imshow(wf_wb.transpose(), cmap='magma_r', aspect="auto", vmin=-120, vmax=-40,zorder=1, extend='both')
cb_data = ax.imshow(wf_wb.transpose(), cmap='greys', aspect="auto", vmin=-120, vmax=-40, extent=[np.max(
    latitude_comb[val_x*n1:val_x*(n1+1)]), np.min(latitude_comb[val_x*n1:val_x*(n1+1)]), len(wf_kuband[0]), 0], zorder=1, extend='both')
ax.format(ultitle='Ka', ylim=(900, 500), xlabel='',
          ylabel='range bins')
ax.set_yticks([600, 700, 800])

ax.axvline(lat_extra[n1_extra], marker=11,
           markersize=6, c='k', zorder=15, linewidth=0.5, linestyle='--')
    
elevation = ds.variables['elevation'][:]
time = ds.variables['two_way_travel_time'][:]
time_conv = (time*c/2)+offset_ka
ALS_Ku_Ka = elevation-ALS_along_radar_track+offset_Ku_Ka
ALS_retrack_gate_Ka = np.ones(len(ALS_Ku_Ka))*np.nan
for i in np.arange(0, len(ALS_Ku_Ka)):
    ALS_retrack_gate_Ka[i] = np.argmin(np.abs(time_conv[i]-ALS_Ku_Ka[i]))
ALS_retrack_gate_Ka[ALS_retrack_gate_Ka==0]=np.nan
ax.plot(lat_extra, ALS_retrack_gate_Ka[val_x * n1:val_x*(n1+1)], c='green', linewidth=0.5)


rtck_gate_kuband = ds.variables['retracking_gate_tfmra40'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_kuband, c=cmap_qual[0], linewidth=0.5)
rtck_gate_kuband = ds.variables['retracking_gate_tfmra50'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_kuband, c=cmap_qual[6], linewidth=0.5)
rtck_gate_kuband = ds.variables['retracking_gate_tfmra80'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_kuband, c=cmap_qual[1], linewidth=0.5)
ax.plot(lat_extra, a_s_peaks_ka[val_x*n1:val_x*(n1+1)], c='brown', linewidth=0.5)
#ax.plot(lat_extra, s_i_centroids_ka[val_x*n1:val_x*(n1+1)], c='orange', linewidth=0.5)
'''
rtck_gate_as = ds.variables['retracking_gate_air_snow_peakiness'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_as, c=cmap_qual[0], linewidth=0.5, linestyle='-')
rtck_gate_is = ds.variables['retracking_gate_snow_ice_peakiness'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_is, c=cmap_qual[1], linewidth=0.5, linestyle='-')

rtck_gate_as = ds.variables['retracking_gate_air_snow_cwt_TN'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_as, c=cmap_qual[6], linewidth=0.5)
rtck_gate_is = ds.variables['retracking_gate_snow_ice_snow_cwt_TN'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_is, c=cmap_qual[5], linewidth=0.5)
'''
#txt = ['Figure 3']
#x = [lat_extra[n1_extra+1]]
#for i, txt in enumerate(txt):
#    ax.annotate(txt, (x[i], 580), c='k')
    
ax = axs[0].panel_axes('b', width=1, space=0)
fn = 'snow_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
wf_kuband = ds.variables['waveform'][val_x*n1:val_x*(n1+1)]
lat_extra = ds.variables['lat'][val_x*n1:val_x*(n1+1)]
wf_wb = 10 * np.log10(wf_kuband)
# cb_data = ax.imshow(wf_wb.transpose(), cmap='magma_r', aspect="auto", vmin=-120, vmax=-40,zorder=1, extend='both')
cb_data = ax.imshow(wf_wb.transpose(), cmap='greys', aspect="auto", vmin=-120, vmax=-40, extent=[np.max(
    latitude_comb[val_x*n1:val_x*(n1+1)]), np.min(latitude_comb[val_x*n1:val_x*(n1+1)]), len(wf_kuband[0]), 0], zorder=1, extend='both')
ax.format(ultitle='C/S', ylim=(900, 500),
          xlabel='latitude (degrees N)', ylabel='')
axs[0].colorbar(cb_data, label='relative power (dB)', loc='t')

ax.axvline(lat_extra[n1_extra], marker=11,
           markersize=6, c='k', zorder=15, linewidth=0.5, linestyle='--')

elevation = ds.variables['elevation'][:]
time = ds.variables['two_way_travel_time'][:]
time_conv = (time*c/2)+offset_snow
ALS_Ku_snow = elevation-ALS_along_radar_track+offset_Ku_snow
ALS_retrack_gate_snow = np.ones(len(ALS_Ku_snow))*np.nan
for i in np.arange(0, len(ALS_Ku_snow)):
    ALS_retrack_gate_snow[i] = np.argmin(np.abs(time_conv[i]-ALS_Ku_snow[i]))
ALS_retrack_gate_snow[ALS_retrack_gate_snow==0]=np.nan
ax.plot(lat_extra, ALS_retrack_gate_snow[val_x * n1:val_x*(n1+1)], c='green', linewidth=0.5)


rtck_gate_as = ds.variables['retracking_gate_air_snow_peakiness'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_as, c=cmap_qual[0], linewidth=0.5, linestyle='-')
rtck_gate_is = ds.variables['retracking_gate_snow_ice_peakiness'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_is, c=cmap_qual[1], linewidth=0.5, linestyle='-')

rtck_gate_as = ds.variables['retracking_gate_air_snow_cwt_TN'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_as, c=cmap_qual[6], linewidth=0.5)
rtck_gate_is = ds.variables['retracking_gate_snow_ice_snow_cwt_TN'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_is, c=cmap_qual[5], linewidth=0.5)

ax.plot(lat_extra, a_s_peaks_snow[val_x * n1:val_x*(n1+1)], c='brown', linewidth=0.5)



ax = axs[1]
idx1, idx2 = np.nanmax(lat_extra), np.nanmin(lat_extra)
data_ALS_scn_filt = data_ALS_scn[(data_ALS_scn[1]<idx1) & (data_ALS_scn[1]>idx2)]
cb = ax.scatter(data_ALS_scn_filt[1], data_ALS_scn_filt[5], c=data_ALS_scn_filt[3], cmap='crest', extend='both' , vmin=np.quantile(data_ALS_scn_filt[3], 0.05), vmax=np.quantile(data_ALS_scn_filt[3], 0.95), s=0.1)
leg1=ax.scatter(lat_ALS_along_radar_track, ALS_scan_number_along_radar_track, s=0.1, c='green')
ax.axvline(lat_extra[n1_extra], marker=11,
          markersize=6, c='k', zorder=15, linewidth=0.5, linestyle='--')
ax.colorbar(cb, label='ALS ellipsoidal elevations (m)', loc='b')
ax.format(xlim=(np.nanmax(lat_extra), np.nanmin(lat_extra)), xlabel='latitude (degrees N)', ylabel='scan number\n(0-251)')
#ax.axvspan(-69.931,-69.935, color='lightgrey', alpha=0.5, edgecolor='grey')
#ax.axvspan(-69.938,-69.942, color='lightgrey', alpha=0.5, edgecolor='grey')
#ax.axvspan(-69.954,-69.972, color='lightgrey', alpha=0.5, edgecolor='grey')
#ax.axvspan(-69.981,-69.998, color='lightgrey', alpha=0.5, edgecolor='grey')
#ax.text(-69.931-0.0001, 210, 'I')
#ax.text(-69.938-0.0001, 210, 'II')
#ax.text(-69.954-0.0001, 210, 'III')
#ax.text(-69.981-0.0001, 210, 'IV')
ax.legend(leg1, 'nadir laser profile', prop=dict(size=8), loc='ur', ncols=4, markersize=10)

ax1 = ax.twiny()
ax1.format(xlim=(np.min(along_track_dist[val_x*n1:val_x*(n1+1)]), np.max(along_track_dist[val_x*n1:val_x*(n1+1)])), xlabel='along-track distance (m)')
#df_ALS_roughness_filt = df_ALS_scan_v1_roughness[(df_ALS_scan_v1_roughness['lat']<idx1) & (df_ALS_scan_v1_roughness['lat']>idx2)]
#leg2=ax1.scatter(df_ALS_roughness_filt['lat'], df_ALS_roughness_filt['elevation_anomaly'], s=0.5, linestyle='-', linewidth=0.1, c='k')
#ax1.format(ylabel='elevation anomaly (m)', ylim=(0,1.5))
#ax1.yaxis.set_label_position("right")
#ax1.yaxis.tick_right()

#ax = axs[1].panel_axes('t', width=1.0, space=0)
#df_ALS_roughness_filt = df_ALS_scan_v1_roughness[(df_ALS_scan_v1_roughness['lat']<idx1) & (df_ALS_scan_v1_roughness['lat']>idx2)]
#leg4 = ax.scatter(df_ALS_roughness_filt['lat'], df_ALS_roughness_filt['roughness_anomaly'], s=0.5, linestyle='--', linewidth=0.5, c='grey')
#leg3 = ax.plot(lat_ALS_along_radar_track, ALS_roughness_along_radar_track,  c=cmap_qual[1], linewidth=0.5, linestyle='-')
#ax.format(xlim=(np.nanmax(lat_extra), np.nanmin(lat_extra)), ylim=(0,1), ylabel='roughness, $\sigma$ (m)', xlabel='latitude (degrees N)')

#ax.legend([leg1, leg2, leg4, leg3], ['nadir laser profile', 'elevation anomaly', '$\sigma_{400m}$','$\sigma_{5m}$'], prop=dict(size=8), loc='ur', ncols=4, markersize=10)



leg1 = Line2D([0], [0], color=cmap_qual[0], linestyle='-',
              label='TFMRA40')
leg2 = Line2D([0], [0], color=cmap_qual[6],  linestyle='-',
              label='TFMRA50')
leg3 = Line2D([0], [0], color=cmap_qual[1],  linestyle='-',
              label='TFMRA80')
#leg4 = Line2D([0], [0], color='orange',  linestyle='-',
#              label='Centroid', marker=8, markersize=10)
leg5 = Line2D([0], [0], color='brown', ls='-',
              label='Max peak')
leg6 = Line2D([0], [0], color='green', linestyle='-', label='ALS')

handles = [leg1, leg2, leg3,  leg5, leg6]
frame = axs[0].legend(handles, loc='b', ncols=3, title='Ka/Ku-band',
                      align='left', titlefontweight='bold')
frame._legend_box.align = "left"
frame.get_title().set_weight('bold')

leg1 = Line2D([0], [0], color=cmap_qual[6], linestyle='-',
              label='a-s CWT')
leg2 = Line2D([0], [0], color=cmap_qual[5],  linestyle='-',
              label='s-i CWT')
leg3 = Line2D([0], [0], color=cmap_qual[0],  linestyle='-',
              label='a-s peakiness')
leg4 = Line2D([0], [0], color=cmap_qual[1],  linestyle='-',
              label='s-i peakiness')

handles = [leg1, leg2, leg3, leg4, leg5, leg6]
frame = axs[0].legend(handles, loc='b', ncols=3, order='F',
                      title='C/S-band', align='right', titlefontweight='bold')
frame._legend_box.align = "right"
frame.get_title().set_weight('bold')
frame.get_frame().set_facecolor('lightgrey')




#handles = [leg5, leg6]
#frame = axs[0].legend(handles, loc='ur', ncols=2)


fig.format(abc="(a)", abcloc='l')


fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\Figure5_waveforms_max.png', dpi=300)



#%% Figure - ONE WAVEFORM EXAMPLE - THE CRYOSPHERE

def plot_waveforms_examples(n_extra, axs, title_spec, ALS_along_radar_track):
    N_obs = val_x*n1+n_extra
    ax = axs

    fn = 'kuband_20221213_02_74_232_002'
    ds = netCDF4.Dataset(
        r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
    wf_kuband = ds.variables['waveform'][:]
    range_40 = ds.variables['range_tfmra40'][:]
    range_50 = ds.variables['range_tfmra50'][:]
    range_80 = ds.variables['range_tfmra80'][:]

    # range_40 = ds.variables['retracking_gate_tfmra40'][:]
    # range_50 = ds.variables['retracking_gate_tfmra50'][:]
    # range_80 = ds.variables['retracking_gate_tfmra80'][:]

    time = ds.variables['two_way_travel_time'][:]
    time_conv = (time*c/2)+offset_ku
    # wf_kuband = 10 * np.log10(wf_kuband)

    idx = np.arange(np.nanargmax(
        wf_kuband[N_obs][:])-100, np.nanargmax(wf_kuband[N_obs][:])+156)
    # noise=np.nanmean(wf_kuband[N_obs][0:500])
    centroid = np.nansum(
        wf_kuband[N_obs][idx]*time_conv[N_obs][idx])/np.nansum(wf_kuband[N_obs][idx])
    # peaks=scipy.find_peaks(wf_kuband[0], height=noise+noise*10, width=2)
    peaks = scipy.find_peaks(wf_kuband[N_obs][idx], prominence=np.nanmax(
        wf_kuband[N_obs][idx])*0.2, distance=25)

    leg1 = ax.axhline(y=(range_40[N_obs]*c/2)+offset_ku,
                      color=cmap_qual[0], linestyle='-', linewidth=1, label='TFMRA40')
    leg2 = ax.axhline(y=(range_50[N_obs]*c/2)+offset_ku,
                      color=cmap_qual[6], linestyle='--', linewidth=1, label='TFMRA50')
    leg3 = ax.axhline(y=(range_80[N_obs]*c/2)+offset_ku,
                      color=cmap_qual[1], linestyle='--', linewidth=1, label='TFMRA80')

    #leg4 = ax.axhline(y=centroid, color='orange', linestyle='-', linewidth=1)
    # ax.scatter(wf_kuband[N_obs][peaks[0][:]+idx[0]],time_conv[N_obs][peaks[0][:]+idx[0]], marker='*', s=10, zorder=2)
    leg5 = ax.axhline(y=time_conv[N_obs][np.nanargmax(
        wf_kuband[N_obs][:])], color='brown', linestyle='--', linewidth=1)
    ax.plot(wf_kuband[N_obs-5:N_obs+5][:, idx],
            time_conv[N_obs-5:N_obs+5][:, idx], zorder=0, c='lightgrey')
    leg6 = ax.plot(wf_kuband[N_obs][idx], time_conv[N_obs]
                   [idx], zorder=0, c='k', linewidth=0.5)
    # ax.scatter(wf_kuband[N_obs][np.nanargmax(wf_kuband[N_obs][:])], time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])],marker='*', s=10, zorder=2)
    # ax.format(ultitle='$\delta$range={:.2f} m'.format((centroid-time_conv[N_obs][peaks[0][0]+idx[0]])))
    ax.format(ylabel='range (m)', xlim=(np.nanmin(wf_kuband[N_obs])+0.05*np.nanmean(wf_kuband[N_obs]), np.nanmax(
        wf_kuband[N_obs])-0.05*np.nanmean(wf_kuband[N_obs])), ylim=(time_conv[N_obs][idx[-1]], time_conv[N_obs][idx[0]]), ultitle='Ku-band'.format(fn[0:2]))

    #ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
    #    range_40[N_obs]*c/2)+offset_ku, color=cmap_qual[0], marker=4, s=30, label='')
    #ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
    #    range_50[N_obs]*c/2)+offset_ku, color=cmap_qual[6], marker=4, s=30, label='')
    #ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
    #    range_80[N_obs]*c/2)+offset_ku, color=cmap_qual[1], marker=4, s=30, label='')
    #ax.scatter(x=np.nanmin(
    #    wf_kuband[N_obs][idx]), y=centroid, color='orange', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=time_conv[N_obs][np.nanargmax(
        wf_kuband[N_obs][:])], color='brown', marker=4, s=30, label='')

    elevation = ds.variables['elevation'][:]
    leg0 = ax.axhline(y=elevation[N_obs] -
                      ALS_along_radar_track[N_obs], c='green')
    
    as_cwt = ds.variables['range_air_snow_cwt_TN'][:]
    si_cwt = ds.variables['range_snow_ice_cwt_TN'][:]
    as_peak = ds.variables['range_air_snow_peakiness'][:]
    si_peak = ds.variables['range_snow_ice_peakiness'][:]
    
    leg1 = ax.axhline(y=(as_cwt[N_obs]*c/2)+offset_ku,
                      color=cmap_qual[6], linestyle='-', linewidth=1, label='TFMRA50')
    leg2 = ax.axhline(y=(si_cwt[N_obs]*c/2)+offset_ku,
                      color=cmap_qual[5], linestyle='--', linewidth=1, label='TFMRA80')

    leg3 = ax.axhline(y=(as_peak[N_obs]*c/2)+offset_ku,
                      color=cmap_qual[0], linestyle='-', linewidth=1, label='TFMRA50')
    leg4 = ax.axhline(y=(si_peak[N_obs]*c/2)+offset_ku,
                      color=cmap_qual[1], linestyle='--', linewidth=1, label='TFMRA80')

    # ax.format(ltitle=r'$h_{}$ = {:.2f} m'.format('{R, ALS}', elevation[N_obs]-ALS_along_radar_track[N_obs]))

    handle0 = r'$h_{}$ = {:.2f} m'.format(
        'R', elevation[N_obs]-ALS_along_radar_track[N_obs])
    handle1 = r'$h_{}$ = {:.2f} m'.format('R', (range_40[N_obs]*c/2)+offset_ku)
    handle2 = r'$h_{}$ = {:.2f} m'.format('R', (range_50[N_obs]*c/2)+offset_ku)
    handle3 = r'$h_{}$ = {:.2f} m'.format('R', (range_80[N_obs]*c/2)+offset_ku)
    #handle4 = r'$h_{}$ = {:.2f} m'.format('R', centroid)
    handle5 = r'$h_{}$ = {:.2f} m'.format(
        'R', time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])])
    #handle6 = r'$h_{}$ = {:.2f} m'.format(
    #    '{s,MaxC}', (centroid-time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])])/params["n_snow"])
    frame = ax.legend([leg0, leg1, leg2, leg3, leg5], [handle0, handle1, handle2, handle3, handle5],
                      loc='lr', handlelength=0, markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, prop={'size': 7})

    ax = axs.panel_axes('r', width=1, space=0)

    fn = 'kaband_20221213_02_74_232_002'
    ds = netCDF4.Dataset(
        r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
    wf_kuband = ds.variables['waveform'][:]
    time = ds.variables['two_way_travel_time'][:]
    range_40 = ds.variables['range_tfmra40'][:]
    range_50 = ds.variables['range_tfmra50'][:]
    range_80 = ds.variables['range_tfmra80'][:]
    time_conv = (time*c/2)+offset_ka+offset_Ku_Ka
    # wf_kuband = 10 * np.log10(wf_kuband)

    idx = np.arange(np.nanargmax(
        wf_kuband[N_obs][:])-100, np.nanargmax(wf_kuband[N_obs][:])+156)
    # noise=np.nanmean(wf_kuband[N_obs][0:500])
    centroid = np.nansum(
        wf_kuband[N_obs][idx]*time_conv[N_obs][idx])/np.nansum(wf_kuband[N_obs][idx])
    # peaks=scipy.find_peaks(wf_kuband[0], height=noise+noise*10, width=2)
    peaks = scipy.find_peaks(wf_kuband[N_obs][idx], prominence=np.nanmax(
        wf_kuband[N_obs][idx])*0.2, distance=25)

    leg1 = ax.axhline(y=(range_40[N_obs]*c/2)+offset_ka+offset_Ku_Ka,
                      color=cmap_qual[0], linestyle='-', linewidth=1, label='TFMRA40')
    leg2 = ax.axhline(y=(range_50[N_obs]*c/2)+offset_ka+offset_Ku_Ka,
                      color=cmap_qual[6], linestyle='--', linewidth=1, label='TFMRA50')
    leg3 = ax.axhline(y=(range_80[N_obs]*c/2)+offset_ka+offset_Ku_Ka,
                      color=cmap_qual[1], linestyle='--', linewidth=1, label='TFMRA80')

    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        range_40[N_obs]*c/2)+offset_ka+offset_Ku_Ka, color='b', marker=4, s=30, label='')

    #leg4 = ax.axhline(y=centroid+offset_Ku_Ka, color='orange',
    #                  linestyle='-', linewidth=1)
    # ax.scatter(wf_kuband[N_obs][peaks[0][:]+idx[0]],time_conv[N_obs][peaks[0][:]+idx[0]], marker='*', s=10, zorder=2)
    leg5 = ax.axhline(y=time_conv[N_obs][np.nanargmax(
        wf_kuband[N_obs][:])]+offset_Ku_Ka, color='brown', linestyle='--', linewidth=1)
    # ax.scatter(wf_kuband[N_obs][np.nanargmax(wf_kuband[N_obs][:])], time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])],marker='*', s=10, zorder=2)
    # ax.format(ultitle='$\delta$range={:.2f} m'.format((centroid-time_conv[N_obs][peaks[0][0]+idx[0]])))
    ax.plot(wf_kuband[N_obs-5:N_obs+5][:, idx], time_conv[N_obs -
            5:N_obs+5][:, idx]+offset_Ku_Ka, zorder=0, c='lightgrey')
    leg6 = ax.plot(wf_kuband[N_obs][idx], time_conv[N_obs]
                   [idx]+offset_Ku_Ka, zorder=0, c='k', linewidth=0.5)

    ax.format(xlabel='power (W)', xlim=(np.nanmin(wf_kuband[N_obs])+0.05*np.nanmean(wf_kuband[N_obs]), np.nanmax(wf_kuband[N_obs])-0.05*np.nanmean(
        wf_kuband[N_obs])), ylim=(time_conv[N_obs][idx[-1]], time_conv[N_obs][idx[0]]), ultitle='Ka-band'.format(fn[0:2]))
    ax.xaxis.labelpad = 15

    #ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
    #    range_40[N_obs]*c/2)+offset_ka+offset_Ku_Ka, color=cmap_qual[0], marker=4, s=30, label='')
    #ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
    #    range_50[N_obs]*c/2)+offset_ka+offset_Ku_Ka, color=cmap_qual[6], marker=4, s=30, label='')
    #ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
    #    range_80[N_obs]*c/2)+offset_ka+offset_Ku_Ka, color=cmap_qual[1], marker=4, s=30, label='')
    #ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=centroid +
    #           offset_Ku_Ka, color='orange', marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=time_conv[N_obs][np.nanargmax(
        wf_kuband[N_obs][:])]+offset_Ku_Ka, color='brown', marker=4, s=30, label='')
    
    as_cwt = ds.variables['range_air_snow_cwt_TN'][:]
    si_cwt = ds.variables['range_snow_ice_cwt_TN'][:]
    as_peak = ds.variables['range_air_snow_peakiness'][:]
    si_peak = ds.variables['range_snow_ice_peakiness'][:]
    
    leg1 = ax.axhline(y=(as_cwt[N_obs]*c/2)+offset_ka+offset_Ku_Ka,
                      color=cmap_qual[6], linestyle='-', linewidth=1, label='TFMRA50')
    leg2 = ax.axhline(y=(si_cwt[N_obs]*c/2)+offset_ka+offset_Ku_Ka,
                      color=cmap_qual[5], linestyle='--', linewidth=1, label='TFMRA80')

    leg3 = ax.axhline(y=(as_peak[N_obs]*c/2)+offset_ka+offset_Ku_Ka,
                      color=cmap_qual[0], linestyle='-', linewidth=1, label='TFMRA50')
    leg4 = ax.axhline(y=(si_peak[N_obs]*c/2)+offset_ka+offset_Ku_Ka,
                      color=cmap_qual[1], linestyle='--', linewidth=1, label='TFMRA80')

    elevation = ds.variables['elevation'][:]
    leg0 = ax.axhline(
        y=elevation[N_obs]-ALS_along_radar_track[N_obs]+offset_Ku_Ka, c='green')

    handle0 = r'$h_{}$ = {:.2f} m'.format(
        'R', elevation[N_obs]-ALS_along_radar_track[N_obs]+offset_Ku_Ka)
    handle1 = r'$h_{}$ = {:.2f} m'.format(
        'R', (range_40[N_obs]*c/2)+offset_ka+offset_Ku_Ka)
    handle2 = r'$h_{}$ = {:.2f} m'.format(
        'R', (range_50[N_obs]*c/2)+offset_ka+offset_Ku_Ka)
    handle3 = r'$h_{}$ = {:.2f} m'.format(
        'R', (range_80[N_obs]*c/2)+offset_ka+offset_Ku_Ka)
    #handle4 = r'$h_{}$ = {:.2f} m'.format('R', centroid+offset_Ku_Ka)
    handle5 = r'$h_{}$ = {:.2f} m'.format(
        'R', time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])]+offset_Ku_Ka)
    #handle6 = r'$h_{}$ = {:.2f} m'.format(
     #   '{s,MaxC}', (centroid-time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])])/params["n_snow"])
    frame = ax.legend([leg0, leg1, leg2, leg3, leg5], [handle0, handle1, handle2, handle3, handle5],
                      loc='lr', handlelength=0, markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, prop={'size': 7})

    ax = axs.panel_axes('r', width=1, space=0)
    ax.patch.set_facecolor('lightgrey')
    fn = 'snow_20221213_02_74_232_002'
    ds = netCDF4.Dataset(
        r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
    wf_kuband = ds.variables['waveform'][:]
    time = ds.variables['two_way_travel_time'][:]
    range_40 = ds.variables['range_tfmra40'][:]
    range_50 = ds.variables['range_tfmra50'][:]
    range_80 = ds.variables['range_tfmra80'][:]

    as_cwt = ds.variables['range_air_snow_cwt_TN'][:]
    si_cwt = ds.variables['range_snow_ice_cwt_TN'][:]
    as_peak = ds.variables['range_air_snow_peakiness'][:]
    si_peak = ds.variables['range_snow_ice_peakiness'][:]
    time_conv = (time*c/2)+offset_snow+offset_Ku_snow
    # wf_kuband = 10 * np.log10(wf_kuband)
    idx = np.arange(np.nanargmax(
        wf_kuband[N_obs][:])-100, np.nanargmax(wf_kuband[N_obs][:])+156)

    # ax.axhline(y=(range_40[N_obs]*c/2)+offset_snow, color='green', linestyle='-', linewidth=1, label='TFMRA40')
    # ax.axhline(y=(range_50[N_obs]*c/2)+offset_snow, color='yellow', linestyle='--', linewidth=1, label='TFMRA50')
    # ax.axhline(y=(range_80[N_obs]*c/2)+offset_snow, color='magenta', linestyle='--', linewidth=1, label='TFMRA80')

    leg1 = ax.axhline(y=(as_cwt[N_obs]*c/2)+offset_snow+offset_Ku_snow,
                      color=cmap_qual[6], linestyle='-', linewidth=1, label='TFMRA50')
    leg2 = ax.axhline(y=(si_cwt[N_obs]*c/2)+offset_snow+offset_Ku_snow,
                      color=cmap_qual[5], linestyle='--', linewidth=1, label='TFMRA80')

    leg3 = ax.axhline(y=(as_peak[N_obs]*c/2)+offset_snow+offset_Ku_snow,
                      color=cmap_qual[0], linestyle='-', linewidth=1, label='TFMRA50')
    leg4 = ax.axhline(y=(si_peak[N_obs]*c/2)+offset_snow+offset_Ku_snow,
                      color=cmap_qual[1], linestyle='--', linewidth=1, label='TFMRA80')

    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        as_cwt[N_obs]*c/2)+offset_snow+offset_Ku_snow, color=cmap_qual[6], marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        si_cwt[N_obs]*c/2)+offset_snow+offset_Ku_snow, color=cmap_qual[5], marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        as_peak[N_obs]*c/2)+offset_snow+offset_Ku_snow, color=cmap_qual[0], marker=4, s=30, label='')
    ax.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=(
        si_peak[N_obs]*c/2)+offset_snow+offset_Ku_snow, color=cmap_qual[1], marker=4, s=30, label='')

    elevation = ds.variables['elevation'][:]
    leg0 = ax.axhline(
        y=elevation[N_obs]-ALS_along_radar_track[N_obs]+offset_Ku_snow, c='green')

    ax.plot(wf_kuband[N_obs-5:N_obs+5][:, idx], time_conv[N_obs -
            5:N_obs+5][:, idx]+offset_Ku_snow, zorder=0, c='grey')
    leg5 = ax.plot(wf_kuband[N_obs][idx], time_conv[N_obs]
                   [idx]+offset_Ku_snow, zorder=0, c='k', linewidth=0.5)
    # ax.scatter(wf_kuband[N_obs][np.nanargmax(wf_kuband[N_obs][:])], time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])],marker='*', s=10, zorder=2)
    # ax.format(ultitle='$\delta$range={:.2f} m'.format((centroid-time_conv[N_obs][peaks[0][0]+idx[0]])))
    # %,title='$\delta$range={:.2f} m\nsd = {:.2f} m'.format((centroid-time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])]), ((centroid-time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])])/params["n_snow"]))

    handle0 = r'$h_{}$ = {:.2f} m'.format(
        'R', elevation[N_obs]-ALS_along_radar_track[N_obs]+offset_Ku_snow)
    handle1 = r'$h_{}$ = {:.2f} m'.format(
        'R', (as_cwt[N_obs]*c/2)+offset_snow+offset_Ku_snow)
    handle2 = r'$h_{}$ = {:.2f} m'.format(
        'R', (si_cwt[N_obs]*c/2)+offset_snow+offset_Ku_snow)
    handle3 = r'$h_{}$ = {:.2f} m'.format(
        'R', (as_peak[N_obs]*c/2)+offset_snow+offset_Ku_snow)
    handle4 = r'$h_{}$ = {:.2f} m'.format(
        'R', (si_peak[N_obs]*c/2)+offset_snow+offset_Ku_snow)
    handle5 = r'$h_{}$ = {:.2f} m'.format(
        '{s,CWT}', ((si_cwt[N_obs]*c/2)-(as_cwt[N_obs]*c/2))/params["n_snow"])
    handle6 = r'$h_{}$ = {:.2f} m'.format(
        '{s,peak}', ((si_peak[N_obs]*c/2)-(as_peak[N_obs]*c/2))/params["n_snow"])
    frame = ax.legend([leg0, leg1, leg2, leg3, leg4, ], [handle0, handle1, handle2, handle3, handle4],
                      loc='lr', handlelength=0, markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, prop={'size': 7})
    frame = ax.legend([leg5, leg5, leg5 ], ['', handle5, handle6],
                      loc='ul', handlelength=0, markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, prop={'size': 7})

    ax.format(xlim=(np.nanmin(wf_kuband[N_obs])+0.05*np.nanmean(wf_kuband[N_obs]), np.nanmax(
        wf_kuband[N_obs])-0.05*np.nanmean(wf_kuband[N_obs])), ultitle='C/S-band'.format(fn[0:2]))
    
    return idx



c = 300e6

fig, axs = pplt.subplots([[1]], axwidth=1, axheight=3, sharex=False, sharey=False)
fig.patch.set_facecolor('white')
# n = 300
# n1=8
# val_x = 300

n = 300
n1 = 70
# n1=8
val_x = 1000

#n0_extra = 880
#idx = plot_waveforms_examples(n0_extra, axs[0], '', ALS_along_radar_track)

n1_extra = 180
idx = plot_waveforms_examples(
    n1_extra, axs[0], '', ALS_along_radar_track)


leg1 = Line2D([0], [0], color=cmap_qual[0], linestyle='-',
              label='TFMRA40', marker=8, markersize=10)
leg2 = Line2D([0], [0], color=cmap_qual[6],  linestyle='--',
              label='TFMRA50', marker=8, markersize=10)
leg3 = Line2D([0], [0], color=cmap_qual[1],  linestyle='--',
              label='TFMRA80', marker=8, markersize=10)
#leg4 = Line2D([0], [0], color='orange',  linestyle='-',
#              label='Centroid', marker=8, markersize=10)
leg5 = Line2D([0], [0], color='brown', ls='--',
              label='Max peak', marker=8, markersize=10)
leg6 = Line2D([0], [0], color='green', linestyle='-', label='ALS')

handles = [leg1, leg2, leg3,  leg5, leg6]
frame = axs[0].legend(handles, loc='t', ncols=3, title='Ka/Ku-band',
                      align='left', titlefontweight='bold', pad=1.)
frame._legend_box.align = "left"
frame.get_title().set_weight('bold')

leg1 = Line2D([0], [0], color=cmap_qual[6], linestyle='-',
              label='a-s CWT', marker=8, markersize=10)
leg2 = Line2D([0], [0], color=cmap_qual[5],  linestyle='--',
              label='s-i CWT', marker=8, markersize=10)
leg3 = Line2D([0], [0], color=cmap_qual[0],  linestyle='-',
              label='a-s peakiness', marker=8, markersize=10)
leg4 = Line2D([0], [0], color=cmap_qual[1],  linestyle='--',
              label='s-i peakiness', marker=8, markersize=10)

handles = [leg1, leg2, leg3, leg4, leg6]
frame = axs[0].legend(handles, loc='b', ncols=3, order='F',
                      title='C/S-band', align='left', titlefontweight='bold', pad=1)
frame._legend_box.align = "left"
frame.get_title().set_weight('bold')
frame.get_frame().set_facecolor('lightgrey')

fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\Figure3_waveform_example_arttu.png', dpi=300)



#%%
fn = 'kuband_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
wf_kuband = ds.variables['waveform'][:]

ppk_ku = ppk(wf_kuband, 10)
max_ku = max_power(wf_kuband)
ppk_l_ku, ppk_r_ku = ppk_left_right(wf_kuband)

#%%

fn = 'kuband_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
elevation_ellipsoid_tfmra40 = ds.variables['elevation_ellipsoid_tfmra40'][:]-offset_ku
elevation_ellipsoid_tfmra50 = ds.variables['elevation_ellipsoid_tfmra50'][:]-offset_ku
elevation_ellipsoid_tfmra80 = ds.variables['elevation_ellipsoid_tfmra80'][:]-offset_ku
flight_alt = ds.variables['elevation']
elevation_ellipsoid_as_peakiness_ku = ds.variables['elevation_air_snow_ellipsoid_peakiness'][:]-offset_ku
elevation_ellipsoid_si_peakiness_ku = ds.variables['elevation_snow_iceellipsoid_peakiness'][:]-offset_ku
elevation_ellipsoid_as_cwt_ku = ds.variables['elevation_air_snow_ellipsoid_cwt_TN'][:]-offset_ku
elevation_ellipsoid_si_cwt_ku = ds.variables['elevation_snow_ice_ellipsoid_cwt_TN'][:]-offset_ku

df_Ku_ALS_diff = pd.DataFrame({'ALS-TFMRA40':ALS_along_radar_track-elevation_ellipsoid_tfmra40,
                              'ALS-TFMRA50':ALS_along_radar_track-elevation_ellipsoid_tfmra50,
                              'ALS-TFMRA80':ALS_along_radar_track-elevation_ellipsoid_tfmra80, 
                              'ALS':ALS_along_radar_track, 
                              'TFMRA40':elevation_ellipsoid_tfmra40, 
                              'TFMRA50':elevation_ellipsoid_tfmra50, 
                              'TFMRA80':elevation_ellipsoid_tfmra80, 
                              'WF_max':a_s_peaks_t, 
                              'WF_centroid':s_i_centroids_t,
                              'WF_max_corr':flight_alt - a_s_peaks_t - offset_ku,
                              'WF_centroid_corr':flight_alt - a_s_peaks_t - offset_ku,
                              'as_CWT':elevation_ellipsoid_as_cwt_ku,
                              'si_CWT':elevation_ellipsoid_si_cwt_ku, 
                              'as_PEAK':elevation_ellipsoid_as_peakiness_ku,
                              'si_PEAK':elevation_ellipsoid_si_peakiness_ku,
})



fn = 'kaband_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
elevation_ellipsoid_tfmra40 = ds.variables['elevation_ellipsoid_tfmra40'][:]-offset_ka
elevation_ellipsoid_tfmra50 = ds.variables['elevation_ellipsoid_tfmra50'][:]-offset_ka
elevation_ellipsoid_tfmra80 = ds.variables['elevation_ellipsoid_tfmra80'][:]-offset_ka
elevation_ellipsoid_as_peakiness_ka = ds.variables['elevation_air_snow_ellipsoid_peakiness'][:]-offset_ka
elevation_ellipsoid_si_peakiness_ka = ds.variables['elevation_snow_iceellipsoid_peakiness'][:]-offset_ka
elevation_ellipsoid_as_cwt_ka = ds.variables['elevation_air_snow_ellipsoid_cwt_TN'][:]-offset_ka
elevation_ellipsoid_si_cwt_ka = ds.variables['elevation_snow_ice_ellipsoid_cwt_TN'][:]-offset_ka

df_Ka_ALS_diff = pd.DataFrame({'ALS-TFMRA40':ALS_along_radar_track-elevation_ellipsoid_tfmra40,
                              'ALS-TFMRA50':ALS_along_radar_track-elevation_ellipsoid_tfmra50,
                              'ALS-TFMRA80':ALS_along_radar_track-elevation_ellipsoid_tfmra80, 
                              'ALS':ALS_along_radar_track, 
                              'TFMRA40':elevation_ellipsoid_tfmra40, 
                              'TFMRA50':elevation_ellipsoid_tfmra50, 
                              'TFMRA80':elevation_ellipsoid_tfmra80, 
                              'WF_max':a_s_peaks_t_ka, 
                              'WF_centroid':s_i_centroids_t_ka,
                              'WF_max_corr':flight_alt - a_s_peaks_t_ka - offset_ka,
                              'WF_centroid_corr':flight_alt - a_s_peaks_t_ka - offset_ka,
                              'as_CWT':elevation_ellipsoid_as_cwt_ka,
                              'si_CWT':elevation_ellipsoid_si_cwt_ka, 
                              'as_PEAK':elevation_ellipsoid_as_peakiness_ka,
                              'si_PEAK':elevation_ellipsoid_si_peakiness_ka,
})


fn = 'snow_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
elevation_ellipsoid_as_peakiness = ds.variables['elevation_air_snow_ellipsoid_peakiness'][:]-offset_snow
elevation_ellipsoid_si_peakiness = ds.variables['elevation_snow_iceellipsoid_peakiness'][:]-offset_snow
elevation_ellipsoid_as_cwt = ds.variables['elevation_air_snow_ellipsoid_cwt_TN'][:]-offset_snow
elevation_ellipsoid_si_cwt = ds.variables['elevation_snow_ice_ellipsoid_cwt_TN'][:]-offset_snow

df_snow_ALS_diff = pd.DataFrame({'ALS-as_CWT':ALS_along_radar_track-elevation_ellipsoid_as_cwt,
                              'ALS-si_CWT':ALS_along_radar_track-elevation_ellipsoid_si_cwt,
                              'ALS-as_PEAK':ALS_along_radar_track-elevation_ellipsoid_as_peakiness,
                              'ALS-si_PEAK':ALS_along_radar_track-elevation_ellipsoid_si_peakiness, 
                              'ALS':ALS_along_radar_track, 
                              'as_CWT':elevation_ellipsoid_as_cwt,
                              'si_CWT':elevation_ellipsoid_si_cwt, 
                              'as_PEAK':elevation_ellipsoid_as_peakiness,
                              'si_PEAK':elevation_ellipsoid_si_peakiness,
                              'WF_max':a_s_peaks_t_snow, 
                              'WF_centroid':s_i_centroids_t_snow,
                              'WF_max_corr':flight_alt - a_s_peaks_t_snow - offset_snow,
                              'WF_centroid_corr':flight_alt - a_s_peaks_t_snow - offset_snow,
                                  
})

#%%

df_test = df_Ku_ALS_diff[(df_Ku_ALS_diff['ALS']-df_Ku_ALS_diff['WF_max_corr']>-0.1) & (df_Ku_ALS_diff['ALS']-df_Ku_ALS_diff['WF_max_corr']<0.1)]
df_test2 = df_test[(df_test['ALS']-df_test['TFMRA50']>-0.1) & (df_test['ALS']-df_test['TFMRA50']<0.1)]
print(len(df_test2)/len(df_test)*100)

# %%
num_rec = 8
min_val, max_val = np.zeros(num_rec), np.zeros(num_rec)
min_val[0], max_val[0] = -65.1901, -65.1908
# min_val[1], max_val[1] = -68.4021, -68.4034
min_val[2], max_val[2] = -68.4051, -68.4059
min_val[3], max_val[3] = -68.4087, -68.4094
min_val[4], max_val[4] = -69.3859, -69.3875
min_val[5], max_val[5] = -69.3892, -69.3907
# min_val[6], max_val[6] = -70.7162, -70.717
min_val[6], max_val[6] = -70.7583, -70.7589
min_val[7], max_val[7] = -69.9682, -69.9696

idx_leads_comb = []

for i in np.arange(0, num_rec):
    idx_lat_Ku = np.where(
        (latitude_comb < min_val[i]) & (latitude_comb > max_val[i]))
    
    idx_leads_comb = np.append(idx_lat_Ku, idx_leads_comb)
    
#%%
def normalize_nan(array):
    return ((array-np.nanmin(array))/(np.nanmax(array)-np.nanmin(array)))

ppk_ku_norm = normalize_nan(ppk_ku)
max_ku_norm = normalize_nan(max_ku)
ppk_r_ku_norm = normalize_nan(ppk_r_ku)
ppk_l_ku_norm = normalize_nan(ppk_l_ku)
idx_leads_comb = idx_leads_comb.astype(int)
ppk_leads = ppk_ku_norm[idx_leads_comb]
max_leads = max_ku_norm[idx_leads_comb]
ppk_l_leads = ppk_l_ku_norm[idx_leads_comb]
ppk_r_leads = ppk_r_ku_norm[idx_leads_comb]



#%% Figure 3 - THE CRYOSPHERE


#fig, ax = pplt.subplots([[1, 1, 2, 3, 3, 4, 4, 5, 5], 
#                         [1, 1, 2, 6, 6, 7, 7, 8, 8]], axwidth=2, axheight=5, sharex=False, sharey=False)

fig, ax = pplt.subplots([[1, 1, 2], 
                         [1, 1, 2], 
                         [1, 1, 2],
                         [3, 3, 3],
                         [3, 3, 3]], axwidth=3, axheight=6.5, sharex=False, sharey=False, proj={1: 'laea'}, proj_kw={'lat_0': -70, 'lon_0': -45})

fig.patch.set_facecolor('white')
#ax1 = ax[1].panel_axes('r', width=1.3, space=0)

n = 300
n1 = 70
val_x = 1000

data_ALS_scn_filt = data_ALS_scn[(data_ALS_scn[1] < np.max(
    latitude_comb[val_x*n1:val_x*(n1+1)])) & (data_ALS_scn[1] > np.min(latitude_comb[val_x*n1:val_x*(n1+1)]))]

axs = ax[0]
#xlim_extent = (np.min(longitude_comb[val_x*n1:val_x*(n1+1)]) -
#               0.005, np.max(longitude_comb[val_x*n1:val_x*(n1+1)])+0.005)
#ylim_extent = (np.min(latitude_comb[val_x*n1:val_x*(n1+1)]) -
#               0.005, np.max(latitude_comb[val_x*n1:val_x*(n1+1)])+0.005)

xlim_extent = (np.min(longitude_comb[val_x*n1:val_x*(n1+1)]), np.max(longitude_comb[val_x*n1:val_x*(n1+1)]))
ylim_extent = (np.min(latitude_comb[val_x*n1:val_x*(n1+1)]), np.max(latitude_comb[val_x*n1:val_x*(n1+1)]))
axs.set_extent([xlim_extent[0], xlim_extent[1], ylim_extent[1],
              ylim_extent[0]], crs=ccrs.PlateCarree())
im = axs.scatter(data_ALS_scn_filt[2], data_ALS_scn_filt[1], c=data_ALS_scn_filt[3], vmin=np.floor(np.quantile(
    data_ALS_scn_filt[3], 0.1)), vmax=np.ceil(np.quantile(data_ALS_scn_filt[3], 0.9)), cmap='crest', s=0.01, label='')
cb = axs.colorbar(im, label='ALS ellipsoidal heights WGS84 (m)', loc='b', length=0.8, labelsize=14)
cb.ax.tick_params(labelsize=14)
axs.scatter(lon_ALS_along_radar_track, lat_ALS_along_radar_track, s=0.05, c='green')
axs.set(frame_on=False)
axs.format(grid=False)

leg_ALS = Line2D([0], [0], color='green', linestyle='-',
              label='Nadir ALS profile')
axs.legend(leg_ALS, loc='ll', frameon=False, prop=dict(size=12))

fontprops = fm.FontProperties(size=14)
scalebar = AnchoredSizeBar(axs.transData,
                           400, '400 m', 'upper right',
                           pad=0.70,
                           sep=5,
                           color='black',
                           frameon=False,
                           size_vertical=4.0)
axs.add_artist(scalebar)         

axs = ax[2]

data = ppk_ku_norm
CDE = np.arange(len(data)) / float(len(data))
ori_data = ppk_ku
sorted_data = np.sort(data)
CDE_mix = np.floor(sorted_data[np.where(CDE>0.6)[0][0]]*(np.nanmax(ori_data)-np.nanmin(ori_data)) + np.nanmin(ori_data))
ax[1].axvline(sorted_data[np.where(CDE>0.60)[0][0]], c=cmap_qual[2], linestyle='--', linewidth=1)

ax[2].plot([0,sorted_data[np.where(CDE>0.60)[0][0]]], [0.6,0.6], c='lightgrey', linestyle='-.', linewidth=1)
ax[2].plot([sorted_data[np.where(CDE>0.60)[0][0]],sorted_data[np.where(CDE>0.60)[0][0]]], [0,0.6], c='lightgrey', linestyle='-.', linewidth=1)


import matplotlib.patches as patches
#rect = patches.Rectangle((sorted_data[np.where(CDE>0.60)[0][0]], 0), 1-sorted_data[np.where(CDE>0.60)[0][0]]-sorted_data[np.where(CDE_leads>0.10)[0][0]], 1.3, facecolor='green', zorder=2)
#ax[2].add_patch(rect)

ori_data = ppk_ku
axs.plot(np.sort(data) ,np.arange(len(data)) / float(len(data)) , c=cmap_qual[0], linestyle='-')
data = ppk_ku_norm[idx_leads_comb]
CDE_leads = np.arange(len(data)) / float(len(data))
CDE_leads_min = np.nanmin(ori_data)
CDE_leads_max = np.nanmax(ori_data)
sorted_data = np.sort(data)
CDE_leads_50 =  np.floor(sorted_data[np.where(CDE_leads>0.10)[0][0]]*(np.nanmax(ori_data)-np.nanmin(ori_data)) + np.nanmin(ori_data))
CDE_leads_50_ppk =  sorted_data[np.where(CDE_leads>0.10)[0][0]]*(np.nanmax(ori_data)-np.nanmin(ori_data)) + np.nanmin(ori_data)
leg1 = axs.plot(np.sort(data) ,np.arange(len(data)) / float(len(data)) , c=cmap_qual[0], linestyle='--', label='min{} = {:.2f}\nmax{} = {:.2f}\n>10%{} = {:.2f}\n>60%{}={:.2f}'.format('$_{PP}$',CDE_leads_min, '$_{PP}$', CDE_leads_max, '$_{PP, leads}$', CDE_leads_50,'$_{PP, mixed}$',CDE_mix))
#plt.hist(ppk_ku_norm, cumulative=True, density=True, histtype='step')
ax[1].axvline(sorted_data[np.where(CDE_leads>0.10)[0][0]], c=cmap_qual[0], linestyle='--', linewidth=1)

ax[2].plot([0,sorted_data[np.where(CDE_leads>0.10)[0][0]]], [0.1,0.1], c='grey', linestyle='-.', linewidth=1)
ax[2].plot([sorted_data[np.where(CDE_leads>0.10)[0][0]],sorted_data[np.where(CDE_leads>0.10)[0][0]]], [0,0.1], c='grey', linestyle='-.', linewidth=1)




#rect = patches.Rectangle((sorted_data[np.where(CDE_leads>0.10)[0][0]], 0), 1-sorted_data[np.where(CDE_leads>0.10)[0][0]], 1.3, facecolor='grey', zorder=1)
#ax[2].add_patch(rect)





#data = ppk_l_ku_norm
#ori_data = ppk_l_ku
#axs.plot(np.sort(data) ,np.arange(len(data)) / float(len(data)) , c='dodgerblue', linestyle='-')
#data = ppk_l_ku_norm[idx_leads_comb]
#CDE_leads = np.arange(len(data)) / float(len(data))
#CDE_leads_min = np.nanmin(ori_data)
#CDE_leads_max = np.nanmax(ori_data)
#sorted_data = np.sort(data)
#CDE_leads_50 =  sorted_data[np.where(CDE_leads>0.3)[0][0]]*(np.nanmax(ori_data)-np.nanmin(ori_data)) + np.nanmin(ori_data)
#CDE_leads_50_ppk_l =  sorted_data[np.where(CDE_leads>0.3)[0][0]]*(np.nanmax(ori_data)-np.nanmin(ori_data)) + np.nanmin(ori_data)
#leg2 = axs.plot(np.sort(data) ,np.arange(len(data)) / float(len(data)) , c='dodgerblue', linestyle='--', label='min{} = {:.2f}\nmax{} = {:.2E}\n>30%{} = {:.2E}'.format('$_{PP_l}$',CDE_leads_min, '$_{PP_l}$',CDE_leads_max, '$_{PP_l, leads}$',CDE_leads_50))
#plt.hist(ppk_ku_norm, cumulative=True, density=True, histtype='step')
#ax1.axvline(sorted_data[np.where(CDE_leads>0.3)[0][0]], c='dodgerblue', linestyle='--', linewidth=1)


data = max_ku_norm
ori_data = max_ku
axs.plot(np.sort(data) ,np.arange(len(data)) / float(len(data)) , c=cmap_qual[6], linestyle='-')
data = max_ku_norm[idx_leads_comb]
CDE_leads = np.arange(len(data)) / float(len(data))
CDE_leads_min = np.nanmin(ori_data)
CDE_leads_max = np.nanmax(ori_data)
sorted_data = np.sort(data)
CDE_leads_50 =  np.floor(sorted_data[np.where(CDE_leads>0.3)[0][0]]*(np.nanmax(ori_data)-np.nanmin(ori_data)) + np.nanmin(ori_data))
CDE_leads_50_max =  sorted_data[np.where(CDE_leads>0.3)[0][0]]*(np.nanmax(ori_data)-np.nanmin(ori_data)) + np.nanmin(ori_data)
leg3 = axs.plot(np.sort(data) ,np.arange(len(data)) / float(len(data)) , c=cmap_qual[6], linestyle='--', label='min{} = {:.2E}\nmax{} = {:.2E}'.format('$_{MAX}$', CDE_leads_min, '$_{MAX}$', CDE_leads_max))
#frame=axs.legend([leg1, leg3], loc='b',  handlelength=0,  markersize=0, linewidth=0, ncols=3, labelcolor='linecolor', alpha=0.5)
axs.legend(leg1, loc='lr',  handlelength=0,  markersize=0, linewidth=0, ncols=3, labelcolor='linecolor', alpha=0.5, frameon=False)
axs.legend(leg3, loc='ur',  handlelength=0,  markersize=0, linewidth=0, ncols=3, labelcolor='linecolor', alpha=0.5, frameon=False)
ax[1].axvline(sorted_data[np.where(CDE_leads>0.3)[0][0]], c=cmap_qual[6], linestyle='--', linewidth=1)
                              

class_leads_seaice = np.zeros(len(ppk_ku))

for i in np.arange(0, len(ppk_ku)):
    if ppk_ku[i] > CDE_mix:     
        if ppk_ku[i] > CDE_leads_50_ppk:
      #      if ppk_l_ku[i] > CDE_leads_50_ppk_l:
                #if max_ku[i] > CDE_leads_50_max:
                    class_leads_seaice[i] = 1
        else: 
            class_leads_seaice[i] = 2
#     if ppk_ku[i] > CDE_leads_50_ppk:
#                 class_leads_seaice[i] = 1               

axs.format(ultitle='Lead classification\n{:.2f}%, mixed: {:.2f}%'.format(len(class_leads_seaice[class_leads_seaice==1])/len(class_leads_seaice)*100, len(class_leads_seaice[class_leads_seaice==2])/len(class_leads_seaice)*100), xlabel='Normalised waveform parameter', ylabel='Cumulative probability', ylim=(0,1.1))


axs = ax[1]

axs.format(ylabel='latitude (degrees N)')

fn = 'kaband_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
lat_extra = ds.variables['lat'][val_x*n1:val_x*(n1+1)]

sort_class = class_leads_seaice[val_x*n1:val_x*(n1+1)]
for i in np.arange(len(sort_class)):
    if sort_class[i] == 1: 
        axs.axhline(lat_extra[i], c='k', alpha=0.2)
    elif sort_class[i] == 2: 
        axs.axhline(lat_extra[i], c='lightgrey', alpha=0.8)

axs.plot(normalize_nan(ppk_ku[val_x*n1:val_x*(n1+1)]), lat_extra, linewidth=0.5, c=cmap_qual[0], label='PP')
#ax1.plot(max_ku[val_x*n1:val_x*(n1+1)], lat_extra, linewidth=0.5)
#ax1.plot(ppk_l_ku[val_x*n1:val_x*(n1+1)], lat_extra, linewidth=0.5)
#ax1.plot(ppk_r_ku[val_x*n1:val_x*(n1+1)], lat_extra, linewidth=0.5)
axs.plot(normalize_nan(max_ku[val_x*n1:val_x*(n1+1)]), lat_extra, linewidth=0.5, c=cmap_qual[6], label='MAX')
#ax1.plot(normalize_nan(ppk_l_ku[val_x*n1:val_x*(n1+1)]), lat_extra, linewidth=0.5, c='dodgerblue', label='PP$_l$')
#ax1.xaxis.tick_top()
#ax1.xaxis.set_label_position('top') 
axs.format(xlabel='Normalised\nparams.')

frame = axs.legend(loc='ur', handlelength=0,  markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False,)
axs.tick_params(labelsize=14)
#axs.set_xlabel('Normalised\nwaveform\nparameters', fontsize=12)
axs.set_ylabel('latitude degrees (N)',fontsize=14)

axs.text(0.52, -69.995, 'PP$_{}$ threshold'.format('{lead}'), c=cmap_qual[0], rotation='vertical')
axs.text(0.28, -69.995, 'PP$_{}$ threshold'.format('{mixed}'), c=cmap_qual[2], rotation='vertical')


fig.format(abc='(a)', abcloc='ul', fontsize=14)
fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\Figure3_new.png', dpi=300)


#%% Figure 4


#fig, ax = pplt.subplots([[1, 1, 2, 3, 3, 4, 4, 5, 5], 
#                         [1, 1, 2, 6, 6, 7, 7, 8, 8]], axwidth=2, axheight=5, sharex=False, sharey=False)

fig, ax = pplt.subplots([[1,2,3], 
                         [4,5,6],
                         [7, 8, 9]], axwidth=2.8, axheight=2.5, sharex=False, sharey=False)

fig.patch.set_facecolor('white')


axs = ax[0]

n_bins = 20
binwidth=0.2
n_bins = np.arange(-3, 2+ binwidth, binwidth)
df = df_Ku_ALS_diff[class_leads_seaice<1]
axs.hist(df['ALS-TFMRA50'], c=cmap_qual[1], edgecolor='k', alpha=1, bins=n_bins, label='{:.2f} $\pm$ {:.2f} m\n5-95% = {:.2f}-{:.2f} m'.format(np.nanmean(df['ALS-TFMRA50']), np.nanstd(df['ALS-TFMRA50']), np.nanquantile(df['ALS-TFMRA50'], 0.05), np.nanquantile(df['ALS-TFMRA50'], 0.95)))
axs.hist(df['ALS-TFMRA40'], c=cmap_qual[0], edgecolor=cmap_qual[0], alpha=0.4, bins=n_bins, histtype='stepfilled', label='{:.2f} $\pm$ {:.2f} m\n5-95% = {:.2f}-{:.2f} m'.format(np.nanmean(df['ALS-TFMRA40']), np.nanstd(df['ALS-TFMRA40']), np.nanquantile(df['ALS-TFMRA40'], 0.05), np.nanquantile(df['ALS-TFMRA40'], 0.95)))
axs.hist(df['ALS-TFMRA80'], c=cmap_qual[2], edgecolor=cmap_qual[2], alpha=0.4, bins=n_bins, histtype='stepfilled', label='{:.2f} $\pm$ {:.2f} m\n5-95% = {:.2f}-{:.2f} m'.format(np.nanmean(df['ALS-TFMRA80']), np.nanstd(df['ALS-TFMRA80']), np.nanquantile(df['ALS-TFMRA80'], 0.05), np.nanquantile(df['ALS-TFMRA80'], 0.95)))
val = df['ALS']-df['WF_max_corr']
axs.hist(val, c=cmap_qual[4], edgecolor='k', alpha=0.8, bins=n_bins, histtype='stepfilled', hatch='///',label='{:.2f} $\pm$ {:.2f} m\n5-95% = {:.2f}-{:.2f} m'.format(np.nanmean(val[(val<5)&(val>-5)]), np.nanstd(val[(val<5)&(val>-5)]), np.nanquantile(val[(val<5)&(val>-5)], 0.05), np.nanquantile(val[(val<5)&(val>-5)], 0.95)))

#frame=axs.legend(loc='ll', ncols=1, handlelength=0.7, title='Ku-band', )
frame = axs.legend(loc='ll', handlelength=0,  markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False,)
frame.get_title().set_weight('bold')
axs.format(xlabel='$\Delta$ALS-TFMRA$_{corr}$ (m)', ultitle='Ku-band (floes)\n$\Delta$Ku = {:.2f} m'.format(offset_ku), xlim=(-3, 2),  ylabel='frequency (counts)', ylim=(0,40000))

axs.axvline(np.nanmean(df['ALS-TFMRA50']), c=cmap_qual[1], linestyle='--', alpha=0.8)
axs.axvline(np.nanmean(df['ALS-TFMRA40']), c=cmap_qual[0], linestyle='--', alpha=0.4)
axs.axvline(np.nanmean(df['ALS-TFMRA80']), c=cmap_qual[2], linestyle='--', alpha=0.5)
axs.axvline(np.nanmean(val[(val<5)&(val>-5)]), c=cmap_qual[4], linestyle='--', alpha=0.8)
print(np.nanmean(val))

axs = ax[1]

n_bins = 20
binwidth=0.2
n_bins = np.arange(-3, 2+ binwidth, binwidth)
df = df_Ka_ALS_diff[class_leads_seaice<1]
leg1=axs.hist(df['ALS-TFMRA50'], c=cmap_qual[1], edgecolor='k', alpha=1, bins=n_bins, label='{:.2f} $\pm$ {:.2f} m\n5-95% = {:.2f}-{:.2f} m'.format(np.nanmean(df['ALS-TFMRA50']), np.nanstd(df['ALS-TFMRA50']), np.nanquantile(df['ALS-TFMRA50'], 0.05), np.nanquantile(df['ALS-TFMRA50'], 0.95)))
leg2=axs.hist(df['ALS-TFMRA40'], c=cmap_qual[0], edgecolor=cmap_qual[0], alpha=0.4, bins=n_bins, histtype='stepfilled', label='{:.2f} $\pm$ {:.2f} m\n5-95% = {:.2f}-{:.2f} m'.format(np.nanmean(df['ALS-TFMRA40']), np.nanstd(df['ALS-TFMRA40']), np.nanquantile(df['ALS-TFMRA40'], 0.05), np.nanquantile(df['ALS-TFMRA40'], 0.95)))
leg3=axs.hist(df['ALS-TFMRA80'], c=cmap_qual[2], edgecolor=cmap_qual[2], alpha=0.4, bins=n_bins, histtype='stepfilled', label='{:.2f} $\pm$ {:.2f} m\n5-95% = {:.2f}-{:.2f} m'.format(np.nanmean(df['ALS-TFMRA80']), np.nanstd(df['ALS-TFMRA80']), np.nanquantile(df['ALS-TFMRA80'], 0.05), np.nanquantile(df['ALS-TFMRA80'], 0.95)))
#frame=axs.legend(loc='ll', ncols=1, handlelength=0.7, title='Ku-band', )
#frame=axs.legend(loc='ll', ncols=1, handlelength=0.7, title='Ka-band')
val = df['ALS']-df['WF_max_corr']
axs.hist(val, c=cmap_qual[4], edgecolor='k', alpha=0.8, bins=n_bins, histtype='stepfilled', hatch='///',label='{:.2f} $\pm$ {:.2f} m\n5-95% = {:.2f}-{:.2f} m'.format(np.nanmean(val[(val<5)&(val>-5)]), np.nanstd(val[(val<5)&(val>-5)]), np.nanquantile(val[(val<5)&(val>-5)], 0.05), np.nanquantile(val[(val<5)&(val>-5)], 0.95)))

frame = axs.legend(loc='ll', handlelength=0,  markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False,)

frame.get_title().set_weight('bold')
axs.format(xlabel='$\Delta$ALS-TFMRA$_{corr}$ (m)', ultitle='Ka-band (floes)\n$\Delta$Ka = {:.2f} m'.format(offset_ka), xlim=(-3, 2), ylabel='frequency (counts)', ylim=(0,40000))
#axs.set_yticklabels([])

axs.axvline(np.nanmean(df['ALS-TFMRA50']), c=cmap_qual[1], linestyle='--', alpha=0.8)
axs.axvline(np.nanmean(df['ALS-TFMRA40']), c=cmap_qual[0], linestyle='--', alpha=0.4)
axs.axvline(np.nanmean(df['ALS-TFMRA80']), c=cmap_qual[2], linestyle='--', alpha=0.5)
axs.axvline(np.nanmean(val[(val<5)&(val>-5)]), c=cmap_qual[4], linestyle='--', alpha=0.8)



axs=ax[2]
n_bins = 20
binwidth=0.2
n_bins = np.arange(-3, 2+ binwidth, binwidth)
df = df_snow_ALS_diff[class_leads_seaice<1]
leg4=axs.hist(df['ALS-as_CWT'], c=cmap_qual[6], edgecolor='k', alpha=0.8, bins=n_bins, label='{:.2f} $\pm$ {:.2f} m\n5-95% = {:.2f}-{:.2f} m'.format(np.nanmean(df['ALS-as_CWT']), np.nanstd(df['ALS-as_CWT']), np.nanquantile(df['ALS-as_CWT'], 0.05), np.nanquantile(df['ALS-as_CWT'], 0.95)))
leg6=axs.hist(df['ALS-as_PEAK'], c=cmap_qual[0], edgecolor='k', alpha=0.8, bins=n_bins, label='{:.2f} $\pm$ {:.2f} m\n5-95% = {:.2f}-{:.2f} m'.format(np.nanmean(df['ALS-as_PEAK']), np.nanstd(df['ALS-as_PEAK']), np.nanquantile(df['ALS-as_PEAK'], 0.05), np.nanquantile(df['ALS-as_PEAK'], 0.95)))
leg5=axs.hist(df['ALS-si_CWT'], c=cmap_qual[5], edgecolor=cmap_qual[5], alpha=0.6, bins=n_bins, histtype='stepfilled', label='{:.2f} $\pm$ {:.2f} m\n5-95% = {:.2f}-{:.2f} m'.format(np.nanmean(df['ALS-si_CWT']), np.nanstd(df['ALS-si_CWT']), np.nanquantile(df['ALS-si_CWT'], 0.05), np.nanquantile(df['ALS-si_CWT'], 0.95)))
leg7=axs.hist(df['ALS-si_PEAK'], c=cmap_qual[1], edgecolor=cmap_qual[1], alpha=0.6, bins=n_bins, histtype='stepfilled', label='{:.2f} $\pm$ {:.2f} m\n5-95% = {:.2f}-{:.2f} m'.format(np.nanmean(df['ALS-si_PEAK']), np.nanstd(df['ALS-si_PEAK']), np.nanquantile(df['ALS-si_PEAK'], 0.05), np.nanquantile(df['ALS-si_PEAK'], 0.95)))
#frame=axs.legend(loc='ur', ncols=1, handlelength=0.7, title='S/C-band')
val = df['ALS']-df['WF_max_corr']
leg8=axs.hist(val, c=cmap_qual[4], edgecolor='k', alpha=0.8, bins=n_bins, histtype='stepfilled', hatch='///',label='{:.2f} $\pm$ {:.2f} m\n5-95% = {:.2f}-{:.2f} m'.format(np.nanmean(val[(val<5)&(val>-5)]), np.nanstd(val[(val<5)&(val>-5)]), np.nanquantile(val[(val<5)&(val>-5)], 0.05), np.nanquantile(val[(val<5)&(val>-5)], 0.95)))
frame = axs.legend(loc='ll', handlelength=0,  markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False)

frame.get_title().set_weight('bold')
axs.format(xlabel='$\Delta$ALS-air-snow/snow-ice$_{CWT/PEAK,corr}$ (m)', ultitle='C/S-band (floes)\n$\Delta$C/S = {:.2f} m'.format(offset_snow), xlim=(-3, 2),  ylabel='frequency (counts)', ylim=(0,40000))
#axs.set_yticklabels([])

axs.axvline(np.nanmean(df['ALS-as_CWT']), c=cmap_qual[6], linestyle='--', alpha=0.8)
axs.axvline(np.nanmean(df['ALS-si_CWT']), c=cmap_qual[5], linestyle='--', alpha=0.6)
axs.axvline(np.nanmean(df['ALS-as_PEAK']), c=cmap_qual[0], linestyle='--', alpha=0.8)
axs.axvline(np.nanmean(df['ALS-si_PEAK']), c=cmap_qual[1], linestyle='--', alpha=0.6)
axs.axvline(np.nanmean(val[(val<5)&(val>-5)]), c=cmap_qual[4], linestyle='--', alpha=0.8)

leg0=axs.plot(0, 0, c='w')
#frame = ax[2].legend([leg2, leg1, leg3], ['TFMRA40$_{corr}$', 'TFMRA50$_{corr}$', 'TFMRA80$_{corr}$'], loc='t', ncols=3, align='left', title='Ka/Ku-band')
#frame._legend_box.align = "left"
#frame.get_title().set_weight('bold')

#frame = ax[4].legend([leg4, leg5, leg6, leg7], ['air-snow$_{CWT,corr}$', 'snow-ice$_{CWT,corr}$', 'air-snow$_{PEAK,corr}$', 'snow-ice$_{PEAK,corr}$'], loc='t', ncols=2, align='right', title='S/C-band')
#frame._legend_box.align = "left"
#frame.get_title().set_weight('bold')


frame = ax[0].legend([leg2, leg1, leg3, leg8, leg4, leg5, leg6, leg7 ], ['TFMRA40$_{corr}$', 'TFMRA50$_{corr}$', 'TFMRA80$_{corr}$', 'MAX', 'air-snow$_{CWT,corr}$', 'snow-ice$_{CWT,corr}$', 'air-snow$_{PEAK,corr}$', 'snow-ice$_{PEAK,corr}$'], loc='t', ncols=4, align='left', title='Ka-, Ku-, or C/S-band (panel a-c)')
#frame = ax[2].legend(*zip(*legend_contents))
frame._legend_box.align = "left"
frame.get_title().set_weight('bold')




axs = ax[3]

n_bins = 20
binwidth=0.1
n_bins = np.arange(-0.2, 1.5+ binwidth, binwidth)
hs = ((df_Ku_ALS_diff['ALS'][class_leads_seaice<1])-(df_Ku_ALS_diff['TFMRA50'][class_leads_seaice<1]))/params['n_snow']
snow_leg1=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} cm'.format(np.nanmean(hs)*100, np.nanstd(hs)*100), edgecolor='k' , c='k', )
hs = ((df_Ku_ALS_diff['ALS'][class_leads_seaice<1])-(df_Ka_ALS_diff['TFMRA50'][class_leads_seaice<1]))/params['n_snow']
snow_leg2=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} cm'.format(np.nanmean(hs)*100, np.nanstd(hs)*100), edgecolor='k', color=cmap_qual[1], alpha=0.9, )
hs = ((df_Ka_ALS_diff['TFMRA50'][class_leads_seaice<1])-(df_Ku_ALS_diff['TFMRA50'][class_leads_seaice<1]))/params['n_snow']
snow_leg3=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} cm'.format(np.nanmean(hs)*100, np.nanstd(hs)*100),  edgecolor='k', color='grey', alpha=0.5,hatch='///')
#snow_leg3=axs.axvline(np.nanmean(hs), linestyle='--', c='b')
hs = ((df_Ka_ALS_diff['WF_max_corr'][class_leads_seaice<1])-(df_Ku_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
hs = hs[(hs>-5) & (hs<5)]
snow_leg4=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} cm'.format(np.nanmean(hs)*100, np.nanstd(hs)*100), edgecolor='k' , c=cmap_qual[4], alpha=0.8,hatch='///' )


hs1 = ((df_Ku_ALS_diff['ALS'][class_leads_seaice<1])-(df_Ku_ALS_diff['TFMRA50'][class_leads_seaice<1]))/params['n_snow']
hs1a = ((df_Ku_ALS_diff['ALS'][class_leads_seaice<1])-(df_Ku_ALS_diff['TFMRA50'][class_leads_seaice<1]))/params['n_snow']
idx = hs1a>0.05
hs1a = hs1a[idx]
hs2 = ((df_Ku_ALS_diff['ALS'][class_leads_seaice<1])-(df_Ka_ALS_diff['TFMRA50'][class_leads_seaice<1]))/params['n_snow']
hs2a = ((df_Ku_ALS_diff['ALS'][class_leads_seaice<1])-(df_Ka_ALS_diff['TFMRA50'][class_leads_seaice<1]))/params['n_snow']
idx = hs2a>0.05
hs2a = hs2a[idx]
hs3 = ((df_Ka_ALS_diff['TFMRA50'][class_leads_seaice<1])-(df_Ku_ALS_diff['TFMRA50'][class_leads_seaice<1]))/params['n_snow']
hs4 = ((df_Ka_ALS_diff['TFMRA50'][class_leads_seaice<1])-(df_Ku_ALS_diff['TFMRA50'][class_leads_seaice<1]))/params['n_snow']
idx = hs4>0.05
hs4 = hs4[idx]

hs5 = ((df_Ka_ALS_diff['WF_max_corr'][class_leads_seaice<1])-(df_Ku_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
hs5a = ((df_Ka_ALS_diff['WF_max_corr'][class_leads_seaice<1])-(df_Ku_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
idx = hs5a>0.05
hs5a = hs5a[idx]


frame=axs.legend([snow_leg1, snow_leg2, snow_leg3, snow_leg4],['h{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} m\n>0.05 m = {:.2f} $\pm$ {:.2f} m'.format('$_{s, ALS-Ku_{corr, TFMRA50}}$',(len(hs)-len(hs1a))/len(hs)*100,np.nanmean(hs1), np.nanstd(hs1),np.nanmean(hs1a), np.nanstd(hs1a)), 'h{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} m\n>0.05 m = {:.2f} $\pm$ {:.2f} m'.format('$_{s, ALS-Ka_{corr, TFMRA50}}$',(len(hs)-len(hs2a))/len(hs)*100,np.nanmean(hs2), np.nanstd(hs2),np.nanmean(hs2a), np.nanstd(hs2a)), 'h{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} m\n>0.05 m = {:.2f} $\pm$ {:.2f} m'.format('$_{s, Ka_{corr, TFMRA50}-Ku_{corr, TFMRA50}}$',(len(hs)-len(hs4))/len(hs)*100,np.nanmean(hs3), np.nanstd(hs3), np.nanmean(hs4), np.nanstd(hs4)), 'h{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} m\n>0.05 m = {:.2f} $\pm$ {:.2f} m\n'.format('$_{s, Ka_{corr, MAX}-Ku_{corr, MAX}}$',(len(hs)-len(hs5a))/len(hs)*100,np.nanmean(hs5[(hs5<5)&(hs5>-5)]), np.nanstd(hs5[(hs5<5)&(hs5>-5)]), np.nanmean(hs5a[(hs5a<5)&(hs5a>-5)]), np.nanstd(hs5a[(hs5a<5)&(hs5a>-5)]))], handlelength=0,  markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, align='left',loc='lr',)
axs.format(ultitle='h$_s$ (frequency-based)', xlim=(-0.25,1.5), xlabel='snow depth, h$_s$ (m)', ylabel='frequency (counts)')
#frame=axs.legend([snow_leg1, snow_leg2, snow_leg3],['h$_{s, ALS-Ku_{corr}}$', 'h$_{s, ALS-Ka_{corr}}$', 'h$_{s, Ka_{corr}-Ku_{corr}}$'], ncols=2, align='left',loc='b', title='h$_s$, freq. combinations, multiple bands (panel g)')
#frame._legend_box.align = "left"
#frame.get_title().set_weight('bold')

axs = ax[4]
hs = ((df_Ku_ALS_diff['ALS'][class_leads_seaice<1])-(df_Ku_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
snow_leg1=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} cm'.format(np.nanmean(hs)*100, np.nanstd(hs)*100), edgecolor='k' , c='k', )
hs = ((df_Ku_ALS_diff['ALS'][class_leads_seaice<1])-(df_Ka_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
snow_leg2=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} cm'.format(np.nanmean(hs)*100, np.nanstd(hs)*100), edgecolor='k', color=cmap_qual[1], alpha=0.9, )
hs = ((df_snow_ALS_diff['ALS'][class_leads_seaice<1])-(df_snow_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
snow_leg3=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} cm'.format(np.nanmean(hs)*100, np.nanstd(hs)*100),  edgecolor='k', color='grey', alpha=0.5,hatch='///')
#snow_leg3=axs.axvline(np.nanmean(hs), linestyle='--', c='b')

hs1 = ((df_Ku_ALS_diff['ALS'][class_leads_seaice<1])-(df_Ku_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
hs1 = hs1[np.abs(hs1)<5]
hs1a = hs1a[np.abs(hs1a)<5]
hs1a = ((df_Ku_ALS_diff['ALS'][class_leads_seaice<1])-(df_Ku_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
idx = hs1a>0.05
hs1a = hs1a[idx]
hs2 = ((df_Ka_ALS_diff['ALS'][class_leads_seaice<1])-(df_Ka_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
hs2a = ((df_Ka_ALS_diff['ALS'][class_leads_seaice<1])-(df_Ka_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
hs2 = hs2[np.abs(hs2)<5]
hs2a = hs2[np.abs(hs2)<5]
idx = hs2a>0.05
hs2a = hs2a[idx]
hs3 = ((df_snow_ALS_diff['ALS'][class_leads_seaice<1])-(df_snow_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
hs4 = ((df_snow_ALS_diff['ALS'][class_leads_seaice<1])-(df_snow_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
hs3 = hs3[np.abs(hs3)<5]
hs4 = hs4[np.abs(hs4)<5]
idx = hs4>0.05
hs4 = hs4[idx]
frame=axs.legend([snow_leg1, snow_leg2, snow_leg3],['h{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} m\n>0.05 m = {:.2f} $\pm$ {:.2f} m'.format('$_{s, ALS-Ku_{corr, MAX}}$',(len(hs)-len(hs1a))/len(hs)*100,np.nanmean(hs1), np.nanstd(hs1),np.nanmean(hs1a), np.nanstd(hs1a)), 'h{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} m\n>0.05 m = {:.2f} $\pm$ {:.2f} m'.format('$_{s, ALS-Ka_{corr, MAX}}$',(len(hs)-len(hs2a))/len(hs)*100,np.nanmean(hs2), np.nanstd(hs2),np.nanmean(hs2a), np.nanstd(hs2a)), 'h{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} m\n>0.05 m = {:.2f} $\pm$ {:.2f} m\n\n\n\n\n'.format('$_{s, ALS-C/S_{corr, MAX}}$',(len(hs)-len(hs4))/len(hs)*100,np.nanmean(hs3), np.nanstd(hs3), np.nanmean(hs4), np.nanstd(hs4))], handlelength=0,  markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, align='left',loc='lr',)
axs.format(ultitle='h$_s$ (frequency-based)', xlim=(-0.25,1.5), xlabel='snow depth, h$_s$ (m)', ylabel='frequency (counts)')
#frame=axs.legend([snow_leg1, snow_leg2, snow_leg3],['h$_{s, ALS-Ku_{corr}}$', 'h$_{s, ALS-Ka_{corr}}$', 'h$_{s, Ka_{corr}-Ku_{corr}}$'], ncols=2, align='left',loc='b', title='h$_s$, freq. combinations, multiple bands (panel g)')
#frame._legend_box.align = "left"
#frame.get_title().set_weight('bold')

axs = ax[5]
hs = ((df_snow_ALS_diff['as_CWT'][class_leads_seaice<1])-(df_snow_ALS_diff['si_CWT'][class_leads_seaice<1]))/params['n_snow']
snow_leg4=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} m'.format(np.nanmean(hs), np.nanstd(hs)), edgecolor='k', c=cmap_qual[6] )
hs = ((df_snow_ALS_diff['as_PEAK'][class_leads_seaice<1])-(df_snow_ALS_diff['si_PEAK'][class_leads_seaice<1]))/params['n_snow']
snow_leg6=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} m'.format(np.nanmean(hs), np.nanstd(hs)),  edgecolor='k', c=cmap_qual[0], alpha=0.8)
#snow_leg6 = axs.axvline(np.nanmean(hs), linestyle='--', c='red', linewidth=1)
#snow_leg7=axs.axvline(np.nanmean(hs), linestyle='-', c='orange', linewidth=1)
axs.format(ultitle='h$_s$ (multiple interfaces)', xlim=(-0.25,1.5),  xlabel='snow depth, h$_s$ (m)', ylabel='frequency (counts)')


hs1 =((df_snow_ALS_diff['as_CWT'][class_leads_seaice<1])-(df_snow_ALS_diff['si_CWT'][class_leads_seaice<1]))/params['n_snow']
hs1a =((df_snow_ALS_diff['as_CWT'][class_leads_seaice<1])-(df_snow_ALS_diff['si_CWT'][class_leads_seaice<1]))/params['n_snow']
idx = hs1a>0.05
hs1a = hs1a[idx]

hs2 = ((df_snow_ALS_diff['as_PEAK'][class_leads_seaice<1])-(df_snow_ALS_diff['si_PEAK'][class_leads_seaice<1]))/params['n_snow']
hs2a =  ((df_snow_ALS_diff['as_PEAK'][class_leads_seaice<1])-(df_snow_ALS_diff['si_PEAK'][class_leads_seaice<1]))/params['n_snow']
idx = hs2a>0.05
hs2a = hs2a[idx]



frame=axs.legend([snow_leg4, snow_leg6],['h{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} m\n>0.05 m = {:.2f} $\pm$ {:.2f} m'.format('$_{s, CWT}$',(len(hs)-len(hs1a))/len(hs)*100,np.nanmean(hs1), np.nanstd(hs1) ,np.nanmean(hs1a), np.nanstd(hs1a)), 'h{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} m\n>0.05 m = {:.2f} $\pm$ {:.2f} m\n\n\n\n\n\n\n\n\n'.format('$_{s, PEAK}$',(len(hs)-len(hs2a))/len(hs)*100,np.nanmean(hs2), np.nanstd(hs2),np.nanmean(hs2a), np.nanstd(hs2a))],handlelength=0,  markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, align='left',loc='lr')
#frame=ax[7].legend([snow_leg4, snow_leg6, snow_leg7], ['h$_{s, CWT_{corr}}$', 'h$_{s, PEAK_{corr}}$', 'h$_{s, MaxC}$'], loc='b', align='left', ncols=2, order='F', title='h$_s$, multiple interfaces, same freq. (panel h)')
frame._legend_box.align = "left"
frame.get_title().set_weight('bold')
fig.format(abc='(a)', abcloc='ul', fontsize=12)
#fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\Figure4_new.png', dpi=300)



#fig, ax = pplt.subplots([[1, 2, 3]], axwidth=2.8, axheight=2.5, spanx=True, sharey=True)

#fig.patch.set_facecolor('white')

axs = ax[6]
hs = ((df_Ku_ALS_diff['TFMRA50'][class_leads_seaice<1])-(df_Ku_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
snow_leg1=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} m'.format(np.nanmean(hs), np.nanstd(hs)), edgecolor='k' , c='k', )
hs = ((df_Ka_ALS_diff['TFMRA50'][class_leads_seaice<1])-(df_Ka_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
snow_leg2=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} m'.format(np.nanmean(hs), np.nanstd(hs)), edgecolor='k', color=cmap_qual[1], alpha=0.75)
hs = ((df_Ka_ALS_diff['TFMRA50'][class_leads_seaice<1])-(df_Ku_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
snow_leg3=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} m'.format(np.nanmean(hs), np.nanstd(hs)),  edgecolor='k', color='grey', alpha=0.5,hatch='///')
#snow_leg3=axs.axvline(np.nanmean(hs), linestyle='--', c='b')

hs1 = ((df_Ku_ALS_diff['TFMRA50'][class_leads_seaice<1])-(df_Ku_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
hs1 = hs1[np.abs(hs1)<5]
hs1a = hs1a[np.abs(hs1a)<5]
hs1a = ((df_Ku_ALS_diff['TFMRA50'][class_leads_seaice<1])-(df_Ku_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
idx = hs1a>0.05
hs1a = hs1a[idx]
hs2 = ((df_Ka_ALS_diff['TFMRA50'][class_leads_seaice<1])-(df_Ka_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
hs2a = ((df_Ka_ALS_diff['TFMRA50'][class_leads_seaice<1])-(df_Ka_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
hs2 = hs2[np.abs(hs2)<5]
hs2a = hs2[np.abs(hs2)<5]
idx = hs2a>0.05
hs2a = hs2a[idx]
hs3 = ((df_Ka_ALS_diff['TFMRA50'][class_leads_seaice<1])-(df_Ku_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
hs4 = ((df_Ka_ALS_diff['TFMRA50'][class_leads_seaice<1])-(df_Ku_ALS_diff['WF_max_corr'][class_leads_seaice<1]))/params['n_snow']
hs3 = hs3[np.abs(hs3)<5]
hs4 = hs4[np.abs(hs4)<5]
idx = hs4>0.05
hs4 = hs4[idx]
frame=axs.legend([snow_leg1, snow_leg2, snow_leg3],['\n\n\nh{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} m\n>0.05 m = {:.2f} $\pm$ {:.2f} m'.format('$_{s, Ku_{corr, TFMRA50}-Ku_{corr, MAX}}$',(len(hs)-len(hs1a))/len(hs)*100,np.nanmean(hs1), np.nanstd(hs1),np.nanmean(hs1a), np.nanstd(hs1a)), 'h{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} m\n>0.05 m = {:.2f} $\pm$ {:.2f} m'.format('$_{s, Ka_{corr, TFMRA50}-Ka_{corr, MAX}}$',(len(hs)-len(hs2a))/len(hs)*100,np.nanmean(hs2), np.nanstd(hs2),np.nanmean(hs2a), np.nanstd(hs2a)), 'h{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} m\n>0.05 m = {:.2f} $\pm$ {:.2f} m'.format('$_{s, Ka_{corr, TFMRA50}-Ku_{corr, MAX}}$',(len(hs)-len(hs4))/len(hs)*100,np.nanmean(hs3), np.nanstd(hs3),np.nanmean(hs4), np.nanstd(hs4))], handlelength=0,  markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, align='left',loc='ur',)
axs.format(ultitle='h$_s$ (multiple interfaces or \nfrequencies)', xlim=(-0.25,1.5), xlabel='snow depth, h$_s$ (m)', ylabel='frequency (counts)')#frame=axs.legend([snow_leg1, snow_leg2, snow_leg3],['h$_{s, ALS-Ku_{corr}}$', 'h$_{s, ALS-Ka_{corr}}$', 'h$_{s, Ka_{corr}-Ku_{corr}}$'], ncols=2, align='left',loc='b', title='h$_s$, freq. combinations, multiple bands (panel g)')
#frame._legend_box.align = "left"
#frame.get_title().set_weight('bold')
fig.format(fontsize=12)
#fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\Figure_MAX_TFMRA50.png', dpi=300)


axs = ax[7]
hs = ((df_Ku_ALS_diff['as_PEAK'][class_leads_seaice<1])-(df_Ku_ALS_diff['si_PEAK'][class_leads_seaice<1]))/params['n_snow']
snow_leg1=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} cm'.format(np.nanmean(hs)*100, np.nanstd(hs)*100), edgecolor='k' , c='k', )
hs = ((df_Ku_ALS_diff['as_CWT'][class_leads_seaice<1])-(df_Ku_ALS_diff['si_CWT'][class_leads_seaice<1]))/params['n_snow']
snow_leg2=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} cm'.format(np.nanmean(hs)*100, np.nanstd(hs)*100), edgecolor='k', color=cmap_qual[1], alpha=0.75)
#snow_leg3=axs.axvline(np.nanmean(hs), linestyle='--', c='b')

hs1 = ((df_Ku_ALS_diff['as_PEAK'][class_leads_seaice<1])-(df_Ku_ALS_diff['si_PEAK'][class_leads_seaice<1]))/params['n_snow']
hs1 = hs1[np.abs(hs1)<5]
hs1a = hs1a[np.abs(hs1a)<5]
hs1a = ((df_Ku_ALS_diff['as_PEAK'][class_leads_seaice<1])-(df_Ku_ALS_diff['si_PEAK'][class_leads_seaice<1]))/params['n_snow']
idx = hs1a>0.05
hs1a = hs1a[idx]
hs2 = ((df_Ku_ALS_diff['as_CWT'][class_leads_seaice<1])-(df_Ku_ALS_diff['si_CWT'][class_leads_seaice<1]))/params['n_snow']
hs2a = ((df_Ku_ALS_diff['as_CWT'][class_leads_seaice<1])-(df_Ku_ALS_diff['si_CWT'][class_leads_seaice<1]))/params['n_snow']
hs2 = hs2[np.abs(hs2)<5]
hs2a = hs2[np.abs(hs2)<5]
idx = hs2a>0.05
hs2a = hs2a[idx]
frame=axs.legend([snow_leg1, snow_leg2, ],['\n\n\nh{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} m\n>0.05 m = {:.2f} $\pm$ {:.2f} m'.format('$_{s, a-s_{PEAK}-s-i_{PEAK}}$',(len(hs)-len(hs1a))/len(hs)*100,np.nanmean(hs1), np.nanstd(hs1),np.nanmean(hs1a), np.nanstd(hs1a)), 'h{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} m\n>0.05 m = {:.2f} $\pm$ {:.2f} m'.format('$_{s, a-s_{CWT}-s-i_{CWT}}$',(len(hs)-len(hs2a))/len(hs)*100,np.nanmean(hs2), np.nanstd(hs2),np.nanmean(hs2a), np.nanstd(hs2a))], handlelength=0,  markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, align='left',loc='ur',)
axs.format(ultitle='h$_s$ (multiple interfaces, Ku)', xlim=(-0.25,1.5), xlabel='snow depth, h$_s$ (m)', ylabel='frequency (counts)')#frame=axs.legend([snow_leg1, snow_leg2, snow_leg3],['h$_{s, ALS-Ku_{corr}}$', 'h$_{s, ALS-Ka_{corr}}$', 'h$_{s, Ka_{corr}-Ku_{corr}}$'], ncols=2, align='left',loc='b', title='h$_s$, freq. combinations, multiple bands (panel g)')
#frame._legend_box.align = "left"
#frame.get_title().set_weight('bold')
fig.format(fontsize=12)


axs = ax[8]
hs = ((df_Ka_ALS_diff['as_PEAK'][class_leads_seaice<1])-(df_Ka_ALS_diff['si_PEAK'][class_leads_seaice<1]))/params['n_snow']
snow_leg1=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} cm'.format(np.nanmean(hs)*100, np.nanstd(hs)*100), edgecolor='k' , c='k', )
hs = ((df_Ka_ALS_diff['as_CWT'][class_leads_seaice<1])-(df_Ka_ALS_diff['si_CWT'][class_leads_seaice<1]))/params['n_snow']
snow_leg2=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} cm'.format(np.nanmean(hs)*100, np.nanstd(hs)*100), edgecolor='k', color=cmap_qual[1], alpha=0.75)
#snow_leg3=axs.axvline(np.nanmean(hs), linestyle='--', c='b')

hs1 = ((df_Ka_ALS_diff['as_PEAK'][class_leads_seaice<1])-(df_Ka_ALS_diff['si_PEAK'][class_leads_seaice<1]))/params['n_snow']
hs1 = hs1[np.abs(hs1)<5]
hs1a = hs1a[np.abs(hs1a)<5]
hs1a = ((df_Ka_ALS_diff['as_PEAK'][class_leads_seaice<1])-(df_Ka_ALS_diff['si_PEAK'][class_leads_seaice<1]))/params['n_snow']
idx = hs1a>0.05
hs1a = hs1a[idx]
hs2 = ((df_Ka_ALS_diff['as_CWT'][class_leads_seaice<1])-(df_Ka_ALS_diff['si_CWT'][class_leads_seaice<1]))/params['n_snow']
hs2a = ((df_Ka_ALS_diff['as_CWT'][class_leads_seaice<1])-(df_Ka_ALS_diff['si_CWT'][class_leads_seaice<1]))/params['n_snow']
hs2 = hs2[np.abs(hs2)<5]
hs2a = hs2[np.abs(hs2)<5]
idx = hs2a>0.05
hs2a = hs2a[idx]
frame=axs.legend([snow_leg1, snow_leg2, ],['\n\n\nh{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} m\n>0.05 m = {:.2f} $\pm$ {:.2f} m'.format('$_{s, a-s_{PEAK}-s-i_{PEAK}}$',(len(hs)-len(hs1a))/len(hs)*100,np.nanmean(hs1), np.nanstd(hs1),np.nanmean(hs1a), np.nanstd(hs1a)), 'h{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} m\n>0.05 m = {:.2f} $\pm$ {:.2f} m'.format('$_{s, a-s_{CWT}-s-i_{CWT}}$',(len(hs)-len(hs2a))/len(hs)*100,np.nanmean(hs2), np.nanstd(hs2),np.nanmean(hs2a), np.nanstd(hs2a))], handlelength=0,  markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, align='left',loc='ur',)
axs.format(ultitle='h$_s$ (multiple interfaces, Ka)', xlim=(-0.2,1.5), xlabel='snow depth, h$_s$ (m)', ylabel='frequency (counts)')#frame=axs.legend([snow_leg1, snow_leg2, snow_leg3],['h$_{s, ALS-Ku_{corr}}$', 'h$_{s, ALS-Ka_{corr}}$', 'h$_{s, Ka_{corr}-Ku_{corr}}$'], ncols=2, align='left',loc='b', title='h$_s$, freq. combinations, multiple bands (panel g)')
#frame._legend_box.align = "left"
#frame.get_title().set_weight('bold')
fig.format(fontsize=12)

fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\Figure_8.png', dpi=300)



#%%% CENTROID - APPENDIX PLOT 

fig, ax = pplt.subplots([[1]], axwidth=1, axheight=3, spanx=True, sharey=True)

fig.patch.set_facecolor('white')

axs = ax[0]

N_obs = val_x*n1+n1_extra

fn = 'kuband_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
wf_kuband = ds.variables['waveform'][:]
time = ds.variables['two_way_travel_time'][:]
time_conv = (time*c/2)+offset_ku

idx = np.arange(np.nanargmax(
    wf_kuband[N_obs][:])-100, np.nanargmax(wf_kuband[N_obs][:])+156)
centroid = np.nansum(
    wf_kuband[N_obs][idx]*time_conv[N_obs][idx])/np.nansum(wf_kuband[N_obs][idx])

leg5 = axs.axhline(y=time_conv[N_obs][np.nanargmax(
    wf_kuband[N_obs][:])], color='brown', linestyle='--', linewidth=1)
axs.plot(wf_kuband[N_obs-5:N_obs+5][:, idx],
        time_conv[N_obs-5:N_obs+5][:, idx], zorder=0, c='lightgrey')
leg6 = axs.plot(wf_kuband[N_obs][idx], time_conv[N_obs]
               [idx], zorder=0, c='k', linewidth=0.5)
axs.format(ylabel='range (m)', xlim=(np.nanmin(wf_kuband[N_obs])+0.05*np.nanmean(wf_kuband[N_obs]), np.nanmax(
    wf_kuband[N_obs])-0.05*np.nanmean(wf_kuband[N_obs])), ylim=(time_conv[N_obs][idx[-1]], time_conv[N_obs][idx[0]]), ultitle='Ku-band'.format(fn[0:2]))

leg7 = axs.axhline(y=centroid, color=cmap_qual[3], linestyle='--', linewidth=1)
axs.scatter(x=np.nanmin(
    wf_kuband[N_obs][idx]), y=centroid, c=cmap_qual[3], marker=4, s=30, label='')
axs.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=time_conv[N_obs][np.nanargmax(
    wf_kuband[N_obs][:])], color='brown', marker=4, s=30, label='')

elevation = ds.variables['elevation'][:]
leg0 = axs.axhline(y=elevation[N_obs] -
                  ALS_along_radar_track[N_obs], c='green')

handle0 = r'$h_{}$ = {:.2f} m'.format(
    'R', elevation[N_obs]-ALS_along_radar_track[N_obs])
handle4 = r'$h_{}$ = {:.2f} m'.format('R', centroid)
handle5 = r'$h_{}$ = {:.2f} m'.format('R', time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])])
handle6 = r'$h_{}$ = {:.2f} m'.format('{s,MaxC}', (centroid-time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])])/params["n_snow"])
frame = axs.legend([leg0, leg5, leg7, leg6], [handle0, handle5, handle4, handle6],
                  loc='lr', handlelength=0, markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, prop={'size': 7})


ax1 = axs.panel_axes('r', width=1, space=0)

fn = 'kaband_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open data
wf_kuband = ds.variables['waveform'][:]
time = ds.variables['two_way_travel_time'][:]
time_conv = (time*c/2)+offset_ka+offset_Ku_Ka


idx = np.arange(np.nanargmax(
    wf_kuband[N_obs][:])-100, np.nanargmax(wf_kuband[N_obs][:])+156)
centroid = np.nansum(
    wf_kuband[N_obs][idx]*time_conv[N_obs][idx])/np.nansum(wf_kuband[N_obs][idx])

leg5 = ax1.axhline(y=time_conv[N_obs][np.nanargmax(
    wf_kuband[N_obs][:])], color='brown', linestyle='--', linewidth=1)
ax1.plot(wf_kuband[N_obs-5:N_obs+5][:, idx],
        time_conv[N_obs-5:N_obs+5][:, idx], zorder=0, c='lightgrey')
leg6 = ax1.plot(wf_kuband[N_obs][idx], time_conv[N_obs]
               [idx], zorder=0, c='k', linewidth=0.5)
ax1.format(ylabel='range (m)', xlim=(np.nanmin(wf_kuband[N_obs])+0.05*np.nanmean(wf_kuband[N_obs]), np.nanmax(wf_kuband[N_obs])-0.05*np.nanmean(wf_kuband[N_obs])), ylim=(time_conv[N_obs][idx[-1]], time_conv[N_obs][idx[0]]), ultitle='Ka-band'.format(fn[0:2]))

leg7 = ax1.axhline(y=centroid, color=cmap_qual[5], linestyle='--', linewidth=1)
ax1.scatter(x=np.nanmin(
    wf_kuband[N_obs][idx]), y=centroid, c=cmap_qual[5], marker=4, s=30, label='')
ax1.scatter(x=np.nanmin(wf_kuband[N_obs][idx]), y=time_conv[N_obs][np.nanargmax(
    wf_kuband[N_obs][:])], color='brown', marker=4, s=30, label='')

elevation = ds.variables['elevation'][:]
leg0 = ax1.axhline(y=elevation[N_obs] -
                  ALS_along_radar_track[N_obs]+offset_Ku_Ka, c='green')

handle0 = r'$h_{}$ = {:.2f} m'.format(
    'R', elevation[N_obs]-ALS_along_radar_track[N_obs]+offset_Ku_Ka)
handle4 = r'$h_{}$ = {:.2f} m'.format('R', centroid)
handle5 = r'$h_{}$ = {:.2f} m'.format('R', time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])])
handle6 = r'$h_{}$ = {:.2f} m'.format('{s,MaxC}', (centroid-time_conv[N_obs][np.nanargmax(wf_kuband[N_obs][:])])/params["n_snow"])
frame = ax1.legend([leg0, leg5, leg7, leg6], [handle0, handle5, handle4, handle6],
                  loc='lr', handlelength=0, markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, prop={'size': 7})

fig.format(xlabel='power (W)', xlabelpad=20)
fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\CENTROID_Fig1.png', dpi=300)


fig, ax = pplt.subplots([[1]], axwidth=2.8, axheight=2.5, sharex=False, sharey=False)

fig.patch.set_facecolor('white')
axs = ax[0]

hs = ((df_Ku_ALS_diff['WF_centroid'][class_leads_seaice<1])-(df_Ku_ALS_diff['WF_max'][class_leads_seaice<1]))/params['n_snow']
snow_leg7 = axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} cm'.format(np.nanmean(hs)*100, np.nanstd(hs)*100),  edgecolor='k', c=cmap_qual[3], )

hs3 = ((df_Ku_ALS_diff['WF_centroid'][class_leads_seaice<1])-(df_Ku_ALS_diff['WF_max'][class_leads_seaice<1]))/params['n_snow']
hs3a = ((df_Ku_ALS_diff['WF_centroid'][class_leads_seaice<1])-(df_Ku_ALS_diff['WF_max'][class_leads_seaice<1]))/params['n_snow']
idx = hs3a>0.05
hs3a = hs3a[idx]

hs = ((df_Ka_ALS_diff['WF_centroid'][class_leads_seaice<1])-(df_Ka_ALS_diff['WF_max'][class_leads_seaice<1]))/params['n_snow']
snow_leg8 = axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} cm'.format(np.nanmean(hs)*100, np.nanstd(hs)*100),  edgecolor='k', c=cmap_qual[5],  alpha=0.5, hatch='///')

hs4 = ((df_Ka_ALS_diff['WF_centroid'][class_leads_seaice<1])-(df_Ka_ALS_diff['WF_max'][class_leads_seaice<1]))/params['n_snow']
hs4a = ((df_Ka_ALS_diff['WF_centroid'][class_leads_seaice<1])-(df_Ka_ALS_diff['WF_max'][class_leads_seaice<1]))/params['n_snow']
idx = hs4a>0.05
hs4a = hs4a[idx]


text_label = 'h{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} cm\n>0.05 m = {:.2f} $\pm$ {:.2f} cm'.format('$_{s, MaxC_{Ku, corr}}$',(len(hs)-len(hs3a))/len(hs)*100,np.nanmean(hs3)*100, np.nanstd(hs3)*100,np.nanmean(hs3a)*100, np.nanstd(hs3a)*100)
text_label2 = 'h{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} cm\n>0.05 m = {:.2f} $\pm$ {:.2f} cm'.format('$_{s, MaxC_{Ka, corr}}$',(len(hs)-len(hs4a))/len(hs)*100,np.nanmean(hs4)*100, np.nanstd(hs4)*100,np.nanmean(hs4a)*100, np.nanstd(hs4a)*100)

frame=axs.legend([snow_leg7, snow_leg8],[text_label, text_label2],handlelength=0,  markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, align='left',loc='ur')
#frame=ax[7].legend([snow_leg4, snow_leg6, snow_leg7], ['h$_{s, CWT_{corr}}$', 'h$_{s, PEAK_{corr}}$', 'h$_{s, MaxC}$'], loc='b', align='left', ncols=2, order='F', title='h$_s$, multiple interfaces, same freq. (panel h)')
frame._legend_box.align = "left"
frame.get_title().set_weight('bold')

axs.format(ultitle='h$_s$ MaxC', xlim=(-0.25,1.5),  xlabel='snow depth, h$_s$ (m)', ylabel='frequency (counts)', ylim=(0, 17000))

fig.format(fontsize=12)
fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\CENTROID_Fig2.png', dpi=300)




#%% statistics

diff_ALS_TMFRA50Ku = ((df_Ku_ALS_diff['ALS'][class_leads_seaice<1])-(df_Ku_ALS_diff['TFMRA80'][class_leads_seaice<1]))
print(len(diff_ALS_TMFRA50Ku))
print('#####: <-0.05 m')
diff_10 = diff_ALS_TMFRA50Ku[diff_ALS_TMFRA50Ku>-0.1]
print(len(diff_ALS_TMFRA50Ku) - len(diff_10))
print(round((1-(len(diff_10)/len(diff_ALS_TMFRA50Ku)))*100,1))
print('#####: >-0.05 m and <0.05 m')
diff_5 = diff_10[(diff_10<0.1)]
print(len(diff_5))
print(round((len(diff_5)/len(diff_ALS_TMFRA50Ku))*100,1))
#print('#####: >-0.05 m and <0.10 m')
#diff_10total = diff_10[diff_10<0.10]
#print(len(diff_10total))
#print(len(diff_10total)/len(diff_ALS_TMFRA50Ku))
print('#####: > 0.05m & <1.5 m')
diff_5more = diff_ALS_TMFRA50Ku[(diff_ALS_TMFRA50Ku>0.1) & (diff_ALS_TMFRA50Ku<1.5)]
print(len(diff_5more))
print(round((len(diff_5more)/len(diff_ALS_TMFRA50Ku))*100,1))
print(round(np.nanmean(diff_5more),2))
print(round(np.nanstd(diff_5more),2))
print('#####: > 1.5m')
diff_5more = diff_ALS_TMFRA50Ku[(diff_ALS_TMFRA50Ku>1.5)]
print(len(diff_5more))
print(round((len(diff_5more)/len(diff_ALS_TMFRA50Ku))*100,1))

#print('#####: > 0.10 m')
#diff_10more = diff_ALS_TMFRA50Ku[diff_ALS_TMFRA50Ku>0.10]
#print(len(diff_10more))
#print(len(diff_10more)/len(diff_ALS_TMFRA50Ku))
#print(np.mean(diff_10more))

#%%
corr_df = df_snow_ALS_diff[class_leads_seaice<1].corr()

#%% PREP airborne data for final analysis 

#[class_leads_seaice<1]

df_airborne = pd.DataFrame({'latitude':latitude_comb, 
                           'longitude':longitude_comb, 
                           'hs_ALS-Ku_MAX':((df_Ku_ALS_diff['ALS'])-(df_Ku_ALS_diff['WF_max_corr']))/params['n_snow'], 
                           'hs_ALS-Ku_TFMRA50':((df_Ku_ALS_diff['ALS'])-(df_Ku_ALS_diff['TFMRA50']))/params['n_snow'],
                           'hs_ALS-Ka_MAX':((df_Ka_ALS_diff['ALS'])-(df_Ka_ALS_diff['WF_max_corr']))/params['n_snow'], 
                           'hs_ALS-Ka_TFMRA50':((df_Ka_ALS_diff['ALS'])-(df_Ka_ALS_diff['TFMRA50']))/params['n_snow'],
                           'hs_ALS-C/S_MAX':((df_snow_ALS_diff['ALS'])-(df_snow_ALS_diff['WF_max_corr']))/params['n_snow'],
                           'hs_PEAK':((df_snow_ALS_diff['as_PEAK'])-(df_snow_ALS_diff['si_PEAK']))/params['n_snow'],
                           'hs_CWT':((df_snow_ALS_diff['as_CWT'])-(df_snow_ALS_diff['si_CWT']))/params['n_snow'],
                           'sea_ice_class':class_leads_seaice, 
                           'scan_line_along':ALS_scan_number_along_radar_track,
                           'roughness_sigma5m':ALS_roughness_along_radar_track, 
                           #'roughness_sigma400m':df_ALS_scan_v1_roughness['roughness_anomaly']
                           })

#df_airborne.to_csv(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\{}_roughness.csv'.format('df_airborne'))
#%%
var = 'hs_PEAK'
hs_example = df_airborne[df_airborne['sea_ice_class']<1][var]
hs_example = hs_example[hs_example.notna()]
print('NaN %: {:.2f}%'.format(((len(df_airborne[df_airborne['sea_ice_class']<1])-len(hs_example))/len(df_airborne[df_airborne['sea_ice_class']<1]))*100))

hs_example = df_airborne[df_airborne['sea_ice_class']<1][var]
hs_example = hs_example[hs_example.notna()]
print('>1.5m %: {:.2f}%'.format(((len(hs_example)-len(hs_example[hs_example<1.5]))/len(df_airborne[df_airborne['sea_ice_class']<1]))*100))

hs_example = df_airborne[df_airborne['sea_ice_class']<1][var]
hs_example = hs_example[hs_example.notna()]
print('<-0.05m %: {:.2f}%'.format(((len(hs_example)-len(hs_example[hs_example>-0.05]))/len(df_airborne[df_airborne['sea_ice_class']<1]))*100))

#%%

fig, ax = pplt.subplots([[1]], axwidth=2, axheight=2, spanx=True, sharey=False)
var2, var1 = 'hs_ALS-Ka_TFMRA50', 'roughness_sigma5m'
ax.scatter(df_airborne[(df_airborne['sea_ice_class']<1)&(df_airborne[var2]>-0.05)&(df_airborne[var1]<5)][var1], df_airborne[(df_airborne['sea_ice_class']<1)&(df_airborne[var2]>-0.05)&(df_airborne[var1]<5)][var2], s=0.5)
ax.format(xlabel='roughness',ylabel='snow depth')



#%%%%%% report CRYO2ICEANT22

c = 300e6

fig, axs = pplt.subplots([[1],
                          [2]], axwidth=7, axheight=1, sharex=False, sharey=False)
fig.patch.set_facecolor('white')
# n = 300
# n1=8
# val_x = 300

## PAPER 
n = 300
n1 = 70
# n1=8
val_x = 1000

## FINAL REPORT
n = 500
n1 = 78
# n1=8
val_x = 1000

n0_extra = 880
#idx = plot_waveforms_examples(n0_extra, axs[0], '', ALS_along_radar_track)

n1_extra = 180
#idx = plot_waveforms_examples_right(    n1_extra, axs[1], '(b)', ALS_along_radar_track)


ax = axs[0]
fn = 'kuband_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
wf_kuband = ds.variables['waveform'][val_x*n1:val_x*(n1+1)]
lat_extra = ds.variables['lat'][val_x*n1:val_x*(n1+1)]
wf_wb = 10 * np.log10(wf_kuband)
# cb_data = ax.imshow(wf_wb.transpose(), cmap='magma_r', aspect="auto", vmin=-120, vmax=-40,zorder=1, extend='both')
cb_data = ax.imshow(wf_wb.transpose(), cmap='greys', aspect="auto", vmin=-120, vmax=-40, extent=[np.max(
    latitude_comb[val_x*n1:val_x*(n1+1)]), np.min(latitude_comb[val_x*n1:val_x*(n1+1)]), len(wf_kuband[0]), 0], zorder=1, extend='both')
ax.set_yticks([600, 700, 800])

#ax.axvline(lat_extra[n1_extra], marker=11,
 #          markersize=6, c='k', zorder=15, linewidth=0.5, linestyle='--')


elevation = ds.variables['elevation'][:]
time = ds.variables['two_way_travel_time'][:]
time_conv = (time*c/2)+offset_ku
ALS_Ku= elevation-ALS_along_radar_track
ALS_retrack_gate_Ku = np.ones(len(ALS_Ku))*np.nan
for i in np.arange(0, len(ALS_Ku)):
    ALS_retrack_gate_Ku[i] = np.argmin(np.abs(time_conv[i]-ALS_Ku[i]))
ALS_retrack_gate_Ku[ALS_retrack_gate_Ku==0]=np.nan
ax.plot(lat_extra, ALS_retrack_gate_Ku[val_x * n1:val_x*(n1+1)], c='green', linewidth=0.5)

#rtck_gate_kuband = ds.variables['retracking_gate_tfmra40'][val_x * n1:val_x*(n1+1)]
#ax.plot(lat_extra, rtck_gate_kuband, c=cmap_qual[0], linewidth=0.5)
#rtck_gate_kuband = ds.variables['retracking_gate_tfmra50'][val_x * n1:val_x*(n1+1)]
#ax.plot(lat_extra, rtck_gate_kuband, c=cmap_qual[6], linewidth=0.5)
#rtck_gate_kuband = ds.variables['retracking_gate_tfmra80'][val_x * n1:val_x*(n1+1)]
#ax.plot(lat_extra, rtck_gate_kuband, c=cmap_qual[1], linewidth=0.5)
ax.plot(lat_extra, a_s_peaks[val_x*n1:val_x*(n1+1)], c='brown', linewidth=0.5)
#ax.plot(lat_extra, s_i_centroids[val_x*n1:val_x*(n1+1)], c='orange', linewidth=0.5)

'''
rtck_gate_as = ds.variables['retracking_gate_air_snow_peakiness'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_as, c=cmap_qual[0], linewidth=0.5, linestyle='-')
rtck_gate_is = ds.variables['retracking_gate_snow_ice_peakiness'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_is, c=cmap_qual[1], linewidth=0.5, linestyle='-')

rtck_gate_as = ds.variables['retracking_gate_air_snow_cwt_TN'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_as, c=cmap_qual[6], linewidth=0.5)
rtck_gate_is = ds.variables['retracking_gate_snow_ice_snow_cwt_TN'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_is, c=cmap_qual[5], linewidth=0.5)
'''
ax.arrow(-69.97, 780, -0.012, 0, facecolor='k', zorder=15, head_length = 0.001, head_width = 20)
ax.text(-69.97, 830, 'snow-ice interface', color='k', zorder=15)

'''
txt = ['Figure 3']
x = [lat_extra[n1_extra+1]]
for i, txt in enumerate(txt):
    ax.annotate(txt, (x[i], 580), c='k')

'''
ax.format(ultitle='Ku', ylim=(900, 500), xlabel='', ylabel='')


ax = axs[0].panel_axes('b', width=1, space=0)
fn = 'kaband_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
wf_kuband = ds.variables['waveform'][val_x*n1:val_x*(n1+1)]
lat_extra = ds.variables['lat'][val_x*n1:val_x*(n1+1)]
time = ds.variables['two_way_travel_time'][:]
#range_res = np.abs((time[0][0] - time[0][1]) * c / 2)

wf_wb = 10 * np.log10(wf_kuband)
# cb_data = ax.imshow(wf_wb.transpose(), cmap='magma_r', aspect="auto", vmin=-120, vmax=-40,zorder=1, extend='both')
cb_data = ax.imshow(wf_wb.transpose(), cmap='greys', aspect="auto", vmin=-120, vmax=-40, extent=[np.max(
    latitude_comb[val_x*n1:val_x*(n1+1)]), np.min(latitude_comb[val_x*n1:val_x*(n1+1)]), len(wf_kuband[0]), 0], zorder=1, extend='both')
ax.format(ultitle='Ka', ylim=(900, 500), xlabel='',
          ylabel='range bins')
ax.set_yticks([600, 700, 800])

#ax.axvline(lat_extra[n1_extra], marker=11,
#           markersize=6, c='k', zorder=15, linewidth=0.5, linestyle='--')
    
elevation = ds.variables['elevation'][:]
time = ds.variables['two_way_travel_time'][:]
time_conv = (time*c/2)+offset_ka
ALS_Ku_Ka = elevation-ALS_along_radar_track+offset_Ku_Ka
ALS_retrack_gate_Ka = np.ones(len(ALS_Ku_Ka))*np.nan
for i in np.arange(0, len(ALS_Ku_Ka)):
    ALS_retrack_gate_Ka[i] = np.argmin(np.abs(time_conv[i]-ALS_Ku_Ka[i]))
ALS_retrack_gate_Ka[ALS_retrack_gate_Ka==0]=np.nan
ax.plot(lat_extra, ALS_retrack_gate_Ka[val_x * n1:val_x*(n1+1)], c='green', linewidth=0.5)


#rtck_gate_kuband = ds.variables['retracking_gate_tfmra40'][val_x * n1:val_x*(n1+1)]
#ax.plot(lat_extra, rtck_gate_kuband, c=cmap_qual[0], linewidth=0.5)
#rtck_gate_kuband = ds.variables['retracking_gate_tfmra50'][val_x * n1:val_x*(n1+1)]
#ax.plot(lat_extra, rtck_gate_kuband, c=cmap_qual[6], linewidth=0.5)
#rtck_gate_kuband = ds.variables['retracking_gate_tfmra80'][val_x * n1:val_x*(n1+1)]
#ax.plot(lat_extra, rtck_gate_kuband, c=cmap_qual[1], linewidth=0.5)
ax.plot(lat_extra, a_s_peaks_ka[val_x*n1:val_x*(n1+1)], c='brown', linewidth=0.5)
#ax.plot(lat_extra, s_i_centroids_ka[val_x*n1:val_x*(n1+1)], c='orange', linewidth=0.5)
'''
rtck_gate_as = ds.variables['retracking_gate_air_snow_peakiness'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_as, c=cmap_qual[0], linewidth=0.5, linestyle='-')
rtck_gate_is = ds.variables['retracking_gate_snow_ice_peakiness'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_is, c=cmap_qual[1], linewidth=0.5, linestyle='-')

rtck_gate_as = ds.variables['retracking_gate_air_snow_cwt_TN'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_as, c=cmap_qual[6], linewidth=0.5)
rtck_gate_is = ds.variables['retracking_gate_snow_ice_snow_cwt_TN'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_is, c=cmap_qual[5], linewidth=0.5)
'''
#txt = ['Figure 3']
#x = [lat_extra[n1_extra+1]]
#for i, txt in enumerate(txt):
#    ax.annotate(txt, (x[i], 580), c='k')
    
ax = axs[0].panel_axes('b', width=1, space=0)
fn = 'snow_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
wf_kuband = ds.variables['waveform'][val_x*n1:val_x*(n1+1)]
lat_extra = ds.variables['lat'][val_x*n1:val_x*(n1+1)]
wf_wb = 10 * np.log10(wf_kuband)
# cb_data = ax.imshow(wf_wb.transpose(), cmap='magma_r', aspect="auto", vmin=-120, vmax=-40,zorder=1, extend='both')
cb_data = ax.imshow(wf_wb.transpose(), cmap='greys', aspect="auto", vmin=-120, vmax=-40, extent=[np.max(
    latitude_comb[val_x*n1:val_x*(n1+1)]), np.min(latitude_comb[val_x*n1:val_x*(n1+1)]), len(wf_kuband[0]), 0], zorder=1, extend='both')
ax.format(ultitle='S/C', ylim=(900, 500),
          xlabel='latitude (degrees N)', ylabel='')
axs[0].colorbar(cb_data, label='relative power (dB)', loc='t')

#ax.axvline(lat_extra[n1_extra], marker=11,
#           markersize=6, c='k', zorder=15, linewidth=0.5, linestyle='--')

elevation = ds.variables['elevation'][:]
time = ds.variables['two_way_travel_time'][:]
time_conv = (time*c/2)+offset_snow
ALS_Ku_snow = elevation-ALS_along_radar_track+offset_Ku_snow
ALS_retrack_gate_snow = np.ones(len(ALS_Ku_snow))*np.nan
for i in np.arange(0, len(ALS_Ku_snow)):
    ALS_retrack_gate_snow[i] = np.argmin(np.abs(time_conv[i]-ALS_Ku_snow[i]))
ALS_retrack_gate_snow[ALS_retrack_gate_snow==0]=np.nan
ax.plot(lat_extra, ALS_retrack_gate_snow[val_x * n1:val_x*(n1+1)], c='green', linewidth=0.5)


rtck_gate_as = ds.variables['retracking_gate_air_snow_peakiness'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_as, c=cmap_qual[0], linewidth=0.5, linestyle='-')
rtck_gate_is = ds.variables['retracking_gate_snow_ice_peakiness'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_is, c=cmap_qual[1], linewidth=0.5, linestyle='-')

rtck_gate_as = ds.variables['retracking_gate_air_snow_cwt_TN'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_as, c=cmap_qual[6], linewidth=0.5)
rtck_gate_is = ds.variables['retracking_gate_snow_ice_snow_cwt_TN'][val_x * n1:val_x*(n1+1)]
ax.plot(lat_extra, rtck_gate_is, c=cmap_qual[5], linewidth=0.5)

ax.plot(lat_extra, a_s_peaks_snow[val_x * n1:val_x*(n1+1)], c='brown', linewidth=0.5)

#txt = ['Figure 3']
#x = [lat_extra[n1_extra+1]]
#for i, txt in enumerate(txt):
#    ax.annotate(txt, (x[i], 580), c='k')

ax = axs[1]
idx1, idx2 = np.nanmax(lat_extra), np.nanmin(lat_extra)
data_ALS_scn_filt = data_ALS_scn[(data_ALS_scn[1]<idx1) & (data_ALS_scn[1]>idx2)]
cb = ax.scatter(data_ALS_scn_filt[1], data_ALS_scn_filt[5], c=data_ALS_scn_filt[3], cmap='crest', extend='both' , vmin=np.quantile(data_ALS_scn_filt[3], 0.05), vmax=np.quantile(data_ALS_scn_filt[3], 0.95), s=0.1)
leg1=ax.scatter(lat_ALS_along_radar_track, ALS_scan_number_along_radar_track, s=0.1, c='green')
#ax.axvline(lat_extra[n1_extra], marker=11,
 #          markersize=6, c='k', zorder=15, linewidth=0.5, linestyle='--')
ax.colorbar(cb, label='ALS ellipsoidal elevations (m)', loc='b')
ax.format(xlim=(np.nanmax(lat_extra), np.nanmin(lat_extra)), xlabel='latitude (degrees N)', ylabel='scan number\n(0-251)')
#ax.axvspan(-69.931,-69.935, color='lightgrey', alpha=0.5, edgecolor='grey')
#ax.axvspan(-69.938,-69.942, color='lightgrey', alpha=0.5, edgecolor='grey')
#ax.axvspan(-69.954,-69.972, color='lightgrey', alpha=0.5, edgecolor='grey')
#ax.axvspan(-69.981,-69.998, color='lightgrey', alpha=0.5, edgecolor='grey')
#ax.text(-69.931-0.0001, 210, 'I')
#ax.text(-69.938-0.0001, 210, 'II')
#ax.text(-69.954-0.0001, 210, 'III')
#ax.text(-69.981-0.0001, 210, 'IV')
ax.legend(leg1, 'nadir laser profile', prop=dict(size=8), loc='ur', ncols=4, markersize=10)

#ax1 = ax.twinx()
#df_ALS_roughness_filt = df_ALS_scan_v1_roughness[(df_ALS_scan_v1_roughness['lat']<idx1) & (df_ALS_scan_v1_roughness['lat']>idx2)]
#leg2=ax1.scatter(df_ALS_roughness_filt['lat'], df_ALS_roughness_filt['elevation_anomaly'], s=0.5, linestyle='-', linewidth=0.1, c='k')
#ax1.format(ylabel='elevation anomaly (m)', ylim=(0,1.5))
#ax1.yaxis.set_label_position("right")
#ax1.yaxis.tick_right()

#ax = axs[1].panel_axes('t', width=1.0, space=0)
#df_ALS_roughness_filt = df_ALS_scan_v1_roughness[(df_ALS_scan_v1_roughness['lat']<idx1) & (df_ALS_scan_v1_roughness['lat']>idx2)]
#leg4 = ax.scatter(df_ALS_roughness_filt['lat'], df_ALS_roughness_filt['roughness_anomaly'], s=0.5, linestyle='--', linewidth=0.5, c='grey')
#leg3 = ax.plot(lat_ALS_along_radar_track, ALS_roughness_along_radar_track,  c=cmap_qual[1], linewidth=0.5, linestyle='-')
#ax.format(xlim=(np.nanmax(lat_extra), np.nanmin(lat_extra)), ylim=(0,1), ylabel='roughness, $\sigma$ (m)', xlabel='latitude (degrees N)')

#ax.legend([leg1, leg2, leg4, leg3], ['nadir laser profile', 'elevation anomaly', '$\sigma_{400m}$','$\sigma_{5m}$'], prop=dict(size=8), loc='ur', ncols=4, markersize=10)


'''
leg1 = Line2D([0], [0], color=cmap_qual[0], linestyle='-',
              label='TFMRA40')
leg2 = Line2D([0], [0], color=cmap_qual[6],  linestyle='-',
              label='TFMRA50')
leg3 = Line2D([0], [0], color=cmap_qual[1],  linestyle='-',
              label='TFMRA80')
#leg4 = Line2D([0], [0], color='orange',  linestyle='-',
#              label='Centroid', marker=8, markersize=10)
leg5 = Line2D([0], [0], color='brown', ls='-',
              label='Max peak')
leg6 = Line2D([0], [0], color='green', linestyle='-', label='ALS')

handles = [leg1, leg2, leg3,  leg5, leg6]
frame = axs[0].legend(handles, loc='b', ncols=3, title='Ka/Ku-band',
                      align='left', titlefontweight='bold')
frame._legend_box.align = "left"
frame.get_title().set_weight('bold')

leg1 = Line2D([0], [0], color=cmap_qual[6], linestyle='-',
              label='a-s CWT')
leg2 = Line2D([0], [0], color=cmap_qual[5],  linestyle='-',
              label='s-i CWT')
leg3 = Line2D([0], [0], color=cmap_qual[0],  linestyle='-',
              label='a-s peakiness')
leg4 = Line2D([0], [0], color=cmap_qual[1],  linestyle='-',
              label='s-i peakiness')

handles = [leg1, leg2, leg3, leg4, leg5, leg6]
frame = axs[0].legend(handles, loc='b', ncols=3, order='F',
                      title='C/S-band', align='right', titlefontweight='bold')
frame._legend_box.align = "right"
frame.get_title().set_weight('bold')
frame.get_frame().set_facecolor('lightgrey')
'''

leg5 = Line2D([0], [0], color='brown', ls='-',
              label='Max peak')
leg6 = Line2D([0], [0], color='green', linestyle='-', label='ALS')

leg1 = Line2D([0], [0], color=cmap_qual[6], linestyle='-',
              label='a-s CWT')
leg2 = Line2D([0], [0], color=cmap_qual[5],  linestyle='-',
              label='s-i CWT')
leg3 = Line2D([0], [0], color=cmap_qual[0],  linestyle='-',
              label='a-s peakiness')
leg4 = Line2D([0], [0], color=cmap_qual[1],  linestyle='-',
              label='s-i peakiness')

handles = [leg1, leg2, leg3, leg4, leg5, leg6]
frame = axs[0].legend(handles, loc='b', ncols=6, order='F',
                      title='', titlefontweight='bold', frameon=False)
#frame._legend_box.align = "right"
frame.get_title().set_weight('bold')
#frame.get_frame().set_facecolor('lightgrey')


#handles = [leg5, leg6]
#frame = axs[0].legend(handles, loc='ur', ncols=2)


fig.format(abc="(a)", abcloc='l')


#fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\Figure5_waveforms_max.png', dpi=300)
fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\Figure_example_CRYO2ICEANT22_report.png', dpi=300)




#%% reports CRYO2ICEANT 22

fig, ax = pplt.subplots([[1, 2]], axwidth=2.8, axheight=2.5, spanx=True, sharey=True)

fig.patch.set_facecolor('white')

axs = ax[0]

df_Ku_ALS_diff_v1 = df_Ku_ALS_diff[val_x*n1:val_x*(n1+1)]
df_Ka_ALS_diff_v1 = df_Ka_ALS_diff[val_x*n1:val_x*(n1+1)]
df_snow_ALS_diff_v1 = df_snow_ALS_diff[val_x*n1:val_x*(n1+1)]
class_leads_seaice_v1 = class_leads_seaice[val_x*n1:val_x*(n1+1)]

n_bins = 20
binwidth=0.075
n_bins = np.arange(-0.25, 1.5+ binwidth, binwidth)
hs = ((df_Ku_ALS_diff_v1['ALS'][class_leads_seaice_v1<1])-(df_Ku_ALS_diff_v1['WF_max_corr'][class_leads_seaice_v1<1]))/params['n_snow']
snow_leg1=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} cm'.format(np.nanmean(hs)*100, np.nanstd(hs)*100), edgecolor='k' , c='k', )
hs = ((df_Ka_ALS_diff_v1['ALS'][class_leads_seaice_v1<1])-(df_Ka_ALS_diff_v1['WF_max_corr'][class_leads_seaice_v1<1]))/params['n_snow']
snow_leg2=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} cm'.format(np.nanmean(hs)*100, np.nanstd(hs)*100), edgecolor='k', color=cmap_qual[4 ], alpha=0.75)
hs = ((df_snow_ALS_diff_v1['ALS'][class_leads_seaice_v1<1])-(df_snow_ALS_diff_v1['WF_max_corr'][class_leads_seaice_v1<1]))/params['n_snow']
snow_leg3=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} cm'.format(np.nanmean(hs)*100, np.nanstd(hs)*100),  edgecolor='k', color='grey', alpha=0.5,hatch='///')
#snow_leg3=axs.axvline(np.nanmean(hs), linestyle='--', c='b')

hs1 = ((df_Ku_ALS_diff_v1['ALS'][class_leads_seaice_v1<1])-(df_Ku_ALS_diff_v1['WF_max_corr'][class_leads_seaice_v1<1]))/params['n_snow']
hs1a = ((df_Ku_ALS_diff_v1['ALS'][class_leads_seaice_v1<1])-(df_Ku_ALS_diff_v1['WF_max_corr'][class_leads_seaice_v1<1]))/params['n_snow']
hs1 = hs1[np.abs(hs1)<5]
hs1a = hs1a[np.abs(hs1a)<5]
idx = hs1a>0.05
hs1a = hs1a[idx]
hs2 = ((df_Ka_ALS_diff_v1['ALS'][class_leads_seaice_v1<1])-(df_Ka_ALS_diff_v1['WF_max_corr'][class_leads_seaice_v1<1]))/params['n_snow']
hs2a = ((df_Ka_ALS_diff_v1['ALS'][class_leads_seaice_v1<1])-(df_Ka_ALS_diff_v1['WF_max_corr'][class_leads_seaice_v1<1]))/params['n_snow']
hs2 = hs2[np.abs(hs2)<5]
hs2a = hs2[np.abs(hs2)<5]
idx = hs2a>0.05
hs2a = hs2a[idx]
hs3 = ((df_snow_ALS_diff_v1['ALS'][class_leads_seaice_v1<1])-(df_snow_ALS_diff_v1['WF_max_corr'][class_leads_seaice_v1<1]))/params['n_snow']
hs4 = ((df_snow_ALS_diff_v1['ALS'][class_leads_seaice_v1<1])-(df_snow_ALS_diff_v1['WF_max_corr'][class_leads_seaice_v1<1]))/params['n_snow']
hs3 = hs3[np.abs(hs3)<5]
hs4 = hs4[np.abs(hs4)<5]
idx = hs4>0.05
hs4 = hs4[idx]
frame=axs.legend([snow_leg1, snow_leg2, snow_leg3],['\n\n\nh{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} cm\n>0.05 m = {:.2f} $\pm$ {:.2f} cm'.format('$_{s, ALS-Ku_{corr, MAX}}$',(len(hs)-len(hs1a))/len(hs)*100,np.nanmean(hs1)*100, np.nanstd(hs1)*100,np.nanmean(hs1a)*100, np.nanstd(hs1a)*100), 'h{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} cm\n>0.05 m = {:.2f} $\pm$ {:.2f} cm'.format('$_{s, ALS-Ka_{corr, MAX}}$',(len(hs)-len(hs2a))/len(hs)*100,np.nanmean(hs2)*100, np.nanstd(hs2)*100,np.nanmean(hs2a)*100, np.nanstd(hs2a)*100), 'h{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} cm\n>0.05 m = {:.2f} $\pm$ {:.2f} cm'.format('$_{s, ALS-C/S_{corr, MAX}}$',(len(hs)-len(hs4))/len(hs)*100,np.nanmean(hs3)*100, np.nanstd(hs3)*100, np.nanmean(hs4)*100, np.nanstd(hs4)*100)], handlelength=0,  markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, align='left',loc='ur',)
axs.format(ultitle='h$_s$ (multiple interfaces or \nfrequencies)', xlim=(-0.25,1.5), xlabel='snow depth, h$_s$ (m)', ylabel='frequency (counts)')#frame=axs.legend([snow_leg1, snow_leg2, snow_leg3],['h$_{s, ALS-Ku_{corr}}$', 'h$_{s, ALS-Ka_{corr}}$', 'h$_{s, Ka_{corr}-Ku_{corr}}$'], ncols=2, align='left',loc='b', title='h$_s$, freq. combinations, multiple bands (panel g)')
#frame._legend_box.align = "left"
#frame.get_title().set_weight('bold')
fig.format(fontsize=12)
#fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\Figure_MAX_TFMRA50.png', dpi=300)


axs = ax[1]

n_bins = 20
binwidth=0.075
n_bins = np.arange(-0.25, 1.5+ binwidth, binwidth)
hs = ((df_snow_ALS_diff_v1['as_PEAK'][class_leads_seaice_v1<1])-(df_snow_ALS_diff_v1['si_PEAK'][class_leads_seaice_v1<1]))/params['n_snow']
snow_leg1=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} cm'.format(np.nanmean(hs)*100, np.nanstd(hs)*100), edgecolor='k' , c='k', )
hs = ((df_snow_ALS_diff_v1['as_CWT'][class_leads_seaice_v1<1])-(df_snow_ALS_diff_v1['si_CWT'][class_leads_seaice_v1<1]))/params['n_snow']
snow_leg2=axs.hist(hs, bins=n_bins, label='{:.2f} $\pm$ {:.2f} cm'.format(np.nanmean(hs)*100, np.nanstd(hs)*100), edgecolor='k', color=cmap_qual[4 ], alpha=0.75)
#snow_leg3=axs.axvline(np.nanmean(hs), linestyle='--', c='b')

hs1 = ((df_snow_ALS_diff_v1['as_PEAK'][class_leads_seaice_v1<1])-(df_snow_ALS_diff_v1['si_PEAK'][class_leads_seaice_v1<1]))/params['n_snow']
hs1 = hs1[np.abs(hs1)<5]
hs1a = ((df_snow_ALS_diff_v1['as_PEAK'][class_leads_seaice_v1<1])-(df_snow_ALS_diff_v1['si_PEAK'][class_leads_seaice_v1<1]))/params['n_snow']
hs1a = hs1a[np.abs(hs1a)<5]
idx = hs1a>0.05
hs1a = hs1a[idx]
hs2 = ((df_snow_ALS_diff_v1['as_CWT'][class_leads_seaice_v1<1])-(df_snow_ALS_diff_v1['si_CWT'][class_leads_seaice_v1<1]))/params['n_snow']
hs2a = ((df_snow_ALS_diff_v1['as_CWT'][class_leads_seaice_v1<1])-(df_snow_ALS_diff_v1['si_CWT'][class_leads_seaice_v1<1]))/params['n_snow']
hs2 = hs2[np.abs(hs2)<5]
hs2a = hs2[np.abs(hs2)<5]
idx = hs2a>0.05
hs2a = hs2a[idx]
frame=axs.legend([snow_leg1, snow_leg2, ],['\n\n\nh{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} cm\n>0.05 m = {:.2f} $\pm$ {:.2f} cm'.format('$_{s, a-s_{PEAK, corr}-s-i_{PEAK, corr}}$',(len(hs)-len(hs1a))/len(hs)*100,np.nanmean(hs1)*100, np.nanstd(hs1)*100,np.nanmean(hs1a)*100, np.nanstd(hs1a)*100), 'h{} ({:.2f}%)\n{:.2f} $\pm$ {:.2f} cm\n>0.05 m = {:.2f} $\pm$ {:.2f} cm'.format('$_{s, a-s_{CWT, corr}-s-i_{CWT, corr}}$',(len(hs)-len(hs2a))/len(hs)*100,np.nanmean(hs2)*100, np.nanstd(hs2)*100,np.nanmean(hs2a)*100, np.nanstd(hs2a)*100)], handlelength=0,  markersize=0, linewidth=0, ncols=1, labelcolor='linecolor', frameon=False, align='left',loc='ur',)
axs.format(ultitle='h$_s$ (multiple interfaces, C/S-band)', xlim=(-0.25,1.5), xlabel='snow depth, h$_s$ (m)', ylabel='frequency (counts)')#frame=axs.legend([snow_leg1, snow_leg2, snow_leg3],['h$_{s, ALS-Ku_{corr}}$', 'h$_{s, ALS-Ka_{corr}}$', 'h$_{s, Ka_{corr}-Ku_{corr}}$'], ncols=2, align='left',loc='b', title='h$_s$, freq. combinations, multiple bands (panel g)')
#frame._legend_box.align = "left"
#frame.get_title().set_weight('bold')
fig.format(fontsize=12)

fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\Figure_CRYO2ICEANTT_final_report.png', dpi=300)


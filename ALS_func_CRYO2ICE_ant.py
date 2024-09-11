# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 12:47:06 2023

@author: rmfha

Functions to support main document for the ALS (airborne laser scanner) study 
for CryoVEx 2017 dual-frequency study by HSK. 

"""
#%% Init


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits
import pandas as pd 
import netCDF4
import os
from matplotlib.lines import Line2D

from scipy.interpolate import griddata
from scipy import stats
from matplotlib.patches import Wedge
import proplot as plot
import pyproj
from astropy.time import Time 
import h5py
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from netCDF4 import Dataset
from itertools import chain
import sys
from scipy import interpolate
from netCDF4 import Dataset,num2date
pd.options.mode.chained_assignment = None
import proplot as pplt


# Say, "the default sans-serif font is COMIC SANS"
mpl.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['figure.facecolor'] = 'white'


os.chdir(r"C:\Users\rmfha\Documents\GitHub\ALS_roughness_CryoVEx2017_dual_freuq_study")



def load_data(fp1):
    ds = netCDF4.Dataset(fp1, 'r')
    
    elevation = ds.variables['Elevation'][:]
    amplitude = ds.variables['Amplitude'][:]
    lat = ds.variables['latitude'][:]
    lon = ds.variables['longitude'][:]
    time = ds.variables['t'][:]
    scan_nr = ds.variables['Scannumber'][:]
    
    df = pd.DataFrame({
        'latitude':lat,
        'longitude':lon,
        'time':time,
        'elevation':elevation,
        'amplitude':amplitude,
        'scan_number':scan_nr
        })
    
    return df

def compute_roughness(df):
    df = df.reset_index(drop=True)
    #scan_line, scan_line_total = [], []
    lat_new, lon_new, elevation_MSS_new = [], [], []
    lat_total, lon_total, elev_anomaly_total, roughess_anomaly_total = [],[],[],[],
    points_pr_scan_total, nr_scan_lines_total = [], []
    scan_number_subset = []
    scan_number_array, scan_line_array = [], []
    anomaly_array, lat_array, lon_array = [], [], []
    elevation_new = [],
    elevation_ellip_total, roughness_ellip_total = [], []
    modal_elevation_array = []
    elev_anomaly = []
    k = 1
    for i in np.arange(0, len(df)):
        try:
            if df['scan_number'][i] < df['scan_number'][i+1]:
                lat_new = np.append(lat_new, df['latitude'][i])
                lon_new = np.append(lon_new, df['longitude'][i])
                elevation_new = np.append(elevation_new, df['elevation'][i])
                scan_number_subset = np.append(scan_number_subset,df['scan_number'][i])
                
                from statistics import mode
                elev_mode = mode(elevation_new)
                elev_anomaly = np.append(elev_anomaly, df['elevation'][i] - elev_mode)
                
                
            else:
                if len(elevation_new) > 0:
                    if len(lat_new) > 25:  # needs more than 25 points to compute statistics
                        lat_mean = np.nanmean(lat_new)
                        lon_mean = np.nanmean(lon_new)
            
                        elev_ellip_mean = np.nanmean(elevation_new)
                        roughness_ellip_mean = np.nanstd(elevation_new)
            
                        elev_anomaly_lim = elev_anomaly[elev_anomaly > 0]
                        elev_anomaly_mean = np.nanmean(elev_anomaly_lim)
                        roughness_anomaly_mean = np.nanstd(elev_anomaly_lim)
            
                    else:  # if there are not more than 25 points pr scan line available, then it will not compute roughness
                        lat_mean = np.nanmean(lat_new)
                        lon_mean = np.nanmean(lon_new)
                        elev_anomaly_mean = np.nan
                        roughness_anomaly_mean = np.nan
                        elev_ellip_mean = np.nan
                        roughness_ellip_mean = np.nan
            
                    lat_total = np.append(lat_total, lat_mean)
                    lon_total = np.append(lon_total, lon_mean)
            
                    elev_anomaly_total = np.append(elev_anomaly_total, elev_anomaly_mean)
                    roughess_anomaly_total = np.append(roughess_anomaly_total, roughness_anomaly_mean)
                    points_pr_scan_total = np.append(points_pr_scan_total, len(lat_new))
                    nr_scan_lines_total = np.append(nr_scan_lines_total, k)
                    anomaly_array = np.append(anomaly_array, elev_anomaly)
            
                    lat_array = np.append(lat_array, lat_new)
                    lon_array = np.append(lon_array, lon_new)
                    scan_number_array = np.append(scan_number_array, scan_number_subset)
            
                    elevation_ellip_total = np.append(elevation_ellip_total, elev_ellip_mean)
                    roughness_ellip_total = np.append(roughness_ellip_total, roughness_ellip_mean)
                    
                    if len(elevation_new) == 0:
                        modal_elevation_array = np.append(modal_elevation_array, np.nan)
                    else:
                        modal_elevation_array = np.append(modal_elevation_array, np.ones(len(elevation_new)) * elev_mode)
                    
                   # print('Elevations array: {}'.format(elevation_new))
                   # print('Scan line array: {}'.format(scan_line_array))
                   # print('Modal elevation array: {}'.format(modal_elevation_array))
            
                    scan_line_array = np.append(scan_line_array, np.ones(len(elevation_new)) * k)
                    
                    if len(scan_line_array) != len(modal_elevation_array):
                        print('ERORR in scan line {}: lengths are not equal'.format(k))
                        print('Scan line array: {}'.format(scan_line_array))
                        print('Modal elevation array: {}'.format(modal_elevation_array))
                        break
                    
                    
                    print('Scan line {} is completed'.format(k))
                    k = k + 1
                lat_new, lon_new, scan_number_subset, elevation_new, elev_anomaly = [], [], [], [], []
                
                
                
        except: 
                print('No more data to finish scan line.')

    print('Lengths of arrays into roughness df:')
    print(len(lat_total))
    print(len(lon_total))
    print(len(points_pr_scan_total))
    print(len(nr_scan_lines_total))
    print(len(elev_anomaly_total))
    print(len(roughess_anomaly_total))
    print(len(elevation_ellip_total))
    print(len(roughness_ellip_total))

    df_roughness = pd.DataFrame({
        'lat':lat_total,
        'lon':lon_total, 

        'points_pr_scan':points_pr_scan_total, 
        'scan_line_nr':nr_scan_lines_total, 
        'elevation_anomaly':elev_anomaly_total, 
        'roughness_anomaly':roughess_anomaly_total, 
        #'elevation_ellip':elevation_ellip_total, 
        #'roughness_ellip':roughness_ellip_total
        })
    
    print('Length of arrays into anomaly df:')
    print(len(lat_array))
    print(len(lon_array))
    print(len(anomaly_array))
    print(len(scan_number_array))
    print(len(scan_line_array))
    print(len(modal_elevation_array))
    
    df_anomaly = pd.DataFrame({
        'lat':lat_array, 
        'lon':lon_array,
        'anomaly':anomaly_array,
        'scan_number':scan_number_array, 
        'scan_line':scan_line_array, 
        'modal_elevation':modal_elevation_array
        
        })
    
    #return df_roughness, df
    return df_roughness, df_anomaly


def compute_roughness_v2(df):
    df = df.reset_index(drop=True)
    
    # Initialisation scan-line
    lat_new, lon_new = [], []
    scan_number_subset = []
    scan_number_array, scan_line_array = [], []
    elevation_array = []
    lat_array, lon_array = [], []
    elevation_new = []
    points_pr_scan_total, nr_scan_lines_total = [], []
    modal_elevation_total = []
    modal_elevation_array = []
    k = 1
    #df['scan_lines_nr']=np.nan
    for i in np.arange(0, len(df)):
        try:
            if df['scan_number'][i] < df['scan_number'][i+1]:
                #scan_line = np.append(scan_line, df2_test['scan_number'][i])
                lat_new = np.append(lat_new, df['latitude'][i])
                lon_new = np.append(lon_new, df['longitude'][i])
                #elevation_MSS_new = np.append(elevation_MSS_new, df['elevation_wrt_MSS'][i])
                elevation_new = np.append(elevation_new, df['elevation'][i])
                scan_number_subset = np.append(scan_number_subset,df['scan_number'][i])
                #df['scan_lines_nr'][i]=k
                
                from statistics import mode
                #elev_mode = np.quantile(elevation_new, 0.10)
                elev_mode = mode(elevation_new)
                #elev_anomaly = elevation_new - elev_mode
                
            else:
                
                points_pr_scan_total=np.append(points_pr_scan_total, len(lat_new))
                nr_scan_lines_total = np.append(nr_scan_lines_total, k)
                elevation_array = np.append(elevation_array, elevation_new)
                lat_array = np.append(lat_array,lat_new)
                lon_array = np.append(lon_array,lon_new)
                scan_number_array = np.append(scan_number_array, scan_number_subset)
                scan_line_array = np.append(scan_line_array, np.ones(len(elevation_new))*k)
                modal_elevation_array = np.append(modal_elevation_array, np.ones(len(elevation_new))*elev_mode)
                modal_elevation_total = np.append(modal_elevation_total, elev_mode)
                lat_new, lon_new, scan_number_subset, elevation_new = [], [], [], [],
                print('Scan line {} completed.'.format(k))
                k = k+1
        except: 
                print('No more data to finish scan line.')
                
    df_modal = pd.DataFrame({'modal':modal_elevation_total})  
    modal_elevation_total_rolling = df_modal.rolling(window=5, center=True).mean()
    points_pr_scan_total_v2, nr_scan_lines_total_v2 = [], []
    lat_total, lon_total, elevation_total, roughness_total, elev_anomaly_total, roughess_anomaly_total = [],[],[],[], [], []
    k = 1
    j = 0
    elevation_ellip_total, roughness_ellip_total = [], []
    anomaly_array = []
    lat_new, lon_new, elevation_new = [], [], []
    for i in np.arange(0, len(df)):
        try:
            if df['scan_number'][i] < df['scan_number'][i+1]:
                #scan_line = np.append(scan_line, df2_test['scan_number'][i])
                lat_new = np.append(lat_new, df['latitude'][i])
                lon_new = np.append(lon_new, df['longitude'][i])
                #elevation_MSS_new = np.append(elevation_MSS_new, df['elevation_wrt_MSS'][i])
                elevation_new = np.append(elevation_new, df['elevation'][i])
        
                elev_mode = modal_elevation_total_rolling['modal'][j]
                if np.isnan(elev_mode):
                    elev_anomaly = np.ones(len(elevation_new))*np.nan
                    lat_mean = np.nanmean(lat_new)
                    lon_mean = np.nanmean(lon_new)
                    elev_anomaly_mean = np.nan
                    roughness_anomaly_mean = np.nan
                    elev_ellip_mean = np.nan
                    roughness_ellip_mean = np.nan
                else: 
                    elev_anomaly = elevation_new - elev_mode
                    if len(lat_new)>25: # needs more than 25 points to compute statistics
                        lat_mean = np.nanmean(lat_new)
                        lon_mean = np.nanmean(lon_new)                
                        elev_ellip_mean = np.nanmean(elevation_new)
                        roughness_ellip_mean = np.nanstd(elevation_new)
            
                        elev_anomaly_lim = elev_anomaly[elev_anomaly > 0]
                        elev_anomaly_mean = np.nanmean(elev_anomaly_lim)
                        roughness_anomaly_mean = np.nanstd(elev_anomaly_lim)
                    else: # if there are not more than 25 points pr scan line available, then it will not compute roughness
                        lat_mean = np.nanmean(lat_new)
                        lon_mean = np.nanmean(lon_new)
                        elev_anomaly_mean = np.nan
                        roughness_anomaly_mean = np.nan
                        elev_ellip_mean = np.nan
                        roughness_ellip_mean = np.nan
            
            points_pr_scan_total_v2=np.append(points_pr_scan_total_v2, len(lat_new))
            nr_scan_lines_total_v2 = np.append(nr_scan_lines_total_v2, k)
            lat_total=np.append(lat_total, lat_mean)
            lon_total=np.append(lon_total, lon_mean)
            elev_anomaly_total=np.append(elev_anomaly_total, elev_anomaly_mean)
            roughess_anomaly_total=np.append(roughess_anomaly_total, roughness_anomaly_mean)
            elevation_ellip_total = np.append(elevation_ellip_total,elev_ellip_mean)
            roughness_ellip_total = np.append(roughness_ellip_total,roughness_ellip_mean)
            modal_elevation_array = np.append(modal_elevation_array, np.ones(len(elevation_new))*elev_mode)
            anomaly_array = np.append(anomaly_array, elev_anomaly)
            lat_new, lon_new, scan_number_subset, elevation_new = [], [], [], []
            print('Scan line {} completed with computation of anomaly and roughness.'.format(k))
            k = k+1
            j = j+1
        except: 
            print('No more data to finish computation of roughness.')
        
    print(len(lat_total))
    print(len(lon_total))
    print(len(elev_anomaly_total))
    print(len(roughess_anomaly_total))
    print(len(elevation_ellip_total))
    print(len(roughness_ellip_total))
    print(len(nr_scan_lines_total_v2))
    print(len(points_pr_scan_total_v2))

    df_roughness = pd.DataFrame({
        'lat':lat_total,
        'lon':lon_total, 
        'points_pr_scan':points_pr_scan_total_v2, 
        'scan_line_nr':nr_scan_lines_total_v2, 
        'elevation_anomaly':elev_anomaly_total, 
        'roughness_anomaly':roughess_anomaly_total, 
        'elevation_ellip':elevation_ellip_total, 
        'roughness_ellip':roughness_ellip_total
        })
    
    print(len(lat_array))
    print(len(lon_array))
    print(len(anomaly_array))
    print(len(scan_number_array))
    print(len(scan_line_array))
    
    df_anomaly = pd.DataFrame({
        'lat':lat_array, 
        'lon':lon_array,
        'anomaly':anomaly_array,
        'scan_number':scan_number_array, 
        'scan_line':scan_line_array, 
        'modal_elevation':modal_elevation_array
        
        })
    
    #return df_roughness, df
    return df_roughness, df_anomaly



def identify_along_orbit(data_find, data_check, param_search, lat1='lat', lon1='lon', lat2='lat', lon2='lon', search_type='NN', dist_req=False):
    from sklearn.neighbors import BallTree
    import numpy as np

    #data_find = dataframe to be collocated with (output size)
    #data_check = dataframe to be checked for to find the correct output (NN or radius search)
    ## Only data within 825 m of CryoSat-2 footprint (find approximate area of coincident data)
    
    
    query_lats = data_find[[lat1]].to_numpy()
    query_lons = data_find[[lon1]].to_numpy()
    
    tree = BallTree(np.deg2rad(data_check[[lat2, lon2]].values),  metric='haversine')
    
    if search_type == 'NN':
        distances, indices = tree.query(np.deg2rad(np.c_[query_lats, query_lons]), k=1)
        new_param = []
        for i in indices:
            new_param = np.append(MSS, data_check[param_search][int(i)])
    elif search_type == 'RADIUS':
        dist_in_metres = dist_req
        earth_radius_in_metres = 6371*1000
        radius = dist_in_metres/earth_radius_in_metres

        is_within, distances = tree.query_radius(np.deg2rad(np.c_[query_lats, query_lons]), r=radius, count_only=False, return_distance=True) 
        distances_in_metres = distances*earth_radius_in_metres
        
        new_param,new_lat_loc,new_lon_loc =  np.empty(len(query_lats)),np.empty(len(query_lats)),np.empty(len(query_lats))
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
                print('Observation point {}/{}: No points within search requirements.'.format(k, len(query_lats)))
            
            new_param[k] = mean_new_param_comp
            new_lat_loc[k] = mean_lat_comp
            new_lon_loc[k] = mean_lon_comp
            k = k+1
    else: 
        print('Search type not correct. Input either "NN" or "RADIUS"')

#    data_check['MSS_DTU21'] = MSS
    
    return new_param, new_lat_loc, new_lon_loc


def identify_close_to_in_situ(data_find, data_check, param_search, param_search2, lat1='lat', lon1='lon', lat2='lat', lon2='lon', dist_req=False):
    from sklearn.neighbors import BallTree
    import numpy as np

    #data_find = dataframe to be collocated with (output size)
    #data_check = dataframe to be checked for to find the correct output (NN or radius search)
    
    
    query_lats = data_find[[lat1]].to_numpy()
    query_lons = data_find[[lon1]].to_numpy()
    
    tree = BallTree(np.deg2rad(data_check[[lat2, lon2]].values),  metric='haversine')
    
    dist_in_metres = dist_req
    earth_radius_in_metres = 6371*1000
    radius = dist_in_metres/earth_radius_in_metres

    is_within, distances = tree.query_radius(np.deg2rad(np.c_[query_lats, query_lons]), r=radius, count_only=False, return_distance=True) 
    #distances_in_metres = distances*earth_radius_in_metres

    for i in is_within:
        array_rel = data_check[param_search].iloc[i]
        array_rel2 = data_check[param_search2].iloc[i]
        array_rel_lat = data_check[lat2].iloc[i]
        array_rel_lon = data_check[lon2].iloc[i]

    
    return array_rel, array_rel2,array_rel_lat, array_rel_lon
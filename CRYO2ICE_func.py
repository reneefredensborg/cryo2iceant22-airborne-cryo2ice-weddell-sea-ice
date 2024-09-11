# -*- coding: utf-8 -*-
"""
Created on Mon May 15 13:15:53 2023

Functions for the CRYO2ICE Antarctic under-flight comparison study. 

@author: rmfha
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits
import pandas as pd 
import netCDF4
import os
from matplotlib.lines import Line2D
os.environ['PROJ_LIB'] = 'C:/Users/renee/anaconda3/Lib/site-packages/mpl_toolkits/basemap'
os.environ['GMT_LIBRARY_PATH']=r'C:\Users\USERNAME\Anaconda3\envs\pygmt\Library\bin'
#from mpl_toolkits.basemap import Basemap 
#from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.interpolate import griddata
from scipy import stats
from matplotlib.patches import Wedge
import proplot as plot
import pyproj
#from astropy.time import Time 
import h5py
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from netCDF4 import Dataset
#from mpl_toolkits.basemap import Basemap
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

import pyproj
#from astropy.time import Time 
import h5py
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from netCDF4 import Dataset
#from mpl_toolkits.basemap import Basemap
from itertools import chain
import sys

def list_files_local(path):
    ''' Get file list from local folder. '''
    from glob import glob
    return glob(path)

def gps2dyr(time):
    '''Convert GPS time to decimal years'''
    return Time(time, format='gps').decimalyear

def track_type(time, lat, tmax=1):
    '''
    Separate tracks into ascending and descending. 
    
    Defines tracks as segments with time breaks > tmax , 
    and tests whether lat increases or decreases w/time.
    '''
    tracks = np.zeros(lat.shape) # generates track segment
    tracks[0:np.argmax(np.abs(lat))] = 1 # set values for segment
    i_asc = np.zeros(tracks.shape, dtype=bool) # output index array 
    
    # loop through individual segments
    for track in np.unique(tracks):
        
        i_track, = np.where(track == tracks) # get all pts from seg
        
        if len(i_track) < 2: continue
            
        # Test if lat increase (asc) or descreases (des) with time
        i_min = time[i_track].argmin()
        i_max = time[i_track].argmax()
        lat_diff = lat[i_track][i_max] - lat[i_track][i_min]
        
        # Determine track trype
        if lat_diff > 0: i_asc[i_track] = True
            
    return i_asc, np.invert(i_asc)

def transform_coord(proj1, proj2, x, y):
    '''
    Transform coordinates from proj1 to proj2 (EPSG num). 
    
    Example EPSG projs: 
        Geodetic (lon/lat): 4326
        Polar Stereo AnIS (x/y): 3031
        Polar Stereo GrIS (x/y): 3413
    '''
    # Set full EPSG projection strings 
    proj1 = pyproj.Proj("+init=EPSG:"+str(proj1))
    proj2 = pyproj.Proj("+init=EPSG:"+str(proj2))
    return pyproj.transform(proj1, proj2, x, y)  # convert

def read_h5(fname, vnames=[]):
    '''Simple HDF5 reader'''
    with h5py.File(fname, 'r') as f:
        return [f[v][:] for v in vnames]

def read_data(fname, bbox=None):
    '''
    Read ATL03 data file and output 6 reduced files. 
    
    Extract variables of interest and separate the ATL03 file
    into each beam (ground track) and ascending/descending orbits. 
    '''
    
    # Each beam is a group 
    group = ['/gt1l', '/gt1r', '/gt2l', '/gt2r', '/gt3l', '/gt3r']
    
    # Loop through beams
    for k,g in enumerate(group):
    
        #-----------------------------------#
        # 1) Read in data for a single beam #
        #-----------------------------------#
    
        # Load variables into memory (more can be added!)
        with h5py.File(fname, 'r') as fi:  

            ''' Only for ATL10-QL
            lat_fb = fi[g+ '/freeboard_beam_segment/beam_freeboard/latitude'][:]
            lon_fb = fi[g+ '/freeboard_beam_segment/beam_freeboard/longitude'][:]
            fb = fi[g+ '/freeboard_beam_segment/beam_freeboard/beam_fb_height'][:]
            gw_fb = fi[g+ '/freeboard_beam_segment/height_segments/height_segment_w_gaussian'][:]
            mss = fi[g+ '/freeboard_beam_segment/geophysical/height_segment_mss'][:]
            mean_tide = fi[g+ '/freeboard_beam_segment/geophysical/height_segment_geoid_free2mean'][:]
            #SET_corr = fi[g+ '/freeboard_beam_segment/geophysical/height_segment_earth'][:]
            SET_corr = fi[g+ '/freeboard_beam_segment/geophysical/height_segment_earth_free2mean'][:]
            #mss_meantide = mss-(mean_tide+SET_corr)#+SET_corr
            mss_meantide = mss+mean_tide+SET_corr
            '''

            lat_fb = fi[g+ '/freeboard_segment/latitude'][:]
            lon_fb = fi[g+ '/freeboard_segment/longitude'][:]
            fb = fi[g+ '/freeboard_segment/beam_fb_height'][:]
            gw_fb = fi[g+ '/freeboard_segment/heights/height_segment_w_gaussian'][:]
            mss = fi[g+ '/freeboard_segment/geophysical/height_segment_mss'][:]
            mean_tide = fi[g+ '/freeboard_segment/geophysical/height_segment_geoid_free2mean'][:]
            #SET_corr = fi[g+ '/freeboard_segment/geophysical/height_segment_earth'][:]
            SET_corr = fi[g+ '/freeboard_segment/geophysical/height_segment_earth_free2mean'][:]
            #mss_meantide = mss-(mean_tide+SET_corr)#+SET_corr
            mss_meantide = mss+mean_tide+SET_corr
            
            #lat_lead = fi[g+ '/leads/latitude'][:]
            #lon_lead = fi[g+ '/leads/longitude'][:]
            #orb = np.full_like(h_ph, k)
        #---------------------------------------------#
        # 2) Filter data according region and quality #
        #---------------------------------------------#
       
        # Test for no data
        #if len(lat_fb) == 0: continue

        #-------------------------------------#
        # 3) Convert time and separate tracks #
        #-------------------------------------#
        
        # Time in GPS seconds (secs since 1980...)
        #t_gps, delta_t_gps = t_ref + t_dt, t_ref + delta_t

        # Time in decimal years
        #t_year, delta_t_year = gps2dyr(t_gps), gps2dyr(delta_t_gps)

        # Determine orbit type
        #i_asc, i_des = track_type(t_year, lat)
        
        #-----------------------#
        # 4) Save selected data #
        #-----------------------#
        
        # Define output file name
        ofile = fname.replace('.h5', '_'+g[1:]+'.h5')
                
        # Save variables
        with h5py.File(ofile, 'w') as f:
            #f['orbit'] = orb
            f['lon_fb'] = lon_fb
            f['lat_fb'] = lat_fb
            f['fb'] = fb - mss_meantide
            f['gw'] = gw_fb
            f['fb_with_mss'] = fb
            f['mss']=mss_meantide
            #f['lon_lead'] = lon_lead
            #f['lat_lead'] = lat_lead
            
            print('out ->', ofile)
            
#def load_data(path, fp):
#    nc_file = netCDF4.Dataset(path + '/' + fp, 'r')
#    sea_ice_freeboard = nc_file.variables['freeboard_20_ku'][:]
#    lat = nc_file.variables['lat_poca_20_ku'][:]
#    lon = nc_file.variables['lon_poca_20_ku'][:]
    
#    return  lon, lat, sea_ice_freeboard

def get_valid_freeboard_flag(flag): 
    """
    flag_masks = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456; // int 
    
    removed flags in dict_error: 
    #        'sarin_bad_velocity'       : 2,
    #        'sarin_out_of_range'       : 4,
    #        'sarin_bad_baseline'       : 8,
    #        'delta_time_error'         :32,
    #        'mispointing_error'        :64,
    #        'sarin_height_ambiguous'   :2048,
    
    """

    dict_error= {
        'calibration_warning'      : 1,
        'sarin_bad_velocity'       : 2,
        'sarin_out_of_range'       : 4,
        'sarin_bad_baseline'       : 8,
        'delta_time_error'         :32,
        'mispointing_error'        :64,
        'sarin_side_redundant'     :256,
        'sarin_rx_2_error'         :512,
        'sarin_rx_1_error'         :1024,
        'sarin_height_ambiguous'   :2048,
        'surf_type_class_ocean'    :32768,
        #'freeboard_error'          :65536,
        #'peakiness_error'          :131072,
        #'ssha_interp_error'        :262144,
        'orbit_discontinuity'      :33554432,
        'orbit_error'              :67108864,
        #'height_sea_ice_error'     :268435456,
        }

    all_flag= {
        'calibration_warning'      : 1,
        'sarin_bad_velocity'       : 2,
        'sarin_out_of_range'       : 4,
        'sarin_bad_baseline'       : 8,
        'lrm_slope_model_invalid'  :16,
        'delta_time_error'         :32,
        'mispointing_error'        :64,
        'surface_model_unavailable':128,
        'sarin_side_redundant'     :256,
        'sarin_rx_2_error'         :512,
        'sarin_rx_1_error'         :1024,
        'sarin_height_ambiguous'   :2048,
        'surf_type_class_undefined':4096,
        'surf_type_class_sea_ice'  :8192,
        'surf_type_class_lead'     :16384,
        'surf_type_class_ocean'    :32768,
        'freeboard_error'          :65536,
        'peakiness_error'          :131072,
        'ssha_interp_error'        :262144,
        'sig0_3_error'             :524288,
        'sig0_2_error'             :1048576,
        'sig0_1_error'             :2097152,
        'height_3_error'           :4194304,
        'height_2_error'           :8388608,
        'height_1_error'           :16777216,
        'orbit_discontinuity'      :33554432,
        'orbit_error'              :67108864,
        'block_degraded'           :134217728,
        'height_sea_ice_error'     :268435456,
        }

    # sea ice class
    flag_seaice = np.bitwise_and(flag, all_flag['surf_type_class_sea_ice'])/all_flag['surf_type_class_sea_ice']

    # errors
    flag_error = np.zeros(flag.size)
    for key in dict_error.keys():
        flag0_error = np.bitwise_and(flag,dict_error[key])/dict_error[key]
        flag_error = np.logical_or(flag0_error,flag_error)

    flag_valid_fb = flag_seaice - flag_error
    flag_valid_fb[flag_valid_fb<0] =0


    return flag_valid_fb

def fit2(lat, lat2,  h, degree):
    p_fit = np.polyfit(lat, h, degree)
    p_val = np.polyval(p_fit, lat2)
    return p_val

def NN_search(df_CS2_new, df_IS2):
    from sklearn.neighbors import BallTree
    import numpy as np
    import pandas as pd

    query_lats = df_CS2_new[['lat']].to_numpy()
    query_lons = df_CS2_new[['lon']].to_numpy()

    tree = BallTree(np.deg2rad(df_IS2[['lat', 'lon']].values), leaf_size =15, metric='haversine')

    distances, indices = tree.query(np.deg2rad(np.c_[query_lats, query_lons]), k=1)
    
    mss_interp = [], [], [],[]
    for i in indices:
        mss_interp = np.append(mss_interp, df_IS2['mss'][int(i)])

    return mss_interp

def load_data(fp1,fp2):
    from scipy.interpolate import interp1d, UnivariateSpline
    import datetime # Python standard library datetime module
    from netCDF4 import Dataset,num2date # http://unidata.github.io/netcdf4-python/

    ds = netCDF4.Dataset(fp1, 'r')
    
    sea_ice_freeboard = ds.variables['radar_freeboard_20_ku'][:]
    lat = ds.variables['lat_poca_20_ku'][:]
    lon = ds.variables['lon_poca_20_ku'][:]
    time = ds.variables['time_20_ku'][:]
    flag = ds.variables[ 'flag_prod_status_20_ku'][:]
    flag_valid = get_valid_freeboard_flag(flag)
    mss = ds.variables['mean_sea_surf_sea_ice_01'][:]
    lat_01 = ds.variables['lat_01'][:]
    lon_01 = ds.variables['lon_01'][:]
    
    
    df_1hz = pd.DataFrame({'lat':lat_01, 'lon':lon_01, 'mss':mss})
    df_20hz = pd.DataFrame({'lat':lat, 'lon':lon})
    mss_interp = NN_search(df_20hz, df_1hz)
    sif_no_mss = sea_ice_freeboard-mss_interp
    
    tname = "time_20_ku"
    nctime = ds.variables[tname][:] # get values
    t_unit = ds.variables[tname].units # get unit  "days since 1950-01-01T00:00:00Z"
    t_cal = ds.variables[tname].calendar
    tvalue = num2date(nctime,units = t_unit,calendar = t_cal)
    #str_time = [i.strftime("%Y-%m-%d %H:%M:%S") for i in tvalue] # to display dates as string
    str_time = [str(i) for i in tvalue]
    
    ds2 = netCDF4.Dataset(fp2, 'r')
    ssd =  ds2.variables['stack_std_20_ku'][:]
    pwr_waveform = ds2.variables['pwr_waveform_20_ku'][:]
    ppk = []

    for i in pwr_waveform:
        calc_pp = []
        noise = np.nanmean(i[10:19])
        i = i[i>noise]

        pp = np.nanmax(i)/np.nanmean(i)
        ppk = np.append(ppk,pp)
    
    df = pd.DataFrame({'sif':sea_ice_freeboard, 'lat':lat, 'lon':lon, 'flag':flag_valid, 'mss':mss_interp, 'ssd':ssd, 'ppk':ppk, 'time':str_time})
    
    df = df[df['lat']<-60]
    df = df[df['flag']==1]  
    df=df.reset_index(drop=True)
    
    lon_filtered = np.array(df['lon'])
    lat_filtered = np.array(df['lat'])
    fb_filtered = np.array(df['sif'])
    mss_filtered = np.array(df['mss'])
    ssd_filtered = np.array(df['ssd'])
    ppk_filtered = np.array(df['ppk'])
    time_filtered = np.array(df['time'])
    return lon_filtered, lat_filtered, fb_filtered, mss_filtered, ssd_filtered,ppk_filtered, time_filtered


def load_all_data(k, files_check_IS2a, files_check_IS2b, files_check_IS2c,files_check_IS2d, files_check_IS2e, files_check_IS2f, files_check_CS2_L2, files_check_CS2_L1b, list_total,dist_req):
    
    ### IS2
    #lon_gt2l, lat_gt2l,fb_gt2l, gw_gt2l, fb_mss, mss = read_h5(files_check_IS2[k], ['lon_fb', 'lat_fb', 'fb', 'gw', 'fb_with_mss', 'mss'])
    #gw_gt2l[fb_gt2l>5] = np.nan
    #fb_gt2l[fb_gt2l>5] = np.nan

    #df_IS2 = pd.DataFrame({'lat':lat_gt2l, 'lon':lon_gt2l, 'fb':fb_gt2l, 'gw':gw_gt2l, 'fb_with_mss':fb_mss, 'mss_IS2':mss})
    #df_IS2 = df_IS2.dropna().reset_index()
    
    #lon_gt2l, lat_gt2l,fb_gt2l, gw_gt2l, fb_mss, mss = read_h5(files_check_IS2[k], ['lon_fb', 'lat_fb', 'fb', 'gw', 'fb_with_mss', 'mss'])
    lon_gt2l, lat_gt2l,fb_gt2l, gw_gt2l, fb_mss, mss = read_h5(files_check_IS2a[k], ['lon_fb', 'lat_fb', 'fb', 'gw', 'fb_with_mss', 'mss'])
    gw_gt2l[fb_gt2l>5] = np.nan
    fb_gt2l[fb_gt2l>5] = np.nan
    df_IS2a = pd.DataFrame({'lat':lat_gt2l, 'lon':lon_gt2l, 'fb':fb_gt2l, 'gw':gw_gt2l, 'fb_with_mss':fb_mss, 'mss_IS2':mss})
    df_IS2a['beam_ID']=1

    lon_gt2l, lat_gt2l,fb_gt2l, gw_gt2l, fb_mss, mss = read_h5(files_check_IS2b[k], ['lon_fb', 'lat_fb', 'fb', 'gw', 'fb_with_mss', 'mss'])
    gw_gt2l[fb_gt2l>5] = np.nan
    fb_gt2l[fb_gt2l>5] = np.nan
    df_IS2b = pd.DataFrame({'lat':lat_gt2l, 'lon':lon_gt2l, 'fb':fb_gt2l, 'gw':gw_gt2l, 'fb_with_mss':fb_mss, 'mss_IS2':mss})
    df_IS2b['beam_ID']=2

    lon_gt2l, lat_gt2l,fb_gt2l, gw_gt2l, fb_mss, mss = read_h5(files_check_IS2c[k], ['lon_fb', 'lat_fb', 'fb', 'gw', 'fb_with_mss', 'mss'])
    gw_gt2l[fb_gt2l>5] = np.nan
    fb_gt2l[fb_gt2l>5] = np.nan
    df_IS2c = pd.DataFrame({'lat':lat_gt2l, 'lon':lon_gt2l, 'fb':fb_gt2l, 'gw':gw_gt2l, 'fb_with_mss':fb_mss, 'mss_IS2':mss})
    df_IS2c['beam_ID']=3
    
    lon_gt2l, lat_gt2l,fb_gt2l, gw_gt2l, fb_mss, mss = read_h5(files_check_IS2d[k], ['lon_fb', 'lat_fb', 'fb', 'gw', 'fb_with_mss', 'mss'])
    gw_gt2l[fb_gt2l>5] = np.nan
    fb_gt2l[fb_gt2l>5] = np.nan
    df_IS2d = pd.DataFrame({'lat':lat_gt2l, 'lon':lon_gt2l, 'fb':fb_gt2l, 'gw':gw_gt2l, 'fb_with_mss':fb_mss, 'mss_IS2':mss})
    df_IS2d['beam_ID']=4

    lon_gt2l, lat_gt2l,fb_gt2l, gw_gt2l, fb_mss, mss = read_h5(files_check_IS2e[k], ['lon_fb', 'lat_fb', 'fb', 'gw', 'fb_with_mss', 'mss'])
    gw_gt2l[fb_gt2l>5] = np.nan
    fb_gt2l[fb_gt2l>5] = np.nan
    df_IS2e = pd.DataFrame({'lat':lat_gt2l, 'lon':lon_gt2l, 'fb':fb_gt2l, 'gw':gw_gt2l, 'fb_with_mss':fb_mss, 'mss_IS2':mss})
    df_IS2e['beam_ID']=5

    lon_gt2l, lat_gt2l,fb_gt2l, gw_gt2l, fb_mss, mss = read_h5(files_check_IS2f[k], ['lon_fb', 'lat_fb', 'fb', 'gw', 'fb_with_mss', 'mss'])
    gw_gt2l[fb_gt2l>5] = np.nan
    fb_gt2l[fb_gt2l>5] = np.nan
    df_IS2f = pd.DataFrame({'lat':lat_gt2l, 'lon':lon_gt2l, 'fb':fb_gt2l, 'gw':gw_gt2l, 'fb_with_mss':fb_mss, 'mss_IS2':mss})
    df_IS2f['beam_ID']=6
    
    frames = [df_IS2a, df_IS2b, df_IS2c, df_IS2d, df_IS2e, df_IS2f]
    #frames = [df_IS2a, df_IS2b, df_IS2c]
    df_IS2 = pd.concat(frames)
    df_IS2 = df_IS2.dropna().reset_index()
    
    ### CS2
    fp1 = files_check_CS2_L2[k]
    fp2 = files_check_CS2_L1b[k]
        
    lon, lat,fb, mss, ssd, ppk, time = load_data(fp1, fp2)
    df_CS2 = pd.DataFrame({'lat':lat, 'lon':lon, 'fb':fb, 'mss':mss, 'fb_no_mss':fb-mss, 'ssd':ssd, 'ppk':ppk, 'time':time})
    df_CS2 = df_CS2[(df_CS2['lat']<-60)].reset_index()
    
    df_CS2_new = CRYO2ICE_smooth_CS2_data(df_CS2,dist_req)
    
    frames = [df_CS2, df_CS2_new]
    df_CS2 = pd.concat(frames, axis=1)
    
    
    basename_without_ext = os.path.splitext(os.path.basename(fp1))[0]
    
    print('File '+str(k)+'/'+str(len(list_total)-1)+ ': ' + basename_without_ext)
    
    #fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(10,8), constrained_layout=True)

    #im = plot_panArctic2(np.array(df_CS2['lon']), np.array(df_CS2['lat']), 'r', 0, 0.5, 'viridis', 'radar freeboard (m)','max', ax[0],fig)
    #im=plot_panArctic2(np.array(df_IS2['lon']), np.array(df_IS2['lat']), 'g', 0, 0.5, 'viridis', 'radar freeboard (m)','max', ax[0],fig)

    #plt.colorbar(im, ax=ax[1], orientation = 'vertical', extend='both', shrink=0.5)
    #plt.show()
    
    return df_IS2, df_CS2, fp1           
            
def CRYO2ICE_identify_IS2_data2(df_CS2, df_IS2, dist_req):
    from sklearn.neighbors import BallTree
    import numpy as np
    import pandas as pd

    query_lats = df_CS2[['lat']].to_numpy()
    query_lons = df_CS2[['lon']].to_numpy()

    tree = BallTree(np.deg2rad(df_IS2[['lat', 'lon']].values),  metric='haversine')

    dist_in_metres = dist_req
    earth_radius_in_metres = 6371*1000
    radius = dist_in_metres/earth_radius_in_metres

    is_within, distances = tree.query_radius(np.deg2rad(np.c_[query_lats, query_lons]), r=radius, count_only=False, return_distance=True) 
    distances_in_metres = distances*earth_radius_in_metres
    

    mean_fb_IS2, mean_gw_IS2, mss_IS2_mean, w_mean_fb_IS2  =  np.empty(len(query_lats)),np.empty(len(query_lats)),np.empty(len(query_lats)),np.empty(len(query_lats))
    dist_std_all, dist_avg_all, dist_min_all, dist_max_all = np.empty(len(query_lats)), np.empty(len(query_lats)),np.empty(len(query_lats)),np.empty(len(query_lats))
    GT1L, GT2L, GT3L, GT1R, GT2R, GT3R = np.empty(len(query_lats)), np.empty(len(query_lats)), np.empty(len(query_lats)), np.empty(len(query_lats)), np.empty(len(query_lats)), np.empty(len(query_lats))
    k = 0
    
    for i in is_within:
        GT1L_count, GT2L_count, GT3L_count, GT1R_count, GT2R_count, GT3R_count = 0,0,0,0,0,0
        beam_ID_all = df_IS2['beam_ID'].iloc[i]
        if len(i)>10:
            data_fb_mean_IS2 = df_IS2['fb'].iloc[i]
            data_gw_mean_IS2 = df_IS2['gw'].iloc[i]
            data_mss_mean_IS2 = df_IS2['mss_IS2'].iloc[i]
        
            mean_fb_val = np.nanmean(data_fb_mean_IS2)

            #masked_data = np.ma.masked_array(data_fb_mean_IS2, np.isnan(data_fb_mean_IS2))
            w_mean_fb_val = np.ma.average(data_fb_mean_IS2, weights=(1/distances_in_metres[k]))
            #w_mean_fb_val = np.ma.average(masked_data, weights=(1/(distances_in_metres[k]/dist_in_metres)))
            #w_mean_fb_val = sum(np.array(data_fb_mean_IS2)/np.array(distances_in_metres[k]))/sum(1/np.array(distances_in_metres[k]))

            #masked_data = np.ma.masked_array(data_mss_mean_IS2, np.isnan(data_fb_mean_IS2))
            mss_IS2_mean_val = np.ma.average(data_mss_mean_IS2, weights=(1/distances_in_metres[k]))
            #mss_IS2_mean_val = np.ma.average(masked_data, weights=(1/(distances_in_metres[k]/dist_in_metres)))
            #w_mean_fb_val = sum(np.array(data_fb_mean_IS2)/np.array(distances_in_metres[k]))/sum(1/np.array(distances_in_metres[k]))

            #mss_IS2_mean_val = np.nanmean(data_mss_mean_IS2)

            #masked_data = np.ma.masked_array(data_gw_mean_IS2, np.isnan(data_fb_mean_IS2))
            mean_gw_val = np.ma.average(data_gw_mean_IS2, weights=(1/distances_in_metres[k]))
            #mean_gw_val = np.ma.average(masked_data, weights=(1/(distances_in_metres[k]/dist_in_metres)))
            #w_mean_fb_val = sum(np.array(data_fb_mean_IS2)/np.array(distances_in_metres[k]))/sum(1/np.array(distances_in_metres[k]))

            #mean_gw_val = np.nanmean(data_gw_mean_IS2)
            
            dist_min = np.nanmin(distances_in_metres[k])
            dist_max = np.nanmax(distances_in_metres[k])
            dist_avg = np.nanmean(distances_in_metres[k])
            dist_std = np.nanstd(distances_in_metres[k])
            
        else:
            mean_fb_val = np.nan
            w_mean_fb_val = np.nan
            mss_IS2_mean_val = np.nan
            mean_gw_val = np.nan
            dist_min = np.nan
            dist_max = np.nan
            dist_avg = np.nan
            dist_std = np.nan
        
        for j in beam_ID_all:
            if j == 1:
                GT1L_count += 1
            elif j == 2:
                GT2L_count += 1
            elif j == 3:
                GT3L_count += 1
            elif j == 4:
                GT1R_count += 1
            elif j == 5:
                GT2R_count += 1
            elif j == 6:
                GT3R_count += 1

        
        mean_fb_IS2[k] = mean_fb_val
        w_mean_fb_IS2[k] = w_mean_fb_val
        mss_IS2_mean[k] = mss_IS2_mean_val
        mean_gw_IS2[k] = mean_gw_val
        dist_avg_all[k] = dist_avg
        dist_min_all[k] = dist_min
        dist_max_all[k] = dist_max
        dist_std_all[k] = dist_std
        
        GT1R[k] = GT1R_count
        GT2R[k] = GT2R_count
        GT3R[k] = GT3R_count
        GT1L[k] = GT1L_count
        GT2L[k] = GT2L_count
        GT3L[k] = GT3L_count
        
        k = k+1
    
    
    df_IS2_new = pd.DataFrame({'IS2_mean_fb':mean_fb_IS2, 'IS2_w_mean_fb':w_mean_fb_IS2,'IS2_mean_gw':mean_gw_IS2, 'IS2_mean_MSS':mss_IS2_mean, 'IS2_dist_min':dist_min_all,
                               'IS2_dist_max':dist_max_all, 'IS2_dist_avg':dist_avg_all, 'IS2_dist_std':dist_std_all, 'GT1L_count':GT1L, 'GT2L_count':GT2L, 'GT3L_count':GT3L, 
                               'GT1R_count':GT1R, 'GT2R_count':GT2R, 'GT3R_count':GT3R})
    
    return df_IS2_new

def numpy_nan_mean(a):
    
    return np.nan if np.all(a!=a) else np.nanmean(a)

def CRYO2ICE_smooth_CS2_data(df_CS2,dist_req,var='fb'):
    from sklearn.neighbors import BallTree
    import numpy as np
    import pandas as pd

    query_lats = df_CS2[['lat']].to_numpy()
    query_lons = df_CS2[['lon']].to_numpy()

    tree = BallTree(np.deg2rad(df_CS2[['lat', 'lon']].values),  metric='haversine')

    dist_in_metres = dist_req
    earth_radius_in_metres = 6371*1000
    radius = dist_in_metres/earth_radius_in_metres

    is_within, distances = tree.query_radius(np.deg2rad(np.c_[query_lats, query_lons]), r=radius, count_only=False, return_distance=True) 
    distances_in_metres = distances*earth_radius_in_metres

    w_mean_fb_IS2 = np.empty(len(query_lats))
    k = 0
    for i in is_within:
        data_fb_mean_IS2 = df_CS2[var].iloc[i]
        w_mean_fb_IS2[k] = numpy_nan_mean(data_fb_mean_IS2)


        k = k+1
    
    df_CS2_new = pd.DataFrame({'CS2_smooth':w_mean_fb_IS2})
    
    return df_CS2_new


def CRYO2ICE_identify(df_CS2, df_IS2,dist_req):
    df_IS2_new = CRYO2ICE_identify_IS2_data2(df_CS2, df_IS2,dist_req)
    
    frames = [df_CS2, df_IS2_new]
    data_check = pd.concat(frames, axis=1)
    
    #data_check = data_check.dropna()
    
    return data_check

def CRYO2ICE_AMSR2_NN(df3_short,df_AMSR2):
    from sklearn.neighbors import BallTree
    import numpy as np
    import pandas as pd

    ## Only data within 825 m of CryoSat-2 footprint (find approximate area of coincident data)
    query_lats = df3_short[['lat']].to_numpy()
    query_lons = df3_short[['lon']].to_numpy()

    tree = BallTree(np.deg2rad(df_AMSR2[['lat', 'lon']].values), leaf_size =15, metric='haversine')

    distances, indices = tree.query(np.deg2rad(np.c_[query_lats, query_lons]), k=1)
    
    data_check = df3_short
    NN_SIC, NN_SD= [], []
    for i in indices:
        NN_SIC = np.append(NN_SIC, df_AMSR2['sea_ice_concentration'][int(i)])
        NN_SD = np.append(NN_SD, df_AMSR2['snow_depth'][int(i)])

    data_check['AMSR2_SIC'] = NN_SIC
    data_check['AMSR2_snow_depth'] = NN_SD
    
    return data_check

def CRYO2ICE_CASSIS_NN(df3_short,df_AMSR2):
    from sklearn.neighbors import BallTree
    import numpy as np
    import pandas as pd

    ## Only data within 825 m of CryoSat-2 footprint (find approximate area of coincident data)
    query_lats = df3_short[['lat']].to_numpy()
    query_lons = df3_short[['lon']].to_numpy()

    tree = BallTree(np.deg2rad(df_AMSR2[['lat', 'lon']].values), leaf_size =15, metric='haversine')

    distances, indices = tree.query(np.deg2rad(np.c_[query_lats, query_lons]), k=1)
    
    data_check = df3_short
    NN_SIC, NN_SD= [], []
    for i in indices:
        #NN_SIC = np.append(NN_SIC, df_AMSR2['sea_ice_concentration'][int(i)])
        NN_SD = np.append(NN_SD, df_AMSR2['snow_depth'][int(i)])

    #data_check['AMSR2_SIC'] = NN_SIC
    data_check['CASSIS_snow_depth'] = NN_SD
    
    return data_check
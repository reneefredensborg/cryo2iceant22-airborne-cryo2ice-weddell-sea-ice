# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:17:49 2024

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
cmap_qual = [rgb2hex(cmap_check((0/8))),rgb2hex(cmap_check((1/8))),rgb2hex(cmap_check((2/8))),
            rgb2hex(cmap_check((5/8))), rgb2hex(cmap_check(6/8)), rgb2hex(cmap_check(7/8)), rgb2hex(cmap_check(8/8))]
cmap_qual2 = LinearSegmentedColormap.from_list('list', cmap_qual, N = len(cmap_qual))
cmap_use = plt.cm.get_cmap('RdYlBu_r', 6) 

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


def read_h5(fname, vnames=[]):
    '''Simple HDF5 reader'''
    with h5py.File(fname, 'r') as f:
        return [f[v][:] for v in vnames]
    
    
from CRYO2ICE_func import load_all_data, CRYO2ICE_identify,CRYO2ICE_AMSR2_NN,CRYO2ICE_CASSIS_NN, CRYO2ICE_smooth_CS2_data

#%%

files = [r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data\ATL10-02_20221213173642_12831701_006_02.h5']


#%% Prep ATL10QL
#
from CRYO2ICE_func import read_data

njobs = 1

# bbox = [lonmin, lonmax, latmin, latmax]
bbox = [0,360, -60, -90]

if njobs == 1:
    print('running in serial ...')
    [read_data(f, bbox) for f in files]

else:
    print('running in parallel (%d jobs) ...' % njobs)
    from joblib import Parallel, delayed
    Parallel(n_jobs=njobs, verbose=5)(delayed(read_data)(f, bbox) for f in files)
    

#%%% Load FF-SAR data

from netCDF4 import Dataset,num2date # http://unidata.github.io/netcdf4-python/


fp1 = r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data\CS_OFFL_SIR1SAR_FR_20221213T201600_20221213T201832_E001_l2__ffsar_seaIce_retLib.nc'
ds = netCDF4.Dataset(fp1, 'r')
ssh_FFSAR = ds.groups['retracking'].variables['ssh'][:]
lat_FFSAR = ds.groups['instrument'].variables['lat'][:]
lon_FFSAR = ds.groups['instrument'].variables['lon'][:]
mss_FFSAR = ds.groups['corrections'].variables['mss_corr'][:]
surftype_FFSAR = ds.groups['focal_point_info'].variables['surfaceType'][:]
sic_FFSAR = ds.groups['focal_point_info'].variables['SIC'][:]


fp1 = r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data\CS_OFFL_SIR1SAR_FR_20221213T201600_20221213T201832_E001_l2__ffsar_seaIce_retLib_freeboard.nc'
ds = netCDF4.Dataset(fp1, 'r')

rfb_FFSAR = ds.variables['radar_freeboard'][:]
rfb_FFSAR[np.abs(rfb_FFSAR)>3]=np.nan
lat_FFSAR = ds.variables['lat'][:]
lon_FFSAR = ds.variables['lon'][:]

df_FFSAR = pd.DataFrame({'lat':lat_FFSAR, 'lon':lon_FFSAR, 'rfb':rfb_FFSAR})
df_FFSAR['rfb_pos'] = copy.deepcopy(df_FFSAR['rfb'])
df_FFSAR['rfb_pos'][df_FFSAR['rfb_pos']<-0.10] = np.nan
dist_req = 3500 # search radius in metres 
df_FFSAR['rfb_smooth']=CRYO2ICE_smooth_CS2_data(df_FFSAR, dist_req, var='rfb_pos')
df_FFSAR['rfb_smooth'][~df_FFSAR['rfb_pos'].notna()]=np.nan
    
#%% CryoSat-2 ESA-E

fp1 = r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data\CS_OFFL_SIR_SAR_1B_20221213T201600_20221213T201832_E001.nc'
ds = netCDF4.Dataset(fp1, 'r')

pwr_wrf_ESAE = ds.variables['pwr_waveform_20_ku'][:]
#pwr_wrf_ESAE = pwr_wrf_ESAE.filled()
echo_scale_pwr_20_ku = ds.variables['echo_scale_pwr_20_ku'][:]
echo_scale_factor_20_ku = ds.variables['echo_scale_factor_20_ku'][:]
lat_ESAE = ds.variables['lat_20_ku'][:]
lon_ESAE = ds.variables['lon_20_ku'][:]

#idx_1 = lat_ESAE[lat_ESAE>np.min(lat_FFSAR)]
idx_ESAE = np.where((lat_ESAE>np.min(lat_FFSAR)) & (lat_ESAE<np.max(lat_FFSAR)))

pwr_wrf_ESAE_cal = [np.array(pwr_wrf_ESAE[i]*((echo_scale_factor_20_ku[i]*2)**echo_scale_pwr_20_ku[i])) for i in np.arange(0,len(pwr_wrf_ESAE))]

fp1 = r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data\CS_OFFL_SIR_SAR_2__20221213T201600_20221213T201832_E001.nc'
ds = netCDF4.Dataset(fp1, 'r')

ssh_ESAE = ds.variables['height_1_20_ku'][:]
rfb_ESAE = ds.variables['radar_freeboard_20_ku'][:]
rfb_ESAE[np.abs(rfb_ESAE)>3]=np.nan
flag_ESAE = ds.variables['flag_prod_status_20_ku'][:]
flag_valid = get_valid_freeboard_flag(flag_ESAE)

#idx_1 = lat_ESAE[lat_ESAE>np.min(lat_FFSAR)]
#idx_ESAE = np.where((lat_ESAE>np.min(lat_FFSAR)) & (lat_ESAE<np.max(lat_FFSAR)))

df_ESAE = pd.DataFrame({'lat':lat_ESAE, 'lon':lon_ESAE, 'rfb':rfb_ESAE, 'ssh':ssh_ESAE})
df_ESAE['rfb_pos'] = copy.deepcopy(df_ESAE['rfb'])
df_ESAE['rfb_pos'][df_ESAE['rfb_pos']<-0.10] = np.nan
dist_req = 3500 # search radius in metres 
df_ESAE['rfb_smooth']=CRYO2ICE_smooth_CS2_data(df_ESAE, dist_req, var='rfb_pos')
df_ESAE['rfb_smooth'][~df_ESAE['rfb_pos'].notna()]=np.nan

#%% load CryoTEMPO data
fp1 = r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data\CS_OFFL_SIR_TDP_SI_ANTARC_20221213T201353_20221213T202031_28_04332_C001.nc'
ds = netCDF4.Dataset(fp1, 'r')

rfb_CryoTEMPO = ds.variables['radar_freeboard'][:]
rfb_CryoTEMPO[np.abs(rfb_CryoTEMPO)>3]=np.nan
lat_CryoTEMPO = ds.variables['latitude'][:]
lon_CryoTEMPO = ds.variables['longitude'][:]

df_CryoTEMPO = pd.DataFrame({'lat':lat_CryoTEMPO, 'lon':lon_CryoTEMPO, 'rfb':rfb_CryoTEMPO})
dist_req = 3500 # search radius in metres 
df_CryoTEMPO['rfb_pos'] = copy.deepcopy(df_CryoTEMPO['rfb'])
df_CryoTEMPO['rfb_pos'][df_CryoTEMPO['rfb_pos']<-0.10] = np.nan
df_CryoTEMPO['rfb_smooth']=CRYO2ICE_smooth_CS2_data(df_CryoTEMPO, dist_req, var='rfb_pos')
df_CryoTEMPO['rfb_smooth'][~df_CryoTEMPO['rfb_pos'].notna()]=np.nan
#%%% Load ATL10 data

import glob

directory = 'C:/Users/rmfha/Documents/GitHub/CRYO2ICE_Antarctic_underflight_comparison/data/'

files_check_IS2a=  glob.glob(directory + '*02_gt1l.h5')
files_check_IS2b =  glob.glob(directory + '*02_gt2l.h5')
files_check_IS2c=  glob.glob(directory + '*02_gt3l.h5')
files_check_IS2d =  glob.glob(directory + '*02_gt1r.h5')
files_check_IS2e =  glob.glob(directory + '*02_gt2r.h5')
files_check_IS2f =  glob.glob(directory + '*02_gt3r.h5')

k = 0
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

#%% Load ATL07 data



#%% CRYO2ICE compute for all three CryoSat-2 products

dist_req = 3500 # search radius in metres 
ds = 0.30
ns = (1+0.51*ds)**(1.5)
                    
df_CryoTEMPO_check = CRYO2ICE_identify(df_CryoTEMPO, df_IS2, dist_req)
df_CryoTEMPO_check['IS2_w_mean_fb'][~df_CryoTEMPO_check['rfb_pos'].notna()]=np.nan
df_CryoTEMPO_check['IS2_w_mean_fb_MSS']=df_CryoTEMPO_check['IS2_w_mean_fb']+df_CryoTEMPO_check['IS2_mean_MSS']
df_CryoTEMPO_check['snow_depth']=(df_CryoTEMPO_check['IS2_w_mean_fb_MSS']-df_CryoTEMPO_check['rfb_smooth'])/ns

df_FFSAR_check = CRYO2ICE_identify(df_FFSAR, df_IS2, dist_req)
df_FFSAR_check['IS2_w_mean_fb'][~df_FFSAR_check['rfb_pos'].notna()]=np.nan
df_FFSAR_check['IS2_w_mean_fb_MSS']=df_FFSAR_check['IS2_w_mean_fb']+df_FFSAR_check['IS2_mean_MSS']
df_FFSAR_check['snow_depth']=(df_FFSAR_check['IS2_w_mean_fb_MSS']-df_FFSAR_check['rfb_smooth'])/ns

df_ESAE_check = CRYO2ICE_identify(df_ESAE, df_IS2, dist_req)
df_ESAE_check['IS2_w_mean_fb'][~df_ESAE_check['rfb_pos'].notna()]=np.nan
df_ESAE_check['IS2_w_mean_fb_MSS']=df_ESAE_check['IS2_w_mean_fb']+df_ESAE_check['IS2_mean_MSS']
df_ESAE_check['snow_depth']=(df_ESAE_check['IS2_w_mean_fb_MSS']-df_ESAE_check['rfb_smooth'])/ns

#%%

print('Extracting AMSR2 data ...')

### prepare and include AMSR2 data
file = h5py.File('C:/Users/rmfha/Documents/GitHub/CRYO2ICE_Antarctic_underflight_comparison/data/AMSR_U2_L3_SeaIce12km_B04_20221213.he5', 'r')
file.keys()
lon = np.array(file['HDFEOS/GRIDS/SpPolarGrid12km/lon'])
lat = np.array(file['HDFEOS/GRIDS/SpPolarGrid12km/lat'])
snow_depth = np.array(file['HDFEOS/GRIDS/SpPolarGrid12km/Data Fields/SI_12km_SH_SNOWDEPTH_5DAY'])
sea_ice_concentration= np.array(file['HDFEOS/GRIDS/SpPolarGrid12km/Data Fields/SI_12km_SH_ICECON_DAY'])
file.close()

missing_snow_depth  = np.where(snow_depth != 110, np.nan, snow_depth)
land_snow_depth  = np.where(snow_depth != 120, np.nan, snow_depth)
OW_snow_depth  = np.where(snow_depth != 130, np.nan, snow_depth)
MYI_snow_depth  = np.where(snow_depth != 140, np.nan, snow_depth)
variability_snow_depth  = np.where(snow_depth != 150, np.nan, snow_depth)
melt_snow_depth  = np.where(snow_depth != 160, np.nan, snow_depth)
snow_depth_only  = np.where((snow_depth > 100) & (snow_depth!=140), np.nan, snow_depth)
snow_depth_only  = np.where(snow_depth_only==140, -9999, snow_depth_only)
sea_ice_concentration_only = np.where(sea_ice_concentration > 100, np.nan, sea_ice_concentration)

df_AMSR2 = pd.DataFrame({'lat':lat.flatten(), 'lon':lon.flatten(), 'snow_depth':snow_depth_only.flatten(), 'sea_ice_concentration':sea_ice_concentration_only.flatten()})
df_CryoTEMPO_check = CRYO2ICE_AMSR2_NN(df_CryoTEMPO_check,df_AMSR2) # update df with AMSR2 data
df_FFSAR_check = CRYO2ICE_AMSR2_NN(df_FFSAR_check,df_AMSR2) # update df with AMSR2 data
df_ESAE_check = CRYO2ICE_AMSR2_NN(df_ESAE_check,df_AMSR2) # update df with AMSR2 data

fn = 'C:/Users/rmfha/Documents/GitHub/CRYO2ICE_Antarctic_underflight_comparison/data/20221213_CASSIS.Depth' 
data_CASSIS = pd.read_csv(fn, delim_whitespace=True, header=None)
#data = data[1:-1]
data_CASSIS = data_CASSIS.dropna()

print('Extract CASSIS model...')

df_CASSIS = pd.DataFrame({'lat':data_CASSIS[2], 'lon':data_CASSIS[3], 'snow_depth':data_CASSIS[4]})
df_CryoTEMPO_check = CRYO2ICE_CASSIS_NN(df_CryoTEMPO_check,df_CASSIS) # update df with AMSR2 data
df_FFSAR_check = CRYO2ICE_CASSIS_NN(df_FFSAR_check,df_CASSIS) # update df with AMSR2 data
df_ESAE_check = CRYO2ICE_CASSIS_NN(df_ESAE_check,df_CASSIS) # update df with AMSR2 data


df_CryoTEMPO_check['CASSIS_snow_depth'][~df_CryoTEMPO_check['rfb_pos'].notna()]=np.nan
df_CryoTEMPO_check['AMSR2_snow_depth'][~df_CryoTEMPO_check['rfb_pos'].notna()]=np.nan

df_FFSAR_check['CASSIS_snow_depth'][~df_FFSAR_check['rfb_pos'].notna()]=np.nan
df_FFSAR_check['AMSR2_snow_depth'][~df_FFSAR_check['rfb_pos'].notna()]=np.nan

df_ESAE_check['CASSIS_snow_depth'][~df_ESAE_check['rfb_pos'].notna()]=np.nan
df_ESAE_check['AMSR2_snow_depth'][~df_ESAE_check['rfb_pos'].notna()]=np.nan

#%%

path = r'C:/Users/rmfha/Documents/GitHub/CRYO2ICE_Antarctic_underflight_comparison/data'

fp1 = r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data\CS_OFFL_SIR_TDP_SI_ANTARC_20221213T201353_20221213T202031_28_04332_C001.nc'
basename_without_ext = os.path.splitext(os.path.basename(fp1))[0]
CS2_proc = 'CryoTEMPO'
df_CryoTEMPO_check.to_hdf(path + '/' +'CRYO2ICE_' +CS2_proc +'_'+basename_without_ext + '_CASSIS_AMSR2.h5', key='CRYO2ICE', mode='w')

fp1 = r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data\CS_OFFL_SIR1SAR_FR_20221213T201600_20221213T201832_E001_l2__ffsar_seaIce_retLib_freeboard.nc'
basename_without_ext = os.path.splitext(os.path.basename(fp1))[0]
CS2_proc = 'FF-SAR'
df_FFSAR_check.to_hdf(path + '/' +'CRYO2ICE_' +CS2_proc +'_'+basename_without_ext + '_CASSIS_AMSR2.h5', key='CRYO2ICE', mode='w')

fp1 = r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data\CS_OFFL_SIR_SAR_2__20221213T201600_20221213T201832_E001.nc'
basename_without_ext = os.path.splitext(os.path.basename(fp1))[0]
CS2_proc = 'ESA-E'
df_ESAE_check.to_hdf(path + '/' +'CRYO2ICE_' +CS2_proc +'_'+basename_without_ext + '_CASSIS_AMSR2.h5', key='CRYO2ICE', mode='w')

#%%
idx_max = []
for i in np.arange(0, len(pwr_wrf_ESAE)):
    idx_max = np.append(np.argmax(pwr_wrf_ESAE[i, :]), idx_max)
    

idx_lat_ESAE = np.argwhere((lat_ESAE<=np.nanmax(df_ESAE_check['lat'])) & (lat_ESAE>=np.nanmin(df_ESAE_check['lat'])))
#%%

fig, axs = pplt.subplots([[1],
                          [2]], share=0,
                         axwidth=5, axheight=1.5, sharex=True, sharey=False)
fig.patch.set_facecolor('white')
'''
ax = axs[0]
C=0.714
ax.plot(lat_FFSAR[np.abs(ssh_FFSAR-mss_FFSAR)<3], ssh_FFSAR[np.abs(ssh_FFSAR-mss_FFSAR)<3], label='CryoSat-2 FF-SAR, N = {}'.format(len(ssh_FFSAR[np.abs(ssh_FFSAR-mss_FFSAR)<3])), linewidth=0.5, c='b')
ax.plot(lat_ESAE, ssh_ESAE-C, label='CryoSat-2 ESA-E, N = {}'.format(len(ssh_ESAE)), linewidth=0.5, c='r')

#ax.plot(lat_FFSAR, mss_FFSAR, label='DTU18MSS', linewidth=0.5)
ax.legend()
ax.format(ylabel='ellipsoidal heights/\nsea surface height (m)')


ax = axs[1]

ax.plot(lat_FFSAR[np.abs(ssh_FFSAR-mss_FFSAR)<3], ssh_FFSAR[np.abs(ssh_FFSAR-mss_FFSAR)<3]-mss_FFSAR[np.abs(ssh_FFSAR-mss_FFSAR)<3], label='CS2 FF-SAR', linewidth=0.5, c='k')
#ax.plot(lat_FFSAR, mss_FFSAR, label='DTU18MSS', linewidth=0.5)
#ax.legend()
ax.format(ylabel='sea surface height\nanomalies (m)')

ax_twin = ax.twinx()
ax_twin.plot(lat_FFSAR, sic_FFSAR, label='SIC', linewidth=0.5, c='red', linestyle='--')
ax_twin.format(ylim=(88, 101), ylabel='sea ice\nconcentration (%)')
ax_twin.set_ylabel(ylabel='sea ice\nconcentration (%)',color='red')
'''
'''
ax2 = axs[1].panel_axes('b', width=0.3, space=0)

for i in np.arange(len(surftype_FFSAR)):
    if surftype_FFSAR[i]==0:
        leg1=ax2.axvline(lat_FFSAR[i], c='grey', alpha=0.5)
    elif surftype_FFSAR[i]==1:
        leg2=ax2.axvline(lat_FFSAR[i], c='blue', alpha=0.5)
    elif surftype_FFSAR[i]==2:
        leg3=ax2.axvline(lat_FFSAR[i], c='red', alpha=0.5)
    elif surftype_FFSAR[i]==3:
        leg4=ax2.axvline(lat_FFSAR[i], c='green', alpha=0.5)
    elif surftype_FFSAR[i]==4:
        leg5=ax2.axvline(lat_FFSAR[i], c='grey')
        
ax.legend([leg2, leg3, leg4, leg5], ['Specular/leads', 'Diffuse/sea ice' , 'Diffuse/ambigious', 'Land'], ncols=4, loc='b', linewidth=8)
'''

ax = axs[0]

ax.format(ylabel='total or radar\nfreeboard (m)', ,xlabel='latitude (degrees N)', lltitle='Freeboards within 3 m from sea level', xlabel='', lefttitle='')
#ax.plot(lat_FFSAR, rfb_FFSAR, label='FF-SAR, N = {}'.format(len(rfb_FFSAR[~np.isnan(rfb_FFSAR)])), linewidth=0.5, markersize=1, linestyle='--', marker='o')
#ax.plot(lat_ESAE[flag_valid==1], rfb_ESAE[flag_valid==1], label='ESA-E, N = {}'.format(len(rfb_ESAE[flag_valid==1])), linewidth=0.5, markersize=1, linestyle='--', marker='o')
leg1=ax.scatter(df_ESAE['lat'], df_ESAE['rfb'], label='ESA-E, N = {} (<-0.10 m: {:.2f}%)'.format(len(df_ESAE[df_ESAE['rfb'].notna()]), (len(df_ESAE[df_ESAE['rfb']<-0.10])/len(df_ESAE[df_ESAE['rfb'].notna()]))*100), linewidth=0.5, markersize=1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[0], alpha=0.2)
leg2=ax.scatter(df_FFSAR['lat'], df_FFSAR['rfb'], label='FF-SAR, N = {} (<-0.10 m: {:.2f}%)'.format(len(df_FFSAR[df_FFSAR['rfb'].notna()]), (len(df_FFSAR[df_FFSAR['rfb']<-0.10])/len(df_FFSAR[df_FFSAR['rfb'].notna()]))*100), linewidth=0.5, markersize=1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[1], alpha=0.2)
leg3=ax.scatter(df_CryoTEMPO['lat'], df_CryoTEMPO['rfb'], label='CryoTEMPO, N = {} (<-0.10 m: {:.2f}%)'.format(len(df_CryoTEMPO[df_CryoTEMPO['rfb'].notna()]), (len(df_CryoTEMPO[df_CryoTEMPO['rfb']<-0.10])/len(df_CryoTEMPO[df_CryoTEMPO['rfb'].notna()]))*100), linewidth=0.5, markersize=1, linestyle='-', marker='o', zorder=-2, c=cmap_qual[2], alpha=0.2)
leg4=ax.scatter(df_CryoTEMPO_check['lat'], df_CryoTEMPO_check['IS2_w_mean_fb']+df_CryoTEMPO_check['IS2_mean_MSS'], label='ICESat-2{}, N = {} (<-0.10 m: {:.2f}%)'.format('$_{CryoTEMPO>-0.1 m}$', len(df_CryoTEMPO_check[df_CryoTEMPO_check['IS2_w_mean_fb'].notna()]), (len(df_CryoTEMPO_check[(df_CryoTEMPO_check['IS2_w_mean_fb']+df_CryoTEMPO_check['IS2_mean_MSS'])<-0.10])/len(df_CryoTEMPO_check[df_CryoTEMPO_check['IS2_w_mean_fb'].notna()]))*100), linewidth=0.5, markersize=0.3, linestyle='-', marker='o', zorder=-1, c=cmap_qual[3])

leg5=ax.scatter(df_ESAE_check['lat'], df_ESAE_check['rfb_smooth'], label='ESA-E{}, N = {}'.format('$_{smooth}$',len(df_ESAE_check[df_ESAE_check['rfb_smooth'].notna()])), linewidth=0.5, markersize=1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[0])
leg6=ax.scatter(df_FFSAR_check['lat'], df_FFSAR_check['rfb_smooth'], label='FF-SAR{}, N = {}'.format('$_{smooth}$',len(df_FFSAR_check[df_FFSAR_check['rfb_smooth'].notna()])), linewidth=0.5, markersize=1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[1])
leg7=ax.scatter(df_CryoTEMPO_check['lat'], df_CryoTEMPO_check['rfb_smooth'], label='CryoTEMPO{}, N = {}'.format('$_{smooth}$',len(df_CryoTEMPO_check[df_CryoTEMPO_check['rfb_smooth'].notna()])), linewidth=0.5, markersize=1, linestyle='-', zorder=-2, marker='o', c=cmap_qual[2])

#ax.scatter(df_IS2[df_IS2['beam_ID']==1]['lat'],df_IS2[df_IS2['beam_ID']==1]['fb_with_mss'], label='IS2 gt1l, N = {}'.format(len(df_IS2[df_IS2['beam_ID']==1]['fb_with_mss'][~np.isnan(df_IS2[df_IS2['beam_ID']==1]['fb_with_mss'])])), s=0.5)
ax.legend([leg1, leg2, leg3, leg4, leg5, leg6, leg7],loc='t', markersize=10, ncols=2, order='F', frameon=False)
text1 = '{:.2f} $\pm$ {:.2f} m'.format(np.nanmean(df_ESAE_check['rfb_smooth']), np.nanstd(df_ESAE_check['rfb_smooth']))
text2 = '{:.2f} $\pm$ {:.2f} m'.format(np.nanmean(df_FFSAR_check['rfb_smooth']), np.nanstd(df_FFSAR_check['rfb_smooth']))
text3 = '{:.2f} $\pm$ {:.2f} m'.format(np.nanmean(df_CryoTEMPO_check['rfb_smooth']), np.nanstd(df_CryoTEMPO_check['rfb_smooth']))
text4 = '{:.2f} $\pm$ {:.2f} m'.format(np.nanmean(df_CryoTEMPO_check['IS2_w_mean_fb']+df_CryoTEMPO_check['IS2_mean_MSS']), np.nanstd(df_CryoTEMPO_check['IS2_w_mean_fb']+df_CryoTEMPO_check['IS2_mean_MSS']))
legend = ax.legend([leg5, leg6, leg7, leg4], [text1, text2, text3, text4], loc='lr', handlelength=0, markersize=0, linewidth=0, ncols=2, frameon=False)
         # ncols=2, labelcolor='linecolor', frameon=False)
for handle, text in zip(legend.legendHandles, legend.get_texts()):
    text.set_color(handle.get_facecolor()[0])

ax1 = axs[0].panel_axes('b', width=1, space=0)
c = ax1.imshow(pwr_wrf_ESAE[idx_lat_ESAE[:,0]].transpose(), cmap='RdYlBu_r', extent=[np.min(df_ESAE_check['lat']), np.max(df_ESAE_check['lat']), len(pwr_wrf_ESAE[0]), 0], aspect="auto", vmin=0, vmax=15000, extend='max')
#ax1.scatter(lat_ESAE, idx_max,  edgecolor='k', facecolor="None", markersize=0.5, linewidth=0.1)

pwr_filt = pwr_wrf_ESAE[idx_lat_ESAE[:,0]]
pwr_max_loc = np.argmax(pwr_filt, axis=1)
leg_pwr_max = ax1.scatter(df_ESAE_check['lat'], pwr_max_loc, marker='o', edgecolor='k', s=0.5, facecolor='k', linewidth=0.1)

ax1.format(lltitle='CryoSat-2 L1B echogram', ylabel='range bins\n(0-256)')
ax1.yaxis.set_label_position("right")
ax1.yaxis.tick_right()
ax1.legend(leg_pwr_max, 'MAX', loc='lr', markersize=10)

ax.colorbar(c, label='normalised power', loc='r', length=0.9)

df_CryoTEMPO_check['snow_depth_zero_freeboard_assumption'] = (df_CryoTEMPO_check['IS2_w_mean_fb_MSS'])/ns
df_FFSAR_check['snow_depth_zero_freeboard_assumption'] = (df_FFSAR_check['IS2_w_mean_fb_MSS'])/ns
df_ESAE_check['snow_depth_zero_freeboard_assumption'] = (df_ESAE_check['IS2_w_mean_fb_MSS'])/ns

ax = axs[1]
CASSIS_snow = df_CryoTEMPO_check['CASSIS_snow_depth'][df_CryoTEMPO_check['snow_depth'].notna()]/1000
AMSR2_snow = df_CryoTEMPO_check['AMSR2_snow_depth'][df_CryoTEMPO_check['snow_depth'].notna()]/100
leg4=ax.scatter(df_CryoTEMPO_check['lat'], df_CryoTEMPO_check['CASSIS_snow_depth']/1000, label='CASSIS{}: {:.2f} $\pm$ {:.2f} m, N = {}'.format('$_{CRYOTEMPO_{CRYO2ICE}}$', np.nanmean(CASSIS_snow), np.nanstd(CASSIS_snow), len(CASSIS_snow)), linewidth=0.5, markersize=1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[4], alpha=0.5)
leg5=ax.scatter(df_CryoTEMPO_check['lat'], df_CryoTEMPO_check['AMSR2_snow_depth']/100, label='AMSR2{}: {:.2f} $\pm$ {:.2f} m, N = {}'.format('$_{CRYOTEMPO_{CRYO2ICE}}$', np.nanmean(AMSR2_snow), np.nanstd(AMSR2_snow), len(AMSR2_snow[~np.isnan(AMSR2_snow)])), linewidth=0.5, markersize=1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[5], alpha=0.5)
leg1=ax.scatter(df_ESAE_check['lat'], df_ESAE_check['snow_depth'], label='ESA-E{}: {:.2f} $\pm$ {:.2f} m'.format('$_{CRYO2ICE}$', np.nanmean(df_ESAE_check['snow_depth']), np.nanstd(df_ESAE_check['snow_depth'])), linewidth=0.5, markersize=1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[0])
leg2=ax.scatter(df_FFSAR_check['lat'], df_FFSAR_check['snow_depth'], label='FF-SAR{}: {:.2f} $\pm$ {:.2f} m'.format('$_{CRYO2ICE}$', np.nanmean(df_FFSAR_check['snow_depth']), np.nanstd(df_FFSAR_check['snow_depth'])), linewidth=0.5, markersize=1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[1])
leg3=ax.scatter(df_CryoTEMPO_check['lat'], df_CryoTEMPO_check['snow_depth'], label='CryoTEMPO{}: {:.2f} $\pm$ {:.2f} m'.format('$_{CRYO2ICE}$', np.nanmean(df_CryoTEMPO_check['snow_depth']), np.nanstd(df_CryoTEMPO_check['snow_depth'])), linewidth=0.5, markersize=1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[2])
leg6=ax.scatter(df_CryoTEMPO_check['lat'], df_CryoTEMPO_check['snow_depth_zero_freeboard_assumption'], label='ICESat-2{}: {:.2f} $\pm$ {:.2f} m'.format('$_{CRYOTEMPO_{CRYO2ICE}, zero-freeboard}$', np.nanmean(df_CryoTEMPO_check['snow_depth_zero_freeboard_assumption']), np.nanstd(df_CryoTEMPO_check['snow_depth_zero_freeboard_assumption'])), linewidth=0.5, markersize=1, linestyle='-', marker='o', zorder=-1, c=cmap_qual[3])
ax.legend([leg1, leg2, leg3, leg6, leg4, leg5],loc='b', markersize=10, order='F', ncols=2, frameon=False) 
ax.format(ylabel='snow depth, h$_s$ (m)',xlabel='latitude (degrees N)',)

fig.format(abc='(a)', abcloc='ul', xlim=(np.max(df_ESAE_check['lat']), np.min(df_ESAE_check['lat'])))

fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\Figure7.png', dpi=300)


#%%

CASSIS_snow_extra = df_CryoTEMPO_check['CASSIS_snow_depth'][(df_CryoTEMPO_check['snow_depth'].notna())&(df_CryoTEMPO_check['AMSR2_snow_depth'].notna())]/1000
CryoTEMPO_snow_extra = df_CryoTEMPO_check['snow_depth'][(df_CryoTEMPO_check['snow_depth'].notna())&(df_CryoTEMPO_check['AMSR2_snow_depth'].notna())]
#%%% Load swath ALS data

fn = r'C:\Users\rmfha\OneDrive - Danmarks Tekniske Universitet\DEFIANT2022\Data\SCANNER\347\Raw\347_183301_1x1.scn'
data_ALS_scn1 = pd.read_csv(fn, header=None, delim_whitespace=True)
fn = r'C:\Users\rmfha\OneDrive - Danmarks Tekniske Universitet\DEFIANT2022\Data\SCANNER\347\Raw\347_194156_1x1.scn'
data_ALS_scn2 = pd.read_csv(fn, header=None, delim_whitespace=True)
fn = r'C:\Users\rmfha\OneDrive - Danmarks Tekniske Universitet\DEFIANT2022\Data\SCANNER\347\Raw\347_203917_1x1.scn'
data_ALS_scn3 = pd.read_csv(fn, header=None, delim_whitespace=True)

frames = [data_ALS_scn1, data_ALS_scn2, data_ALS_scn3]

data_ALS_scn = pd.concat(frames).reset_index()

#%%
fn = 'kuband_20221213_02_74_232_002'
ds = netCDF4.Dataset(
    r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\data'+'/'+fn+'.nc', 'r')  # open CS2 data
latitude_comb = ds.variables['lat'][:]
longitude_comb = ds.variables['lon'][:]


#%%
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
fig, ax = pplt.subplots([[1, 2, 3],
                         [1, 2, 3],
                         [1, 2, 3],
                         [1, 2, 3],
                         [4,4,4],
                         [5,5,5]], axwidth=4, axheight=7, sharex=False, sharey=False, proj={1:'laea',2:'laea',3:'laea'}, proj_kw={'lat_0': -70, 'lon_0': -45})

fig.patch.set_facecolor('white')
fig.format(fontsize=14)
#ax1 = ax[1].panel_axes('r', width=1.3, space=0)


def add_north_arrow(ax, location=(0.11, 0.15), size=16, label='N', arrow_length=0.075, arrow_width=0.01):
    """
    Adds a north arrow to Proplot map axes that adjusts to the map's projection and orientation.
    
    Parameters:
    - ax: Proplot or Matplotlib Axes object with Cartopy projection
    - location: Tuple with the x and y position for the arrow (in axes fraction coordinates)
    - size: Size of the north arrow text label
    - label: Text label for the arrow (usually 'N' for North)
    - arrow_length: Length of the arrow in axes fraction coordinates
    - arrow_width: Width of the arrow in axes fraction coordinates
    """
    # Get the central point of the map in projection coordinates
    center_lon, center_lat = 0, 0  # Assuming this is the center of your map
    center_proj_x, center_proj_y = ax.projection.transform_point(center_lon, center_lat, ccrs.PlateCarree())
    
    # Define a point directly to the north of the center
    north_lat = center_lat + 1  # Move 1 degree north
    north_proj_x, north_proj_y = ax.projection.transform_point(center_lon, north_lat, ccrs.PlateCarree())
    
    # Calculate the angle to rotate the arrow to point to true north
    dx = north_proj_x - center_proj_x
    dy = north_proj_y - center_proj_y
    angle_to_north = np.degrees(np.arctan2(dy, dx))

    # Create an arrow pointing north
    arrow = mpatches.FancyArrow(location[0], location[1], 0, arrow_length, 
                                width=arrow_width, head_width=arrow_width * 3, 
                                transform=ax.transAxes, color='black')

    # Add the arrow to the plot
    ax.add_patch(arrow)

    # Add a label 'N' above the arrow
    ax.text(location[0], location[1] - 0.03, label, 
            transform=ax.transAxes, ha='center', va='center', fontsize=size, color='black')

    # Rotate the arrow to point to true north using `matplotlib.transforms`
    transform = transforms.Affine2D().rotate_deg_around(location[0], location[1], angle_to_north) + ax.transAxes
    arrow.set_transform(transform)

n = 300
n1 = 70
val_x = 1000

data_ALS_scn_filt = data_ALS_scn[(data_ALS_scn[1] < (np.max(latitude_comb[val_x*n1:val_x*(n1+1)])+0.05)) & (data_ALS_scn[1] > (np.min(latitude_comb[val_x*n1:val_x*(n1+1)])-0.05))]

axs = ax[0]
xlim_extent = (np.min(longitude_comb[val_x*n1:val_x*(n1+1)]) -
               0.05, np.max(longitude_comb[val_x*n1:val_x*(n1+1)])+0.05)
ylim_extent = (np.min(latitude_comb[val_x*n1:val_x*(n1+1)]) -
               0.05, np.max(latitude_comb[val_x*n1:val_x*(n1+1)])+0.05)

#xlim_extent = (np.min(longitude_comb[val_x*n1:val_x*(n1+1)]), np.max(longitude_comb[val_x*n1:val_x*(n1+1)]))
#ylim_extent = (np.min(latitude_comb[val_x*n1:val_x*(n1+1)]), np.max(latitude_comb[val_x*n1:val_x*(n1+1)]))
axs.set_extent([xlim_extent[0], xlim_extent[1], ylim_extent[1],
              ylim_extent[0]], crs=ccrs.PlateCarree())
im = axs.scatter(data_ALS_scn_filt[2], data_ALS_scn_filt[1], c=data_ALS_scn_filt[3], vmin=np.floor(np.quantile(
    data_ALS_scn_filt[3], 0.1)), vmax=np.ceil(np.quantile(data_ALS_scn_filt[3], 0.9)), cmap='crest', s=0.01, label='')
axs.set(frame_on=False)
axs.format(latlabels=True)
#axs.format(grid=False)

#leg_ALS = Line2D([0], [0], color='green', linestyle='-',label='Nadir ALS profile')
#axs.legend(leg_ALS, loc='ll', frameon=False, prop=dict(size=12))

fontprops = fm.FontProperties(size=14)
scalebar = AnchoredSizeBar(axs.transData,
                           400, '400 m', 'upper right',
                           pad=0.70,
                           sep=5,
                           color='black',
                           frameon=False,
                           size_vertical=4.0)
axs.add_artist(scalebar)      

df_filt = df_CryoTEMPO_check[(df_CryoTEMPO_check['lon']>xlim_extent[0]) & (df_CryoTEMPO_check['lon']<xlim_extent[1]) & (df_CryoTEMPO_check['lat']<ylim_extent[1]) & (df_CryoTEMPO_check['lat']>ylim_extent[0])]
df_filt = df_filt.reset_index()
N_example = 30
# Width and height in meters
width_m = 1600  # 100 km
height_m = 300  # 50 km

import cartopy.crs as ccrs
from cartopy.geodesic import Geodesic
import shapely
radius_given=3.5
for i in np.arange(0, len(df_filt['lon'])):
    circle_points = Geodesic().circle(lon=df_filt['lon'][i], lat=df_filt['lat'][i], radius=radius_given*10**3, n_samples=200, endpoint=False)
    geom = shapely.geometry.Polygon(circle_points)
    axs.add_geometries((geom,), crs=ccrs.PlateCarree(), facecolor='none', edgecolor='lightgrey', linewidth=0.5)
    
circle_points = Geodesic().circle(lon=df_filt['lon'][N_example], lat=df_filt['lat'][N_example], radius=radius_given*10**3, n_samples=200, endpoint=False)
geom = shapely.geometry.Polygon(circle_points)
axs.add_geometries((geom,), crs=ccrs.PlateCarree(), facecolor='none', edgecolor='b', linewidth=1, zorder=10)

leg1=axs.plot(df_IS2a['lon'], df_IS2a['lat'], c='darkred', s=0.05, label='gt1l')
leg2=axs.plot(df_IS2d['lon'], df_IS2d['lat'], c='coral', s=0.05, alpha=0.5, label='gt1r')

leg3=axs.plot(df_IS2b['lon'], df_IS2b['lat'], c='darkolivegreen', s=0.05, label='gt2l')
leg4=axs.plot(df_IS2e['lon'], df_IS2e['lat'], c='lightgreen', s=0.05, alpha=0.5, label='gt2r')

axs.plot(df_IS2c['lon'], df_IS2c['lat'], c='midnightblue', s=0.05, label='gt3l')
axs.plot(df_IS2f['lon'], df_IS2f['lat'], c='skyblue', s=0.05, alpha=0.5, label='gt3r')

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from geopy import distance
import numpy as np

def meters_to_degrees(center_lat, width_m, height_m):
    # Convert height to degrees latitude
    height_deg_lat = height_m / 111000
    
    # Convert width to degrees longitude
    width_deg_lon = width_m / (111000 * np.cos(np.radians(center_lat)))
    
    return width_deg_lon, height_deg_lat

for i in np.arange(len(df_filt)) :
   # Center of the rectangle (latitude, longitude)
    center_lat = df_filt['lat'][i]  # Example: Equator
    center_lon = df_filt['lon'][i]   # Example: Prime Meridian
    
    
    elevation_height = df_filt['snow_depth'][i]  # Example: 2000 meters
    if np.isnan(elevation_height):
        color='none'
    else:
        # Normalize the elevation value to range 0-1
        elev_min = 0
        elev_max = 0.6
        norm_elevation = (elevation_height - elev_min) / (elev_max - elev_min)
        cmap = plt.get_cmap('_rdylbu_r_copy')  # You can choose any colormap
        color = cmap(norm_elevation)
        
    width_deg_lon, height_deg_lat = meters_to_degrees(center_lat, width_m, height_m)
    min_lat = center_lat - height_deg_lat / 2
    max_lat = center_lat + height_deg_lat / 2
    min_lon = center_lon - width_deg_lon / 2
    max_lon = center_lon + width_deg_lon / 2
    
    # Define the projection
    proj = ccrs.PlateCarree()
    rect = plt.Rectangle((min_lon, min_lat), width_deg_lon, height_deg_lat,linewidth=1, edgecolor='k', facecolor=color, alpha=0.5, transform=proj)
    axs.add_patch(rect)

    
#cm=axs.scatter(df_filt['lon'],df_filt['lat'], edgecolor='k', c=df_filt['snow_depth'], extend='max', vmin=0, vmax=0.6, cmap='_rdylbu_r_copy')
#axs.scatter(df_filt['lon'][N_example],df_filt['lat'][N_example], edgecolor='b', c=df_filt['snow_depth'][N_example], extend='max', vmin=0, vmax=0.6, cmap='_rdylbu_r_copy')

# Center of the rectangle (latitude, longitude)
center_lat = df_filt['lat'][N_example]  # Example: Equator
center_lon = df_filt['lon'][N_example]   # Example: Prime Meridian

elevation_height = df_filt['snow_depth'][N_example]  # Example: 2000 meters
if np.isnan(elevation_height):
    color='none'
else:
    # Normalize the elevation value to range 0-1
    elev_min = 0
    elev_max = 0.6
    norm_elevation = (elevation_height - elev_min) / (elev_max - elev_min)
    cmap = plt.get_cmap('_rdylbu_r_copy')  # You can choose any colormap
    color = cmap(norm_elevation)

width_deg_lon, height_deg_lat = meters_to_degrees(center_lat, width_m, height_m)
min_lat = center_lat - height_deg_lat / 2
max_lat = center_lat + height_deg_lat / 2
min_lon = center_lon - width_deg_lon / 2
max_lon = center_lon + width_deg_lon / 2

# Define the projection
proj = ccrs.PlateCarree()
rect = plt.Rectangle((min_lon, min_lat), width_deg_lon, height_deg_lat,linewidth=1, edgecolor='b', facecolor=color, alpha=1, transform=proj, zorder=10)
axs.add_patch(rect)

#add_north_arrow(axs)

axs = ax[1]
N_example = 12
#xlim_extent = (np.min(longitude_comb[val_x*n1:val_x*(n1+1)]) -
 #              0.05, np.max(longitude_comb[val_x*n1:val_x*(n1+1)])+0.05)
#ylim_extent = (np.min(latitude_comb[val_x*n1:val_x*(n1+1)]) -
 #              0.05, np.max(latitude_comb[val_x*n1:val_x*(n1+1)])+0.05)

xlim_extent = (np.min(longitude_comb[val_x*n1:val_x*(n1+1)]), np.max(longitude_comb[val_x*n1:val_x*(n1+1)]))
ylim_extent = (np.min(latitude_comb[val_x*n1:val_x*(n1+1)]), np.max(latitude_comb[val_x*n1:val_x*(n1+1)]))
axs.set_extent([xlim_extent[0], xlim_extent[1], ylim_extent[1],
              ylim_extent[0]], crs=ccrs.PlateCarree())
im = axs.scatter(data_ALS_scn_filt[2], data_ALS_scn_filt[1], c=data_ALS_scn_filt[3], vmin=np.floor(np.quantile(
    data_ALS_scn_filt[3], 0.1)), vmax=np.ceil(np.quantile(data_ALS_scn_filt[3], 0.9)), cmap='crest', s=0.01, label='')
#axs.set(frame_on=False)
axs.format(grid=False)

area_lon = [xlim_extent[0], xlim_extent[0], xlim_extent[1], xlim_extent[1], xlim_extent[0]]
area_lat = [ylim_extent[0], ylim_extent[1], ylim_extent[1], ylim_extent[0], ylim_extent[0]]
ax[0].plot(area_lon, area_lat, c='red', ls='-', linewidth=1)

#leg_ALS = Line2D([0], [0], color='green', linestyle='-',label='Nadir ALS profile')
#axs.legend(leg_ALS, loc='ll', frameon=False, prop=dict(size=12))

fontprops = fm.FontProperties(size=14)
scalebar = AnchoredSizeBar(axs.transData,
                           400, '400 m', 'upper right',
                           pad=0.70,
                           sep=5,
                           color='black',
                           frameon=False,
                           size_vertical=4.0)
axs.add_artist(scalebar)      

df_filt = df_CryoTEMPO_check[(df_CryoTEMPO_check['lon']>xlim_extent[0]) & (df_CryoTEMPO_check['lon']<xlim_extent[1]) & (df_CryoTEMPO_check['lat']<ylim_extent[1]) & (df_CryoTEMPO_check['lat']>ylim_extent[0])]
df_filt = df_filt.reset_index()

import cartopy.crs as ccrs
from cartopy.geodesic import Geodesic
import shapely
radius_given=3.5 # given in km
for i in np.arange(0, len(df_filt['lon'])):
    circle_points = Geodesic().circle(lon=df_filt['lon'][i], lat=df_filt['lat'][i], radius=radius_given*10**3, n_samples=200, endpoint=False)
    geom = shapely.geometry.Polygon(circle_points)
    axs.add_geometries((geom,), crs=ccrs.PlateCarree(), facecolor='none', edgecolor='lightgrey', linewidth=0.5, zorder=10)
    
circle_points = Geodesic().circle(lon=df_filt['lon'][N_example], lat=df_filt['lat'][N_example], radius=radius_given*10**3, n_samples=200, endpoint=False)
geom = shapely.geometry.Polygon(circle_points)
axs.add_geometries((geom,), crs=ccrs.PlateCarree(), facecolor='none', edgecolor='b', linewidth=1, zorder=10)
    
df_IS2a_filt = df_IS2a[(df_IS2a['lon']>xlim_extent[0]) & (df_IS2a['lon']<xlim_extent[1]) & (df_IS2a['lat']<ylim_extent[1]) & (df_IS2a['lat']>ylim_extent[0])]
df_IS2a_filt = df_IS2a_filt.reset_index()

df_IS2d_filt = df_IS2d[(df_IS2d['lon']>xlim_extent[0]) & (df_IS2d['lon']<xlim_extent[1]) & (df_IS2d['lat']<ylim_extent[1]) & (df_IS2d['lat']>ylim_extent[0])]
df_IS2d_filt = df_IS2d_filt.reset_index()

#leg1=axs.plot(df_IS2a['lon'], df_IS2a['lat'], c='darkred', s=0.05, label='gt1l')
#leg2=axs.plot(df_IS2d['lon'], df_IS2d['lat'], c='coral', s=0.05, alpha=0.5, label='gt1r')
radius_given=17/(10**3)
for i in np.arange(0, len(df_IS2a_filt['lon'])):
    circle_points = Geodesic().circle(lon=df_IS2a_filt['lon'][i], lat=df_IS2a_filt['lat'][i], radius=radius_given*10**3, n_samples=200, endpoint=False)
    geom = shapely.geometry.Polygon(circle_points)
    axs.add_geometries((geom,), crs=ccrs.PlateCarree(), facecolor='none', edgecolor='darkred', linewidth=0.5, zorder=10)

for i in np.arange(0, len(df_IS2d_filt['lon'])):
    circle_points = Geodesic().circle(lon=df_IS2d_filt['lon'][i], lat=df_IS2d_filt['lat'][i], radius=radius_given*10**3, n_samples=200, endpoint=False)
    geom = shapely.geometry.Polygon(circle_points)
    axs.add_geometries((geom,), crs=ccrs.PlateCarree(), facecolor='none', edgecolor='coral', linewidth=0.5, zorder=10)

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from geopy import distance
import numpy as np

def meters_to_degrees(center_lat, width_m, height_m):
    # Convert height to degrees latitude
    height_deg_lat = height_m / 111000
    
    # Convert width to degrees longitude
    width_deg_lon = width_m / (111000 * np.cos(np.radians(center_lat)))
    
    return width_deg_lon, height_deg_lat

for i in np.arange(len(df_filt)) :
   # Center of the rectangle (latitude, longitude)
    center_lat = df_filt['lat'][i]  # Example: Equator
    center_lon = df_filt['lon'][i]   # Example: Prime Meridian
    
    # Width and height in meters
    width_m = 1600  # 100 km
    height_m = 300  # 50 km
    
    elevation_height = df_filt['snow_depth'][i]  # Example: 2000 meters
    if np.isnan(elevation_height):
        color='none'
    else:
        # Normalize the elevation value to range 0-1
        elev_min = 0
        elev_max = 0.6
        norm_elevation = (elevation_height - elev_min) / (elev_max - elev_min)
        cmap = plt.get_cmap('_rdylbu_r_copy')  # You can choose any colormap
        color = cmap(norm_elevation)
        
    width_deg_lon, height_deg_lat = meters_to_degrees(center_lat, width_m, height_m)
    min_lat = center_lat - height_deg_lat / 2
    max_lat = center_lat + height_deg_lat / 2
    min_lon = center_lon - width_deg_lon / 2
    max_lon = center_lon + width_deg_lon / 2
    
    # Define the projection
    proj = ccrs.PlateCarree()
    rect = plt.Rectangle((min_lon, min_lat), width_deg_lon, height_deg_lat,linewidth=1, edgecolor='k', facecolor=color, alpha=0.5, transform=proj)
    axs.add_patch(rect)

    
cm=axs.scatter(df_filt['lon'],df_filt['lat'], edgecolor='k', c=df_filt['snow_depth'], extend='max', vmin=0, vmax=0.6, cmap='_rdylbu_r_copy')
axs.scatter(df_filt['lon'][N_example],df_filt['lat'][N_example], edgecolor='b', c=df_filt['snow_depth'][N_example], extend='max', vmin=0, vmax=0.6, cmap='_rdylbu_r_copy')

# Center of the rectangle (latitude, longitude)
center_lat = df_filt['lat'][N_example]  # Example: Equator
center_lon = df_filt['lon'][N_example]   # Example: Prime Meridian

elevation_height = df_filt['snow_depth'][N_example]  # Example: 2000 meters
if np.isnan(elevation_height):
    color='none'
else:
    # Normalize the elevation value to range 0-1
    elev_min = 0
    elev_max = 0.6
    norm_elevation = (elevation_height - elev_min) / (elev_max - elev_min)
    cmap = plt.get_cmap('_rdylbu_r_copy')  # You can choose any colormap
    color = cmap(norm_elevation)

width_deg_lon, height_deg_lat = meters_to_degrees(center_lat, width_m, height_m)
min_lat = center_lat - height_deg_lat / 2
max_lat = center_lat + height_deg_lat / 2
min_lon = center_lon - width_deg_lon / 2
max_lon = center_lon + width_deg_lon / 2

# Define the projection
proj = ccrs.PlateCarree()
rect = plt.Rectangle((min_lon, min_lat), width_deg_lon, height_deg_lat,linewidth=1, edgecolor='b', facecolor=color, transform=proj)
axs.add_patch(rect)


axs = ax[2]
data_ALS_scn_filt = data_ALS_scn[(data_ALS_scn[1] < (np.max(latitude_comb[val_x*n1:val_x*(n1+1)])+0.05)) & (data_ALS_scn[1] > (np.min(latitude_comb[val_x*n1:val_x*(n1+1)])-0.05))]


#xlim_extent = (np.min(longitude_comb[val_x*n1:val_x*(n1+1)])+0.05, np.max(longitude_comb[val_x*n1:val_x*(n1+1)])-0.05)
#ylim_extent = (np.min(latitude_comb[val_x*n1:val_x*(n1+1)])+0.05, np.max(latitude_comb[val_x*n1:val_x*(n1+1)])-0.05)
ylim_extent = (-69.96, -69.97)
xlim_extent = (-53.73,-53.755)
axs.set_extent([xlim_extent[0], xlim_extent[1], ylim_extent[1],
              ylim_extent[0]], crs=ccrs.PlateCarree())
im = axs.scatter(data_ALS_scn_filt[2], data_ALS_scn_filt[1], c=data_ALS_scn_filt[3], vmin=np.floor(np.quantile(
    data_ALS_scn_filt[3], 0.1)), vmax=np.ceil(np.quantile(data_ALS_scn_filt[3], 0.9)), cmap='crest', s=0.01, label='')
#axs.set(frame_on=False)
axs.format(grid=False)
#axs.format(latlabels='r', lonlabels='b')

area_lon = [xlim_extent[0], xlim_extent[0], xlim_extent[1], xlim_extent[1], xlim_extent[0]]
area_lat = [ylim_extent[0], ylim_extent[1], ylim_extent[1], ylim_extent[0], ylim_extent[0]]
ax[1].plot(area_lon, area_lat, c='red', ls='-', linewidth=1)

#leg_ALS = Line2D([0], [0], color='green', linestyle='-',label='Nadir ALS profile')
#axs.legend(leg_ALS, loc='ll', frameon=False, prop=dict(size=12))

fontprops = fm.FontProperties(size=14)
scalebar = AnchoredSizeBar(axs.transData,
                           400, '400 m', 'upper right',
                           pad=0.70,
                           sep=5,
                           color='black',
                           frameon=False,
                           size_vertical=4.0)
axs.add_artist(scalebar)   
   

df_filt = df_CryoTEMPO_check[(df_CryoTEMPO_check['lon']<xlim_extent[0]) & (df_CryoTEMPO_check['lon']>xlim_extent[1]) & (df_CryoTEMPO_check['lat']>ylim_extent[1]) & (df_CryoTEMPO_check['lat']<ylim_extent[0])]
df_filt = df_filt.reset_index()

import cartopy.crs as ccrs
from cartopy.geodesic import Geodesic
import shapely
radius_given=3.5 # given in km
for i in np.arange(0, len(df_filt['lon'])):
    circle_points = Geodesic().circle(lon=df_filt['lon'][i], lat=df_filt['lat'][i], radius=radius_given*10**3, n_samples=200, endpoint=False)
    geom = shapely.geometry.Polygon(circle_points)
    axs.add_geometries((geom,), crs=ccrs.PlateCarree(), facecolor='none', edgecolor='lightgrey', linewidth=0.5, zorder=10)

df_IS2a_filt = df_IS2a[(df_IS2a['lon']<xlim_extent[0]) & (df_IS2a['lon']>xlim_extent[1]) & (df_IS2a['lat']>(ylim_extent[1]-0.01)) & (df_IS2a['lat']<(ylim_extent[0]+0.01))]
df_IS2a_filt = df_IS2a_filt.reset_index()

df_IS2d_filt = df_IS2d[(df_IS2d['lon']<xlim_extent[0]) & (df_IS2d['lon']>xlim_extent[1]) & (df_IS2d['lat']>(ylim_extent[1]-0.01)) & (df_IS2d['lat']<(ylim_extent[0]+0.01))]
df_IS2d_filt = df_IS2d_filt.reset_index()

radius_given=(17/2)/(10**3)
for i in np.arange(0, len(df_IS2a_filt['lon'])):
    circle_points = Geodesic().circle(lon=df_IS2a_filt['lon'][i], lat=df_IS2a_filt['lat'][i], radius=radius_given*10**3, n_samples=200, endpoint=False)
    geom = shapely.geometry.Polygon(circle_points)
    axs.add_geometries((geom,), crs=ccrs.PlateCarree(), facecolor='none', edgecolor='darkred', linewidth=0.1, zorder=10)

for i in np.arange(0, len(df_IS2d_filt['lon'])):
    circle_points = Geodesic().circle(lon=df_IS2d_filt['lon'][i], lat=df_IS2d_filt['lat'][i], radius=radius_given*10**3, n_samples=200, endpoint=False)
    geom = shapely.geometry.Polygon(circle_points)
    axs.add_geometries((geom,), crs=ccrs.PlateCarree(), facecolor='none', edgecolor='coral', linewidth=0.1, zorder=10)


radar_lon = longitude_comb[val_x*n1:val_x*(n1+1)]
radar_lat = latitude_comb[val_x*n1:val_x*(n1+1)]

radius_given=(5/2)/(10**3)
for i in np.arange(0, len(radar_lon)):
    circle_points = Geodesic().circle(lon=radar_lon[i], lat=radar_lat[i], radius=radius_given*10**3, n_samples=200, endpoint=False)
    geom = shapely.geometry.Polygon(circle_points)
    leg_radar = axs.add_geometries((geom,), crs=ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.1, zorder=10)

axs.legend(leg_radar, 'Ka/Ku/S/C-band radar', loc='ll', frameon=False)
for i in np.arange(len(df_filt)) :
   # Center of the rectangle (latitude, longitude)
    center_lat = df_filt['lat'][i]  # Example: Equator
    center_lon = df_filt['lon'][i]   # Example: Prime Meridian
    
    elevation_height = df_filt['snow_depth'][i]  # Example: 2000 meters
    if np.isnan(elevation_height):
        color='none'
    else:
        # Normalize the elevation value to range 0-1
        elev_min = 0
        elev_max = 0.6
        norm_elevation = (elevation_height - elev_min) / (elev_max - elev_min)
        cmap = plt.get_cmap('_rdylbu_r_copy')  # You can choose any colormap
        color = cmap(norm_elevation)
        
    width_deg_lon, height_deg_lat = meters_to_degrees(center_lat, width_m, height_m)
    min_lat = center_lat - height_deg_lat / 2
    max_lat = center_lat + height_deg_lat / 2
    min_lon = center_lon - width_deg_lon / 2
    max_lon = center_lon + width_deg_lon / 2
    
    # Define the projection
    proj = ccrs.PlateCarree()
    rect = plt.Rectangle((min_lon, min_lat), width_deg_lon, height_deg_lat,linewidth=1, edgecolor='k', facecolor=color, alpha=0.5, transform=proj)
    axs.add_patch(rect)
    
N_example=1

# Center of the rectangle (latitude, longitude)
center_lat = df_filt['lat'][N_example]  # Example: Equator
center_lon = df_filt['lon'][N_example]   # Example: Prime Meridian

elevation_height = df_filt['snow_depth'][N_example]  # Example: 2000 meters
if np.isnan(elevation_height):
    color='none'
else:
    # Normalize the elevation value to range 0-1
    elev_min = 0
    elev_max = 0.6
    norm_elevation = (elevation_height - elev_min) / (elev_max - elev_min)
    cmap = plt.get_cmap('_rdylbu_r_copy')  # You can choose any colormap
    color = cmap(norm_elevation)

width_deg_lon, height_deg_lat = meters_to_degrees(center_lat, width_m, height_m)
min_lat = center_lat - height_deg_lat / 2
max_lat = center_lat + height_deg_lat / 2
min_lon = center_lon - width_deg_lon / 2
max_lon = center_lon + width_deg_lon / 2

# Define the projection
proj = ccrs.PlateCarree()
rect = plt.Rectangle((min_lon, min_lat), width_deg_lon, height_deg_lat,linewidth=1, edgecolor='b', facecolor=color, transform=proj)
axs.add_patch(rect)

cb = fig.colorbar(im, label='ALS ellipsoidal heights WGS84 (m)', loc='t', length=0.8, labelsize=12)
cb.ax.tick_params(labelsize=12)

cb = fig.colorbar(cm, label='CryoTEMPO{} CRYO2ICE snow depth (m)'.format('$_{smooth}$'), loc='t', length=0.8, labelsize=12)
cb.ax.tick_params(labelsize=12)

ax[0].legend([leg1, leg2, leg3, leg4], loc='ll',handlelength=2, markersize=0, linewidth=1, ncols=1, labelcolor='linecolor', frameon=False )


ax[0].format(lefttitle='0.5$^\circ$ zoomed-out from sub-panel (b)')
ax[1].format(lefttitle='Subset as presented in Figure 1')
ax[2].format(lefttitle='Zoomed inset from sub-panel (b)')
fig.format(abc='(a)', abcloc='l')


fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\Figure_comp_CRYO2ICE_air.png', dpi=300)



#fig, ax = pplt.subplots([[1, 1, 1, 1], 
#                          [2, 2, 2, 2]], share=0,
#                         axwidth=4, axheight=1, sharex=True, sharey=False)
#fig.patch.set_facecolor('white')

n = 300
n1 = 70
val_x = 1000

ylim_extent = (np.min(latitude_comb[val_x*n1:val_x*(n1+1)]) -
               0.05, np.max(latitude_comb[val_x*n1:val_x*(n1+1)])+0.05)


axs = ax[3]
#df_CryoTEMPO_check = df_CryoTEMPO_check.sort_values(by=['lat'])
df_CryoTEMPO_check['IS2_w_mean_fb-MSS']=df_CryoTEMPO_check['IS2_w_mean_fb']+df_CryoTEMPO_check['IS2_mean_MSS']
idx = np.where(df_CryoTEMPO_check['IS2_w_mean_fb'].notna())[0]
ci1 = df_CryoTEMPO_check['IS2_dist_avg']-df_CryoTEMPO_check['IS2_dist_min']
ci2 = df_CryoTEMPO_check['IS2_dist_max']-df_CryoTEMPO_check['IS2_dist_avg']
axs.plot(df_CryoTEMPO_check['lat'], df_CryoTEMPO_check['IS2_dist_avg'], label='Average IS2 dist.', c=cmap_qual2(1), zorder=1)
axs.fill_between(df_CryoTEMPO_check['lat'], (df_CryoTEMPO_check['IS2_dist_avg']-ci1), (df_CryoTEMPO_check['IS2_dist_avg']+ci2), alpha=.3, c=cmap_qual2(1), zorder=0)
axs.format(ylabel='ICESat-2 ATL10 distance\n to CryoSat-2 (m)')
axs.axvspan(xmin=ylim_extent[1], xmax=ylim_extent[0], facecolor='r', alpha=0.5)
axs = ax[4]
#ax.area(df_CryoTEMPO_check['lat'], df_CryoTEMPO_check['GT1R_count'])
sum_comb = int(np.sum(df_CryoTEMPO_check['GT1R_count'])) + int(np.sum(df_CryoTEMPO_check['GT2R_count'])) + int(np.sum(df_CryoTEMPO_check['GT3R_count'])) + int(np.sum(df_CryoTEMPO_check['GT1L_count'])) + int(np.sum(df_CryoTEMPO_check['GT2L_count'])) + int(np.sum(df_CryoTEMPO_check['GT3L_count']))
labels = ['gt1r ({:.2f}%)'.format(int(np.sum(df_CryoTEMPO_check['GT1R_count']))/sum_comb*100), 'gt1l ({:.2f}%)'.format(int(np.sum(df_CryoTEMPO_check['GT1L_count']))/sum_comb*100), 'gt2r ({:.2f}%)'.format(int(np.sum(df_CryoTEMPO_check['GT2R_count']))/sum_comb*100), 'gt2l ({:.2f}%)'.format(int(np.sum(df_CryoTEMPO_check['GT2L_count']))/sum_comb*100)]
polys=axs.stackplot(df_CryoTEMPO_check['lat'], df_CryoTEMPO_check['GT1R_count'],df_CryoTEMPO_check['GT1L_count'],df_CryoTEMPO_check['GT2R_count'], df_CryoTEMPO_check['GT2L_count'], labels=labels, colors=cmap_qual)
axs.legend(loc='ul', order='C',handlelength=0, markersize=0, linewidth=0, ncols=4, labelcolor='linecolor', frameon=False, pad=0.2, prop=dict(size=12))
axs.format(ylabel='ATL10 observations used\n per CryoSat-2 point', ylim=(0, 4500))
fig.format(abc='(a)', abcloc='l', xlim=(np.max(df_CryoTEMPO_check['lat'][idx]), np.min(df_CryoTEMPO_check['lat'][idx])), xlabel='latitude (degrees N)')
axs.axvspan(xmin=ylim_extent[1], xmax=ylim_extent[0], facecolor='r', alpha=0.5)
#fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\Appendix_IS2beams.png', dpi=300)

fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\Figure_comp_CRYO2ICE_air.png', dpi=300)


#%%
fig, axs = pplt.subplots(nrows=1, ncols=1, sharey=False)
fig.patch.set_facecolor('white')
import matplotlib.pyplot as plt

cmap_qual3 =  plt.cm.get_cmap('RdBu', 4)
rfb_change = np.full(100, 0.10)
rfb_change2 = np.full(100, 0.30)
rfb_change3 = np.full(100, 0.60)
rfb_change4 = np.full(100, 0.90)

sden_change = np.linspace(0.1, 0.850,100)
sden_change_const = np.full(100, 0.3)
ns = (1+0.51*sden_change)**(1.5)
ns_const = (1+0.51*sden_change_const)**(1.5)
snow_depth = (rfb_change/ns)
snow_depth2 = (rfb_change2/ns)
snow_depth3 = (rfb_change3/ns)
snow_depth4 = (rfb_change4/ns)
snow_depth_const = (rfb_change/ns_const)
snow_depth2_const = (rfb_change2/ns_const)
snow_depth3_const = (rfb_change3/ns_const)
snow_depth4_const = (rfb_change4/ns_const)

ns = (1+0.51*sden_change)**(-1.5)
ns_const = (1+0.51*sden_change_const)**(-1.5)
c = 3e+6
cs = c*ns
rfb_change_sw = rfb_change+((c/cs)-1)*snow_depth
rfb_change2_sw = rfb_change+((c/cs)-1)*snow_depth2
rfb_change3_sw = rfb_change+((c/cs)-1)*snow_depth3
rfb_change4_sw = rfb_change+((c/cs)-1)*snow_depth4
rho_w, rho_i = 1024, 900
t_change_sw = (rho_w/(rho_w-rho_i))*rfb_change_sw+(sden_change/(rho_w-rho_i))*snow_depth
t_change2_sw = (rho_w/(rho_w-rho_i))*rfb_change2_sw+(sden_change/(rho_w-rho_i))*snow_depth2
t_change3_sw = (rho_w/(rho_w-rho_i))*rfb_change3_sw+(sden_change/(rho_w-rho_i))*snow_depth3
t_change4_sw = (rho_w/(rho_w-rho_i))*rfb_change4_sw+(sden_change/(rho_w-rho_i))*snow_depth4
cs = c*ns_const
rfb_change_const_sw = rfb_change+((c/cs)-1)*snow_depth_const
rfb_change2_const_sw = rfb_change+((c/cs)-1)*snow_depth2_const
rfb_change3_const_sw = rfb_change+((c/cs)-1)*snow_depth3_const
rfb_change4_const_sw = rfb_change+((c/cs)-1)*snow_depth4_const
rho_w, rho_i = 1024, 900
t_change_const_sw = (rho_w/(rho_w-rho_i))*rfb_change_const_sw+(sden_change/(rho_w-rho_i))*snow_depth_const
t_change2_const_sw = (rho_w/(rho_w-rho_i))*rfb_change2_const_sw+(sden_change/(rho_w-rho_i))*snow_depth2_const
t_change3_const_sw = (rho_w/(rho_w-rho_i))*rfb_change3_const_sw+(sden_change/(rho_w-rho_i))*snow_depth3_const
t_change4_const_sw = (rho_w/(rho_w-rho_i))*rfb_change4_const_sw+(sden_change/(rho_w-rho_i))*snow_depth4_const



axs[0].plot(sden_change*1000,(snow_depth-snow_depth_const)*100, c=cmap_qual3(0), label=r'$\Delta (h_{a-i}-h_{s-i})$ = 0.1 m', linestyle='-')
axs[0].plot(sden_change*1000,(snow_depth2-snow_depth2_const)*100, c=cmap_qual3(1), label=r'$\Delta (h_{a-i}-h_{s-i})$ = 0.3 m', linestyle='-.')
axs[0].plot(sden_change*1000,(snow_depth3-snow_depth3_const)*100, c=cmap_qual3(2), label=r'$\Delta (h_{a-i}-h_{s-i})$ = 0.6 m')
axs[0].plot(sden_change*1000,(snow_depth4-snow_depth4_const)*100, c=cmap_qual3(3), label=r'$\Delta (h_{a-i}-h_{s-i})$ = 0.9 m', linestyle='--')
axs[0].legend(ncol=2, handlelength=1.5, labelcolor='linecolor', frame=False, loc='t')

leg1=axs[0].axvline(350, linestyle='--', c='lightgrey', linewidth=1)
axs[0].legend(leg1, '{} = 350 (kg m{})'.format(r'$\rho_s$', '$^{-3}$'))
fig.format(xlabel=r'snow density, $\rho_s$ (kg m$^{-3}$)', ylabel='Difference in snow depth, $\Delta$h$_s$ (cm)\n using 300 kg m$^{-3}$ as constant' )

ix = axs[0].inset(
    [180,-18, 250, 10], transform='data', zoom=True,
    zoom_kw={'ec': 'k', 'ls': '--', 'lw': 0.5}
)
ix.format(
    xlim=(280, 380), ylim=(-5, 2), color='k',
    linewidth=0.5, 
)

ix.plot(sden_change*1000,(snow_depth-snow_depth_const)*100, c=cmap_qual3(0), label=r'$\Delta (h_{a-i}-h_{s-i})$ = 0.1 m', linestyle='-')
ix.plot(sden_change*1000,(snow_depth2-snow_depth2_const)*100, c=cmap_qual3(1), label=r'$\Delta (h_{a-i}-h_{s-i})$ = 0.3 m', linestyle='-.')
ix.plot(sden_change*1000,(snow_depth3-snow_depth3_const)*100, c=cmap_qual3(2), label=r'$\Delta (h_{a-i}-h_{s-i})$ = 0.6 m')
ix.plot(sden_change*1000,(snow_depth4-snow_depth4_const)*100, c=cmap_qual3(3), label=r'$\Delta (h_{a-i}-h_{s-i})$ = 0.9 m', linestyle='--')
ix.axvline(350, linestyle='--', c='lightgrey', linewidth=1)

'''
axs[2].plot(sden_change,(t_change_sw-t_change_const_sw)*100, c=cmap_qual(0), label='$\Delta$fb = 0.1 m, fb = 0.1 m')
axs[2].plot(sden_change,(t_change_sw-t_change2_const_sw)*100, c=cmap_qual(1), label='$\Delta$fb = 0.2 m, fb = 0.2 m')
axs[2].plot(sden_change,(t_change_sw-t_change3_const_sw)*100, c=cmap_qual(2), label='$\Delta$fb = 0.3 m, fb = 0.3 m')
axs[2].plot(sden_change,(t_change_sw-t_change4_const_sw)*100, c='red', label='$\Delta$fb = 0.4 m, fb = 0.4 m')
axs[2].legend(ncol=1, handlelength=0, labelcolor='linecolor', frame=False)
fig.format(xlabel='snow density (kg/m$^{3}$)', abc='(a)', abcloc='ul' )


'''

fig.save(r'C:\Users\rmfha\Documents\GitHub\CRYO2ICE_Antarctic_underflight_comparison\figs\Figure5_snowdensity.png', dpi=300)



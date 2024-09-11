# CRYO2ICEANT2022 Antarctic Summer Sea Ice Under-Flight using Multi-Frequency Airborne Altimetry
[![DOI](https://badgen.net/badge/DATA/10.11583%2FDTU.26732227/red)](https://figshare.com/s/9626392bca7b9c2a32e9) [![pySnowRadar](https://badgen.net/badge/pySnowRadar/10.5281%2Fzenodo.4071947/blue)](https://github.com/kingjml/pySnowRadar)

This repository holds the code used to derive and compare airborne (Ka-, Ku-, C/S-band and lidar) observations (snow depth or penetration) acquired along the Weddell Sea CRYO2ICE (CryoSat-2 and ICESat-2) orbit on the 13 December 2022 as part of the CRYO2ICEANT2022 campaign. 

This repository is linked with the following pre-print: URL

Contact: Renée Mie Fredensborg Hansen @ rmfha@space.dtu.dk

## Brief summary 
TO BE INCLUDED 

## Data
Data produced using this code is available at DATA DTU following the DOI above. 
C/S-band radar observations have been re-tracked with pySnowRadar (linked above), however due to issues with installing the package on my own computer, I instead had to use the functions from pySnowRadar locally. This includes the following python-programmes included in this repository: "Peakiness.py" and "Wavelet.py".
For data availability, please check the associated manuscript.

## Code
Conda environment used to run the code is available under "cryo2iceant.yml". Beyond the pySnowRadar package and functions used, the following python documents are included: 
- _tfmra_py3.py_: A threshold-first-maxima-retracker-algorithm (TFMRA) re-tracker.
- _airborne_CRESIS_mat_to_netCDF.py_: Programme to load all relevant .mat frames available after post-processing by CReSIS, and where each frame are re-tracked using the relevant re-trackers and the derived elevations (and other paramters) for the entire under-flight (including all relevant .mat-files) are saved into one file per radar.
- _CRYO2ICE_func.py_: Functions to derive the CRYO2ICE collocated observations for deriving snow depth, and for extracting the nearest neighbouring AMSR2 and CASSIS observations. Based on code used to compute data for Fredensborg Hansen et al (2024), available at following repository: https://github.com/reneefredensborg/CRYO2ICE-Arctic-freeboards-and-snow-depth-2020-2022

For the post-processing including plotting, the following Python documents are relevant:
- _main_CRESIS_comparison.py_: Airborne observations are opened, post-processed and inter-compared. This includes deriving a nadir lidar profile, offset calibration, discrimination of floes and leads, tracking the maximum scattering horizon, and evaluating penetration (to maximum scattering) and derive different snow depth estimates. 
- _CRYO2ICE_CS2_IS2_FF-SAR_comp.py_: CryoSat-2 observations are aligned with ICESat-2 observations usign three different CryoSat-2 products, and following the CVRYO2ICE collocation methodology of Fredensborg Hansen et al (2024). 
- _CRYO2ICE_airborne_comp.py_: The derived CRYO2ICE snow depths and the airborne snow depths are aligned, binned, and compared.

All plots included in the pre-print are available in the post-processing plots. NB! Work still required to clean up code. 

## References
Fredensborg Hansen, R. M., Skourup, H., Rinne, E., Høyland, K. V., Landy, J. C., Merkouriadi, I., & Forsberg, R. (2024). Arctic Freeboard and Snow Depth From Near-Coincident CryoSat-2 and ICESat-2 (CRYO2ICE) Observations: A First Examination of Winter Sea Ice During 2020–2022. Earth and Space Science, 11(4), Article e2023EA003313. https://doi.org/10.1029/2023EA003313

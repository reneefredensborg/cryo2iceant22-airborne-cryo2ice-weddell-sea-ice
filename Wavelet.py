import numpy as np
import pywt

def Wavelet_TN(data, null_2_space, delta_fast_time_range, n_snow, ref_snow_layer, cwt_precision, **kwargs):
    '''
    Function to detect 2 interface layers from a given SnowRadar signal:
        Air-Snow interface
        Snow-Ice Interface

    Uses the Continuous Wavelet Transform (cwt) method originally developed
    by Thomas Newman

    Arguments:
        data: 1D radar data array
        null_2_space: trough to trough distance
        delta_fast_time_range: radar bin range in m
        n_snow: the refractive index of snow
        ref_snow_layer: reference snow depth in m (default 1)
        cwt_precision: precision arg for cwt (default 10)

    Outputs:
        locs_as: Air-snow interface bin index (integer)
        locs_si: Snow-ice interface bin index (integer)

        
    Edits: 
    - 2023/11/13 by rmfha: changes .astype(int) to int(x) 
    '''
    ref_scale_lin_m = 2 * null_2_space
    max_scale_lin = np.ceil(ref_scale_lin_m / delta_fast_time_range)
    lin_scale_vect = np.arange(2, max_scale_lin, 1)[1::2]
    
    snow_layer_opl = ref_snow_layer * n_snow * 2
    ref_scale_log_m = 2 * snow_layer_opl
    max_scale_log = np.ceil(ref_scale_log_m / delta_fast_time_range)
    log_scale_vect = np.arange(2, max_scale_log, 1)[1::2]
    
    lin_coefs = cwt(data, pywt.Wavelet('haar'), lin_scale_vect, cwt_precision)
    log_coefs = cwt(10 * np.log10(data), pywt.Wavelet('haar'), log_scale_vect, cwt_precision)
    
    # Negating edge effects here, we use half the max scale on either end
    # Some discussion is needed on this approach because it can sometimes lead to weird picks
    lin_coefs[:, 0:int(np.ceil(max_scale_lin/2))] = 0
    lin_coefs[:, -int(np.ceil(max_scale_lin/2)):] = 0

    log_coefs[:, 0:int(np.ceil(max_scale_log/2))] = 0
    log_coefs[:, -int(np.ceil(max_scale_log/2)):] = 0
    
    sum_log_coefs = np.sum(log_coefs,axis=0) / log_coefs.shape[0]
    sum_lin_coefs = np.sum(lin_coefs,axis=0) / lin_coefs.shape[0]
    
    locs_si = np.argmax(-sum_lin_coefs)
    locs_as = np.argmax(-sum_log_coefs)
    
    return locs_as, locs_si

def Wavelet_JK(data, scale_vect, **kwargs):
    log_gaus1_coefs, _ =  pywt.cwt(10 * np.log10(data),scale_vect,'gaus1')
    log_gaus1_coefs[:, 0:np.ceil(scale_vect[-1]*2).astype(int)] = np.nan
    log_gaus1_coefs[:, -np.ceil(scale_vect[-1]*2).astype(int):] = np.nan
    sum_log_gaus1_coefs = np.sum(log_gaus1_coefs,axis=0) / log_gaus1_coefs.shape[0]
    locs_as = np.nanargmin(sum_log_gaus1_coefs)

    lin_gaus2_coefs, _ = pywt.cwt(data,scale_vect,'gaus2')
    lin_gaus2_coefs[:, 0:np.ceil(scale_vect[-1]*2).astype(int)] = np.nan
    lin_gaus2_coefs[:, -np.ceil(scale_vect[-1]*2).astype(int):] = np.nan
    sum_lin_gaus2_coefs = np.sum(lin_gaus2_coefs,axis=0) / lin_gaus2_coefs.shape[0]
    locs_si = np.nanargmax(sum_lin_gaus2_coefs)

    return locs_as, locs_si

def cwt(data, wavelet, scales, precision):
    '''
    Implementation of the Continuous Wavelet Transform

    Arguments:
        data: preprocessed snowradar signal data
        wavelet: the specific Wavelet to use (currently the Haar wavelet)
        scales: 
        precision: precision to apply to wavelet operations (default 10)

    Outputs:
        out_coefs:(?)
    '''
    out_coefs = np.zeros((np.size(scales), data.size))
    int_psi, x = pywt.integrate_wavelet(wavelet, precision=precision)
    step = x[1] - x[0]
    x_step = (x[-1] - x[0]) + 1
    
    j_a = [np.arange(scale * x_step) / (scale * step) for scale in scales]
    j_m = [np.delete(j, np.where((j >= np.size(int_psi)))[0]) 
           for j in j_a if np.max(j) >= np.size(int_psi)]
    coef_a = [-np.sqrt(scales[i]) 
               * np.diff(np.convolve(data, int_psi[x.astype(np.int)][::-1]))
              for (i, x) in enumerate(j_m)]
    out_coefs = np.asarray([coef[int(np.floor((coef.size - data.size) / 2))
                            :int(-np.ceil((coef.size - data.size) /2))] for coef in coef_a])
    return out_coefs

def calcpulsewidth(bandwidth, oversample_num=1000, num_nyquist_ts=100):
    '''
    Using the current SnowRadar instance's radar bandwidth (self.bandwidth),
    calculate and set the values for the null-to-null pulse width (self.n2n)
    and the equivalent pulse width (self.epw) 

    The windowing process used is `signal.hann`

    Arguments:
        oversample_num: the bin-amount to oversample the nyquist by
        num_nyquist_ts: the number of nyquist timestamps to use when windowing(?)
    
        
    Edits: 
    - 2023/11/13 by rmfha: changed self to take direct variables to run in the CRESIS code 
    '''

    from scipy import signal
    C = 299792458


    # Time Vector
    nyquist_sf = 2 * bandwidth
    fs = nyquist_sf * oversample_num 
    time_step = 1 / fs 
    max_time = num_nyquist_ts * oversample_num * time_step
    time_vect = np.linspace(-max_time, max_time, int(np.ceil((max_time*2)/time_step)))  

    # Frequency domain object
    half_bandwidth = bandwidth / 2
    n_FFT = len(time_vect)
    f = fs * np.linspace(-0.5, 0.5, n_FFT)
    n_band_points = np.sum(np.abs(f) <= half_bandwidth)

    # Create spectral window 
    spectral_win = signal.hann(n_band_points, sym = False)


    # Frequency domain processing
    # JK: Need to be careful here, f becomes an array if bandwidth is as well.
    # Change it to use f.shape?
    freq_domain_signal = np.zeros(len(f)) 
    try:
        freq_domain_signal[np.abs(f) < half_bandwidth] = spectral_win
    except:
        freq_domain_signal[np.abs(f) < half_bandwidth] = spectral_win[0][0]
    shift_freq_domain_signal = np.fft.ifftshift(freq_domain_signal)
    time_domain_signal = np.fft.ifft(shift_freq_domain_signal) * n_FFT
    time_sig = np.fft.fftshift(time_domain_signal)
    power_signal = np.abs(time_sig ** 2)
    power_signal_norm = power_signal / np.max(power_signal)
    max_idx = np.argmax(power_signal_norm)

    # Calc the equivalent pulse width
    equiv_pulse_width_val = np.sum(power_signal_norm)
    equiv_pulse_width_time = equiv_pulse_width_val * time_step
    epw = equiv_pulse_width_time * C

    # Calc null-to-null pulse width
    with np.errstate(divide = 'ignore'):
        invert_l10_power = -10 * np.log10(power_signal_norm)
    peak_idx, _ = signal.find_peaks(invert_l10_power)
    closest_peaks = np.sort(np.abs(peak_idx - max_idx))

    null_2_width = 2 * np.mean(closest_peaks[0:1])
    null_2_time = null_2_width * time_step
    n2n = null_2_time * C

    return epw, n2n, 


import numpy as np

def R_ratio(corr, og):
    percent_noise = 0.375 # percent of end of signal to use as noise
    sig_len = og.shape[-1]
    noise_i = int(np.ceil(sig_len * percent_noise))
    return float(np.mean(np.std(corr[:,-noise_i:], axis=1) 
                         / np.std(og[:,-noise_i:], axis=1)))

def TTA(signal,samp_rate):
    sampling_rate = samp_rate
    samp_per_ms = int(sampling_rate / 1000)
    percent_noise = 0.375
    sig_len = signal.shape[-1]
    noise_i = int(np.ceil(sig_len*percent_noise))
    signal = abs(np.mean(signal, axis=0))
    sig_avg = np.mean(signal[:samp_per_ms])
    noise_avg = np.mean(signal[-noise_i:])
    return float(sig_avg / noise_avg)
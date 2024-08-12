"""Module for signal data processing functions related to active noise cancellation."""

import numpy as np

def combine_complex(array: np.ndarray) -> np.ndarray:
    """Combine complex components of an array.

    Args:
        array : input array with complex components split along
            axis 0

    Returns:
        output complex array

    """
    return array[0] + 1j * array[1]


def split_complex(array: np.ndarray) -> np.ndarray:
    """Split complex components of an array.

    Args:
        array : input complex array.

    Returns:
        output array with complex components split along axis 0.
    """
    return np.stack((array.real, array.imag), axis=0)


def normalize_by_max(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize input RX/EMI data for CNN correction.

    This function normalizes input data, sample by sample, by dividing each sample by
    its maximum absolute value. 

    Args:
        signal : shape (nb_repetitions, nb_samples_per_shot) input RX/EMI signal
        complex_split : whether to separate complex components along
            new channel (required for inputs to model)

    Returns:
        tuple: a tuple containing
            normalized_signal : shape (nb_repetitions, nb_samples_per_shot) normalized signal
            scale : the maximum absolute value of each sample in the dataset.
    """
    scale = np.expand_dims(np.max(abs(signal), axis=1), axis=1)
    normalized_signal = signal / scale
    return normalized_signal, scale

def split_freq(signal):
    # split input signal into multiple channels multiple channels divided by frequency ranges
    larmor_frequency = 42577
    samp_rate = 25000
    frequencies = (
        np.fft.fftshift(np.fft.fftfreq(signal.shape[-1], 1 / samp_rate)) + larmor_frequency
    ) / 1e3
    fft = np.fft.fftshift(np.fft.fft(signal))
    lp = fft.copy()
    bp = fft.copy()
    hp = fft.copy()
    for i in range(len(frequencies)):
        if frequencies[i] > 40: # set lowpass max frequency
            lp[:,i] = 0.0
        if frequencies[i] > 50 or abs(frequencies[i]) < 35: # set bandpass frequency range
            bp[:,i] = 0.0
        if frequencies[i] < 45: # set highpass min frequency
            hp[:,i] = 0.0
    lp_EMI = np.fft.ifft(np.fft.ifftshift(lp))
    bp_EMI = np.fft.ifft(np.fft.ifftshift(bp))
    hp_EMI = np.fft.ifft(np.fft.ifftshift(hp))
    ax = 2
    split_signal = np.concatenate((lp_EMI,bp_EMI,hp_EMI),axis=ax)
    return split_signal
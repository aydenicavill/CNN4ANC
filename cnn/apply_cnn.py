"""Method that applies active noise cancellation via Pre-Trained CNN."""

import numpy as np

import torch

from  torch.nn import Module

from torch.utils.data import TensorDataset, DataLoader

from utils import combine_complex, normalize_by_max, split_complex

def apply_cnn(
    data_mri: np.ndarray,
    data_noise: np.ndarray,
    model: Module,
    device: str = "cpu",
    batch_size: int = 32,
    num_workers: int = 8,
) -> np.ndarray:
    """Apply CNN to perform active noise cancellation.

    Args:
        data_mri : shape (nb_repetitions, nb_shots, nb_samples_per_shot) MRI data
            to be corrected.
        data_noise : shape (nb_repetitions, nb_shots, nb_samples_per_shot, nb_channels)
            EMI data used for the correction.
        model : pretrained CNN model.
        device : specifies the device to be used for computation.
        batch_size : number of samples to feed the device at a time.
        num_workers :  number of parallel subprocesses to activate during correction.

    Returns:
        array of the same size as data_mri, and contains the corrected data.

    """
    nb_repetitions, nb_shots, nb_samples_per_shot, nb_channels = data_noise.shape

    # Reshape data to group all repeated shots together
    data_mri = data_mri.reshape(-1, nb_samples_per_shot)
    data_noise = data_noise.reshape(-1, nb_samples_per_shot, nb_channels)

    # Prepare data
    data_mri_normalized, scale = normalize_by_max(data_mri)

    data_noise_normalized = []
    nb_channels = data_noise.shape[-1]
    for ch in range(nb_channels):
        data_noise_ch, _ = normalize_by_max(data_noise[:, :, ch])
        data_noise_ch = split_complex(data_noise_ch)
        data_noise_normalized.append(data_noise_ch)
    data_noise_normalized = np.stack((data_noise_normalized), axis=3)
    data_noise_normalized = np.transpose(data_noise_normalized, (1, 0, 2, 3))
    data_noise_normalized = torch.tensor(data_noise_normalized, dtype=torch.float).to(device)

    data_noise_dataset = TensorDataset(data_noise_normalized)
    data_noise_dataloader = DataLoader(
        data_noise_dataset, batch_size=batch_size, num_workers=num_workers
    )

    # Use model to predict MRI noise from EMI data
    predicted_noise_normalized = []
    model.eval()
    with torch.no_grad():
        for data in data_noise_dataloader:
            data = data[0].to(device)
            output = model(data)
            predicted_noise_normalized.append(output)
    predicted_noise_normalized = torch.cat(predicted_noise_normalized, dim=0)
    predicted_noise_normalized = (
        np.squeeze(predicted_noise_normalized, axis=3).cpu().detach().numpy()
    )
    predicted_noise_normalized = np.transpose(predicted_noise_normalized, (1, 0, 2))
    predicted_noise_normalized = combine_complex(predicted_noise_normalized)

    # Remove predicted noise from MRI data
    corrected = (data_mri_normalized - predicted_noise_normalized) * scale
    corrected = corrected.reshape((nb_repetitions, nb_shots, nb_samples_per_shot))
    return corrected
                         
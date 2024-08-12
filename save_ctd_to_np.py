import numpy as np
import tentacles as ttc
import os

job = 'train'

save_path = f'august_data/{job}_data/008/'
os.makedirs(save_path, exist_ok=True)

data_path = '008_10minTxOff/00_data/RawData_noise_only.ctd'


def rephase(RX,EMI):
    length_shot = RX.shape[1]
    n_shots = RX.shape[0]
    freq = 25000
    start_time= 0.004
    TR = 0.1
    times = np.expand_dims(np.array(
        [
            np.arange(length_shot) / (freq) + start_time + i * TR
            for i in range(n_shots)
        ]
    ), axis=2)
    freq_diff = 0.00423528
    print(RX.shape,EMI.shape)
    EMI_rephased = EMI*np.exp(1j*freq_diff*times*2*np.pi)
    return EMI_rephased


dict_tree = ttc.conversion.read_chipiron_to_dict_tree(filepath=data_path)

data = dict_tree['acquisition']['data']
noise = dict_tree['anc']['noise_data']

data = np.squeeze(data,axis=1)
noise = np.squeeze(noise,axis=1)

#noise = rephase(data,noise)

np.save(save_path+f'{job}_Rx.npy',data)
np.save(save_path+f'{job}_EMI.npy',noise)
print(save_path)

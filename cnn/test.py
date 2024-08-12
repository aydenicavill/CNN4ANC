from apply_cnn import apply_cnn
import matplotlib.pyplot as plt
from metrics import R_ratio, TTA
from model import CNN4ANC
import numpy as np
import tentacles as ttc
import torch
from utils import split_freq
import time

SAMP_RATE = 25000

model_path = "models/finetuned-031-model.pth"
data_path = f'../../august_data/test_data/007/'
device = "cpu"

model = CNN4ANC()
model.load_state_dict(torch.load(model_path,map_location=torch.device(device)))

starttime = time.time()
RX = np.load(data_path+'test_Rx.npy')
EMI = np.load(data_path+'test_EMI.npy')
RX_expanded = np.expand_dims(RX,axis=1)
EMI_expanded = np.expand_dims(EMI,axis=1)

#cnn_corrected = apply_cnn(RX_expanded,EMI_expanded,model=model)
cnn_corrected = ttc.anc.apply_cnn(RX_expanded,EMI_expanded,model=model)
editer_corrected = ttc.anc.apply_editer(RX_expanded,EMI_expanded)

cnn_corrected = np.squeeze(cnn_corrected,axis=1)
editer_corrected = np.squeeze(editer_corrected,axis=1)

endtime = time.time()
print(f'time taken: {endtime-starttime}')

cnn_r = R_ratio(cnn_corrected,RX)
editer_r = R_ratio(editer_corrected,RX)

cnn_snr = TTA(cnn_corrected,SAMP_RATE)
editer_snr = TTA(editer_corrected,SAMP_RATE)
original_snr = TTA(RX,SAMP_RATE)

print(f'CNN R Ratio: {cnn_r}')
print(f'EDITER R Ratio: {editer_r}')
print()
print(f'Original SNR: {original_snr}')
print(f'CNN SNR: {cnn_snr}')
print(f'EDITER SNR: {editer_snr}')

'''
sig_length = RX.shape[-1]
time_x = np.linspace(0,sig_length/SAMP_RATE*1000,sig_length)

plt.plot(time_x,abs(np.mean(RX,axis=0)),label='Original',alpha=0.6)
plt.plot(time_x,abs(np.mean(editer_corrected,axis=0)),label='EDITER Corrected',color='r')
plt.plot(time_x,abs(np.mean(cnn_corrected,axis=0)),label='CNN Corrected',color='green')

larmor_frequency = 42577
frequencies = (
        np.fft.fftshift(np.fft.fftfreq(sig_length, 1 / SAMP_RATE)) + larmor_frequency
    ) / 1e3
#plt.plot(frequencies,np.abs(np.fft.fftshift(np.fft.fft(np.mean(RX, axis=0)))),label='Original',alpha=0.6)
#plt.plot(frequencies,np.abs(np.fft.fftshift(np.fft.fft(np.mean(editer_corrected, axis=0)))),label='EDITER Corrected',color='red')
#plt.plot(frequencies,np.abs(np.fft.fftshift(np.fft.fft(np.mean(cnn_corrected, axis=0)))),label='CNN Corrected',color='green')


plt.title(f'CNN SNR: ' + r'$\bf{' + str(round(cnn_snr,4))+'}$\nCNN R Ratio: ' + r'$\bf{' + str(round(cnn_r,4)) + '}$')
#plt.title(f'EDITER SNR: {round(anc_TTA,4)}\nEDITER R Ratio: {round(anc_R,4)} ')
#plt.title(f'CNN R Ratio: {round(cnn_R,4)} EDITER R Ratio: {round(anc_R,4)} ')
#plt.title(f'EDITER SNR: {round(editer_snr,4)}, CNN SNR: ' + r'$\bf{' + str(round(cnn_snr,4))+'}$' 
#          + f'\nEDITER R Ratio: {round(editer_r,4)}, CNN R Ratio: ' + r'$\bf{' + str(round(cnn_r,4)) + '}$')
#plt.title(f'EDITER SNR: {round(editer_snr,4)}, CNN SNR: {round(cnn_snr,4)}\nEDITER R Ratio: {round(editer_r,4)}, CNN R Ratio: {round(cnn_r,4)}')
plt.title('6:15 Model, GRE')

#plt.bar(0,height=0.6,width=1,align='edge',color='k',alpha=0.2,label='TTA Regions')
#plt.bar(50,height=0.6,width=30,align='edge',color='k',alpha=0.2)
plt.xlabel('time (ms)')
#plt.ylim(0,0.4)
#plt.xlabel('frequency (kHz)')
plt.legend(loc='upper right')
plt.grid()
plt.show()
'''
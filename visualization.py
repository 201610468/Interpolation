
#%%
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import numpy as np
from argparse import Namespace
from termcolor import colored

from main import Interpolator
import utils as u
from data import extract_patches

import matplotlib.pyplot as plt
from matplotlib import rcParams
params = {
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 300,  # to adjust notebook inline plot size
    'axes.labelsize': 14, # fontsize for x and y labels (was 10)
    'axes.titlesize': 14,
    'font.size': 14,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'text.usetex':True
}
rcParams.update(params)

true = np.load('/home/jbgpl/seismic_DL/interpolation/deep_prior_interpolation-master/datasets/pohang/observed/observed76_1.npy')
true = true*10000

inter = np.load('/home/jbgpl/seismic_DL/interpolation/deep_prior_interpolation-master/jupyter/resized_image_array.npy')


run = np.load(os.path.join('/home/jbgpl/seismic_DL/interpolation/deep_prior_interpolation-master/results/notebook_2d/', '0_run.npy'), allow_pickle=True).item()
dpi = run['output']
dpi = dpi*10000
dpi = dpi.reshape(1500,-1)

dpi_po = np.load('/home/jbgpl/seismic_DL/interpolation/deep_prior_interpolation-master/results/pohang/result_interval2_pohang1.npy')


opt = dict(cmap="gray", clim=u.clim(true, 98), aspect='auto')
normalized_squared_error = (dpi - inter)**2 / np.max(np.abs(dpi))**2
plt.subplot(133), plt.title('true')
plt.imshow(true.squeeze(), **opt)#, plt.colorbar()
plt.subplot(131), plt.title('resize')
plt.imshow(inter.squeeze(), **opt)#, plt.colorbar()
plt.subplot(132), plt.title('dpi')
plt.imshow(dpi.squeeze(), **opt)#, plt.colorbar()
#plt.subplot(133), plt.title('Residual')
##plt.imshow(normalized_squared_error.squeeze(),**opt)
# %%
#%%
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import numpy as np
from argparse import Namespace
from termcolor import colored

from main_copy import Interpolator
import utils as u
from data_copy import extract_patches

import matplotlib.pyplot as plt
from matplotlib import rcParams
params = {
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 300,  # to adjust notebook inline plot size
    'axes.labelsize': 14, # fontsize for x and y labels (was 10)
    'axes.titlesize': 14,
    'font.size': 14,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'text.usetex':True
}
rcParams.update(params)

po_3d = np.fromfile('/home/jbgpl/seismic_DL/interpolation/deep_prior_interpolation-master/pohang/observed.0001',dtype='float32')
po_3d = po_3d.reshape(38,-1)
po_3d = po_3d.transpose()
po_3d = po_3d.reshape(1,1,-1,38)

for i in range(4,206,3):
    pohang = np.fromfile('/home/jbgpl/seismic_DL/interpolation/deep_prior_interpolation-master/pohang/observed.'+str(i).zfill(4),dtype='float32')
    pohang = pohang.reshape(38,-1)
    pohang = pohang.transpose()
    pohang = pohang.reshape(1,1,-1,38)
    po_3d = np.concatenate([po_3d,pohang],1)
    
print(po_3d.shape)

np.save('/home/jbgpl/seismic_DL/velocity_model/FWIGAN/models/pohang_3D_76_4096_38',po_3d)



opt = dict(cmap="gray", clim=u.clim(pohang, 98), aspect='auto')
plt.imshow(po_3d[:,30,:,:].squeeze(), **opt)#, plt.colorbar()
# %%
import numpy as np

aa = np.fromfile('/home/jbgpl/working/03_3D_TAASMP_v1.0/model/tvp3model_origin.bin',dtype='float32')
aa = aa.reshape(450,)
print(aa.min())
# %%
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import numpy as np
from argparse import Namespace
from termcolor import colored

from main_copy import Interpolator
import utils as u
from data_copy import extract_patches

import matplotlib.pyplot as plt
from matplotlib import rcParams
params = {
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 300,  # to adjust notebook inline plot size
    'axes.labelsize': 14, # fontsize for x and y labels (was 10)
    'axes.titlesize': 14,
    'font.size': 14,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'text.usetex':True
}
rcParams.update(params)



# a = np.fromfile('/home/jbgpl/seismic_DL/SR_denoising/SeismicSuperResolution/data/sleipner2008.bin',dtype='float32')
# a = a.reshape(425,180,53)*10 # (y,x,z)
# a= a.transpose()
a = np.fromfile('/home/jbgpl/seismic_DL/SR_denoising/SeismicSuperResolution/data/sleipner2008.bin',dtype='float32')
a = a.reshape(425,180,53)*10 #(y,x,z)
a = a[155:278,:112,:48]
#a = a[:,1::2]
a = a.transpose()

print(a.shape)
opt = dict(cmap="gray", clim=u.clim(a, 98), aspect='auto')

plt.imshow(a[:,:,5].squeeze(), **opt)#, plt.colorbar()
# %%

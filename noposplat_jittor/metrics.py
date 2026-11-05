import numpy as np


def psnr(gt,pred):
    mse=np.mean((np.clip(gt,0,1)-np.clip(pred,0,1))**2,axis=(-3,-2,-1))
    with np.errstate(divide='ignore'):return -10*np.log10(mse)


def ssim(gt,pred):
    from skimage.metrics import structural_similarity
    return np.asarray([structural_similarity(a,b,win_size=11,gaussian_weights=True,channel_axis=0,data_range=1.0) for a,b in zip(gt,pred)])


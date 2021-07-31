import numpy as np


def tsdf_loss(gt, est, norm='L2', mask=None):

    N = 1
    dim = len(gt.shape)

    for d in range(0, dim):
        N *= gt.shape[d]

    diff = gt - est

    if mask is not None:
        diff = np.multiply(mask, diff)

    # compute loss only where tsdf is positive
    valid = (est > 0).astype(int)
    diff = np.multiply(valid, diff)

    if norm == 'L2':
        loss = 1./N*np.sum(np.power(diff, 2))

    return loss

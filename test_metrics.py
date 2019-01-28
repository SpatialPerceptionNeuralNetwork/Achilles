import numpy as _np

a = _np.load("a.npy")
b = _np.load("b.npy")
m = _np.where(b <= 0.0, 0.0, 1.0)
npix = _np.sum(m)


def _get_threshold(out, gt, delta):
    # out = out.flatten()
    # gt = gt.flatten()
    # mask = m.flatten()
    # out *= mask
    # gt *= mask

    gt = _np.clip(gt, 1e-6, 80.0)
    # out = _np.clip(out, 1e-6, 80.0)
    npix = _np.sum(_np.sign(gt))
    threshold = _np.maximum((out / gt), (gt / out))
    result = _np.greater_equal(threshold, delta).astype(_np.float32)
    acc_val = npix - _np.sum(result)
    ret = acc_val / npix
    return ret


print("Accuracy..... ",_get_threshold(a, b, 1.25))
print("Accuracy..... ",_get_threshold(a, b, 1.25**2))
print("Accuracy..... ",_get_threshold(a, b, 1.25**3))


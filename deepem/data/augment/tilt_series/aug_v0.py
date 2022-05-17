from augmentor import *


def get_augmentation(is_train, recompute=False, flip=False, 
                     tilt_series_src=0, tilt_series_dst=0, **kwargs):
    augs = []

    # Flip & rotate (isotropic)
    if flip:
        augs.append(FlipRotateIsotropic())

    # Tilt series projection & label subsampling
    ts_src = tilt_series_src
    ts_dst = tilt_series_dst
    if ts_src > 0 and ts_dst:
        assert ts_src > ts_dst
        augs.append(TiltSeries(ts_src))
        augs.append(SubsmapleLabels(factor=(ts_dst,1,1)))

    # Recompute connected components
    if recompute:
        augs.append(Label())

    return Compose(augs)

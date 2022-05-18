from augmentor import *


def get_augmentation(is_train, tilt_series=(0,0,0), tilt_series_crop=None,
                     recompute=False, flip=False, **kwargs):
    augs = []

    # Flip & rotate (isotropic)
    if flip:
        augs.append(FlipRotateIsotropic())

    # Tilt series projection & label subsampling
    if tilt_series[0] > 0:
        ts_in = tilt_series[1]
        ts_out = tilt_series[2]
        assert ts_in > 0 and ts_out > 0
        assert ts_in > ts_out
        assert ts_in % ts_out == 0
        augs.append(TiltSeries(ts_in))
        augs.append(SubsampleLabels(factor=(ts_out,1,1)))
        if tilt_series_crop is not None:
            augs.append(CropLabels(tilt_series_crop))

    # Recompute connected components
    if recompute:
        augs.append(Label())

    return Compose(augs)

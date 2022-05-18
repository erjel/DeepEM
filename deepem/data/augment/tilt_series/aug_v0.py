from augmentor import *


def get_augmentation(is_train, recompute=False, flip=False, 
                     tilt_series_in=0, tilt_series_out=0, **kwargs):
    augs = []

    # Flip & rotate (isotropic)
    if flip:
        augs.append(FlipRotateIsotropic())

    # Tilt series projection & label subsampling
    if (tilt_series_in > 0) and (tilt_series_out > 0):
        assert tilt_series_in > tilt_series_out
        augs.append(TiltSeries(tilt_series_in))
        augs.append(SubsmapleLabels(factor=(tilt_series_out,1,1)))

    # Recompute connected components
    if recompute:
        augs.append(Label())

    return Compose(augs)

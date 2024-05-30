import numpy as np


def load_spectrum(path):
    x, y = np.loadtxt(
        path,
        usecols=(0, 1),
        skiprows=3,
        unpack=True,
    )
    return x, y


def file_stitcher(x_lower, y_lower, x_upper, y_upper, stitch_freq):
    l_ind = np.where(x_lower < stitch_freq)[0]
    u_ind = np.where(x_upper >= stitch_freq)[0]
    x_l = x_lower[l_ind]
    y_l = y_lower[l_ind]
    x_u = x_upper[u_ind]
    y_u = y_upper[u_ind]
    x = np.concatenate((x_l, x_u))
    y = np.concatenate((y_l, y_u))
    return x, y


path1 = "/Users/dairenjean/OneDrive/Projects/test_ROT/2022-09-30-12_08_(3-cyano)methylenecyclopropane_235-345GHz_12mtorr.spe"
path2 = "/Users/dairenjean/OneDrive/Projects/test_ROT/2022-09-27-10_10_(3-cyano)methylenecyclopropane_340-500GHz_12mtorr.spe"
x_low, y_low = load_spectrum(path1)
x_up, y_up = load_spectrum(path2)
x_stitched, y_stitched = file_stitcher(x_low, y_low, x_up, y_up, 345000)



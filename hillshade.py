import numpy as np


def hillshade(data, light_angle, light_azimuth, z_factor):
    # Based on QGIS src/core/raster/qgshillshaderenderer.cpp
    zenith_rad = max(0, 90 - light_angle) * np.pi / 180
    azimuth_rad = -light_azimuth * np.pi / 180
    cos_zenith_rad = np.cos(zenith_rad)
    sin_zenith_rad = np.sin(zenith_rad)

    x11 = data[:-2, :-2]
    x21 = data[1:-1, :-2]
    x31 = data[2:, :-2]
    x12 = data[:-2, 1:-1]
    # x22 = data[1:-1, 1:-1]
    x32 = data[2:, 1:-1]
    x13 = data[:-2, 2:]
    x23 = data[1:-1, 2:]
    x33 = data[2:, 2:]
    der_x = ((x13 + x23 + x23 + x33) - (x11 + x21 + x21 + x31)) / 8
    der_y = ((x31 + x32 + x32 + x33) - (x11 + x12 + x12 + x13)) / 8

    slope_rad = np.arctan(z_factor * np.sqrt(der_x * der_x + der_y * der_y))
    aspect_rad = np.arctan2(der_x, -der_y)

    return np.clip(255 * (
        cos_zenith_rad * np.cos(slope_rad) +
        sin_zenith_rad * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)),
        0, 255)

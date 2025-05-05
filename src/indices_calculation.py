'''
Author  : Daniel Bugelnig
Email   : daniel.j.bugelnig@gmail.com
Date    : 2025-05-05
Purpose : AI4EO, indices calculation, and data preparation

TODO change the paths of the data to the new ones
'''
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # oder 'Qt5Agg', wenn du Qt installiert hast
import xarray as xr
import rioxarray as rxa
from indices import EOIndices 
import matplotlib.image as mpimg

'''--------------------------------------------------------------'''
#Loading data again
folder = "../data/treated/nc_files/" 
S1_0418 = xr.load_dataset(folder + "subset_S1A_20230418.nc")
S1_0629 = xr.load_dataset(folder + "subset_S1A_20230629.nc")
S2_0420 = xr.load_dataset(folder + "subset_S2B_20230420.nc")
S2_0624 = xr.load_dataset(folder + "subset1_S2A_20230624.nc")
file_path ="../data/Land_Cover_Pendes.tif"
with rasterio.open(file_path) as src:
    land_cover_data = src.read(1)

# Single plot auxiliar function
def plot_data(data, title="", colorbar=True, **kwargs):
    """
    Plot some specific data
    """
    if 'vmin' not in kwargs:
        kwargs['vmin'] = np.nanpercentile(data, 2)
    if 'vmax' not in kwargs:
        kwargs['vmax'] = np.nanpercentile(data, 98)
    plt.figure(figsize=(12,6))
    img = plt.imshow(data, **kwargs)
    plt.title(title)
    if colorbar:
        plt.colorbar(img)
    plt.tight_layout()
    plt.show()
    


'''--------------------------------------------------------------'''

# print(S1_0418['Amplitude_VH']) # 2146, 2548
# print(S1_0629['Amplitude_VH'])
# print(S2_0420.B2) # 1583, 2509
# print(S2_0624.B2)





'''--------------------------------------------------------------'''
# Sentinel-1 Indices
print(S1_0418.data_vars)
print(S2_0420.data_vars)
S1_0418_VH, S1_0418_VV = EOIndices.radar.extract_VV_VH(S1_0418)
S1_0629_VH, S1_0629_VV = EOIndices.radar.extract_VV_VH(S1_0629)


# Amplitude
S1_0418_VH_A,_ = EOIndices.radar.compute_A(S1_0418_VH)
S1_0418_VV_A,_ = EOIndices.radar.compute_A(S1_0418_VV)
S1_0629_VH_A,_ = EOIndices.radar.compute_A(S1_0629_VH)
S1_0629_VV_A,_ = EOIndices.radar.compute_A(S1_0629_VV)


# VH / VV Ratio
S1_0418_ratio = EOIndices.radar.compute_VH_VV_ratio(S1_0418_VH,S1_0418_VV)
S1_0629_ratio = EOIndices.radar.compute_VH_VV_ratio(S1_0629_VH,S1_0629_VV)


# Sigma Nought
S1_0418_VH_sigma_nought = EOIndices.radar.compute_sigma_nought_log(S1_0418_VH)
S1_0418_VV_sigma_nought = EOIndices.radar.compute_sigma_nought_log(S1_0418_VV)
S1_0629_VH_sigma_nought = EOIndices.radar.compute_sigma_nought_log(S1_0629_VH)
S1_0629_VV_sigma_nought = EOIndices.radar.compute_sigma_nought_log(S1_0629_VV)

# RVI
S1_0418_RVI = EOIndices.radar.compute_RVI(S1_0418_VH,S1_0418_VV)
S1_0629_RVI = EOIndices.radar.compute_RVI(S1_0629_VH,S1_0629_VV)

# RWI
S1_0418_RWI = EOIndices.radar.compute_RWI(S1_0418_VH,S1_0418_VV)
S1_0629_RWI = EOIndices.radar.compute_RWI(S1_0629_VH,S1_0629_VV)

# MPDI
S1_0418_MPDI = EOIndices.radar.compute_MPDI(S1_0418_VH,S1_0418_VV)
S1_0629_MPDI = EOIndices.radar.compute_MPDI(S1_0629_VH,S1_0629_VV)


"""Now you must plot each indice and make sure they look correct."""

# Plot each S1 indice, 0418
plot_data(data=S1_0418_VH_A, title="S1_0418_VH_A")
# plot_data(data=S1_0418_VV_A, title="S1_0418_VV_A")
# plot_data(data=S1_0418_ratio, title="S1_0418_ratio")
# plot_data(data=S1_0418_VH_sigma_nought, title="S1_0418_VH_sigma_nought")
# plot_data(data=S1_0418_VV_sigma_nought, title="S1_0418_VV_sigma_nought")
# plot_data(data=S1_0418_RVI, title="S1_0418_RVI")
# plot_data(data=S1_0418_RWI, title="S1_0418_RWI")
# plot_data(data=S1_0418_MPDI, title="S1_0418_MPDI")

# # Plot each S1 indice, 0629
# plot_data(data=S1_0629_VH_A, title="S1_0629_VH_A")
# plot_data(data=S1_0629_VV_A, title="S1_0629_VV_A")
# plot_data(data=S1_0629_ratio, title="S1_0629_ratio")
# plot_data(data=S1_0629_VH_sigma_nought, title="S1_0629_VH_sigma_nought")
# plot_data(data=S1_0629_VV_sigma_nought, title="S1_0629_VV_sigma_nought")
# plot_data(data=S1_0629_RVI, title="S1_0629_RVI")
# plot_data(data=S1_0629_RWI, title="S1_0629_RWI")
# plot_data(data=S1_0629_MPDI, title="S1_0629_MPDI")


# Sentinel 2 Indices
S2_0420_NDVI = EOIndices.optical.NDVI(S2_0420)
S2_0420_NDWI = EOIndices.optical.NDWI(S2_0420)
S2_0420_AWEI = EOIndices.optical.AWEI(S2_0420)
S2_0420_NDBI = EOIndices.optical.NDBI(S2_0420)
S2_0420_NDSI_snow = EOIndices.optical.NDSI_snow_SWIR(S2_0420)
S2_0420_NBR = EOIndices.optical.NBR(S2_0420)
S2_0420_NDSI_snow_green = EOIndices.optical.NDSI_snow_green_SWIR(S2_0420)

S2_0624_NDVI = EOIndices.optical.NDVI(S2_0624)
S2_0624_NDWI = EOIndices.optical.NDWI(S2_0624)
S2_0624_AWEI = EOIndices.optical.AWEI(S2_0624)
S2_0624_NDBI = EOIndices.optical.NDBI(S2_0624)
S2_0624_NDSI_snow = EOIndices.optical.NDSI_snow_SWIR(S2_0624)
S2_0624_NBR = EOIndices.optical.NBR(S2_0624)
S2_0624_NDSI_snow_green = EOIndices.optical.NDSI_snow_green_SWIR(S2_0624)
print(type(S2_0420_AWEI))

# Plot the indices
plot_data(data= S2_0420_NDVI, title="S2_0420_NDVI")
plot_data(data= S2_0420_NDWI, title="S2_0420_NDWI")
plot_data(data= S2_0420_AWEI, title="S2_0420_AWEI")
plot_data(data= S2_0420_NDBI, title="S2_0420_NDBI")
plot_data(data= S2_0420_NDSI_snow, title="S2_0420_NDSI_snow")
plot_data(data= S2_0420_NBR, title="S2_0420_NBR")
plot_data(data= S2_0420_NDSI_snow_green, title="S2_0420_NDSI_snow_green")

# # Plot the indices
# plot_data(data= S2_0624_NDVI, title="S2_0624_NDVI")
# plot_data(data= S2_0624_NDWI, title="S2_0624_NDWI")
# plot_data(data= S2_0624_AWEI, title="S2_0624_AWEI")
# plot_data(data= S2_0624_NDBI, title="S2_0624_NDBI")
# plot_data(data= S2_0624_NDSI_snow, title="S2_0624_NDSI_snow")
# plot_data(data= S2_0624_NBR, title="S2_0624_NBR")
# plot_data(data= S2_0624_NDSI_snow_green, title="S2_0624_NDSI_snow_green")




"""---

You can already see from the previous plots that the set of valid pixels for each dataset is different.
Generate a mask of the valid pixels for the Sentinel-2 and for the Sentinel-1 images and plot them. Additionally, 
if we want to combine them, we will need a mask containing the valid pixels for both datasets simulatneously.
Tip: You can use [`np.isfinite`](https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html) for this, 
since the invalid pixels are set to `nan`.
"""

msk_S1_20230418 = np.isfinite(S1_0418.Amplitude_VH.data)
msk_S1_20230629 = np.isfinite(S1_0629.Amplitude_VH.data)
msk_S2_20230420 = np.isfinite(S2_0420.B2.data)
msk_S2_20230624 = np.isfinite(S2_0624.B2.data)

# The total mask will be the logical and of both
msk_total = msk_S1_20230418 & msk_S1_20230629 & msk_S2_20230420 & msk_S2_20230624

plt.figure(figsize=(10, 10))

ax = plt.subplot(3, 2, 1)
plt.imshow(msk_S1_20230418)
ax.set_title('Mask for Sentinel-1 20230418 data')

ax = plt.subplot(3, 2, 2)
plt.imshow(msk_S1_20230629)
ax.set_title('Mask for Sentinel-1 20230629 data')

ax = plt.subplot(3, 2, 3)
plt.imshow(msk_S2_20230420)
ax.set_title('Mask for Sentinel-2 20230420 data')

ax = plt.subplot(3, 2, 4)
plt.imshow(msk_S2_20230624)
ax.set_title('Mask for Sentinel-2 20230624 data')

ax = plt.subplot(3, 2, (5, 6))
plt.imshow(msk_total)
ax.set_title('Combined mask for all four datasets')

plt.tight_layout()


"""Here we generate an optical RGB image containing only the valid area of both datasets"""
# Load previous rgb images
rgb_s1_20230418 = mpimg.imread("/home/danielbugelnig/AAU/6.Semester/AI4EO/data/rgb/false_rgb_20230418.png")
rgb_s1_20230629 = mpimg.imread("/home/danielbugelnig/AAU/6.Semester/AI4EO/data/rgb/false_rgb_20230629.png")
rgb_s2_20230420 = mpimg.imread("/home/danielbugelnig/AAU/6.Semester/AI4EO/data/rgb/rgb_20230420.png")
rgb_s2_20230624 = mpimg.imread("/home/danielbugelnig/AAU/6.Semester/AI4EO/data/rgb/rgb_20230624.png")



rgb_20230420_msk = rgb_s2_20230420.copy()
rgb_20230420_msk[~msk_total,:] = np.nan    # all invalid pixel are NAN

plot_data(rgb_20230420_msk, 'Optical image valid area on both datasets')

"""---

## 4. Loading the Land Cover data.

The last data to be loaded is the land cover. In this case we are working with a Tiff file. Therefore we will use rasterio to charge the data.
"""

file_path = "/home/danielbugelnig/AAU/6.Semester/AI4EO/data/Land_Cover_Pendes.tif"
with rasterio.open(file_path) as src:
  land_cover_data = src.read(1)

"""Now, plot the Land Cover and analyse that it is correct."""

# already plotted in data_preparation.py

"""One last check that we must do is to identify if the land cover has the same extension as the S1 and S2 data because we already preprocessed S1 and S2 but not the LC."""

print(S1_0418)
print(S2_0420)
print(land_cover_data.shape) # right now it has the same with S2 but not with S1

# Reproject Land Cover to match S1/S2

# We can use rioxarray library to open the LC data
lc_geotiff = rxa.open_rasterio(file_path)

# First we have to define the CRS in the rasterio format to use its
# functionality
ds = S2_0420    # Using here one S2 dataset but both S1 and S2 should have the same CRS
wkt_string = ds.crs.attrs.get("wkt")
if wkt_string is None:
    raise ValueError("No WKT found in ds.crs attributes.")

# Write the extracted WKT as the CRS for the dataset
ds = ds.rio.write_crs(wkt_string, inplace=False)

# Now ds.rio.crs should reflect the correct CRS, and reproject_match() should work
print(ds.rio.crs)
landcover_aligned = lc_geotiff.rio.reproject_match(ds)

# Assign the aligned LC data array to land_cover_data
land_cover_data = landcover_aligned.isel(band=0).values

landcover_aligned.plot()

# Now they should have the same shape
print(S1_0418)
print(S2_0420)
print(land_cover_data.shape)

# Plot of the aligned LC
plot_data(landcover_aligned.isel(band=0).values, colorbar=True)

"""## 4. Generating and saving the data stack

Finally, we will generate the data stack by merging together all the different parameters and data that we have computed. Also we will generate the labels of each band to add into our final Tiff file.

Notes:
*   To save the RGB image we will directly save the Red, Green and Blue bands instead of the 3d array.
*   We will save the VV and VH amplitudes in dB.

> Add blockquote



Careful! Each index and variable with data is in xarray format; only the values could be stack together.
"""

Penedes_datastack = np.stack([10*np.log10(S1_0418.Amplitude_VV.values), 10*np.log10(S1_0629.Amplitude_VV.values),
                              10*np.log10(S1_0418.Amplitude_VH.values), 10*np.log10(S1_0629.Amplitude_VH.values),
                              S1_0418_ratio.values, S1_0629_ratio.values,
                              S1_0418_VH_sigma_nought.values, S1_0629_VH_sigma_nought.values,
                              S1_0418_VV_sigma_nought.values, S1_0629_VV_sigma_nought.values,
                              S1_0418_RVI.values, S1_0629_RVI.values,
                              S1_0418_RWI.values, S1_0629_RWI.values,
                              S1_0418_MPDI.values, S1_0629_MPDI.values,
                              S2_0420.B4, S2_0420.B3, S2_0420.B2,
                              S2_0624.B4, S2_0624.B3, S2_0624.B2,
                              S2_0420_NDVI.values, S2_0624_NDVI.values,
                              S2_0420_NDWI.values, S2_0624_NDWI.values,
                              S2_0420_AWEI.values, S2_0624_AWEI.values,
                              S2_0420_NDBI.values, S2_0624_NDBI.values,
                              S2_0420_NBR.values, S2_0624_NBR.values,
                              S2_0420_NDSI_snow.values, S2_0624_NDSI_snow.values,
                              land_cover_data,
                              S1_0418.latitude.values, S1_0418.longitude.values], axis=0)

print(Penedes_datastack.shape)

bands_labels = ['Amplitude_VV_20230418', 'Amplitude_VV_20230629',
                'Amplitude_VH_20230418', 'Amplitude_VH_20230629',
                'VH_VV_rate_20230418', 'VH_VV_rate_20230629',
                'Sigma_Nought_VH_20230418', 'Sigma_Nought_VH_20230629',
                'Sigma_Nought_VV_20230418', 'Sigma_Nought_VV_20230629',
                'RVI_20230418', 'RVI_20230629',
                'RWI_20230418', 'RWI_20230629',
                'MPDI_20230418', 'MPDI_20230629',
                'S2_Red_20230420', 'S2_Green_20230420', 'S2_Blue_20230420', 'S2_Red_20230624', 'S2_Green_20230624', 'S2_Blue_20230624',
                'NDVI_20230420', 'NDVI_20230624',
                'NDWI_20230420', 'NDWI_20230624',
                'AWEI_20230420', 'AWEI_20230624',
                'NDBI_20230420', 'NDBI_20230624',
                'NBR_20230420', 'NBR_20230624',
                'NDSI_20230420', 'NDSI_20230624',
                'Land_Cover',
                'Latitude', 'Longitude']

"""Before exporting the data, in order to mask all the layers directly without the need to mask one by one, we will mask directly the whole stack."""

Penedes_datastack_msk = Penedes_datastack.copy()
Penedes_datastack_msk[:, ~msk_total] = np.nan

plot_data(Penedes_datastack_msk[0,:,:])

filepath = folder + "Penedes_datastack_final.tif"
with rasterio.open(
    filepath,
    "w",
    driver="GTiff",
    height=Penedes_datastack_msk.shape[1],
    width=Penedes_datastack_msk.shape[2],
    count=Penedes_datastack_msk.shape[0],
    dtype=Penedes_datastack_msk.dtype,
    crs="EPSG:4326"
) as dst:
    for i in range(Penedes_datastack_msk.shape[0]):
        dst.write(Penedes_datastack_msk[i], i + 1)

    dst.descriptions = tuple(bands_labels)


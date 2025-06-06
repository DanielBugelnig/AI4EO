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
folder = "../data/" 
S1_0823 = xr.load_dataset(folder + "S1B_20210823.nc") 
S1_0108 = xr.load_dataset(folder + "S1A_20220108.nc")
S2_0826 = xr.load_dataset(folder + "S2B_20210826.nc")
S2_0103 = xr.load_dataset(folder + "S2B_20220103.nc")
file_path ="../data/Land_Cover_Palma1.tif"
with rasterio.open(file_path) as src:
    land_cover_data = src.read(1)
    
change_band = xr.load_dataset("../data/LaPalma_mod.nc")

print(S1_0823.data_vars)
print(change_band.data_vars)
print(change_band['Lava_Extent'])
print(S2_0103['B2'])

    
landcover = rxa.open_rasterio(file_path, masked=True).squeeze()
crs_lc = landcover.rio.crs
# Beispiel mit rioxarray
S1_0823= S1_0823.rio.write_crs(crs_lc, inplace=True)
S1_0108 = S1_0108.rio.write_crs(crs_lc, inplace=True)
S2_0826 = S2_0826.rio.write_crs(crs_lc, inplace=True)
S2_0103 = S2_0103.rio.write_crs(crs_lc, inplace=True)
S1_0823= S1_0823.rio.reproject_match(landcover)
S1_0108 = S1_0108.rio.reproject_match(landcover)
S2_0826 = S2_0826.rio.reproject_match(landcover)
S2_0103 = S2_0103.rio.reproject_match(landcover)
change_band = change_band.rio.write_crs(crs_lc, inplace=True)
change_band = change_band.rio.reproject_match(landcover)

    
# Single plot auxiliar function
def plot_data(data, title="", colorbar=True, filename=None, **kwargs):
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
    plt.savefig(filename, dpi=300)
    plt.close()
    


'''--------------------------------------------------------------'''

# print(S1_0823['Amplitude_VH']) # 2146, 2548
# print(S1_0108['Amplitude_VH'])
# print(S2_0826.B2) # 1583, 2509
# print(S2_0103.B2)
# print(landcover)





'''--------------------------------------------------------------'''
# Sentinel-1 Indices
print(S1_0823.data_vars)
print(S2_0826.data_vars)

S1_0823_VH, S1_0823_VV = EOIndices.radar.extract_VV_VH(S1_0823)
S1_0108_VH, S1_0108_VV = EOIndices.radar.extract_VV_VH(S1_0108)

plot_data(data=S1_0823_VH, title="S1_0823_VH", filename="../data/indices/S1_0823_VH.png")
plot_data(data=S1_0823_VV, title="S1_0823_VV", filename="../data/indices/S1_0823_VV.png")
plot_data(data=S1_0108_VH, title="S1_0108_VH", filename="../data/indices/S1_0108_VH.png")
plot_data(data=S1_0108_VV, title="S1_0108_VV", filename="../data/indices/S1_0108_VV.png")

S1_0823_VH = EOIndices.radar.filtering_band(S1_0823_VH, 5)
S1_0823_VV = EOIndices.radar.filtering_band(S1_0823_VV, 5)
S1_0108_VH = EOIndices.radar.filtering_band(S1_0108_VH, 5)
S1_0108_VV = EOIndices.radar.filtering_band(S1_0108_VV, 5)
# Plot the filtered data
plot_data(data=S1_0823_VH, title="S1_0823_VH_filtered", filename="../data/indices/S1_0823_VH_filtered.png")
plot_data(data=S1_0823_VV, title="S1_0823_VV_filtered", filename="../data/indices/S1_0823_VV_filtered.png")
plot_data(data=S1_0108_VH, title="S1_0108_VH_filtered", filename="../data/indices/S1_0108_VH_filtered.png")
plot_data(data=S1_0108_VV, title="S1_0108_VV_filtered", filename="../data/indices/S1_0108_VV_filtered.png")

# Amplitude
S1_0823_VH_A,_ = EOIndices.radar.compute_A(S1_0823_VH)
S1_0823_VV_A,_ = EOIndices.radar.compute_A(S1_0823_VV)
S1_0108_VH_A,_ = EOIndices.radar.compute_A(S1_0108_VH)
S1_0108_VV_A,_ = EOIndices.radar.compute_A(S1_0108_VV)


# VH / VV Ratio
S1_0823_ratio = EOIndices.radar.compute_VH_VV_ratio(S1_0823_VH,S1_0823_VV)
S1_0108_ratio = EOIndices.radar.compute_VH_VV_ratio(S1_0108_VH,S1_0108_VV)


# Sigma Nought
S1_0823_VH_sigma_nought = EOIndices.radar.compute_sigma_nought_log(S1_0823_VH)
S1_0823_VV_sigma_nought = EOIndices.radar.compute_sigma_nought_log(S1_0823_VV)
S1_0108_VH_sigma_nought = EOIndices.radar.compute_sigma_nought_log(S1_0108_VH)
S1_0108_VV_sigma_nought = EOIndices.radar.compute_sigma_nought_log(S1_0108_VV)

# RVI
S1_0823_RVI = EOIndices.radar.compute_RVI(S1_0823_VH,S1_0823_VV)
S1_0108_RVI = EOIndices.radar.compute_RVI(S1_0108_VH,S1_0108_VV)

# RWI
S1_0823_RWI = EOIndices.radar.compute_RWI(S1_0823_VH,S1_0823_VV)
S1_0108_RWI = EOIndices.radar.compute_RWI(S1_0108_VH,S1_0108_VV)

# MPDI
S1_0823_MPDI = EOIndices.radar.compute_MPDI(S1_0823_VH,S1_0823_VV)
S1_0108_MPDI = EOIndices.radar.compute_MPDI(S1_0108_VH,S1_0108_VV)


"""Now you must plot each indice and make sure they look correct."""

# Plot each S1 indice, 0823
plot_data(data=S1_0823_VH_A, title="S1_0823_VH_A", filename="../data/indices/S1_0823_VH_A.png")
plot_data(data=S1_0823_VV_A, title="S1_0823_VV_A", filename="../data/indices/S1_0823_VV_A.png")
plot_data(data=S1_0823_ratio, title="S1_0823_ratio", filename="../data/indices/S1_0823_ratio.png")
plot_data(data=S1_0823_VH_sigma_nought, title="S1_0823_VH_sigma_nought", filename="../data/indices/S1_0823_VH_sigma_nought.png")
plot_data(data=S1_0823_VV_sigma_nought, title="S1_0823_VV_sigma_nought", filename="../data/indices/S1_0823_VV_sigma_nought.png")
plot_data(data=S1_0823_RVI, title="S1_0823_RVI",    filename="../data/indices/S1_0823_RVI.png")
plot_data(data=S1_0823_RWI, title="S1_0823_RWI",   filename="../data/indices/S1_0823_RWI.png")
plot_data(data=S1_0823_MPDI, title="S1_0823_MPDI", filename="../data/indices/S1_0823_MPDI.png")

# Plot each S1 indice, 0108
plot_data(data=S1_0108_VH_A, title="S1_0108_VH_A",  filename="../data/indices/S1_0108_VH_A.png")
plot_data(data=S1_0108_VV_A, title="S1_0108_VV_A", filename="../data/indices/S1_0108_VV_A.png")
plot_data(data=S1_0108_ratio, title="S1_0108_ratio",    filename="../data/indices/S1_0108_ratio.png")
plot_data(data=S1_0108_VH_sigma_nought, title="S1_0108_VH_sigma_nought",    filename="../data/indices/S1_0108_VH_sigma_nought.png")
plot_data(data=S1_0108_VV_sigma_nought, title="S1_0108_VV_sigma_nought", filename="../data/indices/S1_0108_VV_sigma_nought.png")
plot_data(data=S1_0108_RVI, title="S1_0108_RVI", filename="../data/indices/S1_0108_RVI.png")
plot_data(data=S1_0108_RWI, title="S1_0108_RWI", filename="../data/indices/S1_0108_RWI.png")
plot_data(data=S1_0108_MPDI, title="S1_0108_MPDI", filename="../data/indices/S1_0108_MPDI.png")


# Sentinel 2 Indices
S2_0826_NDVI = EOIndices.optical.NDVI(S2_0826)
S2_0826_NDWI = EOIndices.optical.NDWI(S2_0826)
S2_0826_AWEI = EOIndices.optical.AWEI(S2_0826)
S2_0826_NDBI = EOIndices.optical.NDBI(S2_0826)
S2_0826_NDSI_snow = EOIndices.optical.NDSI_snow_SWIR(S2_0826)
S2_0826_NBR = EOIndices.optical.NBR(S2_0826)
S2_0826_NDSI_snow_green = EOIndices.optical.NDSI_snow_green_SWIR(S2_0826)

S2_0103_NDVI = EOIndices.optical.NDVI(S2_0103)
S2_0103_NDWI = EOIndices.optical.NDWI(S2_0103)
S2_0103_AWEI = EOIndices.optical.AWEI(S2_0103)
S2_0103_NDBI = EOIndices.optical.NDBI(S2_0103)
S2_0103_NDSI_snow = EOIndices.optical.NDSI_snow_SWIR(S2_0103)
S2_0103_NBR = EOIndices.optical.NBR(S2_0103)
S2_0103_NDSI_snow_green = EOIndices.optical.NDSI_snow_green_SWIR(S2_0103)
#print(type(S2_0826_AWEI))

# Plot the indices
plot_data(data= S2_0826_NDVI, title="S2_0826_NDVI", filename="../data/indices/S2_0826_NDVI.png")
plot_data(data= S2_0826_NDWI, title="S2_0826_NDWI", filename="../data/indices/S2_0826_NDWI.png")
plot_data(data= S2_0826_AWEI, title="S2_0826_AWEI", filename="../data/indices/S2_0826_AWEI.png")
plot_data(data= S2_0826_NDBI, title="S2_0826_NDBI", filename="../data/indices/S2_0826_NDBI.png")
plot_data(data= S2_0826_NDSI_snow, title="S2_0826_NDSI_snow", filename="../data/indices/S2_0826_NDSI_snow.png")
plot_data(data= S2_0826_NBR, title="S2_0826_NBR", filename="../data/indices/S2_0826_NBR.png")
plot_data(data= S2_0826_NDSI_snow_green, title="S2_0826_NDSI_snow_green", filename="../data/indices/S2_0826_NDSI_snow_green.png")

# Plot the indices
plot_data(data= S2_0103_NDVI, title="S2_0103_NDVI", filename="../data/indices/S2_0103_NDVI.png")
plot_data(data= S2_0103_NDWI, title="S2_0103_NDWI", filename="../data/indices/S2_0103_NDWI.png")
plot_data(data= S2_0103_AWEI, title="S2_0103_AWEI",     filename="../data/indices/S2_0103_AWEI.png")
plot_data(data= S2_0103_NDBI, title="S2_0103_NDBI", filename="../data/indices/S2_0103_NDBI.png")
plot_data(data= S2_0103_NDSI_snow, title="S2_0103_NDSI_snow", filename="../data/indices/S2_0103_NDSI_snow.png")
plot_data(data= S2_0103_NBR, title="S2_0103_NBR", filename="../data/indices/S2_0103_NBR.png")
plot_data(data= S2_0103_NDSI_snow_green, title="S2_0103_NDSI_snow_green", filename="../data/indices/S2_0103_NDSI_snow_green.png")




"""---

You can already see from the previous plots that the set of valid pixels for each dataset is different.
Generate a mask of the valid pixels for the Sentinel-2 and for the Sentinel-1 images and plot them. Additionally, 
if we want to combine them, we will need a mask containing the valid pixels for both datasets simulatneously.
Tip: You can use [`np.isfinite`](https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html) for this, 
since the invalid pixels are set to `nan`.
"""

msk_S1_20210823 = np.isfinite(S1_0823.Amplitude_VH.data)
msk_S1_20220108 = np.isfinite(S1_0108.Amplitude_VH.data)
msk_S2_20210826 = np.isfinite(S2_0826.B2.data)
msk_S2_20220103 = np.isfinite(S2_0103.B2.data)
# print dimesnions of the masks
print(msk_S1_20210823.shape)
print(msk_S1_20220108.shape)
print(msk_S2_20210826.shape)
print(msk_S2_20220103.shape)

# The total mask will be the logical and of both
msk_total = msk_S1_20210823 & msk_S1_20220108 & msk_S2_20210826 & msk_S2_20220103
print(msk_total.shape)

plt.figure(figsize=(10, 10))

ax = plt.subplot(3, 2, 1)
plt.imshow(msk_S1_20210823)
ax.set_title('Mask for Sentinel-1 20210823 data')

ax = plt.subplot(3, 2, 2)
plt.imshow(msk_S1_20220108)
ax.set_title('Mask for Sentinel-1 20220108 data')

ax = plt.subplot(3, 2, 3)
plt.imshow(msk_S2_20210826)
ax.set_title('Mask for Sentinel-2 20210826 data')

ax = plt.subplot(3, 2, 4)
plt.imshow(msk_S2_20220103)
ax.set_title('Mask for Sentinel-2 20220103 data')

ax = plt.subplot(3, 2, (5, 6))
plt.imshow(msk_total)
ax.set_title('Combined mask for all four datasets')

plt.tight_layout()
plt.show()


"""Optical RGB image generation containing only the valid area of both datasets"""
# Generate RGB s2 data
rgb_20220103 = np.dstack((S2_0103.B4, S2_0103.B3, S2_0103.B2))
rgb_20210826 = np.dstack((S2_0826.B4, S2_0826.B3, S2_0826.B2))

for channel in range(3):
    rgb_20220103[:,:,channel] = rgb_20220103[:,:,channel] / np.nanpercentile(rgb_20220103[:,:,channel], 98)
for channel in range(3):
    rgb_20210826[:,:,channel] = rgb_20210826[:,:,channel] / np.nanpercentile(rgb_20210826[:,:,channel], 98)
rgb_20220103 = np.clip(rgb_20220103, 0, 1)
rgb_20210826 = np.clip(rgb_20210826, 0, 1)

# Save the RGB image as a PNG file
plt.imsave("../data/rgb/rgb_20220103.png", rgb_20220103)
plt.imsave("../data/rgb/rgb_20210826.png", rgb_20210826)

#plot_data(rgb_20220103)
#plot_data(rgb_20210826)

#false rgb
rgb_SAR_data_S1_20220108 = np.dstack((S1_0108.Amplitude_VV, S1_0108.Amplitude_VH, S1_0108.Amplitude_VV/S1_0108.Amplitude_VH))
rgb_SAR_data_S1_20220108 = rgb_SAR_data_S1_20220108 / (2.5*np.nanmean(rgb_SAR_data_S1_20220108, axis=(0,1)))

rgb_SAR_data_S1_20210823 = np.dstack((S1_0823.Amplitude_VV, S1_0823.Amplitude_VH, S1_0823.Amplitude_VV/S1_0823.Amplitude_VH))
rgb_SAR_data_S1_20210823 = rgb_SAR_data_S1_20210823 / (2.5*np.nanmean(rgb_SAR_data_S1_20210823, axis=(0,1)))

# Clip the values between 0 and 1
rgb_SAR_data_S1_20220108 = np.clip(rgb_SAR_data_S1_20220108, 0, 1)
rgb_SAR_data_S1_20210823 = np.clip(rgb_SAR_data_S1_20210823, 0, 1)

# Save the RGB image as a PNG file
plt.imsave("../data/rgb/false_rgb_20220108.png", rgb_SAR_data_S1_20220108)
plt.imsave("../data/rgb/false_rgb_20210823.png", rgb_SAR_data_S1_20210823)


#plot_data(rgb_SAR_data_S1_20220108) 

rgb_s1_20210823 = mpimg.imread("../data/rgb/false_rgb_20210823.png")
rgb_s1_20220108 = mpimg.imread("../data/rgb/false_rgb_20220108.png")
rgb_s2_20210826 = mpimg.imread("../data/rgb/rgb_20210826.png")
rgb_s2_20220103 = mpimg.imread("../data/rgb/rgb_20220103.png")



rgb_20210826_msk = rgb_s2_20210826.copy()
rgb_20210826_msk[~msk_total,:] = np.nan    # all invalid pixel are NAN

#plot_data(rgb_20210826_msk, 'Optical image valid area on both datasets')


# already plotted in data_preparation.py

"""One last check that we must do is to identify if the land cover has the same extension as the S1 and S2 data because we already preprocessed S1 and S2 but not the LC."""

print(S1_0823)
print(S2_0826)
print(land_cover_data.shape) # right now it has the same with S2 but not with S1

# Reproject Land Cover to match S1/S2

# We can use rioxarray library to open the LC data
lc_geotiff = rxa.open_rasterio(file_path)

# First we have to define the CRS in the rasterio format to use its
# functionality
ds = S2_0826    # Using here one S2 dataset but both S1 and S2 should have the same CRS
wkt_string = ds.rio.crs.to_wkt()
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
print("Test")
print(np.unique(land_cover_data))  # z. B. [255] oder [230 231 ...]


# Now they should have the same shape
print(S1_0823)
print(S2_0826)
print(land_cover_data.shape)

# Plot of the aligned LC
#plot_data(landcover_aligned.isel(band=0).values, title="alignedLC", colorbar=True)

"""## 4. Generating and saving the data stack

Finally, we will generate the data stack by merging together all the different parameters and data that we have computed. Also we will generate the labels of each band to add into our final Tiff file.

Notes:
*   To save the RGB image we will directly save the Red, Green and Blue bands instead of the 3d array.
*   We will save the VV and VH amplitudes in dB.

> Add blockquote



Careful! Each index and variable with data is in xarray format; only the values could be stack together.
"""
# check the shape
print("Test")
print(change_band['Lava_Extent'].shape)
print("shape of the S1 data")
print(S1_0823.Amplitude_VV.shape)
Palma_datastack = np.stack([10*np.log10(S1_0823.Amplitude_VV.values), 10*np.log10(S1_0108.Amplitude_VV.values),
                              10*np.log10(S1_0823.Amplitude_VH.values), 10*np.log10(S1_0108.Amplitude_VH.values),
                              S1_0823_ratio.values, S1_0108_ratio.values,
                              S1_0823_VH_sigma_nought.values, S1_0108_VH_sigma_nought.values,
                            #   S1_0823_VV_sigma_nought.values, S1_0108_VV_sigma_nought.values,
                              S1_0823_RVI.values, S1_0108_RVI.values,
                              S1_0823_RWI.values, S1_0108_RWI.values,
                              S1_0823_MPDI.values, S1_0108_MPDI.values,
                              S2_0826.B4, S2_0826.B3, S2_0826.B2,
                              S2_0103.B4, S2_0103.B3, S2_0103.B2,
                              S2_0826_NDVI.values, S2_0103_NDVI.values,
                              S2_0826_NDWI.values, S2_0103_NDWI.values,
                              S2_0826_AWEI.values, S2_0103_AWEI.values,
                              S2_0826_NDBI.values, S2_0103_NDBI.values,
                              S2_0826_NBR.values, S2_0103_NBR.values,
                              S2_0826_NDSI_snow.values, S2_0103_NDSI_snow.values,
                              land_cover_data,
                              change_band['Lava_Extent'].values,
                              S1_0823.latitude.values, S1_0823.longitude.values], axis=0)

print(Palma_datastack.shape)

bands_labels = ['Amplitude_VV_20210823', 'Amplitude_VV_20220108',
                'Amplitude_VH_20210823', 'Amplitude_VH_20220108',
                'VH_VV_rate_20210823', 'VH_VV_rate_20220108',
                'Sigma_Nought_VH_20210823', 'Sigma_Nought_VH_20220108',
                # 'Sigma_Nought_VV_20210823', 'Sigma_Nought_VV_20220108',
                'RVI_20210823', 'RVI_20220108',
                'RWI_20210823', 'RWI_20220108',
                'MPDI_20210823', 'MPDI_20220108',
                'S2_Red_20210826', 'S2_Green_20210826', 'S2_Blue_20210826', 'S2_Red_20220103', 'S2_Green_20220103', 'S2_Blue_20220103',
                'NDVI_20210826', 'NDVI_20220103',
                'NDWI_20210826', 'NDWI_20220103',
                'AWEI_20210826', 'AWEI_20220103',
                'NDBI_20210826', 'NDBI_20220103',
                'NBR_20210826', 'NBR_20220103',
                'NDSI_20210826', 'NDSI_20220103',
                'Land_Cover', 'Change_Band',
                'Latitude', 'Longitude']

"""Before exporting the data, in order to mask all the layers directly without the need to mask one by one, we will mask directly the whole stack."""

Palma_datastack_msk = Palma_datastack.copy()
Palma_datastack_msk[:, ~msk_total] = np.nan

#plot_data(Palma_datastack_msk[0,:,:], title="Plotting the first band of the datastack", colorbar=True)

filepath = folder + "Palma_datastack_change_detection.tif"
print(f"Saved to {filepath}")
with rasterio.open(
    filepath,
    "w",
    driver="GTiff",
    height=Palma_datastack_msk.shape[1],
    width=Palma_datastack_msk.shape[2],
    count=Palma_datastack_msk.shape[0],
    dtype=Palma_datastack_msk.dtype,
    crs="EPSG:4326"
) as dst:
    for i in range(Palma_datastack_msk.shape[0]):
        dst.write(Palma_datastack_msk[i], i + 1)

    dst.descriptions = tuple(bands_labels)


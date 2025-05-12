'''
Author  : Daniel Bugelnig
Email   : daniel.j.bugelnig@gmail.com
Date    : 2025-05-05
Purpose : take a first look at the data and prepare it for the next steps
'''



import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  
import xarray as xr
import rioxarray as rxa

LOOKAT_S1 = True
LOOKAT_S2 = True
LOOKAT_TIF = True

# Single plot auxiliar function
def plot_data(data, title="", colorbar=False, **kwargs):
    """
    Plot some specific data
    """
    plt.figure(figsize=(12,6))
    img = plt.imshow(data, **kwargs)
    plt.title(title)
    if colorbar:
        plt.colorbar(img)
    plt.tight_layout()
    plt.show()
    

'''
Data
Three data sources for a common area will be combined. Two Sentinel-1 satellites acquisitions, two Sentinel-2 satellites acquisitions and one Land Cover map.
The three datasets have been previously coregistered to be on the same spatial grid. They are provided in NetCDF format.
The data is available under data/treated/nc
'''


'''
1. Load and display the Sentinel-2 image
The first step is to load and visualize the Sentinel-2 data, in order to better understand the file structure. Sentinel-2 carries 
a single Multi-spectral Instrument (MSI), which provides 13 bands in the visible, near-infrared and shortwave-infrared spectrum 
with different ground resolutions:

#	Name	Spatial Resolution
B1	Coastal aerosol	60
B2	Blue	10
B3	Green	10
B4	Red	10
B5	Red Edge 1	20
B6	Red Edge 2	20
B7	Red Edge 3	20
B8	Near-Infrared	10
B8a	Near-Infrared narrow	20
B9	Water vapor	60
B10	Shortwave-Infrared cirrus	60
B11	Shortwave Infrared 1	20
B12	Shortwave-Infrared 2	20
In the provided file, however, all the bands have been upsampled to the same resolution (10m), for simplicity.

To load the data we can use the load_dataset function from xarray.'''
if LOOKAT_S2:
    print("Loading Sentinel-2 data")
    # Load Sentinel-2 datase
    #S2 Data In NetCDF
    folder = "../data/" 
    data_S2_20220103 = xr.load_dataset(folder + "subset_2_of_S2B_MSIL2A_20220103T120319_N0301_R023_T28RBS_20220103T143548_resampled_reprojected.nc")
    data_S2_20210826 = xr.load_dataset(folder + "subset_3_of_S2B_MSIL2A_20210826T120319_N0500_R023_T28RBS_20230112T122418_resampled_reprojected.nc")

    # Inspecting the dataset
    print(data_S2_20210826)

    plt.figure()
    data_S2_20210826.B8.plot()
    plt.show()

    '''
    Since Sentinel-2 has the Red, Green and Blue channels we can generate an RGB image by forming an image. 
    For this we need to generate an array with a shape (rows, cols, 3), where the last 3 channels correspond 
    to the RGB. We can use np.dstack for this.
    '''

    # Generate RGB, put the corresponding bands here
    rgb_20220103 = np.dstack((data_S2_20220103.B4, data_S2_20220103.B3, data_S2_20220103.B2))
    rgb_20210826 = np.dstack((data_S2_20210826.B4, data_S2_20210826.B3, data_S2_20210826.B2))
    # Clip the values between 0 and 1 --> could be outliers, data should be in that area
    rgb_20220103 = np.clip(rgb_20220103, 0, 1)
    
    

    '''
    In order to reduce the dynamic range shown, instead of clipping between 0 and 1 from the original input data, 
    we will first normalize it between 0 and the 98% percentile (to exclude the 2% with higher values).
    Tip: you can use np.nanpercentile to avoid problems with invalid values, marked as nan.'''

    for channel in range(3):
        rgb_20220103[:,:,channel] = rgb_20220103[:,:,channel] / np.nanpercentile(rgb_20220103[:,:,channel], 98)
    for channel in range(3):
        rgb_20210826[:,:,channel] = rgb_20210826[:,:,channel] / np.nanpercentile(rgb_20210826[:,:,channel], 98)
    rgb_20220103 = np.clip(rgb_20220103, 0, 1)
    rgb_20210826 = np.clip(rgb_20210826, 0, 1)
    
    # Save the RGB image as a PNG file
    plt.imsave("../data/rgb/rgb_20220103.png", rgb_20220103)
    plt.imsave("../data/rgb/rgb_20210826.png", rgb_20210826)

    plot_data(rgb_20220103)
    plot_data(rgb_20210826)
    


'''-------------------------------------------------------------------------------------------------------'''

'''
Loading S1 image
The Sentinel-1 image is a dual-pol SAR data, containing the polarizations VV and VH. 
It corresponds to a GRD product projected to the same grid as the Sentinel-2 data previously seen. 
The included bands contain the amplitude in VV and VH polarizations into the Amplitude_VV and Amplitude_VH bands.

Tip: To visualize this data is easier to show their values in dB (10*np.log10(Amplitude)) due to the large dynamic range of SAR data.
'''
if LOOKAT_S1:
    print("Loading Sentinel-1 data")

    folder = "../data/" 
    data_S1_20220108 = xr.load_dataset(folder + "S1A_20220108.nc")
    data_S1_20210823 = xr.load_dataset(folder + "S1B_20210823.nc")

    
    # Plot amplitudes in VV and VH in dB
    plot_data(10*np.log10(data_S1_20220108.Amplitude_VV), 'Amplitude VV (dB)', True)
    plot_data(10*np.log10(data_S1_20220108.Amplitude_VH), 'Amplitude VH (dB)', True)
    
    #We can also extract the latitude and longitude layer as we must store it on the stack later.
    # TODO - Extract the latitude and longitude data from one of the S1 acquisitions
    print(data_S1_20220108)
    latitude_data = data_S1_20220108['latitude'].values
    longitude_data = data_S1_20220108['longitude'].values
    
    print(data_S1_20210823)
    latitude_data = data_S1_20210823['latitude'].values
    longitude_data = data_S1_20210823['longitude'].values


    # does not work, only one dimensional?
    # plot_data(latitude_data, 'Latitude', True)
    # plot_data(longitude_data, 'Longitude', True)
    
    '''
    We can also generate a false-color RGB image from these SAR images by assigning the amplitude channels (VV, VH, VV / VH) 
    to the RGB channels, respectively. For this we need to generate an array with a shape (rows, cols, 3), where the last 3 channels 
    correspond to the RGB. We can use np.dstack for this.
    In this case, the RGB image will be normalized between 0 and 2.5 times the mean amplitude for each channel.
    '''
    
    rgb_SAR_data_S1_20220108 = np.dstack((data_S1_20220108.Amplitude_VV, data_S1_20220108.Amplitude_VH, data_S1_20220108.Amplitude_VV/data_S1_20220108.Amplitude_VH))
    rgb_SAR_data_S1_20220108 = rgb_SAR_data_S1_20220108 / (2.5*np.nanmean(rgb_SAR_data_S1_20220108, axis=(0,1)))
    
    rgb_SAR_data_S1_20210823 = np.dstack((data_S1_20210823.Amplitude_VV, data_S1_20210823.Amplitude_VH, data_S1_20210823.Amplitude_VV/data_S1_20210823.Amplitude_VH))
    rgb_SAR_data_S1_20210823 = rgb_SAR_data_S1_20210823 / (2.5*np.nanmean(rgb_SAR_data_S1_20210823, axis=(0,1)))

    # Clip the values between 0 and 1
    rgb_SAR_data_S1_20220108 = np.clip(rgb_SAR_data_S1_20220108, 0, 1)
    rgb_SAR_data_S1_20210823 = np.clip(rgb_SAR_data_S1_20210823, 0, 1)

    # Save the RGB image as a PNG file
    plt.imsave("../data/rgb/false_rgb_20220108.png", rgb_SAR_data_S1_20220108)
    plt.imsave("../data/rgb/false_rgb_20210823.png", rgb_SAR_data_S1_20210823)


    plot_data(rgb_SAR_data_S1_20220108) 
    
    '''------------------------------------------------------------------------------------------'''
    '''
    3. Loading the Land Cover data.
    The last data to be loaded is the land cover. In this case we are working with a Tiff file. 
    Therefore we will use rasterio to charge the data.
    '''
if LOOKAT_TIF:
    file_path ="../data/Land_Cover_Palma1.tif"
    with rasterio.open(file_path) as src:
        land_cover_data = src.read(1)
        #plot_data(land_cover_data)
        print(land_cover_data.shape)
        plt.imsave("../data/rgb/cover_rgb.png", land_cover_data)


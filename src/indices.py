'''
Author  : Daniel Bugelnig
Email   : daniel.j.bugelnig@gmail.com
Date    : 2025-03-28
Purpose : Class for computing various indices from radar and optical data.
'''

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  
import xarray as xr
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
import rioxarray as rxa
from scipy.ndimage import uniform_filter
import math


class RadarIndices():
    
    @staticmethod
    def _checking_input(data):
        """
        Validates that the input is an xarray.DataArray.

        Parameters:
        data: Input to check.

        Raises:
        Exception: If input is not of type DataArray.
        """
        if not (isinstance(data, DataArray)):
            raise Exception(f"Wrong datatype. Expected {DataArray}, not {type(data)}")
        
    
    @staticmethod
    def filtering_band(band: DataArray, size: int = 5) -> DataArray:
        """
        Applies a Lee filter to reduce speckle noise in a SAR band (e.g., VV or VH).

        Parameters:
        band (DataArray): The input radar band in linear scale.
        size (int): Size of the moving window used for filtering.

        Returns:
        DataArray: The filtered band as a new DataArray.
        """
        RadarIndices._checking_input(band)
        
        # convert to numpy array
        band_np = band.values.astype(np.float32)
        band_np = np.nan_to_num(band_np, nan=np.nanmedian(band_np)) 
        overall_variance = np.var(band_np)

        # Lee-Filter 
        mean = uniform_filter(band_np, size=size)
        mean_sq = uniform_filter(band_np ** 2, size=size)
        variance = mean_sq - mean**2
        overall_variance = np.var(band_np)
        
        coef = variance / (variance + overall_variance + 1e-8) 
        lee_filtered = mean + coef * (band_np - mean)
        
        return DataArray(lee_filtered, coords=band.coords, dims=band.dims, name=f"{band.name}_filtered")
        
    
   
        
    @staticmethod
    def extract_VV_VH(data):
        """
        Extracts the VV and VH amplitude bands from an xarray.Dataset.

        Parameters:
        data (Dataset): Dataset containing 'Amplitude_VV' and 'Amplitude_VH'.

        Returns:
        tuple: VH and VV bands as DataArrays.
        """
        if not (isinstance(data, Dataset)):
            raise Exception(f"Wrong datatype. Expected {Dataset}, not {type(data)}")
        VH = data['Amplitude_VH']
        VV = data['Amplitude_VV']   
        return VH,VV 
    
    @staticmethod
    def compute_A(data):
        """
        Computes the amplitude A = sqrt(DN) and returns both A and DN.

        Parameters:
        data (DataArray): Input radar band data.

        Returns:
        tuple: Amplitude and original DN as numpy arrays.
        """
        RadarIndices._checking_input(data)
        # compute Amplitude for each pixel
        DN = np.array(data)
        return np.sqrt(data),DN
    
    @staticmethod
    def compute_sigma_nought_lin(data):
        """
        Computes sigma nought (σ⁰) in linear scale from amplitude data.

        Parameters:
        data (DataArray): Input radar band data.

        Returns:
        DataArray: σ⁰ in linear scale.
        """
        RadarIndices._checking_input(data)
        A, DN = RadarIndices.compute_A(data)
        return np.pow(DN,2) / A
    
    @staticmethod
    def compute_sigma_nought_log(data):
        """
        Computes sigma nought (σ⁰) in dB scale from amplitude data.

        Parameters:
        data (DataArray): Input radar band data.

        Returns:
        DataArray: σ⁰ in logarithmic (dB) scale.
        """
        RadarIndices._checking_input(data)
        A, DN = RadarIndices.compute_A(data)
        return 10 * np.log10(np.pow(DN,2)) - A

    @staticmethod
    def compute_VH_VV_ratio(VH, VV):
        """
        Computes the ratio of σ⁰(VH) to σ⁰(VV) in linear scale.

        Parameters:
        VH (DataArray): VH band.
        VV (DataArray): VV band.

        Returns:
        DataArray: The σ⁰(VH) / σ⁰(VV) ratio.
        """
        RadarIndices._checking_input(VH)
        RadarIndices._checking_input(VV)
        return RadarIndices.compute_sigma_nought_lin(VH) / RadarIndices.compute_sigma_nought_lin(VV)

    @staticmethod
    def compute_RVI(VH,VV):
        '''
        Radar Vegetation Index (RVI)
        RVI measures how much vegetation is present based on the radar signal.
        It uses both VV and VH channels to calculate how evenly the radar energy is scattered.
        Higher RVI values usually mean more vegetation, since plants tend to scatter radar signals more.
        Lower RVI values suggest less vegetation or bare ground.
        '''
        RadarIndices._checking_input(VH)
        RadarIndices._checking_input(VV)
        sigVH = RadarIndices.compute_sigma_nought_lin(VH)
        sigVV = RadarIndices.compute_sigma_nought_lin(VV)
        return 4 * sigVH / (sigVV + sigVH)

    @staticmethod
    def compute_RWI(VH,VV):
        '''
        Radar Water Index
        It looks at how much radar signal is reflected in the VH (cross-polarized) channel, compared to the total reflection 
        from both VH and VV (vertical) channels. 
        If RWI is high, the surface is likely water or very wet.
        If RWI is low, it's likely dry ground, vegetation, or built-up area.
        '''
        RadarIndices._checking_input(VH)
        RadarIndices._checking_input(VV)
        sigVH = RadarIndices.compute_sigma_nought_lin(VH)
        sigVV = RadarIndices.compute_sigma_nought_lin(VV)
        return sigVH / (sigVH + sigVV)

    @staticmethod
    def compute_MPDI(VH,VV):
        '''
        Modified Polarization Difference Index (MPDI)
        It compares how different the radar reflection is between the VV and VH channels. 
        If MPDI is high, VV reflection is much stronger than VH ; this usually means dry soil or urban areas.
        If MPDI is close to 0, both polarizations are similar ;this suggests vegetation.
        A negative MPDI can indicate wet vegetation or unusual surface conditions.
        '''
        RadarIndices._checking_input(VH)
        RadarIndices._checking_input(VV)
        sigVH = RadarIndices.compute_sigma_nought_lin(VH)
        sigVV = RadarIndices.compute_sigma_nought_lin(VV)
        return (sigVV - sigVH) / (sigVV + sigVH)

class OpticalIndices():
    '''
    Sentinel-2 Indices: Value Ranges and Meaning

NDVI (Normalized Difference Vegetation Index)
    Value range: -1.0 to +1.0
    Description: Measures vegetation density and health.
                > 0.2 → Vegetation present
                > 0.6 → Healthy, dense vegetation
                < 0.0 → Water, snow, or clouds

NDWI (Normalized Difference Water Index)
    Value range: -1.0 to +1.0
    Description: Detects water surfaces using green and NIR reflection.
                > 0.3 → Likely water
                < 0.0 → Dry areas or vegetation

AWEI (Automated Water Extraction Index)
    Value range: approx. -1 to +3
    Description: Robust water index for urban or shadowed areas.
                > 0.0 → Water
                < 0.0 → No water

NDBI (Normalized Difference Built-up Index)
    Value range: -1.0 to +1.0
    Description: Detects urban or built-up areas.
                > 0.2 → Likely urban area or concrete
                < 0.0 → Vegetation or water

NDSI_snow_SWIR (Normalized Difference Snow Index: SWIR variant)
    Value range: -1.0 to +1.0
    Description: Detects snow using the difference between B11 and B12.
                > 0.4 → Likely snow
                < 0.0 → No snow

NDSI_snow_green_SWIR (NDSI: variant using green and SWIR bands)
    Value range: -1.0 to +1.0
    Description: Alternative method for snow detection using B3 and B11.
                > 0.3 to 0.5 → Snow likely

NBR (Normalized Burn Ratio)
    Value range: -1.0 to +1.0
    Description: Detects burned areas (e.g., after forest fires).
                > 0.5 → Healthy vegetation
                < 0.1 → Burned or severely damaged areas
    '''
    
    
    @staticmethod
    def _check_input(data):
        if not (isinstance(data, Dataset)):
            raise Exception(f"Wrong datatype. Expected {Dataset}, not {type(data)}")
    
    @staticmethod
    def NDVI(data):
        '''
        Normalized Difference Vegetation Index (NDVI)
        NDVI measures the amount of vegetation by comparing near-infrared (B8) and red light (B4).
        Healthy vegetation reflects more NIR and absorbs more red light.
        High NDVI → lots of vegetation. Low or negative NDVI → bare soil or water.
        '''
        b8 = data['B8']
        b4 = data['B4']
        return (b8 - b4) / (b8 + b4)

    @staticmethod
    def NDWI(data):
        '''
        Normalized Difference Water Index (NDWI)
        NDWI detects water by comparing green (B3) and near-infrared (B8) reflectance.
        Water reflects green and absorbs NIR, so high NDWI values indicate water.
        '''
        b3 = data['B3']
        b8 = data['B8']
        return (b3 - b8) / (b3 + b8 + 1e-6)
    
    @staticmethod
    def AWEI(data):
        '''
        Automated Water Extraction Index (AWEI)
        AWEI is a more advanced water index that helps detect water even in shadowed or urban areas.
        It uses multiple bands to reduce false positives and highlight water bodies more clearly.
        '''
        b3 = data['B3']
        b8 = data['B8']
        b11 = data['B11']
        b12 = data['B12']
        return 4 * (b3 - b11) - (0.25 * b8 + 2.75 * b12)
    
    @staticmethod
    def NDBI(data):
        '''
        Normalized Difference Built-up Index (NDBI)
        NDBI identifies urban or built-up areas by comparing SWIR (B11) and NIR (B8).
        Urban surfaces reflect more SWIR than vegetation, so high NDBI suggests buildings.
        '''
        b11 = data['B11']
        b8 = data['B8']
        return (b11 - b8) / (b11 + b8 + 1e-6)
    
    @staticmethod
    def NDSI_snow_SWIR(data):
        '''
        Normalized Difference Snow Index (NDSI) ; SWIR version
        This version of NDSI helps detect snow using shortwave infrared bands (B11 and B12).
        Snow reflects B11 strongly and absorbs B12, so high values may indicate snow cover.
        '''
        b11 = data['B11']
        b12 = data['B12']
        return (b11 - b12) / (b11 + b12 + 1e-6)

    @staticmethod
    def NDSI_snow_green_SWIR(data):
        '''
        Normalized Difference Snow Index (NDSI) ; green & SWIR version
        An alternative version of NDSI using green (B3) and SWIR (B11) bands.
        Also useful for snow detection in visible-SWIR contrast.
        '''
        b3 = data['B3']
        b11 = data['B11']
        return (b3 - b11) / (b3 + b11 + 1e-6)

    @staticmethod
    def NBR(data):
        '''
        Normalized Burn Ratio (NBR)
        NBR helps identify burned areas and wildfire damage by comparing NIR (B8) and SWIR2 (B12).
        Lower values usually indicate burned or damaged vegetation.
        '''
        b8 = data['B8']
        b12 = data['B12']
        return (b8 - b12) / (b8 + b12 + 1e-6)

class EOIndices():
    radar = RadarIndices()
    optical = OpticalIndices()
    
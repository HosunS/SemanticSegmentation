################## used to check values within .tif files
import rasterio
import numpy as np

def print_tiff_values(file_path):
    with rasterio.open(file_path) as dataset:
        # read the dataset's valid data mask as a ndarray.
        mask = dataset.dataset_mask()
        
        # read the dataset values as an ndarray.
        data = dataset.read(1)  # Read the first band
        
        # print metadata
        print("Metadata:", dataset.meta)
        
        # print mask
        print("Mask:", mask)
        
        # print data
        print("Data:")
        print(data)

file_path = 'C:/Users/nkson/OneDrive/Desktop/cs175/final-project-california-roll/data/raw/Train/Tile4/settlement_gt.tif'
print_tiff_values(file_path)


# ######################## used to check values of .nc files after preprocessing and subtiling
# import xarray as xr

# def print_nc_values(file_path):
#     # open the NetCDF file
#     dataset = xr.open_dataset(file_path)
    
#     # print dataset metadata
#     print("Metadata:")
#     print(dataset)
    
#     # print data variables
#     print("\nData variables:")
#     for var in dataset.data_vars:
#         print(f"Variable '{var}':")
#         print(np.max(dataset[var].values))
#         print(np.min(dataset[var].values))
#         print(dataset[var].values)


# file_path = 'C:/Users/nkson/OneDrive/Desktop/cs175/final-project-california-roll/data/processed/Train/subtiles/Tile4/0_0/gt.nc'
# print_nc_values(file_path)
"""
This module contains functions for loading satellite data from a directory of
tiles.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import tifffile
import xarray as xr

# local modules
sys.path.append(".")
from src.utilities import SatelliteType


def process_viirs_filename(file_path: Path) -> Tuple[str, str]:
    """
    This function takes in the file_path of a VIIRS file and outputs
    a tuple containing two strings, in the format (date, band). It parses
    the date using the datetime.strptime function with the corresponding
    format: https://docs.python.org/3/library/datetime.html#format-codes
    and then converts it to a string.

    Example input: C:/users/foo/data/DNB_VNP46A1_A2020221.tif
    Example output: ("2020-08-08", "0")

    The format of the date within the filename is {year}{day}, as
    the day here is special, it is the day of the year as a zero-padded
    decimal number like: [001, 002, …, 365] (the format code for this is %j).
    For the example input above, 2020221 is the 221st day of the year 2020 (don't ask
    why they format it like this I got no idea), which translates to August 8th, 2020

    Parameters
    ----------
    file_path : Path
        The Path of the VIIRS file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    date_str = file_path.stem.split("_")[2][1:]
    date = datetime.strptime(date_str, "%Y%j").strftime("%Y-%m-%d")
    band = "0"
    return date, band



def process_s1_filename(file_path: Path) -> Tuple[str, str]:
    """
    This function takes in the file_path of a Sentinel-1 file and outputs
    a tuple containing two strings, in the format (date, band). It parses
    the date using the datetime.strptime function with the corresponding
    format: https://docs.python.org/3/library/datetime.html#format-codes
    and then converts it to a string.

    Example input: C:/users/foo/data/S1A_IW_GRDH_20200804_VV.tif
    Example output: ("2020-8-4", "VV")

    The format of the date within the filename is {year}{month}{day}.

    Parameters
    ----------
    file_path : Path
        The Path of the Sentinel-1 file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    parts = file_path.stem.split('_')
    date_str = parts[3]
    band = parts[-1].split('.')[0]
    date = datetime.strptime(date_str, "%Y%m%d").date()
    return (str(date), band)



def process_s2_filename(file_path: Path) -> Tuple[str, str]:
    """
    This function takes in the file_path of a Sentinel-2 file and outputs
    a tuple containing two strings, in the format (date, band). It parses
    the date using the datetime.strptime function with the corresponding
    format: https://docs.python.org/3/library/datetime.html#format-codes
    and then converts it to a string.

    Example input: C:/users/foo/data/L2A_20200816_B01.tif
    Example output: ("2020-8-16", "01")

    The format of the date within the filename is {year}{month}{day}.

    Parameters
    ----------
    file_path : Path
        The Path of the Sentinel-2 file.

    Returns
    -------
    Tuple[str, str]
    """
    date_str = file_path.stem.split("_")[1]
    date = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
    band = file_path.stem.split("_")[-1][1:]
    return date, band


def process_landsat_filename(file_path: Path) -> Tuple[str, str]:
    """
    This function takes in the file_path of a Landsat file and outputs
    a tuple containing two strings, in the format (date, band). It parses
    the date using the datetime.strptime function with the corresponding
    format: https://docs.python.org/3/library/datetime.html#format-codes
    and then converts it to a string.

    Example input: C:/users/foo/data/LC08_L1TP_2020-08-30_B9.tif
    Example output: ("2020-8-30", "9")

    The format of the date within the filename is {year}-{month}-{day}.

    Parameters
    ----------
    file_path : Path
        The Path of the Landsat file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    parts = file_path.stem.split('_')
    date_str = parts[2]
    band = parts[-1][1:]
    date = datetime.strptime(date_str, "%Y-%m-%d").date()
    return (str(date), band)


def process_ground_truth_filename(file_path: Path) -> Tuple[str, str]:
    """
    This function takes in the file_path of the ground truth file and returns
    ("0", "0"), as there is only one ground truth file. The date is set to the
    datetime.min.date() and converted to a string.

    Example input: C:/users/foo/data/groundTruth.tif
    Example output: ("0001-01-01, "0")

    The date for the ground truth will just be the datetime.min.date().

    Parameters
    ----------
    file_path: Path
        The Path of the ground truth file, and though we will ignore it,
        we still need to pass it.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    return (str(datetime.min.date()), "0")


def get_filename_pattern(satellite_type: SatelliteType) -> str:
    """
    Return the file_path pattern for the given satellite type
    using the SatelliteType.value as a key for the patterns dict.

    Parameters
    ----------
    satellite_type : SatelliteType
        The type of satellite being used. SatelliteType is an Enum

    Returns
    -------
    str
        The file_path pattern for the given satellite type.
    """
    patterns = {
        SatelliteType.VIIRS: "DNB_VNP46A1_",
        SatelliteType.S1: "S1A_IW_GRDH_",
        SatelliteType.S2: "L2A_",
        SatelliteType.LANDSAT: "LC08_L1TP_",
        SatelliteType.GT: "groundTruth.tif",
    }
    return patterns[satellite_type]



def get_satellite_files(tile_dir: Path, satellite_type: SatelliteType) -> List[Path]:
    """
    Retrieve all satellite files in the tile directory matching the satellite type pattern using
    the get_filename_pattern function.

    Parameters
    ----------
    tile_dir : Path
        The directory containing the satellite tiles.
    satellite_type : SatelliteType
        The type of satellite, should be one of SatelliteType.VIIRS,
        SatelliteType.S1, SatelliteType.S2, SatelliteType.LANDSAT, or SatelliteType.GT.

    Returns
    -------
    List[Path]
        A list of Path objects for each satellite file that was in the tile directory.
    """
    pattern = get_filename_pattern(satellite_type)
    return [f for f in tile_dir.iterdir() if f.is_file() and pattern in f.name and f.suffix == ".tif"]


def get_grouping_function(satellite_type: SatelliteType) -> Callable:
    """
    Return the function to group satellite files by date and band using the
    SatelliteType.value as a key for the patterns dict.

    Parameters
    ----------
    satellite_type : SatelliteType
        The type of satellite, should be one of SatelliteType.VIIRS,
        SatelliteType.S1, SatelliteType.S2, SatelliteType.LANDSAT, or SatelliteType.GT.

    Returns
    -------
    Callable (function)
        The function to group satellite files by date and band.
    """
    functions = {
        SatelliteType.VIIRS: process_viirs_filename,
        SatelliteType.S1: process_s1_filename,
        SatelliteType.S2: process_s2_filename,
        SatelliteType.LANDSAT: process_landsat_filename,
        SatelliteType.GT: process_ground_truth_filename,
    }
    return functions[satellite_type]

def get_unique_dates_and_bands(
    tile_dir: Path, satellite_type: SatelliteType
) -> Tuple[List[str], List[str]]:
    """
    Extracts unique dates and bands from tile directory.
    Generates a set of dates and a set of bands using the grouping function to extract
    the date and band from each file in the satellite files. Those 2 sets are then sorted
    and returned.

    Parameters
    ----------
    tile_dir : Path
        The Tile directory containing the satellite tif files.
    satellite_type : SatelliteType
        The type of satellite, should be one of SatelliteType.VIIRS,
        SatelliteType.S1, SatelliteType.S2, SatelliteType.LANDSAT, or SatelliteType.GT.

    Returns
    -------
    Tuple[List[str], List[str]]
        A tuple containing two lists of the unique dates and bands.
        Each list is a sorted set of the dates and bands.
        The dates list is index 0; The bands list is index 1.
    """
    satellite_files, grouping_function = get_satellite_files(
        tile_dir, satellite_type
    ), get_grouping_function(satellite_type)
    satellite_files = get_satellite_files(tile_dir, satellite_type)
    grouping_function = get_grouping_function(satellite_type)
    dates = set()
    bands = set()
    for file in satellite_files:
        date, band = grouping_function(file)
        dates.add(date)
        bands.add(band)
    return (sorted(list(dates)), sorted(list(bands)))


def get_parent_tile_id(tile_dir: Path) -> str:
    """
    Returns the name (parent_tile_id) of the tile_dir as a string.

    Parameters
    ----------
    tile_dir : Path
        The Tile directory containing the satellite tif files.

    Returns
    -------
    str
        The parent_tile_id of the path
    """
    return tile_dir.name


def read_satellite_file(satellite_file: Path) -> np.ndarray:
    """
    Reads the satellite file into a np.ndarray with dtype=np.float32 using tifffile.imread.

    Normally, you would use a library like rasterio: https://rasterio.readthedocs.io/en/stable/
    to read the files, but because the dataset does NOT contain any metadata (IEEE has chosen to remove it),
    we can just use tifffile to easily read the data.

    Parameters
    ----------
    satellite_file : Path
        A Path object to the satellite file.

    Returns
    -------
    2 dimensional np.ndarray of shape (height, width)

    """
    return tifffile.imread(satellite_file).astype(np.float32)

def load_satellite(tile_dir: Path, satellite_type: SatelliteType) -> xr.DataArray:
    """
    Load all bands for a given satellite type from a directory of tile files. Loads the data by
    stacking it into a single 4D np.ndarray of dimensions (band, date, height, width). This is then
    stored in an xarray: https://docs.xarray.dev/en/stable/
    along with the relevant metadata as attributes of the xarray.

    Parameters
    ----------
    tile_dir : Path
        The Tile directory containing the satellite tif files.
    satellite_type : SatelliteType
        The type of satellite, should be one of SatelliteType.VIIRS,
        SatelliteType.S1, SatelliteType.S2, SatelliteType.LANDSAT, or SatelliteType.GT.

    Returns
    -------
    xr.DataArray
        An xr.DataArray containing the satellite data with dimensions (date, band, height, width)
        and the satellite_type, tile_dir, and parent_tile_id attributes.
    """
    satellite_files = get_satellite_files(tile_dir, satellite_type)
    grouping_function = get_grouping_function(satellite_type)
    parent_tile_id = get_parent_tile_id(tile_dir)
    dates, bands = get_unique_dates_and_bands(tile_dir, satellite_type)
    
    data_list = []
    for date in dates:
        band_data = []
        for band in bands:
            for file in satellite_files:
                file_date, file_band = grouping_function(file)
                if file_date == date and file_band == band:
                    data = read_satellite_file(file)
                    band_data.append(data)
                    break
        if band_data:
            data_list.append(np.stack(band_data))
    
    all_data = np.stack(data_list)
    data_array = xr.DataArray(
        all_data, dims=("date", "band", "height", "width"),
        coords={"date": dates, "band": bands, "height": range(all_data.shape[2]), "width": range(all_data.shape[3])}
    )
    data_array.attrs["satellite_type"] = satellite_type.value
    data_array.attrs["tile_dir"] = str(tile_dir)
    data_array.attrs["parent_tile_id"] = parent_tile_id
    return data_array

def load_satellite_list(
    tile_dir: Path, satellite_type_list: List[SatelliteType]
) -> list[xr.DataArray]:
    """
    Loads all the satellites from the tile directory based on the satellite type list.

    Parameters
    ----------
    tile_dir : Path
        The Tile directory containing the satellite tif files.
    satellite_type_list : List[SatelliteType]
        List of the type of satellite, each should be one of SatelliteType.VIIRS,
        SatelliteType.S1, SatelliteType.S2, SatelliteType.LANDSAT, or SatelliteType.GT.

    Returns
    -------
    List[xr.DataArray]
        List of data arrays for each SatelliteType in the satellite type list.
    """
    return [load_satellite(tile_dir, satellite_type) for satellite_type in satellite_type_list]


# given
def load_satellite_dir(
    data_dir: Path, satellite_type_list: List[SatelliteType]
) -> List[List[xr.DataArray]]:
    """
    Load all bands for a given satellite type from a directory of multiple
    tile files.

    Parameters
    ----------
    data_dir : Path
        The directory containing all of the satellite tiles.
    satellite_type_list : List[SatelliteType]
        A list of the type of satellite, should be one of SatelliteType.VIIRS,
        SatelliteType.S1, SatelliteType.S2, SatelliteType.LANDSAT, or SatelliteType.GT.
    Returns
    -------
    List[List[xr.DataArray]]
        A list of tiles, each element containing a satellite_list (list of each satellites data_array
        for that tile).
    """
    return [
        load_satellite_list(tile_dir, satellite_type_list)
        for tile_dir in sorted(data_dir.iterdir())
        if tile_dir.is_dir()
    ]


def create_satellite_dataset_list(
    list_of_data_array_list: List[List[xr.DataArray]],
    satellite_type_list: List[SatelliteType],
    list_of_preprocess_func_list: List[List[Callable]] = None,
):
    """
    Creates the satellite_dataset_list of all the data (gotten from load_satellite_dir). This function takes
    the data arrays from each tile and combines them into a single dataset by satellite.
    Each satellite gets its own xr.Dataset: https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html

    Parameters
    ----------
    list_of_data_array_list : List[List[xr.DataArray]]
        A list of tiles, each element containing a satellite_list (list of each satellites data_array
        for that tile)
    satellite_type_list : List[SatelliteType]
        A list of the type of satellite, should be one of SatelliteType.VIIRS,
        SatelliteType.S1, SatelliteType.S2, SatelliteType.LANDSAT, or SatelliteType.GT.
    list_of_preprocess_func_list :  List[List[Callable]]
        A list of lists. Inside every list, each element is a function to apply to the data_array
        as a preprocessing function.

    Returns
    -------
    List[xr.Dataset]
        List of xr.Dataset, where each element is the data for a single satellite over all the tiles.
    """
    data_dict_list = [dict() for _ in satellite_type_list]
    for satellite_list in list_of_data_array_list:
        for index, data_array in enumerate(satellite_list):
            if list_of_preprocess_func_list != None:
                if list_of_preprocess_func_list[index] != None:
                    for func in list_of_preprocess_func_list[index]:
                        data_array = func(data_array)
            (data_dict_list[index])[data_array.attrs["parent_tile_id"]] = data_array

    data_set_list = []
    for index, data_dict in enumerate(data_dict_list):
        dataset = xr.Dataset(data_dict)
        dataset.attrs["satellite_type"] = satellite_type_list[index].value
        data_set_list.append(dataset)
        
    return data_set_list
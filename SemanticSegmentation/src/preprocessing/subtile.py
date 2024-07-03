import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import xarray as xr

# local modules
sys.path.append(".")
from src.utilities import SatelliteType


class Subtile:
    def __init__(
        self,
        satellite_list: List[xr.DataArray],
        ground_truth: xr.DataArray | None = None,
        slice_size: tuple = (4, 4),
        parent_tile_id: int = None
    ):
        """
        This class handles saving and loading of subtiles for a parent image (Tile#).
        
        Parameters:
            slice_size: the slice size of the image ( (4, 4) = 4x4 subtiles )
        """
        self.satellite_list = satellite_list
        self.ground_truth = ground_truth
        self.slice_size = slice_size
        self.parent_tile_id = satellite_list[0].attrs["parent_tile_id"] if parent_tile_id is None else parent_tile_id
        # print(f"Initialized Subtile with parent_tile_id: {self.parent_tile_id}")

    def __calculate_slice_index(self, x: int, y: int, slice_size: tuple, length: tuple):
        start_index = (
            int(np.divide(length[0], slice_size[0]) * x),
            int(np.divide(length[1], slice_size[1]) * y),
        )
        end_index = (
            int(np.divide(length[0], slice_size[0]) * (x + 1)),
            int(np.divide(length[1], slice_size[1]) * (y + 1)),
        )

        # print(f"Calculated slice indices for ({x}, {y}) -> start: {start_index}, end: {end_index}")

        if start_index[0] > length[0] or start_index[1] > length[1]:
            raise IndexError(
                f"Start index {start_index} out of range for img of shape {length}"
            )

        if end_index[0] > length[0] or end_index[1] > length[1]:
            raise IndexError(
                f"End index {end_index} out of range for img of shape {length}"
            )

        return start_index, end_index

    def get_subtile_from_parent_image(self, x: int, y: int) -> Tuple[List[xr.DataArray], xr.DataArray]:
        img_length = self.satellite_list[0].shape[2:]
        label_length = self.ground_truth.shape[2:]

        # print(f"Image length: {img_length}, Label length: {label_length}")

        start_index_img, end_index_img = self.__calculate_slice_index(
            x, y, self.slice_size, img_length
        )
        start_index_label, end_index_label = self.__calculate_slice_index(
            x, y, self.slice_size, label_length
        )

        new_satellite_list = list()
        for data_array in self.satellite_list:
            sliced_data_array = data_array[
                :,
                :,
                start_index_img[0] : end_index_img[0],
                start_index_img[1] : end_index_img[1],
            ]
            sliced_data_array.attrs["x"] = x
            sliced_data_array.attrs["y"] = y

            new_satellite_list.append(sliced_data_array)
            # print(f"Sliced satellite data array for ({x}, {y}), shape: {sliced_data_array.shape}")

        new_ground_truth = self.ground_truth[
            :,
            :,
            start_index_label[0] : end_index_label[0],
            start_index_label[1] : end_index_label[1],
        ]
        new_ground_truth.attrs["x"] = x
        new_ground_truth.attrs["y"] = y

        # print(f"Sliced ground truth data array for ({x}, {y}), shape: {new_ground_truth.shape}")

        return (
            new_satellite_list,
            new_ground_truth,
        )

    def _save_image(
        self, subtiled_data_array: xr.DataArray, subtile_directory: Path, x: int, y: int
    ):
        # print(f"Saving image subtile at ({x}, {y}) to {subtile_directory}, shape: {subtiled_data_array.shape}")
        subtiled_data_array.to_netcdf(
            subtile_directory
            / subtiled_data_array.attrs["parent_tile_id"]
            / f"{x}_{y}"
            / f"{subtiled_data_array.attrs['satellite_type']}.nc"
        )

    def _save_label(
        self,
        subtiled_ground_truth: xr.DataArray,
        subtile_directory: Path,
        x: int,
        y: int,
    ):
        # print(f"Saving label subtile at ({x}, {y}) to {subtile_directory}, shape: {subtiled_ground_truth.shape}")
        subtiled_ground_truth.to_netcdf(
            subtile_directory
            / subtiled_ground_truth.attrs["parent_tile_id"]
            / f"{x}_{y}"
            / f"{SatelliteType.GT.value}.nc"
        )

    def save(self, directory_to_save: Path) -> None:
        directory_to_save.mkdir(parents=True, exist_ok=True)
        # print(f"Created directory {directory_to_save}")

        subtile_directory = directory_to_save / "subtiles"
        subtile_directory.mkdir(parents=True, exist_ok=True)
        # print(f"Created subtile directory {subtile_directory}")

        for x in range(self.slice_size[0]):
            for y in range(self.slice_size[1]):
                subtiled_list, subtiled_ground_truth = (
                    self.get_subtile_from_parent_image(x, y)
                )

                Path(subtile_directory / self.parent_tile_id).mkdir(exist_ok=True)
                assert Path(subtile_directory / self.parent_tile_id).exists()

                Path(subtile_directory / self.parent_tile_id / f"{x}_{y}").mkdir(
                    exist_ok=True
                )
                assert Path(
                    subtile_directory / self.parent_tile_id / f"{x}_{y}"
                ).exists()

                # print(f"Saving subtiles at ({x}, {y})")

                for subtiled_data_array in subtiled_list:
                    self._save_image(subtiled_data_array, subtile_directory, x, y)
                self._save_label(subtiled_ground_truth, subtile_directory, x, y)

        self.satellite_list = None
        self.ground_truth = None
        # print("Cleared satellite_list and ground_truth after saving")

    def load_subtile(
        self,
        directory_to_load: Path,
        satellite_type_list: List[SatelliteType],
        x: int,
        y: int,
    ) -> List[xr.DataArray]:
        tile_dir = directory_to_load / "subtiles" / self.parent_tile_id
        # print(f"Loading subtile from {tile_dir} at ({x}, {y})")
        list_of_subtiled_data_array = list()
        for satellite_type in satellite_type_list:
            subtile_file = tile_dir / f"{x}_{y}" / f"{satellite_type.value}.nc"
            assert subtile_file.exists(), f"{subtile_file} does not exist"
            # print(f"Loading {subtile_file}")

            data_array = xr.load_dataarray(subtile_file)
            assert data_array.attrs["x"] == np.int32(x), f"{data_array.attrs['x']}, {x}"
            assert data_array.attrs["y"] == np.int32(y), f"{data_array.attrs['y']}, {y}"
            # print(f"Loaded data array shape: {data_array.shape}")

            list_of_subtiled_data_array.append(data_array)
        return list_of_subtiled_data_array

    @staticmethod
    def load_subtile_by_dir(
        directory_to_load: Path,
        satellite_type_list: List[SatelliteType],
        slice_size: Tuple[int, int] = (4, 4),
        has_gt: bool = True,
    ) -> List[xr.DataArray]:
        # print(f"Loading subtiles by directory from {directory_to_load}")
        list_of_subtiled_data_array = list()
        for satellite_type in satellite_type_list:
            subtile_file = directory_to_load / f"{satellite_type.value}.nc"
            assert subtile_file.exists(), f"{subtile_file} does not exist"
            # print(f"Loading {subtile_file}")

            data_array = xr.load_dataarray(subtile_file)
            # print(f"Loaded data array shape: {data_array.shape}")

            list_of_subtiled_data_array.append(data_array)

        if has_gt:
            gt_data_array = xr.load_dataarray(
                directory_to_load / f"{SatelliteType.GT.value}.nc"
            )
            # print(f"Loaded ground truth data array shape: {gt_data_array.shape}")
        else:
            gt_data_array = None

        subtile = Subtile(
            satellite_list=list_of_subtiled_data_array,
            ground_truth=gt_data_array,
            slice_size=slice_size,
        )
        return subtile

    def restitch(
        self, directory_to_load: Path, satellite_type_list: List[SatelliteType]
    ) -> None:
        satellite_type_list_with_gt = satellite_type_list + [SatelliteType.GT]
        # print(f"Restitching subtiles from {directory_to_load}")

        list_of_data_array = list()
        for satellite_type in satellite_type_list_with_gt:
            row = []
            for x in range(self.slice_size[0]):
                col = []
                for y in range(self.slice_size[1]):
                    data_array = self.load_subtile(
                        directory_to_load, [satellite_type], x, y
                    )[0]
                    del data_array.attrs["x"]
                    del data_array.attrs["y"]

                    col.append(data_array)
                    # print(f"Restitched column data array for satellite type {satellite_type} at ({x}, {y}), shape: {data_array.shape}")
                row.append(xr.concat(col, dim="width"))
            data_array = xr.concat(row, dim="height")
            list_of_data_array.append(data_array)
            # print(f"Restitched row data array for satellite type {satellite_type}, shape: {data_array.shape}")

        self.satellite_list = list_of_data_array[:-1]
        self.ground_truth = list_of_data_array[-1]
        # print("Completed restitching")

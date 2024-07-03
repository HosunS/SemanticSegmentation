from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataclasses import dataclass, field

import xarray as xr


# given
class SatelliteType(Enum):
    VIIRS_MAX_PROJ = "viirs_max_projection"
    VIIRS = "viirs"
    S1 = "sentinel1"
    S2 = "sentinel2"
    LANDSAT = "landsat"
    GT = "gt"


ROOT = Path.cwd()
PROJ_NAME = "FINAL_PROJ_SWEEPS"
MODEL = "UNet"



@dataclass
class ESDConfig:
    processed_dir: Path = ROOT / "data" / "processed"
    raw_dir: Path = ROOT / "data" / "raw" / "Train"
    results_dir: Path = ROOT / "data" / "predictions" / MODEL
    test_raw_dir: Path = ROOT / "data" / "raw" / "Test"  # Add this line for test raw data
    test_processed_dir: Path = ROOT / "data" / "processed_test"  # Add this line for test processed data
    selected_bands = {
        SatelliteType.VIIRS: ["0"],
        # SatelliteType.S1: ["VV", "VH"],
        SatelliteType.S2: [
            # "12",
            # "11",
            # "09",
            # "8A",
            # "08",
            # "07",
            # "06",
            # "05",
            "04",
            "03",
            "02",
            # "01",
        ],
        # SatelliteType.LANDSAT: [
        #     "11",
        #     "10",
        #     "9",
        #     "8",
        #     "7",
        #     "6",
        #     "5",
        #     "4",
        #     "3",
        #     "2",
        #     "1",
        # ],
        # SatelliteType.VIIRS_MAX_PROJ: ["0"],
    }



    accelerator: str = "gpu"
    batch_size: int = 12
    ## set depth to 1 for SegmentationCNN, doesn't matta for the rest
    depth: int = 1
    devices: int = 1
    embedding_size: int = 32
    in_channels: int = 21 # num_dates * num_bands  (max_viirs , viirs, S2 (RGB ~ 2,3,4))
    kernel_size: int = 6
    learning_rate: float = 0.0003747936327140576
    # 0.00034752640083101196
    max_epochs: int = 8
    model_path: Path = ROOT / "models" / MODEL / "last.ckpt"
    model_type: str = MODEL
    n_encoders: int = 4
    num_workers: int = 24 #can change
    out_channels: int = 4 #channels for each class 
    pool_sizes: str = "5,5,2"
    seed: int = 12378921
    # for dilatedUNet, slice size should be (2,2) or within powers of 2, too big  of a slicesize will give tensor mishapes. '
    # adjust in dilatedUNet  and Deeplabv3 to match the output tensor size to the size of the mask
    slice_size: tuple = (2,2)
    power: float = 0.9
    max_iters: int = 1000
    backbone: str = "resnet101"
    
    ## added for deeplabv3 / dilatedunet
    wandb_run_name: str | None = None

# given
def get_satellite_dataset_size(
    data_set: xr.Dataset, dims: List[str] = ["date", "band", "height", "width"]
):
    """
    Gets the shape of a dataset

    Parameters
    ----------
    data_set : xr.Dataset
        A satellite dataset
    dims: List[str]
        A list of dimensions of the data_set data_arrays
    Returns
    -------
    Tuple:
        Shape of the data_set, default is (date, band, height, width)
    """
    return tuple(data_set.sizes[d] for d in dims)

import sys
from argparse import ArgumentParser

from pathlib import Path
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    EarlyStopping
)

sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.utilities import ESDConfig, PROJ_NAME

ROOT = Path.cwd()

# ensure GPU is available
import torch
# print(torch.cuda.is_available())  # Should print True if GPU is available
# print(torch.cuda.device_count())  # Should print the number of GPUs available
# print(torch.cuda.get_device_name(0))  # Should print the name of the first GPU


def train(options: ESDConfig):
    # wandb.finish()
    # initialize wandb
    wandb.init(project="final_project_metrics" , entity="california_roll",config=options)
    # setup the wandb logger
    wandb_logger = pl.loggers.WandbLogger()
    # initialize the datamodule
    data_module = ESDDataModule(
        processed_dir=options.processed_dir,
        raw_dir=options.raw_dir,
        batch_size=options.batch_size,
        seed=options.seed,
        selected_bands=options.selected_bands,
        slice_size=options.slice_size,
        num_workers=options.num_workers,
        )

    # prepare the data
    data_module.prepare_data()
    data_module.setup(stage="fit")

    # create a model params dict to initialize ESDSegmentation
    # note: different models have different parameters
    model_params = {}
    if options.model_type == "DilatedUNet":
        pass
    elif options.model_type == "DeepLabV3":
        pass
    elif options.model_type == "FCNResnetTransfer":
        model_params={
            "backbone": options.backbone
        }
    elif options.model_type =="UNet":
        model_params ={
        "n_encoders": options.n_encoders,
        "embedding_size" : options.embedding_size
        }
    elif options.model_type == "SegmentationCNN":
        model_params = {
        "depth":options.depth,
        "embedding_size": options.embedding_size,
        "pool_sizes": [int(pool_size) for pool_size in options.pool_sizes.split(",")],
        "kernel_size": options.kernel_size,
        }
    
    
    # initialize the ESDSegmentation model
    model = ESDSegmentation(
                            model_type=options.model_type,
                            in_channels=options.in_channels,
                            out_channels=options.out_channels,
                            learning_rate=options.learning_rate,
                            model_params=model_params
                            )
    
    # Use the following callbacks, they're provided for you,
    # but you may change some of the settings
    # ModelCheckpoint: saves intermediate results for the neural network
    # in case it crashes
    # LearningRateMonitor: logs the current learning rate on weights and biases
    # RichProgressBar: nicer looking progress bar (requires the rich package)
    # RichModelSummary: shows a summary of the model before training (requires rich)
    
    callbacks = [
        ModelCheckpoint(
            dirpath=ROOT / "models" / options.model_type,
            filename="{epoch}-{val_loss:.2f}-{other_metric:.2f}",
            save_top_k=0,
            save_last=True,
            verbose=True,
            monitor="val_f1",
            mode="max",
            every_n_train_steps=1000,
        ),
        LearningRateMonitor(),
        RichProgressBar(),
        RichModelSummary(max_depth=3),
        # EarlyStopping(
        #     monitor="val_loss",  # monitoring this because lower val_loss in our models will correlate to higher val_f1 for now
        #     patience=10,  #epochs with no improvement
        #     verbose=True,
        #     mode="min"
        # ),
    ]
    
    
    # Determine the accelerator and devices to use
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = [0]  # List of GPU device ids, here using the first GPU
        torch.set_float32_matmul_precision("high")
    else:
        accelerator = 'cpu'
        devices = 1  # Use only the CPU
        
    
        
    # initialize trainer, set accelerator, devices, number of nodes, logger
    # max epochs and callbacks
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,     
        logger=wandb_logger,
        max_epochs=options.max_epochs,
        callbacks=callbacks,
        log_every_n_steps=12
    )
    # run trainer.fit
    trainer.fit(model,datamodule=data_module)
    # raise NotImplementedError


if __name__ == "__main__":
    # load dataclass arguments from yml file

    config = ESDConfig()
    parser = ArgumentParser()

    parser.add_argument(
        "--model_type",
        type=str,
        help="The model to initialize.",
        default=config.model_type,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="The learning rate for training model",
        default=config.learning_rate,
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        help="Number of epochs to train for.",
        default=config.max_epochs,
    )
    parser.add_argument(
        "--raw_dir", type=str, default=config.raw_dir, help="Path to raw directory"
    )
    parser.add_argument(
        "-p", "--processed_dir", type=str, default=config.processed_dir, help="."
    )

    parser.add_argument(
        "--in_channels",
        type=int,
        default=config.in_channels,
        help="Number of input channels",
    )
    
    parser.add_argument(
        "--depth",
        type=int,
        help="Depth of the encoders (CNN only)",
        default=config.depth,
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=config.out_channels,
        help="Number of output channels",
    )
    parser.add_argument(
        "--n_encoders",
        type=int,
        help="Number of encoders (Unet only)",
        default=config.n_encoders,
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        help="Embedding size of the neural network (CNN/Unet)",
        default=config.embedding_size,
    )

    parser.add_argument(
        "--kernel_size",
        help="Kernel size of the convolutions",
        type=int,
        default=config.kernel_size,
    )

    parse_args = parser.parse_args()

    config = ESDConfig(**parse_args.__dict__)
    train(config)

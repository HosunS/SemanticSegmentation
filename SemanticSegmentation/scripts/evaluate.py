import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.utilities import ESDConfig
from src.visualization.restitch_plot import restitch_and_plot


def main(options):
    # initialize datamodule
    dm = ESDDataModule(
        raw_dir=options.raw_dir,
        processed_dir=options.processed_dir,
        selected_bands=options.selected_bands,
        batch_size=options.batch_size,
        seed=options.seed,
        slice_size=options.slice_size
    )

    # prepare data
    dm.prepare_data()
    dm.setup("fit")


    # load model from checkpoint
    model = ESDSegmentation.load_from_checkpoint(options.model_path)
    # set model to eval mode
    model.eval()

    # get a list of all processed tiles
    # processed_tiles = list(options.processed_dir.rglob("Tile*"))
    processed_tiles = list(Path(options.processed_dir/"Val/subtiles").glob('Tile*'))
    # for each tile
    for tile in processed_tiles:
        # print(tile)
        
        # run restitch and plot
        restitch_and_plot(options=options,
                        datamodule=dm,
                        model=model,
                        parent_tile_id=tile.parts[-1],
                        results_dir=Path(options.results_dir),
                        accelerator='gpu')




if __name__ == "__main__":
    config = ESDConfig()
    parser = ArgumentParser()

    parser.add_argument(
        "--model_path", type=str, help="Model path.", default=config.model_path
    )
    parser.add_argument(
        "--raw_dir", type=str, default=config.raw_dir, help="Path to raw directory"
    )
    parser.add_argument(
        "-p", "--processed_dir", type=str, default=config.processed_dir, help="."
    )
    parser.add_argument(
        "--results_dir", type=str, default=config.results_dir, help="Results dir"
    )
    main(ESDConfig(**parser.parse_args().__dict__))


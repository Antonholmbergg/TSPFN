from lightning.pytorch.cli import LightningCLI

from tspfn.data.datamodule import TSPFNDataModule
from tspfn.model import SimpleModel


class StantardModelCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        pass


if __name__ == "__main__":
    cli = StantardModelCLI(SimpleModel, TSPFNDataModule)

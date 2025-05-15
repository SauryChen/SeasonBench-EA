from torch.utils.data import Dataset, DataLoader
from .dataset import NWP_Dataset
import lightning.pytorch as pl

class NWPDataModule(pl.LightningDataModule):
    def __init__(self, data_args):
        super().__init__()
        self.data_args = data_args

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = NWP_Dataset(
                data_dir=self.data_args['data_dir'],
                center=self.data_args['center'],
                input_vars=self.data_args['input_vars'],
                input_cons=self.data_args['input_cons'],
                output_vars=self.data_args['output_vars'],
                status='train',
                crop_size=self.data_args['crop_size'],
                is_normalized_nwp=True,
                is_normalized_era5=True,
            )

            self.val_dataset = NWP_Dataset(
                data_dir=self.data_args['data_dir'],
                center=self.data_args['center'],
                input_vars=self.data_args['input_vars'],
                input_cons=self.data_args['input_cons'],
                output_vars=self.data_args['output_vars'],
                status='val',
                crop_size=self.data_args['crop_size'],
                is_normalized_nwp=True,
                is_normalized_era5=True,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_args['batch_size'],
            num_workers=self.data_args['num_workers'],
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_args['batch_size'],
            num_workers=self.data_args['num_workers'],
            shuffle=False,
        )

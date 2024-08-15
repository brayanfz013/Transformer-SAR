import math
from pathlib import Path
from typing import Any

from src.lib.class_load import LoadFiles
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image


def create_dataloader(dataset: Dataset, dataset_opt: dict) -> DataLoader:
	"""Create dataloader.

	Args:
	    dataset (torch.utils.data.Dataset): Dataset.
	    dataset_opt (dict): Dataset options. It contains the following keys:
	        phase (str): 'train' or 'val'.
	        num_worker_per_gpu (int): Number of workers for each GPU.
	        batch_size_per_gpu (int): Training batch size for each GPU.
	    num_gpu (int): Number of GPUs. Used only in the train phase.
	        Default: 1.
	    dist (bool): Whether in distributed training. Used only in the train
	        phase. Default: False.
	    sampler (torch.utils.data.sampler): Data sampler. Default: None.
	    seed (int | None): Seed. Default: None
	"""
	phase = dataset_opt["phase"]
	if phase == "train":
		dataloader_args = dict(
			dataset=dataset,
			batch_size=dataset_opt["batch_size"],
			shuffle=dataset_opt["shuffle"],
			num_workers=dataset_opt["num_worker"],
			drop_last=True,
		)

	elif phase in ["val", "test"]:  # validation
		dataloader_args = dict(
			dataset=dataset, batch_size=1, shuffle=False, num_workers=0
		)
	else:
		raise ValueError(
			f"Wrong dataset phase: {phase}. "
			"Supported ones are 'train', 'val' and 'test'."
		)

	dataloader_args["pin_memory"] = dataset_opt.get("pin_memory", False)
	return DataLoader(**dataloader_args)


class CustomDataset(Dataset):
	def __init__(
		self, dataset_path: Path | str, channel: int = 3, train: bool = True
	) -> None:
		self.handler_files = LoadFiles()

		if train:
			self.noisy_path = Path(dataset_path).joinpath("Noisy/train")
			self.gt_path = Path(dataset_path).joinpath("GTruth/train")
		else:
			self.noisy_path = Path(dataset_path).joinpath("Noisy/test")
			self.gt_path = Path(dataset_path).joinpath("GTruth/test")
		self._noisy_images = self.handler_files.search_load_files_extencion(
			path_search=self.noisy_path.as_posix(), ext=["png"]
		)["png"][1]
		self._gt_imges = self.handler_files.search_load_files_extencion(
			path_search=self.gt_path.as_posix(), ext=["png"]
		)["png"][1]
		self.channel = channel

	def __getitem__(self, index) -> dict[str, Any]:
		if self.channel == 3:
			img_gt: Tensor = read_image(
				self._gt_imges[index], mode=ImageReadMode.RGB
			)  # ImageReadMode.UNCHANGED
			img_lq: Tensor = read_image(
				self._noisy_images[index], mode=ImageReadMode.RGB
			)
		elif self.channel == 1:
			img_gt: Tensor = read_image(self._gt_imges[index], mode=ImageReadMode.GRAY)
			img_lq: Tensor = read_image(
				self._noisy_images[index], mode=ImageReadMode.GRAY
			)
		return {
			"lq": img_lq,
			"gt": img_gt,
			"lq_path": self._noisy_images[index],
			"gt_path": self.gt_path[index],
		}

	def __len__(self):
		return len(self._noisy_images)


def load_dataloader(opt):
	if opt["dataloader"]["phase"] == "train":
		sar_dataset = CustomDataset(
			dataset_path=opt["preprocessing"]["output"], channel=3, train=True
		)
		dataloader = create_dataloader(
			dataset=sar_dataset, dataset_opt=opt["dataloader"]
		)
	elif opt["dataloader"]["phase"] == "val":
		sar_dataset = CustomDataset(
			dataset_path=opt["preprocessing"]["output"], channel=3, train=False
		)
		dataloader = create_dataloader(
			dataset=sar_dataset,
			dataset_opt=opt["dataloader"],
		)
	num_iter_per_epoch = math.ceil(
		len(sar_dataset) / opt["dataloader"]["batch_size_per_gpu"]
	)
	total_iters = int(opt["dataloader"]["iters"])
	total_epochs = math.ceil(total_iters / (num_iter_per_epoch))

	return dataloader, total_epochs, total_iters


if __name__ == "__main__":
	import yaml

	file_config = Path(
		"/home/arthemis/Documents/pytorch_env/pytorch_env/transformers_ruben/src/data/config/config.yml"
	)
	configuration = yaml.safe_load(file_config.open("r"))

	dataloader_, total_epochs, total_iters = load_dataloader(configuration)

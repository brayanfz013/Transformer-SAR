import datetime
import logging
import math
import os
import random
import time
from os import path as osp
from pathlib import Path

import numpy as np
import torch
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (
	MessageLogger,
	get_env_info,
	get_root_logger,
	init_tb_logger,
	init_wandb_logger,
	make_exp_dirs,
)
from basicsr.utils.options import dict2str
from torch.utils.data import DataLoader, Dataset


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

	#  prefetch_mode=None: Normal dataloader
	# prefetch_mode='cuda': dataloader for CUDAPrefetcher
	return DataLoader(**dataloader_args)


def parse_options():
	import yaml

	file_config = Path(
		"/home/arthemis/Documents/pytorch_env/pytorch_env/transformers_ruben/src/data/config/config.yml"
	)
	return yaml.safe_load(file_config.open("r"))


def init_loggers(opt):
	log_file = "/home/arthemis/Documents/pytorch_env/pytorch_env/transformers_ruben/src/data/logs/file.log"  # osp.join(opt["path"]["log"], f"train_{opt['name']}_{get_time_str()}.log")
	logger = get_root_logger(
		logger_name="basicsr", log_level=logging.INFO, log_file=log_file
	)
	logger.info(get_env_info())
	logger.info(dict2str(opt))

	# initialize wandb logger before tensorboard logger to allow proper sync:
	if (
		(opt["logger"].get("wandb") is not None)
		and (opt["logger"]["wandb"].get("project") is not None)
		and ("debug" not in opt["name"])
	):
		assert (
			opt["logger"].get("use_tb_logger") is True
		), "should turn on tensorboard when using wandb"
		init_wandb_logger(opt)
	tb_logger = None
	if opt["logger"].get("use_tb_logger") and "debug" not in opt["name"]:
		tb_logger = init_tb_logger(log_dir=osp.join("tb_logger", opt["name"]))
	return logger, tb_logger


from typing import Any

from src.lib.class_load import LoadFiles
from torch import Tensor
from torchvision.io import ImageReadMode, read_image

handler_files = LoadFiles()


class CustomDataset(Dataset):
	def __init__(
		self, opt: dict, dataset_path: Path | str, channel: int = 3, train: bool = True
	) -> None:
		self.opt = opt
		if train:
			self.noisy_path = Path(dataset_path).joinpath("val/groundtruth")
			self.gt_path = Path(dataset_path).joinpath("GTruth/groundtruth")
		else:
			self.noisy_path = Path(dataset_path).joinpath("val/input")
			self.gt_path = Path(dataset_path).joinpath("GTruth/input")
		self._noisy_images = handler_files.search_load_files_extencion(
			path_search=self.noisy_path.as_posix(), ext=["png"]
		)["png"][1]
		self._gt_imges = handler_files.search_load_files_extencion(
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
			"lq": img_lq.float(),
			"gt": img_gt.float(),
			"lq_path": self._noisy_images[index],
			"gt_path": self._gt_imges[index],
		}

	def __len__(self):
		return len(self._noisy_images)


# ImageReadMode.UNCHANGED


def load_dataloader(opt):
	if opt["dataloader"]["phase"] == "train":
		sar_dataset = CustomDataset(
			opt, dataset_path=opt["preprocessing"]["output"], channel=1, train=True
		)
		dataloader = create_dataloader(
			dataset=sar_dataset, dataset_opt=opt["dataloader"]
		)
	elif opt["dataloader"]["phase"] == "val":
		sar_dataset = CustomDataset(
			opt, dataset_path=opt["preprocessing"]["output"], channel=1, train=False
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


def create_train_val_dataloader(opt):
	# create train and val dataloaders
	opt["dataloader"]["phase"] = "train"
	train_loader, total_epochs, total_iters = load_dataloader(opt)
	opt["dataloader"]["phase"] = "val"
	val_loader, _, _ = load_dataloader(opt)

	return train_loader, val_loader, total_epochs, total_iters


def main():
	# parse options, set distributed setting, set ramdom seed
	opt = parse_options()
	opt["dist"] = False
	torch.backends.cudnn.benchmark = True
	# torch.backends.cudnn.deterministic = True

	# automatic resume ..
	state_folder_path = "experiments/{}/training_states/".format(opt["name"])

	try:
		states = os.listdir(state_folder_path)
	except:
		states = []

	resume_state = None
	if len(states) > 0:
		max_state_file = "{}.state".format(max([int(x[0:-6]) for x in states]))
		resume_state = os.path.join(state_folder_path, max_state_file)
		opt["path"]["resume_state"] = resume_state

	# load resume states if necessary
	if opt["path"].get("resume_state"):
		device_id = torch.cuda.current_device()
		resume_state = torch.load(
			opt["path"]["resume_state"],
			map_location=lambda storage, loc: storage.cuda(device_id),
		)
	else:
		resume_state = None

	# mkdir for experiments and logger
	if resume_state is None:
		make_exp_dirs(opt)
		# if opt["logger"].get("use_tb_logger") and "debug" not in opt["name"] and 0 == 0:
		#     mkdir_and_rename(osp.join("tb_logger", opt["name"]))

	# initialize loggers
	logger, tb_logger = init_loggers(opt)

	# create train and validation dataloaders
	result = create_train_val_dataloader(opt)
	train_loader, val_loader, total_epochs, total_iters = result

	# # create model
	# if resume_state:  # resume training
	#     # check_resume(opt, resume_state["iter"])
	#     model = create_model(opt)

	#     model.resume_training(resume_state)  # handle optimizers and schedulers
	#     logger.info(
	#         f"Resuming training from epoch: {resume_state['epoch']}, "
	#         f"iter: {resume_state['iter']}."
	#     )
	#     start_epoch = resume_state["epoch"]
	#     current_iter = resume_state["iter"]
	# else:
	model = create_model(opt)
	start_epoch = 0
	current_iter = 0

	for name, parameter in model.net_g.named_parameters():
		parameter.requires_grad = False
		# print(name)
		if name in "output.weight":
			parameter.requires_grad = True
		if "refinement.3" in name:
			parameter.requires_grad = True
	from prettytable import PrettyTable

	def count_parameters(model):
		table = PrettyTable(["Modules", "Parameters"])
		total_params = 0
		for name, parameter in model.named_parameters():
			if not parameter.requires_grad:
				continue
			params = parameter.numel()
			table.add_row([name, params])
			total_params += params
		# print(table)
		# print(f"Total Trainable Params: {total_params}")
		return total_params

	total_parameter = count_parameters(model.net_g)
	# print(total_parameter)
	# print(sum(p.numel() for p in model.net_g.parameters() if p.requires_grad))

	# print(summary(model.net_g, input_size=[(1, 512, 512)]))

	# create message logger (formatted outputs)
	msg_logger = MessageLogger(opt, current_iter, tb_logger)

	# dataloader prefetcher
	prefetch_mode = opt["datasets"]["train"].get("prefetch_mode")
	if prefetch_mode is None or prefetch_mode == "cpu":
		prefetcher = CPUPrefetcher(train_loader)
	elif prefetch_mode == "cuda":
		prefetcher = CUDAPrefetcher(train_loader, opt)
		logger.info(f"Use {prefetch_mode} prefetch dataloader")
		opt["datasets"]["train"]["pin_memory"] = True
		if opt["datasets"]["train"].get("pin_memory") is not True:
			raise ValueError("Please set pin_memory=True for CUDAPrefetcher.")
	else:
		raise ValueError(
			f"Wrong prefetch_mode {prefetch_mode}."
			"Supported ones are: None, 'cuda', 'cpu'."
		)

	# training
	logger.info(f"Start training from epoch: {start_epoch}, iter: {current_iter}")
	data_time, iter_time = time.time(), time.time()
	start_time = time.time()

	# for epoch in range(start_epoch, total_epochs + 1):

	iters = opt["datasets"]["train"].get("iters")
	batch_size = opt["datasets"]["train"].get("batch_size_per_gpu")
	mini_batch_sizes = opt["datasets"]["train"].get("mini_batch_sizes")
	gt_size = opt["datasets"]["train"].get("gt_size")
	mini_gt_sizes = opt["datasets"]["train"].get("gt_sizes")

	groups = np.array([sum(iters[0 : i + 1]) for i in range(0, len(iters))])

	logger_j = [True] * len(groups)

	scale = opt["scale"]

	epoch = start_epoch
	while current_iter <= total_iters:
		prefetcher.reset()
		train_data = prefetcher.next()

		while train_data is not None:
			data_time = time.time() - data_time

			current_iter += 1
			if current_iter > total_iters:
				break
			# update learning rate
			model.update_learning_rate(
				current_iter, warmup_iter=opt["train"].get("warmup_iter", -1)
			)

			### ------Progressive learning ---------------------
			j = ((current_iter > groups) != True).nonzero()[0]
			if len(j) == 0:
				bs_j = len(groups) - 1
			else:
				bs_j = j[0]

			mini_gt_size = mini_gt_sizes[bs_j]
			mini_batch_size = mini_batch_sizes[bs_j]

			if logger_j[bs_j]:
				logger.info(
					"\n Updating Patch_Size to {} and Batch_Size to {} \n".format(
						mini_gt_size,
						mini_batch_size * torch.cuda.device_count(),
					)
				)
				logger_j[bs_j] = False

			lq = train_data["lq"]
			gt = train_data["gt"]

			if mini_batch_size < batch_size:
				indices = random.sample(range(0, batch_size), k=mini_batch_size)
				lq = lq[indices]
				gt = gt[indices]

			if mini_gt_size < gt_size:
				x0 = int((gt_size - mini_gt_size) * random.random())
				y0 = int((gt_size - mini_gt_size) * random.random())
				x1 = x0 + mini_gt_size
				y1 = y0 + mini_gt_size
				lq = lq[:, :, x0:x1, y0:y1]
				gt = gt[:, :, x0 * scale : x1 * scale, y0 * scale : y1 * scale]
			###-------------------------------------------

			model.feed_train_data({"lq": lq, "gt": gt})
			print(current_iter)
			model.optimize_parameters(current_iter)

			iter_time = time.time() - iter_time
			# log
			if current_iter % opt["logger"]["print_freq"] == 0:
				log_vars = {"epoch": epoch, "iter": current_iter}
				log_vars.update({"lrs": model.get_current_learning_rate()})
				log_vars.update({"time": iter_time, "data_time": data_time})
				log_vars.update(model.get_current_log())
				msg_logger(log_vars)

			# save models and training states
			if current_iter % opt["logger"]["save_checkpoint_freq"] == 0:
				logger.info("Saving models and training states.")
				model.save(epoch, current_iter)

			# validation
			if opt.get("val") is not None and (
				current_iter % opt["val"]["val_freq"] == 0
			):
				rgb2bgr = opt["val"].get("rgb2bgr", True)
				# wheather use uint8 image to compute metrics
				use_image = opt["val"].get("use_image", True)
				model.validation(
					val_loader,
					current_iter,
					tb_logger,
					opt["val"]["save_img"],
					rgb2bgr,
					use_image,
				)

			data_time = time.time()
			iter_time = time.time()
			train_data = prefetcher.next()
		# end of iter
		epoch += 1

	# end of epoch

	consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
	logger.info(f"End of training. Time consumed: {consumed_time}")
	logger.info("Save the latest model.")
	model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
	if opt.get("val") is not None:
		model.validation(val_loader, current_iter, tb_logger, opt["val"]["save_img"])
	if tb_logger:
		tb_logger.close()


if __name__ == "__main__":
	main()

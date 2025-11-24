# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
from datasets.CVCDataset import CVCDataset
from datasets.KvasirDataset import get_KvasirDataset
from datasets.CrossDomain_dataset import get_crossdomain_dataset

def generate_dataset(dataset_name, cfg, data_split):

	####  single domain setting ######
	if dataset_name == 'cvc' or dataset_name == 'CVC':
		return CVCDataset('CVC', cfg, data_split)

	elif dataset_name == 'Kvasir':
		if data_split == 'train':
			IMG_ROOT = cfg.TRAIN_IMG_ROOT
			GT_ROOT = cfg.TRAIN_GT_ROOT
		else:
			IMG_ROOT = cfg.TEST_IMG_ROOT
			GT_ROOT = cfg.TEST_GT_ROOT
		return get_KvasirDataset(IMG_ROOT,GT_ROOT, cfg.DATA_SIZE,data_split)
	####  single domain setting ######

	####  cross-domain setting ######
	elif dataset_name == 'CrossDomain' and data_split == 'train':
		return get_crossdomain_dataset(cfg.TRAIN_IMG_ROOT, cfg.TRAIN_GT_ROOT, cfg.DATA_SIZE,data_split)
	elif dataset_name == 'ETIS' and data_split == 'test':
		return get_crossdomain_dataset(cfg.ETIS_IMG_TXT, cfg.ETIS_LABEL_TXT, cfg.DATA_SIZE)
	elif dataset_name == 'CVC_ColonDB' and data_split == 'test':
		return get_crossdomain_dataset(cfg.CVC_ColonDB_IMG_TXT, cfg.CVC_ColonDB_LABEL_TXT, cfg.DATA_SIZE,data_split)

	else:
		raise ValueError('generate_dataset.py: dataset %s is not support yet'%dataset_name)

	####  cross-domain setting ######

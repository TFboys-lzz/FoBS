# ----------------------------------------
# Written by Xiaoqing Guo
# ----------------------------------------

import torch
from model.ori_deeplabv3plus   import ori_deeplabv3plus
from model.deeplabv3plus       import deeplabv3plus
from model.deeplabv3plusggmmix import deeplabv3plusggmmix
from model.deeplabv3plusConv   import deeplabv3plusConv
from model.deeplabv3p          import deeplabv3p

def generate_net(cfg):

	if cfg.MODEL_NAME == 'ori_deeplabv3plus'   or cfg.MODEL_NAME == 'ori_deeplabv3+':
		return ori_deeplabv3plus(cfg)
	if cfg.MODEL_NAME == 'deeplabv3plus'       or cfg.MODEL_NAME == 'deeplabv3+':
		return deeplabv3plus(cfg)
	if cfg.MODEL_NAME == 'deeplabv3plusggmmix' or cfg.MODEL_NAME == 'deeplabv3+ggmmix':
		return deeplabv3plusggmmix(cfg)
	if cfg.MODEL_NAME == 'deeplabv3plusConv'   or cfg.MODEL_NAME == 'deeplabv3+Conv':
		return deeplabv3plusConv(cfg)
	if cfg.MODEL_NAME == 'deeplabv3p':
		return deeplabv3p(cfg)

	else:
		raise ValueError('generateNet.py: network %s is not support yet'%cfg.MODEL_NAME)

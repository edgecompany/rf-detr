import supervision as sv
from rfdetr import RFDETRBase, RFDETRLarge
import onnx
import os
from pathlib import Path
import torch
from copy import deepcopy
from glob import glob
from coco_classes import TEC_COCO_CATEGORIES, COCO_CATEGORIES

pretrain_weights = 'logs/rfdetr_train_output-2025-06-10_16-17-41/checkpoint_best_regular.pth'
model = RFDETRBase(resolution=728, pretrain_weights=pretrain_weights)

model.export(
    resolution=728, 
    simplify=True,  
    backbone_only=False,
    device='cuda')

onnx_path = 'output/inference_model.sim.onnx'

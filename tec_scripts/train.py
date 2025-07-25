from datetime import datetime
import os

from clearml import Task
from rfdetr import RFDETRBase, RFDETRLarge
from coco_classes import TEC_COCO_CATEGORIES, COCO_CATEGORIES
from utils import clearml_log_fit_history



task = Task.init(
    project_name="detector-bird",
    task_name="RF-DETR-B train - Coco Categories",
    task_type="training",
    tags=["bird", "detect", "bird-standing-optimized", "coco-categories", "fp32"],
)

config = {
    "model": {
        "pretrain_weights":"logs/rfdetr_train_output-2025-06-18_21-33-22/checkpoint.pth",
        "resolution": 728,
        "num_classes": len(TEC_COCO_CATEGORIES)-1, # Non coco classes dataset
        "device": "cuda",
    },
    "train": {
        "resume": False,
        "dataset_dir": "../../localDatasets/TECCOCO_Dataset_Detection_32Frame_Tiles-imgsize_1920-tilesize_728-classes_21-bgsample_3.0-min_box_size_6/",
        "epochs": 10,
        "batch_size": 32,
        "grad_accum_steps": 1,
        "num_workers": 12,
        "lr": 1e-3,
        "lr_encoder": 1e-4,
        "tensorboard": True,
        "output_dir": f"logs/rfdetr_train_output-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    },
}
task.connect(config)

os.makedirs(config['train']['output_dir'], exist_ok=True)


history = []
def callback2(data):
    history.append(data)
    clearml_log_fit_history(data)


model = RFDETRBase(**config['model'])
model.callbacks["on_fit_epoch_end"].append(callback2)

model.train(**config['train'])

from datetime import datetime
import os
# import faster_coco_eval

# # Replace pycocotools with faster_coco_eval
# faster_coco_eval.init_as_pycocotools()

from clearml import Task
from rfdetr import RFDETRBase, RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES
from utils import clearml_log_fit_history



task = Task.init(
    project_name="detector-bird",
    task_name="RF-DETR-B train - Coco Categories",
    task_type="training",
    tags=["bird", "detect", "bird-standing-optimized", "coco-categories", "fp32"],
)

config = {
    "model": {
        "pretrain_weights":"./logs/rfdetr_train_output-2025-04-27_23-27-24/checkpoint.pth",
        "resolution": 728,
        # "num_classes": 1, # Non coco classes dataset
        "device": "cuda",
        "num_queries": 100,
    },
    "train": {
        # "resume": "./logs/rfdetr_train_output-2025-04-26_09-28-57/checkpoint.pth",
        "dataset_dir": "../localDatasets/COCO_Dataset_Detection_24Frame_Tiles-imgsize_1920-tilesize_728-classes_90-bgsample_2.0/",
        "epochs": 50,
        "batch_size": 48,
        "grad_accum_steps": 1,
        "num_workers": 2,
        "lr": 1e-4,
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

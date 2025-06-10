from pathlib import Path
from clearml import Task
from rfdetr import RFDETRBase, RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES
from utils import clearml_log_fit_history
from datetime import datetime
import os
from export_onnx import export as to_onnx
import subprocess


EXPORT = True

config = {
    "model": {
        "pretrain_weights":'./logs/rfdetr_train_output-2025-06-02_23-32-44/checkpoint.pth',
        "resolution": 728,
        "device": "cuda",
        "num_queries": 100,
    },
    "train": {
        "eval": True,
        "dataset_dir": "../localDatasets/COCO_Dataset_Detection_24Frame_Tiles-imgsize_1920-tilesize_728-classes_90-bgsample_2.0/",
        "batch_size": 48,
        "num_workers": 4,
        "output_dir": f"logs/rfdetr_eval_output-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    },
}

task = None
task = Task.init(
    project_name="detector-bird",
    task_name=f"RF-DETR-B eval - {config['model']['pretrain_weights']}",
    task_type="testing",
    tags=["bird", "detect", "bird-standing-optimized", "fp32"],
)

task.connect(config)

os.makedirs(config['train']['output_dir'], exist_ok=True)


history = []
def callback2(data):
    history.append(data)
    clearml_log_fit_history(data)


model = RFDETRBase(**config['model'])
model.callbacks["on_fit_epoch_end"].append(callback2)

# model.train(**config['train'])

if EXPORT:
    if task is not None: task.mark_started(force=True)

    model_name = f"RFDETR-{task.id if task else  ''}" 
    assert 'resolution' in config['model'] and config["model"]['resolution'], "Model resolution must be specified in configuration during exports."
    onnx_path = to_onnx(model_name, model, config["model"]['resolution'], dynamic_axes=True, save_dir=config["train"]["output_dir"], device='cuda', slim=True, clearml_task=task, upload=True)
    print(f"Export with triton model navigator onnx model: {onnx_path}")
    
    onnx_path = Path(onnx_path)
    
    
    print("Triton Model Navigator Package export...")
    model_navigator_out = onnx_path.parent / f'nav-RFDETRBase-onnx-trt-{task.id}.nav'
    command = [
        "python",  #  ensure the 'python' command is in your PATH
        "export_nav_package.py",  #  the name of your script
        str(onnx_path.resolve()),
        str(model_navigator_out.resolve())
    ]
    
    try:
        # Use subprocess.run for more control and error handling
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    
        print("Model navigtor conversion executed successfully.")
        print("Output:")
        print(result.stdout)  # Print the standard output of the script

    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
        print("Return Code:", e.returncode)
        print("Standard Output:", e.stdout)  # Print the standard output
        print("Standard Error:", e.stderr)    # Print the standard error, which is crucial for debugging
    
    print("Triton Model Navigator Package export... Done.")
    if model_navigator_out.exists():
        if task is not None:
            task.update_output_model(
                model_path=str(model_navigator_out), 
                name=f"nav-RFDETRBase-onnx-trt-{task.id}.nav",
                model_name=f"nav-RFDETRBase-onnx-trt-{task.id}.nav",
                comment=f"Triton Model Navigator Package exported model from model id = {task.id} of type = rfdetr-base",
                auto_delete_file=False
            )
    else:
        print(f"|ERR| Triton Model Navigator Package not genereated in {model_navigator_out}")
    
    if task is not None:
        task.flush(wait_for_uploads=True)
        task.mark_completed()
        task.publish()

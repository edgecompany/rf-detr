from PIL import Image
import io
import requests
import json
import pandas as pd

from pathlib import Path
from clearml import Task
from rfdetr import RFDETRBase, RFDETRLarge
from utils import clearml_log_fit_history
from datetime import datetime
import os
import subprocess
from coco_classes import TEC_COCO_CATEGORIES, COCO_CATEGORIES
import shutil

EXPORT = True

config = {
    "model": {
        "pretrain_weights":'logs/rfdetr_train_output-2025-06-18_21-33-22/checkpoint_best_regular.pth',
        "resolution": 728,
        "device": "cuda",
        "num_classes": len(TEC_COCO_CATEGORIES)-1, # Non coco classes dataset
    },
    "train": {
        "eval": True,
        "dataset_dir": "../../localDatasets/TECCOCO_Dataset_Detection_32Frame_Tiles-imgsize_1920-tilesize_728-classes_21-bgsample_3.0-min_box_size_6/",
        "batch_size": 48,
        "num_workers": 16,
        "output_dir": f"logs/rfdetr_eval_output-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    },
}


task = None
task = Task.init(
    project_name="detector-bird",
    task_name=f"RF-DETR-B eval - {config['model']['pretrain_weights']}",
    task_type="testing",
    tags=["bird", "detect", "tec-coco-classes", "fp32"],
)

task.connect(config)

os.makedirs(config['train']['output_dir'], exist_ok=True)


history = []
def callback2(data):
    history.append(data)
    clearml_log_fit_history(data)


model = RFDETRBase(**config['model'])
model.callbacks["on_train_end"].append(callback2)

model.train(**config['train'])

if EXPORT:
    if task is not None: task.mark_started(force=True)
    
    results_json = os.path.join(config['train']['output_dir'], 'results.json')
    if os.path.isfile(results_json):
        with open(results_json) as f:
            results = json.load(f)
        logger = task.get_logger()

        # --- Overall Metrics Reporting ---
        logger.report_scalar(
            title='Overall Metrics',
            series='mAP',
            value=results['map'],
            iteration=0
        )
        logger.report_scalar(
            title='Overall Metrics',
            series='Precision',
            value=results['precision'],
            iteration=0
        )
        logger.report_scalar(
            title='Overall Metrics',
            series='Recall',
            value=results['recall'],
            iteration=0
        )

        # --- Class-wise Metrics Reporting using a Table ---
        class_data = [item for item in results['class_map'] if item['class'] != 'all']

        # Convert class_data to a pandas DataFrame for easy reporting
        df_class_metrics = pd.DataFrame(class_data)

        # Report the DataFrame as a table to ClearML
        logger.report_table(
            title='Class-wise COCO Evaluation Metrics',
            series='Metrics Table',
            table_plot=df_class_metrics
        )
    
    model_name = f"RFDETR-{task.id if task else  ''}" 

    save_dir = Path(config["train"]["output_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    model.export(
        output_dir=save_dir,
        resolution=728, 
        simplify=True,  
        backbone_only=False,
        device='cuda', 
        opset_version=20)
 
    onnx_path =  save_dir / 'inference_model.onnx'
    onnx_sim_path =  save_dir / 'inference_model.sim.onnx'   
    
    # if onnx_path.exists():
    #     shutil.copy(onnx_path, save_dir / 'inference_model.onnx')
    # if onnx_sim_path.exists():
    #     shutil.copy(onnx_sim_path, save_dir / 'inference_model.sim.onnx')
    
    print("Triton Model Navigator Package export...")
    model_navigator_out = save_dir / f'nav-RFDETRBase-onnx-trt-{task.id}.nav'
    command = [
        "python",  #  ensure the 'python' command is in your PATH
        "export_nav_package.py",  #  the name of your script
        str(onnx_sim_path.resolve()),
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
    if onnx_sim_path.exists():
        task.update_output_model(
                model_path=str(onnx_sim_path.resolve()), 
                name=f"nav-RFDETRBase-onnx-{task.id}.sim.onnx",
                model_name=f"nav-RFDETRBase-onnx-{task.id}.sim.onnx",
                comment=f"ONNX exported model from model id = {task.id} of type = rfdetr-base",
                auto_delete_file=False
            )
        
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

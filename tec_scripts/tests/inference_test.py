from pathlib import Path
import sys
sys.path.append('.')
from rfdetr_onnx import RTDETR_ONNX
from rfdetr_nav import RTDETR_NAV

# Get model and image
image_path = "./images/cars-people-walking_0.jpg"
onnx_model_path = "../logs/rfdetr_eval_output-2025-07-23_08-26-54/inference_model.sim.onnx"
nav_model_path = "../logs/rfdetr_eval_output-2025-07-23_08-26-54/nav-RFDETRBase-onnx-trt-4de91248976b42d7b8c1743a3421e85b-fp32.nav"

# Initialize the model
onnx_model = RTDETR_ONNX(onnx_model_path=onnx_model_path)
nav_model = RTDETR_NAV(nav_model_path=nav_model_path)

# Run inference and get detections
_, nav_labels, nav_boxes = nav_model.run_inference(image_path)

# Run inference and get detections
_, onnx_labels, onnx_boxes = onnx_model.run_inference(image_path)

# Draw and display the detections
onnx_model.save_detections(image_path, onnx_boxes, onnx_labels, f"out/onnx_{Path(image_path).name}_{Path(onnx_model_path).name}.jpg")
nav_model.save_detections(image_path, nav_boxes, nav_labels, f"out/nav_{Path(image_path).name}_{Path(nav_model_path).name}.jpg")
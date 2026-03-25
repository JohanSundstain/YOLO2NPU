def pth2onnx(model_path):
	import torch
	import onnx
	import onnxruntime as ort
	import numpy as np
	from ultralytics import YOLO
 
 	# Load the PyTorch model
	model = YOLO(model_path)
	
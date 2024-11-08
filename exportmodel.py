from ultralytics import YOLOv10
# Load a model
model = YOLOv10('yolov10n.pt') #Load an official model
model = YOLOv10('/root/yolov10-main/runs/detect/train_v1019/weights/best.pt') #load a custom trained model

# Export the model
model.export(format='onnx')
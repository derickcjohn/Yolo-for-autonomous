from ultralytics import YOLO

# Load a model
# det_model = YOLO("yolo11n.pt")  # load an official model
seg_model = YOLO("runs/segment/train2/weights/best.pt")  # load a custom-trained model

# Export the model
# det_model.export(format="onnx", dynamic=True, device="cpu", imgsz=640)
seg_model.export(format="onnx", dynamic=True, device="cpu", imgsz=640)
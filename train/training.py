from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)

# Train the model 
results = model.train(data="data.yaml", epochs=900, patience=100, imgsz=640)

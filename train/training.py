from ultralytics import YOLO

# # Load a segmentation model
# model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)

# # Train the model 
# results = model.train(data="data.yaml", epochs=900, patience=100, imgsz=640)

# Load semantic segmentation model
model = YOLO("yolo26n-sem.pt")  # load a pretrained model (recommended for training)
results = model.train(data="mapillary.yaml", epochs=900, patience=100, imgsz=1024, lr0=0.001)

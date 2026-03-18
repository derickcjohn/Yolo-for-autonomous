from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model

results = model("street.jpg", classes=[0,1,2,3,5,7,9,10,11,12])  # return a list of Results objects 

results[0].show()
results[0].save(filename="result_street.jpg")  # save to disk
from ultralytics import YOLO


model=YOLO('runs/detect/train11/weights/best.pt')
results = model.predict(source='0', show=True)

print(results)
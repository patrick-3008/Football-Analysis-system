from ultralytics import YOLO

model = YOLO('models/best.pt')

results = model.predict('input_videos/08fd33_4.mp4', save=True)
print(results[0])
print('=====================================')
for result in results:
    print(result.boxes.xyxy.cpu().numpy())
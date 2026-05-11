import torch
import cv2

# Model disabled for Azure deployment
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(img_path):
    if model:
        results = model(img_path)

        img = results.render()[0]

        output_path = "static/output.jpg"
        cv2.imwrite(output_path, img)

        detections = results.pandas().xyxy[0]

        objects = []
        object_names = []

        for _, row in detections.iterrows():
            label = row['name']
            confidence = round(row['confidence'], 2)

            objects.append(f"{label} ({confidence})")
            object_names.append(label)

        return objects, output_path, object_names

    else:
        # Fallback when model is not available
        output_path = "static/output.jpg"

        # Just copy input image as output (or skip processing)
        img = cv2.imread(img_path)
        if img is not None:
            cv2.imwrite(output_path, img)

        return ["Model not loaded"], output_path, []
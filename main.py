# Install required packages
!pip install -q opencv-python-headless torch torchvision transformers==4.39.3

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, SiglipForImageClassification
from google.colab import files
from IPython.display import HTML
from base64 import b64encode

# Upload video
print("Upload an MP4 video for analysis:")
uploaded = files.upload()
video_path = next(iter(uploaded))

# Load model
model_id = "prithivMLmods/deepfake-detector-model-v1"
processor = AutoImageProcessor.from_pretrained(model_id)
model = SiglipForImageClassification.from_pretrained(model_id)
model.eval()

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Analyze video
cap = cv2.VideoCapture(video_path)
predictions = []
bounding_boxes = []
frames = []

frame_limit = 60
count = 0

while count < frame_limit:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    confidence = 0.5
    box = []

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        inputs = processor(images=face_rgb, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).logits
            confidence = torch.softmax(output, dim=-1)[0][1].item()
        box.append((x, y, w, h))
        break

    predictions.append(confidence)
    bounding_boxes.append(box)
    frames.append(frame)
    count += 1

cap.release()

# Generate verdict
score = np.mean(predictions)
result = "FAKE" if score > 0.5 else "REAL"
box_color = (0, 0, 255) if result == "FAKE" else (0, 255, 0)

print(f"Result: {result} (Avg confidence: {score:.2f})")

# Generate output video
output_file = "processed_output.mp4"
h, w, _ = frames[0].shape
video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))

for i, frame in enumerate(frames):
    for (x, y, bw, bh) in bounding_boxes[i]:
        cv2.rectangle(frame, (x, y), (x+bw, y+bh), box_color, 2)
        cv2.putText(frame, result, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    video_writer.write(frame)

video_writer.release()

# Show confidence histogram
plt.figure(figsize=(8, 4))
plt.hist(predictions, bins=10, color="skyblue", edgecolor="black")
plt.title("Fake Confidence Per Frame")
plt.xlabel("Confidence (0 = Real, 1 = Fake)")
plt.ylabel("Number of Frames")
plt.grid(True)
plt.tight_layout()
plt.show()

# Preview and download link
with open(output_file, "rb") as f:
    data = f.read()
    b64 = b64encode(data).decode()

HTML(f"""
<video width="400" controls>
  <source src="data:video/mp4;base64,{b64}" type="video/mp4">
</video><br>
<a download="processed_output.mp4" href="data:video/mp4;base64,{b64}">Download Processed Video</a>
""")

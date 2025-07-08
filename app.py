
from flask import Flask, request, render_template, send_file
import os
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

processor = AutoImageProcessor.from_pretrained("prithivMLmods/deepfake-detector-model-v1")
model = SiglipForImageClassification.from_pretrained("prithivMLmods/deepfake-detector-model-v1")
model.eval()

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_file = request.files['video']
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.mp4')
        video_file.save(video_path)

        cap = cv2.VideoCapture(video_path)
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed.mp4')
        frame_preds = []
        frame_faces = []
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= 60:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            found = False
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                if face.size == 0:
                    continue
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                inputs = processor(images=face_rgb, return_tensors="pt")
                with torch.no_grad():
                    logits = model(**inputs).logits
                    confidence = torch.softmax(logits, dim=-1)[0][1].item()
                frame_preds.append(confidence)
                frame_faces.append((x, y, w, h))
                found = True
                break
            if not found:
                frame_preds.append(0.5)
                frame_faces.append(None)
            frames.append(frame)
            frame_count += 1
        cap.release()

        avg = np.mean(frame_preds)
        verdict = "FAKE" if avg > 0.5 else "REAL"
        color = (0, 0, 255) if verdict == "FAKE" else (0, 255, 0)

        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

        for i in range(len(frames)):
            if frame_faces[i]:
                x, y, w, h = frame_faces[i]
                cv2.rectangle(frames[i], (x, y), (x+w, y+h), color, 2)
                cv2.putText(frames[i], verdict, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            out.write(frames[i])
        out.release()

        return send_file(out_path, as_attachment=True)

    return render_template('index.html')

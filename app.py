from flask import Flask, render_template, Response
import cv2
import torch
from collections import Counter

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

stats_counter = Counter()

def detect_and_count(frame):
    global stats_counter
    stats_counter.clear()
    results = model(frame)
    for det in results.xyxy[0]:
        cls_id = int(det[5])
        cls_name = model.names[cls_id]
        stats_counter[cls_name] += 1
        x1, y1, x2, y2 = map(int, det[:4])
        label = f"{cls_name}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    return frame

def generate():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = detect_and_count(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template("index.html", stats=dict(stats_counter))

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

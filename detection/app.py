from flask import Flask, render_template, Response
import cv2

from detect_emotion import detect_emotion
from detect_gaze import detect_gaze
from detect_head import detect_head_movement
from detect_objects import detect_objects

app = Flask(__name__)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# A route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# This function will serve the webcam video feed
def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame with detection functions
        emotion = detect_emotion(frame)
        # gaze = detect_gaze(frame)
        # head = detect_head_movement(frame)
        frame = detect_objects(frame)

        # Add text to the frame
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(frame, f"Gaze: {gaze}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(frame, f"Head: {head}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert the frame to JPEG for rendering in HTML
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route to stream the video feed
@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)


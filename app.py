# app.py
from flask import Flask, render_template, Response, redirect, url_for
import cv2
import numpy as np
from datetime import datetime
import threading
from deepface import DeepFace
import time

app = Flask(__name__)

# Global variables
camera = None
is_analyzing = False
expression_times = {
    'sad': 0,
    'surprise': 0,
    'neutral': 0
}
start_time = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def analyze_expression(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True)
        emotion = result[0]['dominant_emotion']
        
        if emotion in ['sad', 'surprise', 'neutral']:
            return emotion
        return 'neutral'
    except:
        return 'neutral'

def generate_frames():
    global is_analyzing, expression_times, start_time
    
    camera = get_camera()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        if is_analyzing:
            # Analyze expression and update times
            current_expression = analyze_expression(frame)
            if current_time := time.time():
                if start_time is None:
                    start_time = current_time
                
                if current_expression in expression_times:
                    expression_times[current_expression] += 1
            
            # Draw text on frame
            cv2.putText(frame, f"Expression: {current_expression}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
        
        # Convert frame to jpg
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start():
    global is_analyzing, expression_times, start_time
    is_analyzing = True
    expression_times = {'sad': 0, 'surprise': 0, 'neutral': 0}
    start_time = None
    return 'OK'

@app.route('/stop')
def stop():
    global is_analyzing
    is_analyzing = False
    release_camera()
    return redirect(url_for('results'))

@app.route('/results')
def results():
    total_frames = sum(expression_times.values())
    if total_frames > 0:
        results = {
            expression: (count / total_frames) * 100
            for expression, count in expression_times.items()
        }
    else:
        results = {expression: 0 for expression in expression_times}
    
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
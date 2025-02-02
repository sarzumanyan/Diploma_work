from flask import Flask, render_template, request, jsonify
import cv2
import sqlite3
from datetime import datetime
import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database import db, Detection
from detection.detect_emotion import detect_emotion
from detection.detect_head import detect_head_and_eye_movement
from detection.detect_objects import detect_objects



# Database setup
def setup_database():
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            emotion TEXT,
            head_movement TEXT,
            objects TEXT
        )
    """)
    conn.commit()
    return conn

# Save detection results to the database
def save_to_database(conn, emotion, head_movement, objects):
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO detections (timestamp, emotion, head_movement, objects)
        VALUES (?, ?, ?, ?)
    """, (timestamp, emotion, head_movement, ", ".join(objects)))
    conn.commit()

# Main function for real-time detection
def main():
    # Set up database
    conn = setup_database()

    # Initialize webcam feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Resize frame for better performance (optional)
        resized_frame = cv2.resize(frame, (640, 480))
        
        # Call detection functions
        emotion = detect_emotion(resized_frame)
        head_movement = detect_head_and_eye_movement(resized_frame)
        objects = detect_objects(resized_frame)

        # Display results on the frame
        display_text = f"Emotion: {emotion}\nHead Movement: {head_movement}\nObjects: {', '.join(objects)}"
        cv2.putText(resized_frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save results to database
        save_to_database(conn, emotion, head_movement, objects)

        # Show the frame with overlaid results
        cv2.imshow('Real-Time Detection', resized_frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main()

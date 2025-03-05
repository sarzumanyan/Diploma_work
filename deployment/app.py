from flask import Flask, render_template, request, jsonify
import threading
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
from detection.detect_gaze import detect_head_and_eye_movement
from detection.detect_objects import detect_objects

# Database setup
def setup_database():
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            firstname TEXT NOT NULL,
            lastname TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            emotion TEXT,
            head_movement TEXT,
            objects TEXT,
            timestamp TEXT,
            FOREIGN KEY (user_id) REFERENCES user (id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    return conn

def add_dummy_user(conn):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO user (firstname, lastname, email)
        VALUES (?, ?, ?)
    """, ("Sona", "Arzumanyan", "sona.arzumanyan@example.com"))
    conn.commit()

    cursor.execute("SELECT id FROM user WHERE email = ?", ("sona.arzumanyan@example.com",))
    return cursor.fetchone()[0]

def save_to_database(conn, user_id, emotion, head_movement, objects):
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO detection (user_id, emotion, head_movement, objects, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, emotion, head_movement, ", ".join(objects), timestamp))
    conn.commit()

def main():
    conn = setup_database()
    user_id = add_dummy_user(conn)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        resized_frame = cv2.resize(frame, (640, 480))

        # Threading setup
        emotion = None
        head_movement = None
        objects = None
        detection_done = threading.Event()  # Event to signal when all detections are done

        def detect_emotion_thread():
            nonlocal emotion
            emotion = detect_emotion(resized_frame)
            detection_done.set()  # Signal completion

        def detect_head_movement_thread():
            nonlocal head_movement
            head_movement = detect_head_and_eye_movement(resized_frame)
            detection_done.set()

        def detect_objects_thread():
            nonlocal objects
            objects = detect_objects(resized_frame)
            detection_done.set()

        # Creating threads
        emotion_thread = threading.Thread(target=detect_emotion_thread)
        head_movement_thread = threading.Thread(target=detect_head_movement_thread)
        objects_thread = threading.Thread(target=detect_objects_thread)

        # Starting threads
        emotion_thread.start()
        head_movement_thread.start()
        objects_thread.start()

        # Wait for all threads to complete
        emotion_thread.join()
        head_movement_thread.join()
        objects_thread.join()

        # Display results
        display_text = [
            f"Emotion: {emotion}",
            f"Head Movement: {head_movement}",
        ]

        x, y = 10, 30
        line_height = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (0, 255, 0)
        thickness = 2

        for line in display_text:
            cv2.putText(resized_frame, line, (x, y), font, font_scale, font_color, thickness)
            y += line_height  

        save_to_database(conn, user_id, emotion, head_movement, objects)
        cv2.imshow('Real-Time Detection', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main()

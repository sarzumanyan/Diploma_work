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
from detection.detect_gaze import detect_head_and_eye_movement
from detection.detect_objects import detect_objects

# Database setup
def setup_database():
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()

    # Create 'users' table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            firstname TEXT NOT NULL,
            lastname TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE
        )
    """)

    # Create 'exam' table with a foreign key to 'users'
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS exam (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            exam_question TEXT NOT NULL,
            emotion TEXT,
            head_movement TEXT,
            objects TEXT,
            timestamp TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    return conn

# Add a dummy user (only for testing purposes)
def add_dummy_user(conn):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO users (firstname, lastname, email)
        VALUES (?, ?, ?)
    """, ("Sona", "Arzumanyan", "sona.arzumanyan@example.com"))
    conn.commit()

    # Get the user_id of the dummy user
    cursor.execute("SELECT id FROM users WHERE email = ?", ("sona.arzumanyan@example.com",))
    return cursor.fetchone()[0]

# Save detection results to the database
def save_to_database(conn, user_id, exam_question, emotion, head_movement, objects):
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO exam (user_id, exam_question, emotion, head_movement, objects, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, exam_question, emotion, head_movement, ", ".join(objects), timestamp))
    conn.commit()

# Main function for real-time detection
def main():
    conn = setup_database()

    # Add a dummy user and get their user_id
    user_id = add_dummy_user(conn)

    # Use a dummy exam question
    exam_question = "How do you feel about this test?"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
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
        display_text = [
            f"Emotion: {emotion}",
            f"Head Movement: {head_movement}",
        ]

        # Set the starting position for the text
        x, y = 10, 30
        line_height = 30  # Adjust based on font size

        # Define font properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (0, 255, 0)
        thickness = 2

        # Render each line of text separately
        for line in display_text:
            cv2.putText(resized_frame, line, (x, y), font, font_scale, font_color, thickness)
            y += line_height  # Move to the next line position

        
        # Save results to database
        save_to_database(conn, user_id, exam_question, emotion, head_movement, objects)

        # Show the frame with overlaid results
        cv2.imshow('Real-Time Detection', resized_frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main()

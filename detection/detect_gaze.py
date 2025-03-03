import cv2
import mediapipe as mp
import numpy as np
import winsound

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh

def calculate_gaze_direction(landmarks, img_width, img_height):
    # Extract eye landmarks
    left_eye_indices = [33, 133, 160, 144, 153, 154, 155, 133]
    right_eye_indices = [362, 263, 387, 373, 380, 374, 382, 362]

    def get_eye_center(eye_points):
        x = np.mean([point[0] for point in eye_points])
        y = np.mean([point[1] for point in eye_points])
        return x, y

    # Convert landmarks to pixel coordinates
    left_eye = [(int(landmarks[i].x * img_width), int(landmarks[i].y * img_height)) for i in left_eye_indices]
    right_eye = [(int(landmarks[i].x * img_width), int(landmarks[i].y * img_height)) for i in right_eye_indices]

    left_center = get_eye_center(left_eye)
    right_center = get_eye_center(right_eye)

    # Calculate gaze ratios
    left_ratio = (left_center[0] - left_eye[0][0]) / (left_eye[3][0] - left_eye[0][0])
    right_ratio = (right_center[0] - right_eye[0][0]) / (right_eye[3][0] - right_eye[0][0])

    # Determine gaze direction
    if left_ratio > 0.6 and right_ratio > 0.6:
        return "Looking Right"
    elif left_ratio < 0.4 and right_ratio < 0.4:
        return "Looking Left"
    elif np.mean([left_center[1], right_center[1]]) < np.mean([left_eye[1][1], right_eye[1][1]]) - 5:
        return "Looking Up"
    elif np.mean([left_center[1], right_center[1]]) > np.mean([left_eye[4][1], right_eye[4][1]]) + 5:
        return "Looking Down"
    else:
        return "Looking Center"

def detect_head_and_eye_movement(image):
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        img_height, img_width, _ = image.shape
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Gaze detection
                gaze_direction = calculate_gaze_direction(face_landmarks.landmark, img_width, img_height)

                # Alert conditions
                if gaze_direction in ["Looking Down", "Looking Left", "Looking Right"]:
                    winsound.Beep(500, 200)  # Beep sound for alert
                    return f"ALERT: Gaze: {gaze_direction}"

                return f"Gaze: {gaze_direction}"
        return "No Face Detected"

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    frame_count = 0  # Initialize frame counter

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1  # Increment frame counter

        if frame_count % 3 != 0:
            # Skip the frame if it's not the 3rd frame
            continue

        # Detect head and eye movement
        status = detect_head_and_eye_movement(frame)

        # Display status
        cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if "ALERT" not in status else (0, 0, 255), 2)

        cv2.imshow("Head and Eye Movement Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

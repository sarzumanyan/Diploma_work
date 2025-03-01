import cv2
import dlib
import numpy as np
import winsound

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/ASUS/workspace/Polytech/Diploma/Diploma_work/models/head_movement_model/shape_predictor_68_face_landmarks.dat")

def detect_head_and_eye_movement(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Get the coordinates of the eyes
        left_eye_y = (landmarks.part(37).y + landmarks.part(41).y) / 2
        right_eye_y = (landmarks.part(43).y + landmarks.part(47).y) / 2
        face_top_y = face.top()
        face_bottom_y = face.bottom()
        face_height = face_bottom_y - face_top_y

        # Get nose position and face width
        nose = landmarks.part(30).x
        face_width = face.right() - face.left()

        # Check for head movement
        if nose < face.left() + face_width // 3:
            return "Not Looking to Screen (Left)"
        elif nose > face.right() - face_width // 3:
            return "Not Looking to Screen (Right)"
        else:
            # Check for eye movement if the head is centered
            if left_eye_y > face_top_y + face_height // 2 and right_eye_y > face_top_y + face_height // 2:
                return "Looking Down (Cheating)"
            else:
                return "Looking Center"

    return "No Face Detected"

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        head_status = detect_head_and_eye_movement(frame)

        # Display an alert if the person is not looking at the screen or is looking down
        if "Not Looking" in head_status:
            alert_text = "ALERT: Not Looking to Screen"
            cv2.putText(frame, alert_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red text for alert
            winsound.Beep(1000, 500)  # Beep sound at 1000 Hz for 500 ms
        elif "Looking Down" in head_status:
            alert_text = "ALERT: Looking Down (Possible Cheating)"
            cv2.putText(frame, alert_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red text for alert
            winsound.Beep(1000, 500)  # Beep sound at 1000 Hz for 500 ms
        else:
            cv2.putText(frame, f"Head: {head_status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Head and Eye Movement Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

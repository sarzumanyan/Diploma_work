import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/ASUS/workspace/Polytech/Diploma/Diploma_work/models/head_movement_model/shape_predictor_68_face_landmarks.dat")  # Update with correct path

def detect_head_movement(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        nose = landmarks.part(30).x
        face_width = face.right() - face.left()

        if nose < face.left() + face_width // 3:
            return "Looking Left"
        elif nose > face.right() - face_width // 3:
            return "Looking Right"
        else:
            return "Looking Center"

    return "No Face"

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        head_status = detect_head_movement(frame)
        cv2.putText(frame, f"Head: {head_status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Head Movement Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

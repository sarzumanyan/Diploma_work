from deepface import DeepFace
import cv2

def detect_emotion(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion
    except:
        return "Unknown"

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

        emotion = detect_emotion(frame)
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

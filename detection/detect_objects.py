from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # Pre-trained YOLO model

def detect_objects(frame):
    detected_objects = []
    results = model(frame)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            if label in ["cell phone", "book", "laptop"]:
                detected_objects.append(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return detected_objects

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_objects = detect_objects(frame)

        # Combine the detected objects into a single string for display
        display_text = f"Objects: {', '.join(detected_objects)}"
        cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

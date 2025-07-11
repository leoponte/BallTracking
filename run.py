import cv2
import numpy as np
from ultralytics import YOLO

def load_yolo_model():
    """
    Load the YOLO model using the ultralytics library.
    """
    model = YOLO("yolov8n.pt")  # Replace with the appropriate YOLO model file
    return model

def process_frame(frame, model):
    """
    Process a single frame to detect objects using the YOLO model.
    """
    results = model(frame)  # Perform inference on the frame
    ball_coordinates = None

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Class ID
            confidence = box.conf[0]   # Confidence score
            if confidence > 0.5:  # Filter detections with confidence > 0.5
                center_x = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                center_y = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                label = f"Class {class_id}"  # Replace with actual class names if available
                color = (0, 255, 0) if class_id == 32 else (255, 0, 0)  # Green for ball, blue for others
                cv2.circle(frame, (center_x, center_y), 10, color, 2)
                cv2.putText(frame, label, (center_x + 10, center_y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if class_id == 32:  # Assuming '32' is the class ID for the ball
                    ball_coordinates = (center_x, center_y)
    return ball_coordinates

def track_ball(video_url):
    """
    Track the ball in a soccer livestream using YOLO.
    """
    import pdb;
    pdb.set_trace()
    model = load_yolo_model()
    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        print("Error: Unable to open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or error reading frame.")
            break

        ball_coordinates = process_frame(frame, model)
        if ball_coordinates:
            cv2.circle(frame, ball_coordinates, 10, (0, 255, 0), 2)
            cv2.putText(frame, "Ball", (ball_coordinates[0] + 10, ball_coordinates[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Ball Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Main function to execute the ball tracking.
    """
    video_url = input("Enter the soccer livestream URL: ")
    track_ball(video_url)

if __name__ == "__main__":
    main()
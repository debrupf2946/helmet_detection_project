import cv2
import math
from ultralytics import YOLO

def detect_and_display_realtime(model_path, confidence_threshold=0.1):
    """
    Perform real-time object detection using Ultralytics YOLO and display results using OpenCV.

    :param model_path: Path to the YOLO model.
    :param confidence_threshold: Minimum confidence threshold for displaying detections.
    """
    # Load YOLO model
    model = YOLO(model_path)

    # Open webcam/video stream
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam. Replace with video file path for pre-recorded video.

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # Class names from the YOLO model
    class_labels = model.names

    print("Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the camera.")
            break

        # Run YOLO model inference
        results = model.predict(frame, conf=confidence_threshold)

        # Iterate over results and draw bounding boxes with confidence and class labels
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Calculate width and height of the bounding box
                w, h = x2 - x1, y2 - y1

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

                # Extract confidence and class ID
                conf = math.ceil((box.conf[0] * 100)) / 100  # Round confidence to 2 decimal places
                cls = int(box.cls[0])  # Class ID

                # Add label and confidence above the bounding box
                if conf > confidence_threshold:  # Display only if above confidence threshold
                    label = f"{class_labels[cls]} {conf:.2f}"
                    cv2.putText(
                        frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                    )

        # Display the resulting frame
        cv2.imshow("YOLO Real-Time Object Detection", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

# Example Usage
detect_and_display_realtime(
    model_path="best_TT.pt",  # Replace with your YOLO model path
    confidence_threshold=0.1
)

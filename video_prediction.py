import cv2
import math
from ultralytics import YOLO

def detect_and_display_cv2(video_path, model_path, confidence_threshold=0.1):
    """
    Perform object detection on a video using Ultralytics YOLO and display results in real-time.

    :param video_path: Path to the input video file.
    :param model_path: Path to the YOLO model.
    :param confidence_threshold: Minimum confidence threshold for displaying detections.
    """
    # Load YOLO model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video at {video_path}")
        return

    # Process video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Run YOLO model inference on the current frame
        results = model.predict(frame, conf=confidence_threshold)

        # Class names from the YOLO model
        class_labels = model.names

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

        # Display the frame with bounding boxes and labels
        cv2.imshow("YOLO Object Detection", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Example Usage
detect_and_display_cv2(
    video_path="he2.mp4",  # Replace with the path to your video file
    model_path="best_TT.pt",
    confidence_threshold=0.1
)

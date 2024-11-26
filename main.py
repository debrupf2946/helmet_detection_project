import cv2
import math
from ultralytics import YOLO

def detect_and_display_cv2(image_path, model_path, confidence_threshold=0.1):
    """
    Perform object detection using Ultralytics YOLO and display results using OpenCV's imshow.

    :param image_path: Path to the input image.
    :param model_path: Path to the YOLO model.
    :param confidence_threshold: Minimum confidence threshold for displaying detections.
    """
    # Load YOLO model
    model = YOLO(model_path)

    # Read the input image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    # Run YOLO model inference
    results = model.predict(img, conf=confidence_threshold)

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
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            # Extract confidence and class ID
            conf = math.ceil((box.conf[0] * 100)) / 100  # Round confidence to 2 decimal places
            cls = int(box.cls[0])  # Class ID

            # Add label and confidence above the bounding box
            if conf > confidence_threshold:  # Display only if above confidence threshold
                label = f"{class_labels[cls]} {conf:.2f}"
                cv2.putText(
                    img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                )

    # Display the image with bounding boxes and labels using OpenCV
    cv2.imshow("YOLO Object Detection", img)

    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example Usage
detect_and_display_cv2(
    image_path="/Users/debruppaul/Downloads/helmet_detection/test/images/BikesHelmets101_png.rf.b741dfc6f95a416770fe9089a4d2cb33.jpg",
    model_path="best_TT.pt",
    confidence_threshold=0.1
)


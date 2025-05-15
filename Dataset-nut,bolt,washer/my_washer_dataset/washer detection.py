import cv2
from ultralytics import YOLO

def live_washer_detection():
    """
    Real-time detection of washers using a trained YOLOv8 model and webcam.
    Press 'Esc' or 'q' to exit.
    """

    # 1. Load your trained YOLOv8 model
    #    Replace 'washer_detection_model.pt' with the actual path to your model file
    model = YOLO("C:/Users/rajes/Downloads/mechanical parts (Bolt,Nut, Washer,Pin) detection/mechanical parts (Bolt,Nut, Washer,Pin) detection/my_washer_dataset/best.pt")

    # 2. Open the webcam (device=0 for default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("✅ Starting real-time washer detection. Press 'Esc' or 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from camera. Exiting...")
            break

        # 3. Run YOLO inference on the current frame
        #    Adjust 'conf=0.5' if you want a different confidence threshold
        results = model.predict(frame, conf=0.5)

        # 4. YOLOv8 returns a list of result objects for each image
        #    We typically just look at results[0]
        if len(results) > 0:
            detections = results[0].boxes  # bounding boxes, classes, confidences, etc.

            # 5. Draw bounding boxes & labels on each detection
            for box in detections:
                # [x1, y1, x2, y2]
                coords = box.xyxy[0]
                x1, y1, x2, y2 = [int(val) for val in coords]

                cls_id = int(box.cls[0])   # predicted class index
                conf = float(box.conf[0])  # confidence score

                # If your model has just one class, you can label everything as 'washer'
                # If multiple classes exist, fetch the name from model.names
                class_name = model.names[cls_id] if model.names else "washer"

                # Draw a bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add text label: class + confidence
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 6. Show the annotated frame
        cv2.imshow("Washer Detection (YOLOv8)", frame)

        # 7. Break the loop if 'q' or 'Esc' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # 27 is Esc
            break

    # 8. Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_washer_detection()

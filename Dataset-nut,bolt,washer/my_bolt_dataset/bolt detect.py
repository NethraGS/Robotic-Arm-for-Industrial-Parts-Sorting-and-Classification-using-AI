import cv2
from ultralytics import YOLO

def live_bolt_detection():
    """
    Real-time detection of bolts using a trained YOLOv8 model and webcam.
    Press 'Esc' or 'q' to exit.
    """

    # 1. Load your trained YOLOv8 model
    #    Replace 'bolt_detection_model.pt' with the path/name to your model file
    model = YOLO("E:/Projects on board/mechanical parts (Bolt,Nut, Washer,Pin) detection/my_bolt_dataset/bolt_detection_model.pt")

    # 2. Open the webcam (0 = default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("✅ Starting real-time bolt detection. Press 'Esc' or 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from camera. Exiting...")
            break

        # 3. Run YOLO inference on the current frame
        #    Adjust the conf threshold as needed (defaulted to 0.5)
        results = model.predict(frame, conf=0.5)

        # 4. Extract bounding boxes (if any). YOLOv8 returns a list of results for each image.
        if len(results) > 0:
            detections = results[0].boxes

            for box in detections:
                # box.xyxy -> [x1, y1, x2, y2]
                coords = box.xyxy[0]
                x1, y1, x2, y2 = [int(val) for val in coords]

                # box.cls -> predicted class index, box.conf -> confidence
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # If you have multiple classes, get the class name from model.names
                class_name = model.names[cls_id] if model.names else "bolt"

                # 5. Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{class_name} {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # 6. Show the annotated frame
        cv2.imshow("Bolt Detection (YOLOv8)", frame)

        # 7. Exit on 'Esc' or 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    # 8. Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_bolt_detection()

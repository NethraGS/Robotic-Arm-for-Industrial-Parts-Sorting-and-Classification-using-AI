import cv2
from ultralytics import YOLO

def live_nut_detection():
    """
    Real-time detection of the class named 'Nut Detection - v1 Nut Dataset' ONLY,
    with a minimum confidence threshold of 0.85.
    Press 'Esc' or 'q' to exit.
    """

    # 1. Load your YOLOv8 model
    model = YOLO("E:/Projects on board/mechanical parts (Bolt,Nut, Washer,Pin) detection/my_nut_dataset/nut_detection_model.pt")

    # 2. Identify the class index for "Nut Detection - v1 Nut Dataset"
    nut_index = None
    for class_idx, class_name in model.names.items():
        # Compare case-insensitively
        if class_name.lower() == "nut detection - v1 nut dataset":
            nut_index = class_idx
            break

    if nut_index is None:
        print("❌ Could not find a class named 'Nut Detection - v1 Nut Dataset' in the model. Check model.names!")
        return

    # 3. Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("✅ Starting real-time detection: Only 'Nut Detection - v1 Nut Dataset' at >= 85% confidence will be labeled.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame capture failed, exiting...")
            break

        # 4. Run YOLO inference, specifying confidence >= 0.85 and classes=[nut_index]
        results = model.predict(
            source=frame,
            conf=0.55,                 # Minimum confidence
            classes=[nut_index],       # Only return bounding boxes for that specific class
        )

        # 5. Draw bounding boxes for all returned detections
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                conf = float(box.conf[0])

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Label
                label_text = f"Nut Detection - v1 Nut Dataset {conf:.2f}"
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 6. Show the annotated frame
        cv2.imshow("Nut Detection (>= 85% Conf)", frame)

        # 7. Exit on 'q' or 'Esc'
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_nut_detection()

import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 1. Verify a sample image from the train set
train_images_path = "E:/Projects on board/mechanical parts (Bolt,Nut, Washer,Pin) detection/my_nut_dataset/train/images"
image_files = [f for f in os.listdir(train_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if len(image_files) == 0:
    print("❌ No training images found!")
else:
    sample_image = os.path.join(train_images_path, image_files[0])
    img = cv2.imread(sample_image)
    if img is None:
        print("❌ Error reading sample image.")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Sample Nut Image")
        plt.axis("off")
        plt.show()

# 2. Train YOLOv8 model
model = YOLO("E:/Projects on board/mechanical parts (Bolt,Nut, Washer,Pin) detection/my_nut_dataset/yolov8n.pt")  # or yolov8s.pt, etc.
model.train(
    data="E:/Projects on board/mechanical parts (Bolt,Nut, Washer,Pin) detection/my_nut_dataset/data.yaml",  # data.yaml (with nc=1, names=['nut'])
    epochs=10,
    batch=16,
    imgsz=640
)

# 3. Save trained model
model.save("nut_detection_model.pt")
print("✅ Training complete. Model is saved as nut_detection_model.pt")

# 4. Validate or test on a separate set
val_results = model.val()
print("Validation Results:", val_results)

# 5. Inference function
def detect_nut_in_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: File {image_path} not found.")
        return
    
    trained_model = YOLO("nut_detection_model.pt")
    results = trained_model(image_path)
    results.show()

# Example usage:
# detect_nut_in_image("D:/path_to/my_nut_dataset/test/images/test_nut.jpg")

# 6. Export (if needed)
# model.export(format="onnx")
# model.export(format="tflite")
# model.export(format="pt")

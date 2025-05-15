import os
from ultralytics import YOLO

def train_bolt_model():
    """
    Train a YOLOv8n model on your custom bolt dataset.
    Adjust hyperparameters (epochs, batch, imgsz) to meet your needs.
    """

    # 1. Load the YOLOv8 nano model (pretrained on COCO)
    model = YOLO("E:/Projects on board/mechanical parts (Bolt,Nut, Washer,Pin) detection/my_nut_dataset/yolov8n.pt")  # or "yolov8s.pt" etc., but user specifically wants yolov8n

    # 2. Train the model on your bolt dataset
    #    data.yaml must point to the correct train/val images
    results = model.train(
        data="E:/Projects on board/mechanical parts (Bolt,Nut, Washer,Pin) detection/my_bolt_dataset/data.yaml",   # e.g. path to your data.yaml
        epochs=20,       # number of training epochs
        batch=16,        # batch size
        imgsz=640,       # image size
        name="bolt_model_v1"  # runs/<name>/ for training logs & weights
    )

    # 3. Save the final model weights
    #    The best and last weights are saved automatically in the "runs" folder.
    #    You can also explicitly save or rename them if you want:
    model.save("bolt_detection_model.pt")
    print("✅ Training complete. Model saved as bolt_detection_model.pt")

if __name__ == "__main__":
    train_bolt_model()

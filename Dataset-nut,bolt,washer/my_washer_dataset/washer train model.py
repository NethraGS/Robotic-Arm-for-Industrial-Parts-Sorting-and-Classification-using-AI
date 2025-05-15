import os
from ultralytics import YOLO

def train_washer_model():
    """
    Train a YOLOv8n model on your custom washer dataset.
    Adjust hyperparameters (epochs, batch, imgsz) as needed.
    """

    # 1. Load the YOLOv8 nano checkpoint (pretrained on COCO)
    model = YOLO()  # You can use 'yolov8s.pt' etc. if desired

from ultralytics import YOLO

def train_washer_only():
    model = YOLO("E:/Projects on board/mechanical parts (Bolt,Nut, Washer,Pin) detection/my_nut_dataset/yolov8n.pt")
    model.train(
        data="E:/Projects on board/mechanical parts (Bolt,Nut, Washer,Pin) detection/my_washer_dataset/data.yaml",
        epochs=20,
        batch=16,
        imgsz=640,
        # This line is crucial:
        classes=[2],  # only train on class index=2 => 'washer'
        name="washer_only_model"
    )

if __name__ == "__main__":
    train_washer_only()

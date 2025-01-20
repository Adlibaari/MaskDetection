from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    
    # Train the model
    results = model.train(data="data.yaml", epochs=100, imgsz=1280)

if __name__ == "__main__":
    main()

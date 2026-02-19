from ultralytics import YOLO

def main():
    #modelo base
    model = YOLO("yolov8s.pt") 

    model.train(
        data="dataset/data.yaml", 
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,  #GPU
        name="modelo_v2",
        patience=15 
    )

if __name__ == "__main__":
    main()
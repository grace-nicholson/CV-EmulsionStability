from ultralytics import YOLO

def main():
    #model = YOLO("yolo26m.pt")   # pre-trained
    model = YOLO("yolo26m.yaml") # randomised weights

    model.train(
        data="./data_2/data.yaml",
        epochs=600,
        imgsz=640,
        batch=8,          # explicit; try 8, drop to 4 if it still OOMs
        device="mps",
        workers=0,
        cache=False,      # or "disk" if I/O is the bottleneck
        seed=0,
        patience=100,
        close_mosaic=10,
        project="runs",
        name="yolo26m_yaml"
    )

if __name__ == "__main__":
    main()
import yaml
from ultralytics import YOLO

def main():
    # Requires best_hyperparameters.yaml to be in ./
    with open("best_hyperparameters_mpt.yaml") as f:
        hyp = yaml.safe_load(f)

    model = YOLO("yolo26m.pt")

    model.train(
        data="./data_2/data.yaml",
        epochs=500,
        imgsz=640,
        batch=16,
        device=0,
        workers=6,
        cache="ram",
        seed=0,
        patience=100,
        optimizer="AdamW",
        project="runs",
        name="yolo26m_pt",
        **hyp,
    )

if __name__ == "__main__":
    main()

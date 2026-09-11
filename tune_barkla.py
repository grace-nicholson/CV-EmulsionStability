from ultralytics import YOLO

def main():
  model = YOLO("yolo26m.pt")
  #model = YOLO("/users/sggnich2/scratch/yolo26m_yaml-7/weights/best.pt")

  model.tune(
      data="/mnt/scratch/users/sggnich2/data_2/data.yaml",
      epochs=30,
      imgsz=640,
      iterations=100,
      optimizer="AdamW",
      plots=False,
      batch=16,
      device=0,
      workers=6,
      cache="ram",
      project="runs",
      name="yolo26mpt_tune",
  )

if __name__ == "__main__":
    main()

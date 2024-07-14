
!nvidia-smi
pip install ultralytics==8.0.26
from ultralytics import YOLO

!yolo task=detect mode=train model=yolov8l.pt data=../content/drive/MyDrive/Datasets/tube_data/data.yaml epochs=20 imgsz=660
// here in data , u need to add 'content' also but no need to add that in path which is in data.yaml file 
// in free version we can train upto only 20 epochs


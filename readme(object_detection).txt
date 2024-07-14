Upload your dataset in google drive , mount your google drive in google colab

Change your data.yaml file:
add path on the Top:
path: ../drive/MyDrive/Datasets/tube_data


In google colab , select runtime ->change runtime type ->T4 GPU

Code to run on google colab line by line

!nvidia-smi
pip install ultralytics==8.0.26
from ultralytics import YOLO

!yolo task=detect mode=train model=yolov8l.pt data=../content/drive/MyDrive/Datasets/tube_data/data.yaml epochs=20 imgsz=660
// here in data , u need to add 'content' also but no need to add that in path which is in data.yaml file 
// in free version we can train upto only 20 epochs

After completing the training , we will get a 'runs' folder
runs -> detect -> train -> weights 
Download the best.pt file . That is your trained model.

Now on pycharm, make a project folder:

install this libraries in pycharm's termial:
// version is important

cvzone==1.5.6
ultralytics==8.0.26
hydra-core>=1.2.0
matplotlib>=3.2.2
numpy>=1.18.5
opencv-python==4.5.4.60
Pillow>=7.1.2
PyYAML>=5.3.1
requests>=2.23.0
scipy>=1.4.1
torch>=1.7.0
torchvision>=0.8.1
tqdm>=4.64.0
filterpy==1.4.5
scikit-image==0.19.3
lap==0.4.0

Run this code in pycharm:

from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 480)
cap.set(4, 480)
#cap = cv2.VideoCapture("tubev.mp4")  # For Video

model = YOLO("best.pt")

classNames = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90']
myColor = (0, 0, 255)
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            # w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,y1-20)))

    cv2.imshow("Image", img)
    cv2.waitKey(1)





























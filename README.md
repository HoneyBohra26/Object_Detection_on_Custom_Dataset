This project involves training a YOLOv8 model for object detection using Google Colab, followed by deployment in a local environment using PyCharm. The process begins with uploading the dataset to Google Drive and mounting it in Google Colab. The dataset path is specified in a configuration file (`data.yaml`) with the line `path: ../drive/MyDrive/Datasets/tube_data` added at the top. Google Colab is then configured to use a T4 GPU for faster training.

In Colab, the environment is set up by checking the GPU with `!nvidia-smi` and installing the `ultralytics` package version 8.0.26. The YOLOv8 model is then imported from `ultralytics`. The training process is initiated using a specific command that specifies the model, dataset path, number of epochs (20), and image size (660). Upon completion, the training results are saved in a 'runs' directory. The trained model, `best.pt`, is downloaded from the `weights` folder within the `runs` directory.

For deployment, a new project folder is created in PyCharm, and specific versions of the required libraries are installed. These include `cvzone`, `ultralytics`, `hydra-core`, `matplotlib`, `numpy`, `opencv-python`, `Pillow`, `PyYAML`, `requests`, `scipy`, `torch`, `torchvision`, `tqdm`, `filterpy`, `scikit-image`, and `lap`. The PyCharm environment is prepared for real-time object detection.

The deployment code involves using the `ultralytics` library to load the trained YOLO model and capture video feed either from a webcam or a video file. The video feed is processed frame by frame, and the model detects objects in each frame, drawing bounding boxes and displaying confidence scores on the video feed. The detected objects' coordinates are printed to the console, and the resulting video is displayed using OpenCV functions.

This project demonstrates the end-to-end workflow of training an object detection model in the cloud and deploying it locally, integrating various machine learning, computer vision, and software development techniques.

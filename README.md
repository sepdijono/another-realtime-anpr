# another-realtime-anpr

Link to: [another-realtime-anpr-ocr](https://github.com/sepdijono/another-realtime-anpr-ocr)

We regret to inform you that we only provide guides for Linux users, specifically Ubuntu.

**Automatic Number Plate Recognition Using Yolov3**

**Detector** : _yolov.weights_

Important : This repository contains a two-part project for license plate recognition using YOLOv3 and FastAPI. You must create two environment one for "Easy OCR Server" and one for "License Plate Recognition Program"

**Part 1 - Easy OCR Server**
The "Easy OCR" server is an API program built using FastAPI. It has an endpoint at "/ocr" that can be used to obtain vehicle license plates from an image.

Environment:
The following environment is required for running the Easy OCR Server:
  - Python 3.9 or higher (syarat minimum FastAPI)
  - easyocr
  - FastAPI 
  - uvicorn
  
To install the required packages, run the following command:

```pip install -r api-requirements.txt``

If you in trouble with easyocr dependencies I suggest to install easyocr first before others, here is how to install  easyocr :
```pip install easyocr```

Usage:
```uvicorn ocrs_fast:app --reload```

**Part 2 - License Plate Recognition Program** - You are here !!!
The main program reads from a camera or video file and iterates to display the detected license plate (using YOLOv3) in a box/frame on the screen.

Environment
The following environment is required for running the License Plate Recognition Program:
  - Python 3.9 or higher
  - OpenCV
  - YOLOv3
To install the required packages, run the following command:
 
```pip install -r app-requirements.txt```

Usage :

Before that you want to create folder name "foto-plat-nomor" inside the project, this folder will contains all detected number plates. 

Using camera:

```python main.py -icam0```
You can change device by changing number behind -icam0 for camera number 1, -icam1 for camera number 2 etc

Using video file:

```python main.py -i/home/pyy/Videos/fps10/out7.mp4```

To maximize program performance in accessing videos, use fps and resolution that are compatible with your hardware capabilities. If the video appears choppy when running the application, resize the video accordingly. Here an example to resize the video using ffmpeg:

```ffmpeg -i mobil_7.mp4 -vf scale=480:-1 -filter fps=10 /home/pyy/Videos/fps10/out7.mp4```

Meaning : convert mobil_7.mp4 by scaling width 480px, maintain aspect ratio, set fps to 10 and save it as : /home/pyy/Videos/fps10/out7.mp4


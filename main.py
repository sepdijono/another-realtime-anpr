import asyncio
import base64
import os
import pickle
import sys
import time
from datetime import datetime as dt
import cv2
import pyshine as ps
import requests
import ujson

from utils import *

# darknet-yolov3
model_cfg_path = os.path.join('.', 'models', 'cfg', 'darknet-yolov3.cfg')
model_weights_path = os.path.join('.', 'models', 'weights', 'model.weights')

# yolov3-tiny
model_tiny_cfg_path = os.path.join('.', 'models', 'cfg', 'yolov3-tiny.cfg')
model_tiny_weights_path = os.path.join('.', 'models', 'weights', 'yolov3-tiny.weights')

ITS_TINY = False


async def append_box(frame, detections):
    frame = np.array(frame)
    H, W, _ = frame.shape
    bboxes = []
    bbox_confidences = []
    class_ids = []
    scores = []
    l = 8
    t = 4
    rgb = (255, 0, 255)
    # rgb = (0, 200, 0)
    for detection in detections:
        # [x1, x2, x3, x4, x5, x6, ..., x85]
        bbox = detection[:4]

        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]
        bbox_confidence = detection[4]
        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        bboxes.append(bbox)
        bbox_confidences.append(bbox_confidence)
        class_ids.append(class_id)
        scores.append(score)

    bboxes, class_ids, scores = await NMS(bboxes, class_ids, bbox_confidences)
    await asyncio.create_task(apply_NMS(frame, bboxes, rgb, t, l))


async def im2json(im):
    """Convert a Numpy array to JSON string"""
    imdata = pickle.dumps(im)
    jstr = ujson.dumps({"image": base64.b64encode(imdata).decode('ascii')})
    return jstr


async def request_ocr(img=cv2.imread('image.png')):
    try:
        input_json = await im2json(img)
        r = requests.post('http://127.0.0.1:8000/ocr', json=input_json)
        return r
    except Exception as e:
        return "ERROR"


async def apply_NMS(frame, bboxes, rgb, t, l):
    async def draw_frame(frm):
        cv2.imshow("Result", frm)

    async def save_license_num(license_plt):
        if license_plt.size != 0:
            try:
                foto_plat_nomor_path = os.path.join('.', 'foto-plat-nomor')
                filename = str(dt.now()).replace(" ", "")
                image = Image.fromarray(license_plt)
                filename = f'{foto_plat_nomor_path}/{filename}.jpg'
                # Enable this code below to activate saving number plates screenshot
                # image.save(filename)
            except Exception as e:
                print(f'Error : {e}')

    async def draw_text(img, text,
                        font=cv2.FONT_HERSHEY_PLAIN,
                        pos=(0, 0),
                        font_scale=0.5,
                        background_RGB=rgb,
                        text_RGB=(255, 250, 250)
                        ):
        if img.size == 0:
            return
        x, y = pos
        ps.putBText(img, text, text_offset_x=x, text_offset_y=y, font=font,
                          vspace=5, hspace=10, font_scale=font_scale,
                          background_RGB=background_RGB, text_RGB=text_RGB)
        return

    frame = np.array(frame)
    for bbox_, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox
        license_plate = frame[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)),
                        :].copy()

        x = int(xc - (w / 2))
        y = int(yc - (h / 2))
        x1, y1 = x + w, y + h
        cv2.rectangle(frame, (x, y), (x + w, y + h), rgb, 0)
        # Top left x,y
        cv2.line(frame, (x, y), (x + l, y), rgb, t)
        cv2.line(frame, (x, y), (x, y + l), rgb, t)
        # Top right x1,y
        cv2.line(frame, (x1, y), (x1 - l, y), rgb, t)
        cv2.line(frame, (x1, y), (x1, y + l), rgb, t)
        # Bottom left x,y1
        cv2.line(frame, (x, y1), (x + l, y1), rgb, t)
        cv2.line(frame, (x, y1), (x, y1 - l), rgb, t)
        # Bottom right x1,y1
        cv2.line(frame, (x1, y1), (x1 - l, y1), rgb, t)
        cv2.line(frame, (x1, y1), (x1, y1 - l), rgb, t)
        # draw stick
        cv2.line(frame, (x, y), (x , y - 7), rgb, 1)
        cv2.line(frame, (x, y - 7), (x + 12, y - 12), rgb, 1)
        r = await request_ocr(license_plate)
        if r.content is not None:
            try:
                data = r.content.decode('utf-8')
                data = ujson.loads(data)
                await asyncio.create_task(draw_text(frame, f"{data['prefix']} {data['nk']} {data['suffix']}",
                                                    font_scale=1.5,
                                                    pos=(x + 22, y - 25),
                                                    background_RGB=rgb,
                                                    text_RGB=(255, 250, 250)))
            except Exception as e:
                # print(e)
                pass
        # Enable this code below to activate saving number plates screenshot
        # await asyncio.create_task(save_license_num(license_plate))
    await asyncio.create_task(draw_frame(frame))


async def main():
    async def print_fps(frm, fps_txt):
        cv2.putText(frm, fps_txt, (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 1)

    async def dnn_blob(frm):
        return cv2.dnn.blobFromImage(frm, 1 / 255, (416, 416), (0, 0, 0), True)

    if len(sys.argv) < 2:
        print("Parameter error!")
        sys.exit(1)

    l = 8
    t = 4
    fps_start_time = 0
    rgb = (255, 0, 255)
    cap = None
    for i in sys.argv:
        if i.strip().lower()[0:2] == '-i':
            if i.strip().lower()[2:5] == 'cam':
                if len(i[2:]) > 3:
                    try:
                        target_cam = int(i[5:])
                        cap = cv2.VideoCapture(target_cam)
                        if cap.isOpened():
                            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, -1)
                            cap.set(cv2.CAP_PROP_FPS, 10)
                        else:
                            print("Parameter [cam] error!")
                            sys.exit(-1)
                    except Exception as e:
                        print(e)
                        sys.exit(-1)
                else:
                    print("Parameter error!")
                    sys.exit(-1)
            else:
                target_file = i[2:].strip()
                cap = cv2.VideoCapture(f"{target_file}")

    # load models
    if ITS_TINY:
        net = cv2.dnn.readNetFromDarknet(model_tiny_cfg_path, model_tiny_weights_path)
    else:
        net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    while True:
        success, frame = cap.read()
        if success:
            try:
                ori_size = frame.shape[:2]
                ratio = (ori_size[0] * 1.5) / ori_size[1]
                new_size = (int(ori_size[1] * ratio), int(ori_size[0] * ratio))
                frame = cv2.resize(frame, new_size)
            except Exception as e:
                print(e)
            fps_end_time = time.time()
            time_diff = fps_end_time - fps_start_time
            fps = 1 / time_diff
            fps_start_time = fps_end_time
            fps_text = f"FPS :{fps:.2f}"
            asyncio.create_task(print_fps(frame, fps_text))

            H, W, _ = frame.shape

            # convert image
            blob = await dnn_blob(frame)

            # get detections
            net.setInput(blob)
            detections = await get_outputs(net)
            await asyncio.create_task(append_box(frame, detections))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())

from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import base64
import requests


start_time=time.time()
oldclass=''

cap = cv2.VideoCapture("ven/video/Fending Off The Enemy _ Elephant_ King of the Kalahari.mp4")
cap.set(3, 1280)
cap.set(4, 720)
model = YOLO("ven/finalYolo/best_L-model2.pt")
classnames = ['Boar','Cheetah','Deer','Elephant','Hyena','Jaguar','Leopard','Lion','Panda','Snake','Tiger','Wolf']


flag =True
def send_telegram_msg(message):
    TOKEN = "6263063793:AAGo22xawSSiz9ciYnhFGPRa0c1MGPPrKfE"
    chat_id = "5341914866"
    # message = "Hello,World!"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"

    response = requests.get(url)

    return response.json()



while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if conf >0.85:
              cvzone.putTextRect(img, f'{classnames[cls]} {conf}', (max(0, x1), max(35, y1)),scale=1,thickness=1)

              """a=classnames[cls]
              while True:
                    send_telegram_msg(f'{classnames[cls]}' + " Please, Go to safe place" )
                        break
                    if a!=classnames[cls]:
                    break"""

              if flag :#and oldclass!= classnames[cls] :
                send_telegram_msg(f'{classnames[cls]}' + " Please, Go to safe place")
                oldclass = classnames[cls]
                flag = False

              cu_time = time.time()
              e_time = cu_time - start_time
              if e_time >= 10:
                    send_telegram_msg(f'{classnames[cls]}' + " Please, Go to safe place")
                    start_time = cu_time



    cv2.imshow('Animal', img)
    cv2.waitKey(1)


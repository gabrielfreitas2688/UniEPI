from ultralytics import YOLO
import cv2
import cvzone

cap = cv2.VideoCapture("testes/epi-3.mp4")  #Carregar o video

model = YOLO("best.pt") # carregar o modelo

classNames = ['com capacete', 'sem capacete', 'sem colete', 'pessoa', 'com colete']

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
            #Confiança
            conf = int(box.conf[0]*100)
            #Extrair Classe
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if conf>60:
                if currentClass =='sem capacete' or currentClass =='sem colete':
                    myColor = (0, 0,255) #cor vermelha
                elif currentClass =='com capacete' or currentClass =='com colete':
                    myColor =(0,255,0) #cor verde
                else:
                    myColor = (255, 0, 0)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',(x1,y1-5), scale=1, thickness=1,colorB=myColor,colorT=(255,255,255),colorR=myColor)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

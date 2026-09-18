import cv2
from ultralytics import YOLO
from securitytrainer import embedding, same
vid = cv2.VideoCapture(0)
model = YOLO("yolo26n.pt")

faceembed = embedding(cv2.imread(r"C:\Users\hembo\Pictures\Screenshots\Screenshot 2026-06-14 131730.png"))

while vid.isOpened():
    success, frame = vid.read()
    if not success: break

    results = model(frame, verbose = False)

    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                height = y2-y1 
                heady2 = y1+int(height*.55)
                headcrop = frame[y1:heady2, x1:x2]

                if headcrop.size == 0: continue

                frameperson = embedding(headcrop)

                if same(frameperson, faceembed):
                    cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                    font = cv2.FONT_HERSHEY_SIMPLEX 
                    cv2.putText(frame, "Aryan", (x1,y1-10), font, .9, (0,255,0), 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,0),2)
                    font = cv2.FONT_HERSHEY_SIMPLEX 
                    cv2.putText(frame, "UNKNOWN", (x1,y1-10), font, .9, (0,0,0), 2, cv2.LINE_AA)

    cv2.imshow("SECURITY", frame)
    if cv2.waitKey(1) == ord("q"): break 
vid.release()
cv2.destroyAllWindows()
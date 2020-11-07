import sys
import cv2


xmlpath = sys.path[0] + '/xml/haarcascade_frontalface_alt2.xml'

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Fail to open video')
    sys.exit()

# Object of CascadeClassifier
# Load xml
classifier = cv2.CascadeClassifier(xmlpath)

if classifier.empty():
    print('Fail to load xml file')
    sys.exit()

while True:
    
    retval, frame = cap.read()

    if not retval:
        break
    
    tm = cv2.TickMeter()
    tm.start()
    
    faces = classifier.detectMultiScale(frame, scaleFactor=1.3, minSize=(200, 200))
    
    tm.stop()
    ms = round(tm.getAvgTimeMilli(), 4)
    
    for (x, y, w, h) in faces:

		# Draw rectangle of face and put text
        cv2.rectangle(frame, (x, y, w, h), (255, 0, 0), 2)
        cv2.putText(frame, str(ms)+'ms', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        msg = '====> detect face{}x{} time:{}'.format(w, h, ms)
        print(msg)

    cv2.imshow('frame', frame)
    
	# ESC
    if cv2.waitKey(100) == 27:
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import time

print('OpenCV version:', cv2.__version__)

cap = cv2.VideoCapture(0)
print('Camera opened:', cap.isOpened())
ret, frame = cap.read()
print('Read frame:', ret, 'shape:', None if frame is None else frame.shape)

try:
    cv2.imshow('Test Imshow', frame)
    print('Called imshow(); waiting for 2 seconds...')
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    print('Destroyed windows')
except Exception as e:
    print('imshow error:', e)

cap.release()
print('Done')

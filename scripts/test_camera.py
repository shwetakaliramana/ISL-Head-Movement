import cv2
import time

CAMERAS = range(4)
print('Testing camera indices:', list(CAMERAS))
for idx in CAMERAS:
    print('\n--- testing index', idx, '---')
    # Try with DirectShow first then fallback
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_DSHOW):
        try:
            cap = cv2.VideoCapture(idx, backend)
        except Exception as e:
            print('  open error backend', backend, e)
            cap = None
        if cap is None or not cap.isOpened():
            print('  not opened with backend', backend)
            continue
        ok, frame = cap.read()
        print('  first read ok=', ok, 'frame_shape=', None if frame is None else frame.shape)
        # try reading a few frames
        n_ok = 0
        for i in range(5):
            ok, frame = cap.read()
            if ok and frame is not None:
                n_ok += 1
        print('  successful reads (next 5):', n_ok)
        # save a sample frame if available
        if frame is not None:
            path = f'tmp_camera_{idx}.jpg'
            cv2.imwrite(path, frame)
            print('  saved sample to', path)
        cap.release()
        break

print('\nDone.\nNote: If no cameras open, try increasing indices or closing other apps using the camera.')

import datetime
import time

import cv2
import numpy as np

# from astropy.io import fits
from PIL import Image as PILImage
from PIL import ImageTk

from asi_tracker_camera import ASITrackerCameraManager
from utils.camera_utils import scale_histo

camera = ASITrackerCameraManager()

camera.init_camera()

entryExp = 1e-3  # 1e-6
camera.set_exposure_time(int(float(entryExp) * 1e6))
camera.set_gain(253)

print(camera.camera_info.keys())
exit()
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output-1.mp4", fourcc, 20.0, (1920, 1080))

while True:
    img = camera.read()
    print(img.shape)
    # img = np.moveaxis(img, 0, 2)
    # img = img.astype(np.uint8)
    # img = (img-np.min(img))/(np.max(img)-np.min(img))*255
    # img = scale_histo(img)
    print(f"min: {np.min(img)}, max: {np.max(img)}")
    
    # img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # img = cv2.flip(img, 0)
    out.write(img)
    
    cv2.imshow("img", img)
    # cv2.imwrite("img.jpg", img)
    if cv2.waitKey(1) == ord("q"):
        break

out.release()
cv2.destroyAllWindows()

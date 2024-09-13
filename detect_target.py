import time
from pathlib import Path

import cv2 as cv
import numpy as np

from asi_tracker_camera import ASITrackerCameraManager
from utils.camera_utils import scale_histo

live_video = False
file_path = Path("./test.avi")
if live_video:
    video_stream = ASITrackerCameraManager()

    video_stream.init_camera()

    entryExp = 1e-3  # 1e-6
    video_stream.set_exposure_time(int(float(entryExp) * 1e6))
    video_stream.set_gain(253)
else:
    file_path = Path("./output.avi")

    video_stream = cv.VideoCapture(str(file_path))

fourcc = cv.VideoWriter_fourcc(*"MP4V")
out = cv.VideoWriter(file_path.with_name(file_path.stem + "_circles.mp4"), fourcc, 20.0, (1920, 1080))

seconds_to_skip = 0
if not live_video:
    fps = video_stream.get(cv.CAP_PROP_FPS)
else:
    fps = 20
frame_idx = 0
while True:
    if not live_video:
        _ret, frame = video_stream.read()
    else:
        frame = video_stream.read()
        frame = scale_histo(frame)
        frame = frame.astype(np.uint8)

    if frame_idx < seconds_to_skip * fps:
        frame_idx += 1
        continue
    if frame is None:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    # gray = cv.medianBlur(gray, 5)
    # bin = cv.adaptiveThreshold(
    #     gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
    # )
    ret,bin = cv.threshold(gray,150,255,cv.THRESH_BINARY)
    n_pix = sum(sum(bin))
    print(n_pix)
    # if n_pix > 0:
    #     print(f"Found target! {n_pix} pixels")
    # bin = cv.medianBlur(bin, 3)
    # contours, heirs = cv.findContours(bin, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    # try:
    #     heirs = heirs[0]
    # except:
    #     heirs = []
    # print(len(frame))
    # # for frame in frames:
    # #     cv2.im

    rows = bin.shape[0]
    circles = cv.HoughCircles(
        bin,
        cv.HOUGH_GRADIENT,
        1,
        rows / 8,
        param1=40,
        param2=10,
        minRadius=0,
        maxRadius=0,
    )
    # for cnt, heir in zip(contours, heirs):
    #     _, _, _, outer_i = heir
    #     if outer_i >= 0:
    #         continue
    #     x, y, w, h = cv.boundingRect(cnt)
    #     if not (16 <= h <= 64  and w <= 1.2*h):
    #         continue
    #     pad = max(h-w, 0)
    #     x, w = x - (pad // 2), w + pad
    #     cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))

    #     bin_roi = bin[y:,x:][:h,:w]

    #     m = bin_roi != 0
    #     if not 0.1 < m.mean() < 0.4:
    #         continue

    # circles = cv.HoughCircles(
    #     bin, cv.HOUGH_GRADIENT, dp=1 / (frame.shape[0] * frame.shape[1]), minDist=200
    # )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # for i in circles[0, :]:
        #     center = (i[0], i[1])
        #     # circle center
        #     cv.circle(frame, center, 1, (0, 100, 100), 3)
        #     # circle outline
        #     radius = i[2]
        #     cv.circle(frame, center, radius, (255, 0, 255), 3)
        i = circles[0,0]
        center = (i[0], i[1])
        # circle center
        cv.circle(frame, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv.circle(frame, center, radius, (255, 0, 255), 3)

        print("Found circles!")
        # time.sleep(0.2)

        # s = 1.5*float(h)/SZ
        # m = cv.moments(bin_roi)
        # c1 = np.float32([m['m10'], m['m01']]) / m['m00']
        # c0 = np.float32([SZ/2, SZ/2])
        # t = c1 - s*c0
        # A = np.zeros((2, 3), np.float32)
        # A[:,:2] = np.eye(2)*s
        # A[:,2] = t
        # bin_norm = cv.warpAffine(bin_roi, A, (SZ, SZ), flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR)
        # bin_norm = deskew(bin_norm)
        # if x+w+SZ < frame.shape[1] and y+SZ < frame.shape[0]:
        #     frame[y:,x+w:][:SZ, :SZ] = bin_norm[...,np.newaxis]

        # sample = preprocess_hog([bin_norm])
        # digit = model.predict(sample)[1].ravel()
        # cv.putText(frame, '%d'%digit, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (200, 0, 0), thickness = 1)

    # time.sleep(0.1)
    cv.imshow("frame", gray)
    out.write(frame)
    # cv.imshow('bin', bin)
    if cv.waitKey(1) == ord("q"):
        break
    frame_idx += 1

out.release()
cv.destroyAllWindows()

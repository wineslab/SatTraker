import argparse
import time

import cv2
import numpy as np

try:
    from asi_tracker_camera import ASITrackerCameraManager
    asi_tracker_camera = True
except ImportError as e:
    asi_tracker_camera = False
    print(e)

from utils.camera_utils import detect_target

focal_length_mm = 200
pixel_size_um = 2.9
image_scale_arcs = 206 * pixel_size_um / focal_length_mm
image_scale_deg = image_scale_arcs / 3600

antenna_bw_deg = .3 # deg
antenna_bw_ps = antenna_bw_deg / image_scale_deg  # px

def test_detect_target(file_path=None):
    if file_path is None and asi_tracker_camera:
        camera = ASITrackerCameraManager()
        camera.init_camera()

        camera.set_exposure_time(1e3)
        camera.set_gain(253)

        videostream = camera
        skip_frames = 0
    else:
        videostream = cv2.VideoCapture(file_path)
        fps = videostream.get(cv2.CAP_PROP_FPS)

        skip_seconds = 19
        skip_frames = fps * skip_seconds

    frame_width = videostream.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = videostream.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_center = (int(frame_width // 2), int(frame_height // 2))

    frame_size_deg = frame_width * image_scale_deg
    print(f"Frame size: {frame_size_deg:.2f} deg")

    frame_count = 0
    while True:
        _, frame = videostream.read()
        frame_count += 1
        if frame_count < skip_frames:
            continue
        frame = cv2.flip(frame, 1)
        if frame is None:
            break
        target_found, center, radius = detect_target(frame, target_type="circle")
        if target_found:
            cv2.circle(frame, center, radius, (255, 100, 100), 3)
            cv2.line(
                frame,
                (frame_center[0], frame_center[1]),
                center,
                (255, 255, 255),
                3,
            )
            targetX, targetY = center

            print(f"Target X: {targetX:.2f}, Target Y: {targetY:.2f}")

            move_x_px = frame_center[0] - targetX
            move_y_px = frame_center[1] - targetY

            move_x_deg = move_x_px * image_scale_deg
            move_y_deg = move_y_px * image_scale_deg

            cv2.putText(
                frame,
                f"Move x: {move_x_deg:.3f} deg ({move_x_px} ps), Move y: {move_y_deg:.3f} deg ({move_y_px} ps)",
                (0,frame.shape[0] -80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Target diameter: {2*radius*image_scale_deg:.3f} deg",
                (0, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # crosshair
            cv2.circle(frame, frame_center, int(antenna_bw_ps), (100, 100, 255), 3)
            cv2.line(
                frame,
                (frame_center[0] - int(antenna_bw_ps), frame_center[1]),
                (frame_center[0] + int(antenna_bw_ps), frame_center[1]),
                (100, 100, 255),
                3,
            )
            cv2.line(
                frame,
                (frame_center[0], frame_center[1] - int(antenna_bw_ps)),
                (frame_center[0], frame_center[1] + int(antenna_bw_ps)),
                (100, 100, 255),
                3,
            )
            print(f"Move x: {move_x_deg:.2f}, Move y: {move_y_deg:.2f}")
            time.sleep(1)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    videostream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default=None)
    args = parser.parse_args()
    
    file_path = args.file_path
    
    test_detect_target(file_path=file_path)

import datetime as dt
import time

import cv2
import numpy as np
from pytz import timezone
from skyfield import almanac
from skyfield.api import N, W, load, wgs84

from asi_tracker_camera import ASITrackerCameraManager
from RotaryClient import RotaryClient
from utils.camera_utils import detect_target, get_x_y, scale_histo


class Target:
    def __init__(self, name) -> None:

        self.name = name
        self.timescale = load.timescale()
        self.eph = load("de440s.bsp")
        self.earth = self.eph["earth"]

        if name == "Sun" or name == "Moon":  # used for polar alignment
            self.target = self.eph[name]
            self.type = "body"
        else:  # we assume it is a satellite
            self.stations_url = "http://celestrak.org/NORAD/elements/stations.txt"
            satellites = load.tle_file(self.stations_url)
            self.target = [s for s in satellites if name in s.name][0]
            self.type = "satellite"

    def set_location(self, lat, lon, alt):
        self.location = wgs84.latlon(lat, lon, alt)
        return self.location
    
    def get_azimuth_elevation(self, time):
        self.location_geocentric = self.earth + self.location

        diff_vector = self.target - self.location_geocentric
        topocentric = diff_vector.at(time)
        alt, az, distance = topocentric.altaz()
        return alt.degrees, az.degrees

    def is_visible(self):
        time = self.timescale.now()
        alt, az = self.get_azimuth_elevation(time)
        return alt > 0

class Telescope:
    def __init__(self) -> None:
        self.mount = RotaryClient()

        self.camera = ASITrackerCameraManager()
        self.camera.init_camera()

        self.exposure_time = 1e-3  # 1e-6
        self.camera.set_exposure_time(int(float(self.exposure_time) * 1e6))
        self.camera.set_gain(253)

        # self.gps = mavRtk()
        # self.location = self.gps.get_location()
        self.location = (42.337408, -71.087856, 40)

        self.Tracking = False
        self.Connected = False

        self.mount.connect("/dev/ttyUSB0")

        self.Slewing = False

        self.rtk_geo_azimuth = 0
        self.mount_rel_azimuth = 0
        self.azimuth_prediction = []
        self.elevation_prediction = []
        self.azimuth_estimation = []  # stores the list of estimated azimuths
        self.elevation_estimation = []  # stores the list of estimated elevations
        self.Azimuth = 0
        self.Altitude = 0

        self.time = time.time()
        
        self.focal_length_mm = 200
        
        self.image_scale_arcs = 206 * self.camera.pixel_size_um / self.focal_length_mm
        self.image_scale_deg = self.image_scale_arcs / 3600
        
        print(f"Image scale: {self.image_scale_deg} deg/pixel")

    def _set_target(self, target_name):
        self.target = Target(target_name)
        self.target.set_location(*self.location)

    def calibrate(self):
        # check if it is day or night
        target = Target("Sun")
        target.set_location(*self.location)
        if target.is_visible():
            self._set_target("Sun")
        else:
            self._set_target("Moon")

        self.target.set_location(*self.location)

        self.scan_sector()

    def scan_sector(
        self,
        azimuth_step=0.5,
        elevation_step=0.5,
        azimuth_max=10,
        elevation_max=10,
        clockwise=True,
        save_video=True,
    ) -> None:
        stop = False
        self.current_fps = self.camera.get_fps()
        self.current_exposure = self.camera.get_exposure_time()

        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(f"calibration_rough_{dt.datetime.now()}.mp4", fourcc, self.current_fps, (1920, 1080))

        if clockwise:
            orientation = -1
        else:
            orientation = 1

        for _ in np.arange(0, azimuth_max, azimuth_step):
            for _ in np.arange(0, elevation_max, elevation_step):
                t00 = time.time()
                frame = self.camera.read()

                cv2.imshow("img", frame)
                t01 = time.time()
                target_found, _ = detect_target(frame)
                t02 = time.time()
                if target_found:
                    if save_video:
                        out.release()
                        cv2.destroyAllWindows()
                    self.go_to_target(save_video=save_video)
                    if save_video:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out = cv2.VideoWriter(f"calibration_rough_{dt.datetime.now()}.mp4", fourcc, self.current_fps, (1920, 1080))
                if cv2.waitKey(1) == ord("q"):
                    stop = True
                    break
                t10 = time.time()
                self.mount.MoveAxis(1, elevation_step)
                self.mount.Altitude += elevation_step
                t11 = time.time()
                print(
                    f"Time taken: {(t11-t00):.2f} s (find_target: {(t01-t00):.2f} s, MoveAxis: {(t11-t10):.2f} s)"
                )
                if 1 / self.current_fps > ((t11 - t00) + self.current_exposure):
                    time.sleep(
                        1 / self.current_fps - ((t11 - t00) + self.current_exposure)
                    )
            if stop:
                break
            self.mount.MoveAxis(1, -elevation_max)
            self.mount.MoveAxis(0, orientation*azimuth_step)
            self.mount.Azimuth += orientation*azimuth_step
            if save_video:
                out.write(frame)

            cv2.imshow("img", frame)
            if cv2.waitKey(1) == ord("q"):
                if save_video:
                    out.release()
                cv2.destroyAllWindows()
                return

    def go_to_target(self, save_video=True):
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(f"calibration_precise_{dt.datetime.now()}.mp4", fourcc, self.current_fps, (1920, 1080))
        err = np.inf
        roibox = []
        imageroi = []
        while err>1e-3:
            frame = self.camera.read()
            t0 = time.time()
            # if imageroi == []:
            #     imageroi = frame.copy()
            # roibox, imageroi, foundtarget = get_x_y(
            #     frame,
            #     roibox_y1=0,
            #     roibox_x1=0,
            #     roiwidth=frame.shape[1] / 2,
            #     roiheight=frame.shape[0] / 2,
            #     # roibox,
            #     # imageroi,
            #     trackingtype="Bright",
            #     minbright=150,
            # )

            # roiheight, roiwidth = imageroi.shape[:2]
            # targetX = roibox[0][0] + (roiwidth / 2)
            # targetY = roibox[0][1] + (roiheight / 2)
            
            _, center = detect_target(frame)
            if center is None:
                return
            targetX, targetY = center
            
            print(f"Target X: {targetX}, Target Y: {targetY}")

            move_x_px = frame.shape[1] / 2 - targetX
            move_y_px = frame.shape[0] / 2 - targetY
            
            move_x_deg = move_x_px * self.image_scale_deg
            move_y_deg = move_y_px * self.image_scale_deg
            
            print(f"Move x: {move_x_deg}, Move y: {move_y_deg}")

            self.mount.MoveAxis(0, move_x_deg)
            self.mount.MoveAxis(1, -move_y_deg)

            err = np.sqrt(move_x_deg**2 + move_y_deg**2)
            cv2.imshow("img", frame)
            if cv2.waitKey(1) == ord("q"):
                out.release()
                cv2.destroyAllWindows()
                return
            t1 = time.time()
            if 1 / self.current_fps > ((t1 - t0) + self.current_exposure):
                time.sleep(
                    1 / self.current_fps - ((t1 - t0) + self.current_exposure)
                )

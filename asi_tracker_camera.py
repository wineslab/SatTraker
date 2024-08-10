import os
import sys
import time

import numpy as np
import zwoasi as asi


class ASITrackerCameraManager():
    def __init__(self):
        env_filename = os.getenv("ZWO_ASI_LIB")
        asi.init(env_filename)
        self.camera = None
        self.camera_id = 0
        self.camera_info = None
        self.cameras_found = None
        self.camera_controls = None

    def init_camera(self):
        self.connect_camera()

    def connect_camera(self):
        num_cameras = asi.get_num_cameras()
        if num_cameras == 0:
            print('No cameras found')
            sys.exit(0)

        cameras_found = asi.list_cameras()  # Models names of the connected cameras

        if num_cameras == 1:
            print('Found one camera: %s' % cameras_found[0])
        else:
            print('Found %d cameras' % num_cameras)
            for n in range(num_cameras):
                print('    %d: %s' % (n, cameras_found[n]))
            # TO DO: allow user to select a camera
            print('Using #%d: %s' % (self.camera_id, cameras_found[self.camera_id]))

        self.camera = asi.Camera(self.camera_id)
        self.camera_info = self.camera.get_camera_property()

        print("Camera controls:")
        self.camera_controls = self.camera.get_controls()
        # for cn in sorted(self.camera_controls.keys()):
        #     print("    %s:" % cn)
        #     for k in sorted(self.camera_controls[cn].keys()):
        #         print("        %s: %s" % (k, repr(self.camera_controls[cn][k])))
        self.camera.start_video_capture()
        if self.camera_info['IsColorCam']:
            print('Capturing a single color frame')
            self.camera.set_image_type(asi.ASI_IMG_RGB24)
        else:
            print('Capturing a single 8-bit mono frame')
            self.camera.set_image_type(asi.ASI_IMG_RAW8)
        self.configure_camera()

    def set_exposure_time(self, exposure_time):
        self.camera.set_control_value(asi.ASI_EXPOSURE, exposure_time)

    def get_image(self):
        img = self.camera.capture_video_frame()

        return np.moveaxis(img, 2, 0)

    def configure_camera(self):
        # Use minimum USB bandwidth permitted
        self.camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, self.camera.get_controls()['BandWidth']['MinValue'])

        # Set some sensible defaults. They will need adjusting depending upon
        # the sensitivity, lens and lighting conditions used.
        self.camera.disable_dark_subtract()

        # Can autoexposure be used?
        if 'Exposure' in self.camera_controls and self.camera_controls['Exposure']['IsAutoSupported']:
            print('Enabling auto-exposure mode')
            self.camera.set_control_value(asi.ASI_EXPOSURE,
                                    self.camera_controls['Exposure']['DefaultValue'],
                                    auto=True)

            if 'Gain' in self.camera_controls and self.camera_controls['Gain']['IsAutoSupported']:
                print('Enabling automatic gain setting')
                self.camera.set_control_value(
                    asi.ASI_GAIN, self.camera_controls["Gain"]["DefaultValue"], auto=True
                )

            # Keep max gain to the default but allow exposure to be increased to its maximum value if necessary
            self.camera.set_control_value(
                self.camera_controls["AutoExpMaxExpMS"]["ControlType"],
                self.camera_controls["AutoExpMaxExpMS"]["MaxValue"],
            )

            print('Waiting for auto-exposure to compute correct settings ...')
            sleep_interval = 0.100
            df_last = None
            gain_last = None
            exposure_last = None
            matches = 0
            while True:
                time.sleep(sleep_interval)
                settings = self.camera.get_control_values()
                df = self.camera.get_dropped_frames()
                gain = settings['Gain']
                exposure = settings['Exposure']
                if df != df_last:
                    print('   Gain {gain:d}  Exposure: {exposure:f} Dropped frames: {df:d}'
                        .format(gain=settings['Gain'],
                                exposure=settings['Exposure'],
                                df=df))
                    if gain == gain_last and exposure == exposure_last:
                        matches += 1
                    else:
                        matches = 0
                    if matches >= 5:
                        break
                    df_last = df
                    gain_last = gain
                    exposure_last = exposure

        # Set the timeout, units are ms
        timeout = (self.camera.get_control_value(asi.ASI_EXPOSURE)[0] / 1000) * 2 + 500
        self.camera.default_timeout = timeout

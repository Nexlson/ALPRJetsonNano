import logging # For logging message in console
import threading 
import subprocess
import numpy as np
import cv2 

""" Camera only run onboard camera """

def open_cam(width, height):
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, you might need to add
        # 'flip-method=2' into gst_str below.
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1920, height=(int)1080, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def grab_img(cam):
    # grab new image and put to global 'img_handle' on sub thread
    while cam.thread_running:
        _, cam.img_handle = cam.cap.read()
        if cam.img_handle is None:
            #logging.warning('Camera: cap.read() returns None...')
            break
    cam.thread_running = False

class Camera():
    def __init__(self): 
        self.is_opened = False
        self.cap = None
        self.img_handle = None
        self.thread_running = False
        self.img_width = 640
        self.img_height = 480
        self.thread = None
        self._open()

    def _open(self):
        if self.cap is not None:
            raise RuntimeError("camera is already opened!")
        logging.info("Camera: using Jetson onboard camera")
        self.cap = open_cam(640, 480) # width and height
        self._start()
    
    def isOpened(self):
        return self.is_opened
    
    def _start(self):
        if not self.cap.isOpened():
            logging.warning("Camera: starting while cap is not opened!")
            return
        _, self.img_handle = self.cap.read()
        if self.img_handle is None:
            logging.warning('Camera: cap.read() returns no image!')
            self.is_opened = False
            return
        self.is_opened = True
        self.img_height, self.img_width, _ = self.img_handle.shape
        assert not self.thread_running 
        self.thread_running = True
        self.thread = threading.Thread(target=grab_img, args=(self,))
        self.thread.start()
    
    def _stop(self):
        # Turn thread running to false if true
        if self.thread_running:
            self.thread_running = False
    
    def read(self):
        # Return none if camera not opened
        if not self.is_opened:
            return None
        return self.img_handle
    
    def release(self):
        self._stop()
        try:
            self.cap.release()
        except:
            pass
        self.is_opened = False
    
    def __del__(self):
        self.release()
        

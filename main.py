import os
import time
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
from utils.display import open_window, set_display, show_fps
from utils.camera import Camera
from utils.className import alprClassNames
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from utils import alpr

def main():
    # initialize camera class
    cam = Camera()
    if not cam.is_opened():
        raise SystemExit('EROR: failed to open camera!')
    cls_dict = alprClassNames()
    
    # Yolo dimensions (416x416)
    yolo_dim = 416
    h = w = int(yolo_dim)

    # Initialized model and tools
    cwd = os.getcwd()
    model_yolo = str(cwd) + '/weights/yolov4-tiny-416.trt'
    model_crnn = str(cwd) + '/weights/crnn.pth'
    trt_yolo = TrtYOLO(model_yolo, (h,w), category_num=1) # category number is number of classes
    crnn = alpr.AutoLPR(decoder='bestPath', normalise=True)
    crnn.load(crnn_path=model_crnn)
    open_window(WINDOW_NAME, TITLE, cam.img_width, cam.img_height)
    vis = BBoxVisualization(cls_dict)

    # Loop and detect
    full_scrn = False
    fps = 0.0 
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        # Detect car plate
        boxes confs, clss = trt_yolo.detect(img, conf_th=0.5)
        # Crop and preprocess car plate 
        cropped = vis.crop_plate(img, boxes, confs, clss)
        # Recognize car plate
        lp_plate = ''
        fileLocate = str(cwd) + '/detections/detection1.jpg'
        if os.path.exists(fileLocate):
           lp_plate = lpr.predict(fileLocate)

        # Draw boxes and fps
        img = vis.draw_bboxes(img, boxes, confs, clss, lp=lp_plate)
        img = show_fps(img, fps)

        # Show image
        cv2.imshow(WINDOW_NAME, img)

        # Calculate fps
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc

        # Exit key
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break

    # Release capture and destroy all windows
    cam.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    main()
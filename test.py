import os

cwd = os.getcwd()
model = str(cwd) + '/weights/carplate_detector.trt'
print(model)
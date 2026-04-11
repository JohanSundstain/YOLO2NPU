import sys
import os
import numpy as np

from preprocess_yolov8 import preprocess_yolov8_detection

#_abs_path = os.path.join(os.getcwd(), ".")
sys.path.append(os.getcwd())
from utils.image_process import imagenet_preprocess_method1
from utils.tools import get_file_list


# Get a list of images from the provided path
images_path = "coco128/images/train2017"
images_list = get_file_list(images_path)
datas = []
for image_path in images_list:
    input = preprocess_yolov8_detection(image_path, target_size=(640, 640))
    datas.append(input)
# concat the datas and save calib dataset
datas = np.concatenate(datas, axis=0)
datas = datas.transpose(0, 3, 1, 2)  
print(datas.shape)
np.save("datasets/calib_data.npy", datas)
print("Generate calib dataset succ.")
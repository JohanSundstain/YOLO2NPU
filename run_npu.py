import cv2
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm

from utils.tools import get_file_list
from utils.image_process import preprocess_object_detect_method1
from utils.object_detect_postprocess import postprocess_yolo, xywh2xyxy
from utils.draw import draw_coco as draw
from utils.NOE_Engine import EngineInfer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        default="test_data",
        help="path to the image file",
    )
    parser.add_argument(
        "--model_path",
        default="yolov8n.cix",
        help="path to the quant model file",
    )
    parser.add_argument(
        "--output_dir", default="./output", help="path to the result output"
    )
    parser.add_argument(
        "--conf_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--nms_thr", type=float, default=0.45, help="NMS threshould to the object."
    )
    parser.add_argument(
        "--benchmark",
        default=False,
        help="benchmark on COCO val dataset.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    benchmark = args.benchmark
    os.makedirs(args.output_dir, exist_ok=True)
    # Get list of images from the specified path
    image_list = get_file_list(args.image_path)
    model = EngineInfer(args.model_path)
    if benchmark:
        from utils.evaluate.coco_metric import COCO_Metric

        save_pred_json = "pred_yolov8_n_npu.json"
        coco_metric = COCO_Metric(saved_json_path=save_pred_json)
        # get all test image path
        image_list = coco_metric.get_image_ids()
    else:
        image_list = get_file_list(args.image_path)

    for img_name in tqdm(image_list):
        if benchmark:
            # image info and image path
            img_id = img_name
            img_name = coco_metric.get_image_path(img_id)

        # Preprocess the image for inference
        src_shape, new_shape, show_image, data = preprocess_object_detect_method1(
            img_name, target_size=(640, 640), mode="BGR"
        )

        # Run inference and get predictions
        pred = model.forward(data.astype(np.float32))[0]
        pred = np.reshape(pred, (84, 8400))
        pred = np.transpose(pred, (1, 0))

        # bboxes, conf, class_id
        results = postprocess_yolo(pred, args.conf_thr, args.nms_thr)
        if len(results) == 0:
            if benchmark:
                continue
            else:
                output_path = os.path.join(args.output_dir,"npu_" + os.path.basename(img_name))
                # Save the resulting image to the output directory
                cv2.imwrite(output_path, show_image)
                continue

        bbox_xywh = results[:, :4]
        bbox_xyxy = xywh2xyxy(bbox_xywh)
        x_scale = src_shape[1] / new_shape[1]
        y_scale = src_shape[0] / new_shape[0]
        bbox_xyxy *= (x_scale, y_scale, x_scale, y_scale)

        if benchmark:
            coco_metric.append_bboxes(img_id, bbox_xyxy, results[:, 5], results[:, 4])
        else:
            ret_img = draw(show_image, bbox_xyxy, results[:, 5], results[:, 4])
            output_path = os.path.join(args.output_dir, "npu_" + os.path.basename(img_name))
            # Save the resulting image to the output directory
            cv2.imwrite(output_path, ret_img)

    if benchmark:
        coco_metric.saved_json()
        coco_metric.evaluate()

    model.clean()
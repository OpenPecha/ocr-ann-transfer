import json
import logging
import os

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.special import expit

from ocr_ann_transfer.monlam_ocr.utils import (
    generate_line_images,
    pad_image,
    patch_image,
    resize_image,
    unpatch_image,
    unpatch_prediction,
)


class LineDetection:
    """
    Handles layout detection
    Args:
        - config_file: json file with the following parameters:
            - onnx model file
            - image input width and height
            - classes

    """

    def __init__(
        self,
        config_file: str,
        binarize_output: bool = False,
        mode: str = "cpu",
    ) -> None:
        self._config_file = config_file
        self._onnx_model_file = None
        self._patch_size = 512
        self._binarize_output = binarize_output
        self._inference = None
        # add other Execution Providers if applicable, see: https://onnxruntime.ai/docs/execution-providers
        self.mode = mode

        if self.mode == "cuda":
            execution_providers = ["CUDAExecutionProvider"]
        else:
            execution_providers = ["CPUExecutionProvider"]

        self.execution_providers = execution_providers

        self._init()

    def _init(self) -> None:
        model_dir = os.path.dirname(self._config_file)

        _file = open(self._config_file, encoding="utf-8")
        json_content = json.loads(_file.read())

        if self.mode == "cuda":
            self._onnx_model_file = f"{model_dir}/{json_content['gpu-model']}"
        else:
            self._onnx_model_file = f"{model_dir}/{json_content['cpu-model']}"

        self._patch_size = json_content["patch_size"]

        if self._onnx_model_file is not None:
            try:
                self._inference = ort.InferenceSession(
                    self._onnx_model_file, providers=self.execution_providers
                )
                self.can_run = True
            except Exception as error:
                print(f"Failed to load model file: {error}")
                self.can_run = False
        else:
            self.can_run = False

        logging.debug(f"Line Inference -> Init(): {self.can_run}")

    def predict(
        self,
        original_image: np.array,
        unpatch_type: int = 0,
        class_threshold: float = 0.8,
    ) -> np.array:
        image, _ = resize_image(original_image)
        padded_img, (pad_x, pad_y) = pad_image(image, self._patch_size)
        image_patches, y_splits = patch_image(padded_img, self._patch_size)
        image_batch = np.array(image_patches)
        image_batch = image_batch.astype(np.float32)
        image_batch /= 255.0

        image_batch = np.transpose(image_batch, axes=[0, 3, 1, 2])  # make B x C x H xW

        ort_batch = ort.OrtValue.ortvalue_from_numpy(image_batch)
        prediction = self._inference.run_with_ort_values(
            ["output"], {"input": ort_batch}
        )
        prediction = prediction[0].numpy()
        prediction = np.squeeze(prediction, axis=1)

        prediction = expit(prediction)
        prediction = np.where(prediction > class_threshold, 1.0, 0.0)
        pred_list = [prediction[x, :, :] for x in range(prediction.shape[0])]

        if unpatch_type == 0:
            unpatched_image = unpatch_image(image, pred_list)
        else:
            unpatched_image = unpatch_prediction(image, y_splits)

        cropped_image = unpatched_image[
            : unpatched_image.shape[0] - pad_y, : unpatched_image.shape[1] - pad_x
        ]
        back_sized_image = cv2.resize(
            cropped_image, (original_image.shape[1], original_image.shape[0])
        )
        back_sized_image = back_sized_image.astype(np.uint8)

        # TODO: remove this into a post-processing module
        line_images, sorted_contours, bbox, peaks = generate_line_images(
            original_image, back_sized_image
        )
        return back_sized_image, line_images, sorted_contours, bbox, peaks

    def predict_batch(self, image_list: list[np.array]):
        pass


if __name__ == "__main__":
    config_file = "/home/tenzin3/ocr-ann-transfer/model/line_model_config.json"
    line_detect_obj = LineDetection(config_file)

    image_file = "/home/tenzin3/ocr-ann-transfer/images_dir/W2PD17382-I1KG81275/I1KG812750004.jpg"

    image = Image.open(image_file)
    image_array = np.array(image)
    res = line_detect_obj.predict(image_array)
    line_images = res[1]

    for i, line_image in enumerate(line_images):
        # Load an image file directly
        image = Image.fromarray(line_image)
        # Display the image
        image.save(f"line_image_{i:05}.jpg")

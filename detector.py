""" Contains a filter for detection using OPENVINO """

import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin


class Detector:
    """Filter to perform car detection on given image"""

    # pylint: disable=too-many-instance-attributes
    # Ten is reasonable in this case.
    def __init__(self):
        # Plugin initialization for specified device and load extensions library if specified
        plugin = IEPlugin(device="CPU")
        plugin.add_cpu_extension('openvino/lib/libcpu_extension_avx2.so')

        # Read detector IR
        detector_bin = 'openvino/model/person-vehicle-bike-detection-crossroad-0078.bin'
        detector_xml = 'openvino/model/person-vehicle-bike-detection-crossroad-0078.xml'
        detector_net = IENetwork.from_ir(model=detector_xml, weights=detector_bin)

        self.d_in = next(iter(detector_net.inputs))
        self.d_out = next(iter(detector_net.outputs))
        detector_net.batch_size = 1

        # Read and pre-process input images
        self.d_n, self.d_c, self.d_h, self.d_w = detector_net.inputs[self.d_in]
        self.d_images = np.ndarray(shape=(self.d_n, self.d_c, self.d_h, self.d_w))

        # Loading models to the plugin
        self.d_exec_net = plugin.load(network=detector_net)

    def find_all(self, frame):
        # pylint: disable=too-many-locals
        height, width = frame.shape[:-1]

        if height * self.d_w > self.d_h * width:
            new_width = self.d_w * height / self.d_h
            new_height = height
            border_size = int((new_width - width) / 2)
            frame = cv2.copyMakeBorder(frame, top=0, bottom=0, left=border_size, right=border_size,
                                       borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif height * self.d_w < self.d_h * width:
            new_width = width
            new_height = self.d_h * width / self.d_w
            border_size = int((new_height - height) / 2)
            frame = cv2.copyMakeBorder(frame, top=border_size, bottom=border_size, left=0, right=0,
                                       borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            new_width = width
            new_height = height

        if (new_width, new_height) != (self.d_w, self.d_h):
            d_frame = cv2.resize(frame, (self.d_w, self.d_h))
        else:
            d_frame = frame

        # Change data layout from HWC to CHW
        self.d_images[0] = d_frame.transpose((2, 0, 1))

        result = []
        raw_res = self.d_exec_net.infer(inputs={self.d_in: self.d_images})[self.d_out][0][0]
        for _, label, confidence, left, top, right, bottom in raw_res:
            left = max(0, int(left * new_width - (new_width - width) / 2))
            right = min(int(right * new_width - (new_width - width) / 2), width - 1)

            top = max(0, int(top * new_height - (new_height - height) / 2))
            bottom = min(int(bottom * new_height - (new_height - height) / 2), height - 1)
            if confidence >= 0.2:
                result.append({
                    'xmin': left,
                    'ymin': top,
                    'xmax': right,
                    'ymax': bottom,
                    'label': label,
                    'confidence': confidence
                })
        return result


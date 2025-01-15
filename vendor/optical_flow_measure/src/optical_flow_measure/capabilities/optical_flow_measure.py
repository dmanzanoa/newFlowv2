from typing import Dict, Optional, Tuple, Union, List

from highlighter.agent.capabilities import Capability, StreamEvent
from highlighter.client import DataFile
from highlighter.core import LabeledUUID


from typing import Dict, Optional, Tuple, Union, List
import os
import cv2
import torch
import numpy as np
import csv
import pandas as pd
import torchvision.transforms as T
import sys
from neuflowv2 import NeuFlowV2

__all__ = ["OpticalFlowMeasure"]

ATTRIBUTE_UUID = LabeledUUID(int=2, label="response")

class OpticalFlowMeasure(Capability):
    """Does something cool
    """

    def __init__(self, context):
        context.get_implementation("PipelineElement").__init__(self, context)
        super().__init__(context)
        model_path = getattr(context, "model_path", "neuflow_sintel.onnx")
        self.estimator = NeuFlowV2(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preprocess_frame = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            T.Resize((224, 224))
        ])
        self.frame_idx = 0

    def start_stream(self, stream, stream_id) -> Tuple[StreamEvent, Optional[str]]:
        self.frame_idx = 0
        return StreamEvent.OKAY, None

    def process_frame(self, stream, data_files: List[DataFile]) -> Tuple[StreamEvent, Union[Dict, str]]:
        assert len(data_files) == 1
        for df in data_files:
            self.logger.info(f"DataFile.content: {df.content.shape}")
            frame = df.content
            if frame is None:
                return StreamEvent.ERROR, "No frame content provided."

            filtered_frame = cv2.GaussianBlur(frame, (3, 3), 0)
            frame_tensor = self.preprocess_frame(cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2RGB)).to(self.device)

            if self.frame_idx == 0:
                self.prev_frame_tensor = frame_tensor
                self.frame_idx += 1
                continue

            prev_frame_np = (self.prev_frame_tensor.cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            curr_frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

            result = self.estimator(prev_frame_np, curr_frame_np)
            result_array = np.array(result)
            result_resized = cv2.resize(result_array, (frame.shape[1], frame.shape[0]))
            self.prev_frame_tensor = frame_tensor
            self.frame_idx += 1

        return StreamEvent.OKAY, {"optical_motion_vector": result_resized}

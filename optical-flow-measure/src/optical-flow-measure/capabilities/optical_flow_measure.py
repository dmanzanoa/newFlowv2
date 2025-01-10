from typing import Dict, Optional, Tuple, Union, List

from highlighter.agent.capabilities import Capability, StreamEvent
from highlighter.client import DataFile
from highlighter.core import LabeledUUID
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

# Compute project root by moving up several levels for ONNX model in github report.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
sys.path.append(project_root)
# Append the NeuFlow path dynamically
default_model_path = os.path.join(project_root,"models", "neuflow_sintel.onnx")
from neuflowv2 import NeuFlowV2

__all__ = ["OpticalFlowMeasure"]

ATTRIBUTE_UUID = LabeledUUID(int=2, label="response")

class OpticalFlowMeasure(Capability):
    """Does something cool
    """

    def __init__(self, context):
        context.get_implementation("PipelineElement").__init__(self, context)
        super().__init__(context)
        model_path = getattr(context, "model_path", default_model_path)
        self.estimator = NeuFlowV2(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preprocess_frame = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            T.Resize((224, 224))
        ])
        self.frame_idx = 0
        self.motion_data = {} 

    def start_stream(self, stream, stream_id) -> Tuple[StreamEvent, Optional[str]]:
        self.motion_data = {} 
        self.frame_idx = 0
        return StreamEvent.OKAY, None

    def normalize_motion_data(self):
        raw_totals = [data["raw_magnitude_total"] for data in self.motion_data.values()]
        if not raw_totals:  # Check if the list is empty
            self.logger.warning("No motion data available for normalization.")
            return  # Exit the function if no data is present
        min_total = min(raw_totals)
        max_total = max(raw_totals)
        for frame_idx, data in self.motion_data.items():
            if max_total > min_total:  # Avoid division by zero
                data["Normalized Total Motion"] = (data["raw_magnitude_total"] - min_total) / (max_total - min_total)
            else:
                data["Normalized Total Motion"] = 0.0

    def process_frame(self, stream, data_files: List[DataFile]) -> Tuple[StreamEvent, Union[Dict, str]]:
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
            # Estimate motion using NeuFlowV2
            prev_frame_np = (self.prev_frame_tensor.cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            curr_frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            result = self.estimator(prev_frame_np, curr_frame_np)
            u, v = result[..., 0], result[..., 1]
            magnitude = np.sqrt(u**2 + v**2)
            raw_magnitude_total = np.sum(magnitude)
            # Save motion data to the dictionary
            self.motion_data[self.frame_idx] = {
                "raw_magnitude_total": raw_magnitude_total
            }
            self.prev_frame_tensor = frame_tensor
            self.frame_idx += 1
        self.normalize_motion_data()
        return StreamEvent.OKAY, self.motion_data
    
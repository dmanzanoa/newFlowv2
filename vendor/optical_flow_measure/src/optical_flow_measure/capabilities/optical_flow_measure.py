from typing import List, Tuple, Dict, Any, Optional
from typing import Union, List
from uuid import uuid4
from highlighter.core import LabeledUUID
from highlighter.agent.capabilities import Capability, StreamEvent
from highlighter.client.base_models import (
    Entity,
    Annotation,
    Observation,
    ObjectClassTypeConnection,
    DatumSource,
)
from highlighter.core import paginate
from highlighter import OBJECT_CLASS_ATTRIBUTE_UUID
from highlighter.client import DataFile
import torch
import cv2
import numpy as np
from torchvision import transforms
from datetime import datetime, timedelta
import torchvision.transforms as T 
import torch
from highlighter.client import HLClient

"""
OpticalFlowMeasure

This module implements a capability to process video streams and analyze optical flow between two consecutive frames using a 
NeuFlowV2 model.
It calculates motion vectors, horizontal and vertical (H,W,2), and creates entities that had movement measure for each frame.

Key Functions:
- __init__: Initializes the capability, setting up the processing device, preprocessing pipeline, and loading the NeuFlowV2 model.
- load_model: Static method to load the optical flow model (NeuFlowV2) using a specified path from the context.
- start_stream: Prepares the stream for processing, initializes object class configurations for inference, and validates their presence.
- process_frame: Processes each frame in the video stream, calculating motion magnitude using optical flow and returning results.
- stop_stream: Cleans up resources after the stream ends (implementation placeholder).
- entities_for_object_class: Generates entities based on optical flow results.
- create_motion_entity: Create a motion entity with the amount of movement in a video stream.
"""


__all__ = ["OpticalFlowMeasure"]

ATTRIBUTE_UUID = LabeledUUID(int=2, label="response")

class OpticalFlowMeasure(Capability):
    """Does something cool
    """

    class DefaultStreamParameters(Capability.DefaultStreamParameters):
        pass
    def __init__(self, context):
        """
        Initialize the OpticalFlowMeasure capability.

        Args:
            context: The execution context providing access to configuration and utilities.
        """

        context.get_implementation("PipelineElement").__init__(self, context)
        super().__init__(context)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preprocess_frame = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.frame_idx = 0
        self.entities = {}
        self.estimator = self.load_model(context)
        self.frame_id_of_last_detection = {}
        print(f"Initialized OpticalFlowMeasure on device: {self.device}")

    @staticmethod
    def load_model(context):
        """
        Load the NeuFlowV2 optical flow model.

        Args:
            context: The execution context containing the model path.

        Returns:
            The loaded NeuFlowV2 model instance.
        """
        model_path = getattr(context, "model_path", "neuflow_sintel.onnx")
        # Assume NeuFlowV2 is the estimator class from an imported module
        from neuflowv2 import NeuFlowV2
        return NeuFlowV2(model_path)

    def start_stream(self, stream, stream_id):
        """
        Start processing a video stream.

        Args:
            stream: The video stream to process.
            stream_id: The unique identifier for the stream.

        Returns:
            StreamEvent indicating the result of starting the stream.
        """
        super().start_stream(stream, stream_id)
        self.frame_idx = 0
        self.prev_frame_tensor = None
        self.entities.clear()
        self.client = HLClient.get_client()
        self.entity_ids = {}
        self.track_ids = {}
        print(f"Started stream {stream_id} with frame index reset.") 
        self.object_classes_for_inference = {
            "Chicken": {"inference_index": 1}
        }
        generator = paginate(
            self.client.objectClassConnection,
            ObjectClassTypeConnection,
        )

        for object_class_result in generator:
            if object_class_result.name in self.object_classes_for_inference:
                print(object_class_result.name)
                self.object_classes_for_inference[object_class_result.name][
                    "uuid"
                ] = object_class_result.uuid
                print(object_class_result.uuid)
                
        for item in self.object_classes_for_inference.values():
            if not all('uuid' in item and item['uuid'] for item in self.object_classes_for_inference.values()):
                raise ValueError(
                f"Not all required object classes present, found: {', '.join(self.object_classes_for_inference.keys())}"
                )

        return StreamEvent.OKAY, {}

    def process_frame(self, stream, data_files: List[DataFile]) -> Tuple[StreamEvent, Dict[str, Any]]:
        """
            Process a single video frame to calculate motion magnitude and create entities.

            Args:
                stream: The video stream object containing metadata and state.
                data_files: A list of DataFile objects containing frame data.

            Returns:
                A tuple containing a StreamEvent and a dictionary with measurements results entity.
        """
        print(f"Processing frame: {stream.frame_id}")
        entities = {}

        for df in data_files:
            frame = df.content
            if frame is None:
                return StreamEvent.ERROR, "No frame content provided."

        filtered_frame = cv2.GaussianBlur(frame, (3, 3), 0)
        frame_tensor = self.preprocess_frame(cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2RGB)).to(self.device)

        if self.frame_idx == 0:
            self.prev_frame_tensor = frame_tensor
            self.frame_idx += 1
            return StreamEvent.OKAY, {"optical_motion_vector": 0.0}

        prev_frame_np = (self.prev_frame_tensor.cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        curr_frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        result = self.estimator(prev_frame_np, curr_frame_np)
        u, v = result[..., 0], result[..., 1]
        magnitude = np.sqrt(u**2 + v**2)
        motion_magnitude = np.sum(magnitude)
        for object_class_value in self.object_classes_for_inference.values():
            frame_entities = self.entities_for_object_class(
                motion_magnitude,
                object_class_value["inference_index"],
                stream,
                object_class_value["uuid"]
            )
        
        entities.update(frame_entities)

        self.prev_frame_tensor = frame_tensor
        self.frame_idx += 1

        return StreamEvent.OKAY, {"entities": entities}

    def entities_for_object_class(self, motion_magnitude, output_idx, stream, object_class_uuid):
        """
        Generate entities based on motion magnitude for a specific object class.

        Args:
            motion_magnitude: The calculated magnitude of motion.
            output_idx: The index of the output in the inference model (Just Chicken).
            stream: The video stream object.
            object_class_uuid: The unique identifier for the object class.

        Returns:
            A dictionary of generated entities.
        """

        entities = self.create_motion_entity(
            stream, motion_magnitude, object_class_uuid
        )
        return entities

    def create_motion_entity(self, stream, motion_magnitude, object_class_uuid):
        """
        Create a motion entity with the amount of movement in a video stream.

        Args:
            motion_magnitude (float): The calculated magnitude of motion between two consecutive frames.
            stream: The current video stream object, likely containing metadata like frame ID.
            object_class_uuid: A unique identifier categorizing the type of detected motion (e.g., Chickens).

        Returns:
            dict: A dictionary where the key is the unique ID of the entity, and the value is the constructed Entity object containing metadata about the amount of movement.

        Workflow:
            1. Create a new observation and annotation to represent the motion measurement.
            2. Assign unique IDs to the entity, observation, and annotation.
            3. Maintain continuity across frames by assigning the same track ID for motion detected in consecutive frames, or a new track ID for distinct motion events.
            4. Return the constructed motion entity for further processing or storage.
        Note: Confidence = motion_magnitude
        """
        entities = {}

        entity_id = uuid4()
        object_class_obs_id = uuid4()
        object_class_attr_id = OBJECT_CLASS_ATTRIBUTE_UUID

        if not object_class_uuid in self.entity_ids:
            self.entity_ids[object_class_uuid] = uuid4()

        motion_entity = Entity(id=self.entity_ids[object_class_uuid])
        annotation_id = uuid4()

        if object_class_uuid not in self.frame_id_of_last_detection or (
            stream.frame_id > self.frame_id_of_last_detection[object_class_uuid] + 1
        ):
            
            self.track_ids[object_class_uuid] = uuid4()
            print(f"New track_id generated: {self.track_ids[object_class_uuid]}")

        self.frame_id_of_last_detection[object_class_uuid] = stream.frame_id
        
        #motion_entity = Entity(id=entity_id)

        annotation = Annotation(
            id=annotation_id,
            datum_source=DatumSource(frame_id=stream.frame_id, confidence=motion_magnitude),
            track_id=self.track_ids[object_class_uuid] 
        )
        motion_entity.annotations.add(annotation)

        annotation.observations.add(
            Observation(
                id=object_class_obs_id,
                annotation_id=annotation_id,
                entity_id=self.entity_ids[object_class_uuid],
                attribute_id=object_class_attr_id,  
                value=object_class_uuid,
                datum_source=DatumSource(frame_id=stream.frame_id, confidence=motion_magnitude),
            )
        )
        entities[motion_entity.id] = motion_entity

        return entities
 
    def stop_stream(self, stream, stream_id) -> Tuple[StreamEvent, Dict[str, Any]]:
        """
        Stop the stream and reset the state.
        """
        self.prev_frame_tensor = None
        self.entities.clear()
        print(f"Stopped stream {stream_id}. Reset state.") 
        return StreamEvent.OKAY, {}

    def debug_to_file(self, value):
        print(value)
        with open("debug.txt", "w") as file:
            file.write(value)

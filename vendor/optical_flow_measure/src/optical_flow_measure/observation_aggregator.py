import numpy as np
import uuid

from datetime import datetime, timezone
from highlighter.client import EAVT, PixelLocationAttributeValue, Point2d, Polygon
from highlighter.core import PIXEL_LOCATION_ATTRIBUTE_UUID
from .observation_writer import ObservationWriter
from shapely.geometry import Polygon as ShapelyPolygon
from typing import Any, Dict, List

__all__ = ["ObservationAggregator"]


# ToDo: Move EAVT to highlighter_sdk.Observation
# see https://gitlab.com/silverpond/products/highlighter_all/-/issues/928?work_item_iid=972
Observation = EAVT


def average_embedding(embeddings: np.ndarray) -> np.ndarray:
    return embeddings.mean(axis=0)


def days_since_epoch(time):
    assert isinstance(time, datetime), f"created_at must be a datetime object, got {time}"
    return (time - datetime.fromtimestamp(0, timezone.utc)).days

class ObservationAggregator():
    """Aggregates Observations (EAVTs)

    Observations (EAVTs) are updated according to their
    attribute_id.

    Args:
        minimum_track_frame_length (int): Only include tracks in the final
        result if the track has at least this number of frames.

        minimum_embedding_in_track_frame_length (int): Only include aggregate
        embeddings and include them in the final result if the track the at
        least this number of frames.

    ---

    To add a custom update function simply create a function with
    the correct interface (see the staticmethods `update_*_entity` below) and add it to the
    ObservationAggregator.UPDATE_ENTITY_FNS dict with the attibute_id as the key
    and the function as the value, ie:

      ObservationAggregator.UPDATE_ENTITY_FNS["ATTRIBUTE_ID"] = update_my_attr_entity

    Example Usage:
        ```
        from highlighter.task import update_task_result

        aggregator = ObservationAggregator()

        for d in data:

           # predictions happen ...
           observations: List[Observation] = my_model.predict(data)
           aggregator.update_with_observations(observations)

        # after all observations have been aggregated you can use a pre
        # build writer or role your own. The following uses the S3AvroObservationWriter

        filename = "some-file.avro"
        hl_client = ...
        writer = S3AvroObservationWriter(ENTITY_AVRO_SCHEMA, fileanme,
                                         client=hl_client)

        shrine_file_info = aggregator.write(writer)

        task_id = ...
        update_task_result(hl_client,
                           task_id,
                           "SUCCESS",
                           background_info_layer_file_data=shrine_file_info,
                           )
        ```


    TODO: This should probably be a function we pass a collection of observations to
    and it does the rest. I don't see a need to incrementally update
    the state of this class if we're already aggregating either in a dedicated
    PipelineElelemt or within the pipeline framework. One consideration is that
    we'll likely need to add ways to update various attributes depending on
    their type or ID. This still needs some design thought

    """

    # TODO do this filtering in the tracker and make configurable
    DEFAULT_MINIMUM_TRACK_FRAME_LENGTH = 1
    DEFAULT_MINIMUM_EMBEDDING_TRACK_FRAME_LENGTH = 4

    @staticmethod
    def update_pixel_location_entity(observation: Observation, entity: Dict, frame_id: int) -> Dict:
        track = entity['tracks'][-1]
        if isinstance(observation.value, ShapelyPolygon):
            x0, y0, x1, y1 = observation.value.bounds
        elif isinstance(observation.value, Polygon):
            x0, y0, x1, y1 = observation.value.get_top_left_bottom_right_coordinates()
        elif isinstance(observation.value, Point2d):
            x0, y0 = observation.value.x, observation.value.y
            x1, y1 = x0, y0
        elif isinstance(observation.value, PixelLocationAttributeValue):
            x0, y0, x1, y1 = observation.value.value.bounds
        else:
            raise ValueError(f"Invalid pixel_location: {observation.value}")
        track['detections'].append({
            "frame_id": frame_id,
            "bounds": {
                "min": {"x": float(x0), "y": float(y0)},
                "max": {"x": float(x1), "y": float(y1)}
            },
            "geometry_type": 0,
            "confidence": observation.datum_source.confidence
        })
        return entity

    @staticmethod
    def update_embeddings_entity(observation: Observation, entity: Dict, frame_id: int) -> Dict:
        entity['embeddings'].append(observation.value)
        return entity


    @staticmethod
    def update_object_class_entity(observation: Observation, entity: Dict, frame_id: int) -> Dict:
        entity['object_class'] = str(observation.value)
        return entity

    @staticmethod
    def update_entity_with_enum_observation(observation: Observation, entity: Dict, frame_id: int) -> Dict:
        track = entity["tracks"][0]
        track["eavts"].append({
            "entityId": str(observation.entity_id),
            "entityAttributeId": str(observation.attribute_id),
            "entityAttributeEnumId": str(observation.value),
            "entityDatumSource": {
                "confidence": observation.datum_source.confidence,
                "frameId": observation.datum_source.frame_id,
            },
            "value": None,
            "time": days_since_epoch(observation.time),
        })
        return entity


    UPDATE_ENTITY_FNS = {
        "594fcdba-c3dc-4fad-b1c1-f5f537e1d16c": update_pixel_location_entity,
        "8beab557-8d83-4257-82d0-101341236c5b": update_embeddings_entity,
        "df10b67d-b476-4c4d-acc2-c1deb5a0e4f4": update_object_class_entity,
        }

    @staticmethod
    def make_track_observation(entity_id: str, attribute_id: str,
                        attribute_enum_id: str, value: Any,
                        confidence: float, frame_id: int,
                        created_at: datetime) -> Dict:

        if attribute_enum_id != "null":
            # assert attribute_enum_id is a valid uuid
            _ = uuid.UUID(attribute_enum_id)

        return dict(entityId=entity_id,
                    entityAttributeId=attribute_id,
                    entityAttributeEnumId=attribute_enum_id,
                    value=value,
                    entityDatumSource=dict(confidence=confidence,
                                           frameId=frame_id),
                    time=days_since_epoch(created_at))


    def __init__(self,
                 minimum_track_frame_length=DEFAULT_MINIMUM_TRACK_FRAME_LENGTH,
                 minimum_embedding_in_track_frame_length=DEFAULT_MINIMUM_EMBEDDING_TRACK_FRAME_LENGTH):

        self._minimum_track_frame_length = minimum_track_frame_length
        self._minimum_embedding_in_track_frame_length = minimum_embedding_in_track_frame_length
        self._final_result = []
        self._result = {}
        self._end_tracks = {}
        self._previous_frame_id = None

    def _settle(self, frame_id, after: int = 50):
        entity_to_drop = []
        entity_to_finalise = []
        for entity_id, last_frame in self._end_tracks.items():
            if any([x is None for x in [frame_id, last_frame, after]]):
                raise ValueError("Cannot settle observations if any of "
                                 "frame_id|last_frame|after are None")
            if frame_id < last_frame + after:
                continue

            if len(self._result[entity_id]['tracks'][0]['detections']) >= self._minimum_track_frame_length:
                entity_to_finalise.append(entity_id)
            entity_to_drop.append(entity_id)

        for entity_id in entity_to_finalise:
            entity = self._result[entity_id]
            entity['embeddings'] = []

            self._final_result.append(self._result[entity_id])

        for entity_id in set(entity_to_drop + entity_to_finalise):
            del self._end_tracks[entity_id]
            del self._result[entity_id]

    def _update_with_a_single_observation(self, observation: Observation):
        entity_id = str(observation.entity_id)
        if entity_id not in self._result:
            self._result[entity_id] = {
                'id': entity_id,
                'object_class': "",
                'tracks': [{'track_id': str(uuid.uuid4()), 'detections': [], 'eavts': []}],
                'embeddings': [],
            }

        attribute_id = str(observation.attribute_id)
        update_entity_fn = ObservationAggregator.UPDATE_ENTITY_FNS.get(attribute_id, None)

        frame_id = observation.datum_source.frame_id
        if update_entity_fn is not None:
            self._result[entity_id] = update_entity_fn(observation, self._result[entity_id],
                                                       frame_id)
        else:
            self._result[entity_id] = ObservationAggregator.update_entity_with_enum_observation(
                observation, self._result[entity_id], frame_id)

        if attribute_id == str(PIXEL_LOCATION_ATTRIBUTE_UUID):
            self._end_tracks[entity_id] = frame_id

    def update_with_observations(self, observations: List[Observation]):
        if not observations:
            return

        for observation in observations:
            if observation.datum_source.frame_id is None:
                raise ValueError(f"To use the {self.__class__.__name__} you "
                                 "must specify 'frame_id' on the datum_source")
            self._update_with_a_single_observation(observation)

        last = observations[-1]
        last_frame_id = last.datum_source.frame_id
        if last_frame_id != self._previous_frame_id:
            self._settle(last_frame_id, after=50)
            self._previous_frame_id = last_frame_id


    def write(self, writer: ObservationWriter):
        self._settle(self._previous_frame_id, after=0)
        return writer.write(self._final_result)

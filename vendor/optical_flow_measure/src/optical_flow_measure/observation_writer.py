import json
from abc import ABC, abstractmethod
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import fastavro
from highlighter.client import HLClient, aws_s3 as hls3

__all__ = ["ObservationWriter",
           "BaseAvroObservationWriter",
           "FileAvroObservationWriter",
           "ByteAvroObservationWriter",
           "S3AvroObservationWriter",
        ]

class ObservationWriter(ABC):

    @abstractmethod
    def write(self, final_result: List) -> Any:
        # ToDo: Define the data structure of final_result
        pass

class BaseAvroObservationWriter(ObservationWriter):

    class CompressionCodec(str, Enum):
        NULL = "null"
        SNAPPY = "snappy"

    def __init__(self, schema: Union[str, Path, List],
                 compression_codec: CompressionCodec=CompressionCodec.NULL):

        if not isinstance(compression_codec, BaseAvroObservationWriter.CompressionCodec):
            raise ValueError(f"Expected a CompressionCodec enum, got: {compression_codec}")
        self.compression_codec = compression_codec

        if isinstance(schema, (str, Path)):
            self._schema = json.load(Path(schema).open("r"))
        elif isinstance(schema, list):
            self._schema = schema
        else:
            raise ValueError(f"Invaid schema dict or schema path, got: {schema}")


class FileAvroObservationWriter(BaseAvroObservationWriter):

    def __init__(self, schema: Union[str, Path, List],
                 output_path: Union[str, Path],
                 compression_codec: BaseAvroObservationWriter.CompressionCodec=BaseAvroObservationWriter.CompressionCodec.NULL,
                 ):
        super().__init__(schema, compression_codec=compression_codec)
        self.output_path = Path(output_path)


    def write(self, final_result):
        with self.output_path.open('wb') as fp:
            fastavro.writer(fp, self._schema, final_result, codec=self.compression_codec)

class ByteAvroObservationWriter(BaseAvroObservationWriter):

    def write(self, final_result) -> bytes:
        with BytesIO() as fp:
            fastavro.writer(fp, self._schema, final_result, codec=self.compression_codec)
            return bytes(fp.getbuffer())


class S3AvroObservationWriter(ByteAvroObservationWriter):

    def __init__(self, schema: Union[str, Path, List],
                 filename: str,
                 client: Optional[HLClient] = None,
                 compression_codec: BaseAvroObservationWriter.CompressionCodec=BaseAvroObservationWriter.CompressionCodec.NULL,
                 ):
        super().__init__(schema, compression_codec=compression_codec)
        self.filename = filename

        if client is None:
            self.client = HLClient.from_env()
        else:
            self.client: HLClient = client

    def write(self, final_result) -> Dict:
        content = super().write(final_result)
        shrine_file = hls3.upload_file_to_s3_in_memory(self.client,
                                                  content,
                                                  self.filename,
                                                  mimetype="application/avro")
        return shrine_file


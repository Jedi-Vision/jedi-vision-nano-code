from queue import Queue
from queue import Empty
from ..representation import ObjectRepData
from ..pb.objectrep_pb2 import ObjectRepData as PBObjectRepData  # type: ignore
import zmq
import threading
from typing import Literal
import struct


class AudioBuffer:
    """
        A one way audio buffer for transferring object data from Python
        to the Steam Audio Library process.
    """

    def __init__(
        self,
        size: int = 0,
        addr: str = "localhost",
        port: int = 5555
    ) -> None:

        self.q = Queue(maxsize=size)
        self.ctx = zmq.Context()
        self.start_client(self.ctx, addr, port)

    def start_client(self, ctx, addr, port):
        self.socket = ctx.socket(zmq.REQ)
        self.socket.connect(f"tcp://{addr}:{port}")
        self.running = False

        threading.Thread(target=self.send_message_worker).start()

    def kill_client(self):
        self.q.join()
        self.running = False
        self.ctx.term()

    def queue_message(self, message: ObjectRepData) -> None:
        self.q.put(message, timeout=0.001)

    def serialize_message(
        self,
        message: ObjectRepData,
        type: Literal['protobuf', 'struct'] = 'struct'
    ) -> bytes:
        """
        Function for serializing the ObjectRepData dataclass into a sendable
        bytes string or bytes object.

        Arguments:
            type (Literal["protobuf", "struct"]): The type of serialization to
                use. Protobuf has a very slow overhead and requires using a
                Protobuf enabled server for receiving the messages. Struct uses
                the built in Python struct library for converting between
                Python objects and C structs.

        Returns:
            bytes | str
        """

        if type == "protobuf":
            serialized_message = message.to_protobuf(PBObjectRepData)
            return serialized_message.SerializeToString()
        elif type == "struct":
            return struct.pack("<h", 20)

    def send_message_worker(self) -> None:
        """
        Worker for sending messages in buffer to the spatial audio server.
        """

        running = True
        while running:

            try:
                message = self.q.get(timeout=0.001)
                serialized_message = self.serialize_message(message)
                print("Sending message.")
                self.socket.send(serialized_message)
                self.socket.recv()  # must wait for audio server to recieve message
                self.q.task_done()
            except Empty as e:
                print(f"Audio buffer empty: {e}")
                pass
            except Exception as e:
                raise Exception(f"Message send failure: {e}")

        return

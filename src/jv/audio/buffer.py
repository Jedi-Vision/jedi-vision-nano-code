from queue import Queue
from queue import Empty
from ..representation import ObjectRepData
from ..pb.objectrep_pb2 import ObjectRepData as PBObjectRepData  # type: ignore
import zmq
from zmq.sugar.socket import Socket
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
        addr: str = "/tmp/jv/audio/0.sock"
    ):
        """
        Initializes the buffer with a queue, a ZeroMQ context, and starts a client connection.

        Args:
            size (int, optional): The maximum size of the queue. Defaults to 0, which means the queue is unbounded.
            addr (str, optional): The pathname for the socket. Default is "/tmp/jv/audio/0".
        """

        self.q = Queue(maxsize=size)
        self.ctx = zmq.Context()
        self.start_client(self.ctx, addr)

    def start_client(self, ctx: zmq.Context[Socket[bytes]], addr: str):
        self.socket = ctx.socket(zmq.REQ)
        self.socket.connect(f"ipc://{addr}")
        self.running = False

        threading.Thread(target=self.send_message_worker).start()

    def kill_client(self):
        self.q.join()
        self.running = False
        self.ctx.term()

    def __del__(self):
        self.kill_client()

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

        self.running = True
        while self.running:

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

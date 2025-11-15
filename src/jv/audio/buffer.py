from queue import Queue, Empty
from ..representation import ObjectRepData
from ..pb.objectrep_pb2 import ObjectRepData as PBObjectRepData  # type: ignore
from ..serial import serialize_dataclass
import zmq
import threading
from typing import Literal


class ObjectBuffer:
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

        self.max_size = size
        self.q = Queue(size)
        self.ctx = zmq.Context()
        self.addr = addr
        self.thread = None

    def start(self):
        self.socket = self.ctx.socket(zmq.REQ)
        self.socket.connect(f"ipc://{self.addr}")
        self.running = False
        self.thread = threading.Thread(target=self._send_message)
        self.thread.start()

    def stop(self):
        self.q.join()
        self.running = False
        if self.thread is not None:
            self.thread.join()
        self.ctx.term()

    def put(self, message: ObjectRepData) -> None:
        if self.q.full():
            self.q.get(timeout=0.001)  # Remove the oldest frame to make space
        self.q.put(message, timeout=0.001)

    def _serialize_message(
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
            return serialize_dataclass(message)

    def _send_message(self) -> None:
        """
        Worker for sending messages in buffer to the spatial audio server.
        """

        self.running = True
        while self.running:

            try:
                if self.q.full():
                    self.q.get()  # Remove the oldest frame to make space

                message = self.q.get(timeout=0.001)
                serialized_message = self._serialize_message(message)

                # print("Sending message.")
                self.socket.send(serialized_message)
                self.socket.recv()  # must wait for audio server to recieve message
                self.q.task_done()
            except Empty:
                # print("Object buffer empty")
                pass
            except Exception as e:
                raise Exception(f"Message send failure: {e}")

        return

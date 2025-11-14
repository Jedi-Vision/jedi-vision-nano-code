"""
Request-and-reply client example code using ZeroMQ.
"""

import zmq


def main():
    context = zmq.Context()

    #  Socket to talk to server
    print("Connecting to hello world server…")
    socket = context.socket(zmq.REQ)
    socket.connect("ipc:///tmp/jv/audio/0")

    #  Do 10 requests, waiting each time for a response
    for request in range(10):
        print(f"Sending request {request} …")
        socket.send(b"Hello")

        #  Get the reply.
        message = socket.recv()
        print(f"Received reply {request} [ {message} ]")


if __name__ == "__main__":
    main()

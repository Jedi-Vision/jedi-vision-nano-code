# Camera

The `FrameBuffer` class continually retrieves frames from a video stream and stores it in a queue.

```python
buffer = FrameBuffer(...)
buffer.start()
frame = buffer.get()
buffer.stop()
```
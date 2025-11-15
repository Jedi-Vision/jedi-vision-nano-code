from jv import Driver

driver = Driver(
    "yolo11",
    "mps",
    # camera_index="examples/videos/sidewalk_pov.mp4",
    object_buffer_size=1,
    frame_buffer_size=1,
    frame_skip=1
)

driver.run()

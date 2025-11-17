from jv import Driver

driver = Driver(
    "mps",
    camera_index="examples/videos/sidewalk_pov.mp4",
    object_buffer_size=1,
    frame_buffer_size=1,
    frame_skip=2,
    frame_rate=30,
    show_det=True,
)

driver.run()

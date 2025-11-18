from jv import Driver

driver = Driver(
    device="mps",
    output_to="none",
    camera_index="examples/videos/sidewalk_pov.mp4",
    object_buffer_size=1,
    frame_buffer_size=1000,
    frame_skip=10,
    frame_rate=30,
    show_det=True,
    depth=False
)

driver.run()

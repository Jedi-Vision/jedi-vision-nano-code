from jv.representation import YoloEnvironmentRepresentationModel
import cv2
from argparse import ArgumentParser
import os
import time


parser = ArgumentParser()
parser.add_argument("-d", "--device",
                    choices=['cpu', 'mps', 'cuda'], default="mps")
parser.add_argument("-v", "--video",
                    default="../examples/videos/sidewalk_pov.mp4")
parser.add_argument("-w", "--webcam",
                    action="store_true", default=False, help="Use webcam instead of input video.")
parser.add_argument("-p", "--phone",
                    action="store_true", default=False, help="Use phone instead of input video.")
parser.add_argument("-t", "--text",
                    action="store_true", default=False, help="Add text labels for classes on seg-masks.")
args = parser.parse_args()

if (
    args.video == "../examples/videos/sidewalk_pov.mp4"
    and not (os.path.exists(args.video) or args.webcam or args.phone)
):
    print(
        "Please download example video from https://oregonstate.app.box.com/file/2035963501758 "
        "and put in `examples/videos`"
    )
    exit(1)

model = YoloEnvironmentRepresentationModel("yolo11", device=args.device)

if args.webcam:
    cap = cv2.VideoCapture(0)  # run with webcam
elif args.phone:
    cap = cv2.VideoCapture(1)  # run with phone (if on mac with continuity camera)
else:
    cap = cv2.VideoCapture(args.video)  # sidewalk video

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if not cap.isOpened():
    print("Error: Unable to open video stream.")
    exit(1)
frame_skip = 2  # skip every second frame

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read frame.")
        break

    # Skip frames
    frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if frame_count % frame_skip != 0:
        continue

    # Run model and add mask to image
    start = time.time()
    out = model.run(frame)
    end = time.time() - start
    print(f"Inference took {end*1000:.4f}ms")

    # Iterate over detected objects
    for obj in out.object_coordinates:
        x, y = obj.x, obj.y
        label = obj.label if obj.label else ""
        object_id = obj.object_id if obj.object_id else 0

        # Draw points on an image same size as input.shape
        cv2.circle(
            img=frame,
            center=(int(x), int(y)),
            radius=5,
            color=(0, 255, 0),  # Green color
            thickness=-1
        )

        # Add text for label and object id
        cv2.putText(
            img=frame,
            text=f"{label}:{object_id}",
            org=(int(x) + 10, int(y) - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255),  # White color
            thickness=1,
            lineType=cv2.LINE_AA
        )

    cv2.imshow("Tracked Objects", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

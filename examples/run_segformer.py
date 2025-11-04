from jv.representations import SegFormerEnvironmentRepresentationModelClass
import cv2
import time
import torch
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--device", choices=['cpu', 'mps', 'cuda'], default="mps")
parser.add_argument("--video", default="./videos/sidewalk_pov.mp4")
parser.add_argument("--webcam", action="store_true", default=False, help="Use webcam instead of input video.")
parser.add_argument("--phone", action="store_true", default=False, help="Use phone instead of input video.")
parser.add_argument("--text", action="store_true", default=False, help="Add text labels for classes on seg-masks.")
args = parser.parse_args()

if args.video == "./videos/sidewalk_pov.mp4" and not (os.path.exists(args.video) or args.webcam or args.phone):
    print(
        "Please download example video from https://oregonstate.app.box.com/file/2035963501758 "
        "and put in `examples/videos`"
    )
    exit(1)

model = SegFormerEnvironmentRepresentationModelClass("ade-small", k=1, device=args.device)

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

    # Run frame through segformer and apply mask
    preproc_time = time.time()
    inputs = model.preprocess(torch.tensor(frame))
    preproc_time = time.time() - preproc_time

    inference_time = time.time()
    out = model.process(inputs)
    inference_time = time.time() - inference_time

    postproc_time = time.time()
    masked_frame = model.postprocess_to_image(out, frame, add_text_labels=args.text)
    postproc_time = time.time() - postproc_time

    if frame_count % 30 == 0:
        print(
            f"""
                Frame {frame_count}/{total_frames} --
                Preprocess: {preproc_time*1000:.4f}ms,
                Inference: {inference_time*1000:.4f}ms,
                Postprocess: {postproc_time*1000:.4f}ms,
                Total: {(preproc_time + inference_time + postproc_time)*1000:.4f}ms
             """
        )

    # Display the result
    cv2.imshow("Segmentation", masked_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

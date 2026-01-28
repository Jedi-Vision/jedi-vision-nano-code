from PIL import Image
import depth_pro
import cv2
import time
# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms(device='mps')
model.eval()

# Open video file using OpenCV
video_path = "../../examples/videos/sidewalk_pov.mp4"  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no more frames

    # Convert frame to PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess the frame
    image = transform(image)

    # Run inference and time it
    start_time = time.time()
    prediction = model.infer(image)
    inference_time = time.time() - start_time

    # Extract depth and focal length
    depth = prediction["depth"].cpu().numpy()  # Depth in [m].
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.

    # Normalize depth for visualization
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
    depth_colormap = (depth_normalized * 255).astype('uint8')

    # Convert depth to a color image
    depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)

    # Display the depth map
    cv2.imshow("Depth Prediction", depth_colormap)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Print inference time
    print(f"Inference time: {inference_time:.4f} seconds")

# Release video capture
cap.release()

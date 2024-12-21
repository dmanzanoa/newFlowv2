import numpy as np
import os
import sys
import pandas as pd
import torch
import torchvision.transforms as T
from neuflowv2 import NeuFlowV2
import cv2
import csv

# Initialize the NeuFlow model
model_path = "models/neuflow_sintel.onnx"
estimator = NeuFlowV2(model_path)

# Ensure the script receives a video filename as an argument
if len(sys.argv) != 2:
    print("Usage: python process_video.py <video_filename>")
    sys.exit(1)

# Input video filename
filename = sys.argv[1]

# Derived paths and settings
folder_name = filename.replace(".mp4", "")
os.makedirs(folder_name, exist_ok=True)

output_csv_path = os.path.join(folder_name, "motion_data.csv")
output_normalized_csv_path = os.path.join(folder_name, "normalized_data.csv")
output_video_path = os.path.join(folder_name, "output_with_magnitudes.mp4")

# Target dimensions for the output video
target_video_width = 1280  # Width of the resized video
font_scale = 1.1  # Increase font size
font_thickness = 3

# Torch preprocessing pipeline for the model input
def preprocess_frame(frame):
    transform = T.Compose([
        T.ToTensor(),  # Convert to PyTorch tensor
        T.ConvertImageDtype(torch.float32),  # Normalize to [0, 1]
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Map to [-1, 1]
        T.Resize((224, 224))  # Resize for the model input
    ])
    return transform(frame)

# Function to find the last processed frame
def last_frame_processed(csvpath):
    if not os.path.exists(csvpath):
        return -1
    else:
        df = pd.read_csv(csvpath)
        return int(df.iloc[-1]["Frame_index"])

# Open the input video
cap = cv2.VideoCapture(filename)
if not cap.isOpened():
    print(f"Error: Could not open video file {filename}")
    sys.exit(1)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Calculate new dimensions to preserve aspect ratio
aspect_ratio = frame_width / frame_height
target_video_height = int(target_video_width / aspect_ratio)

# Initialize VideoWriter for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_video_width, target_video_height))

# Prepare CSV file
if os.path.exists(output_csv_path):
    motion_data = pd.read_csv(output_csv_path)
    frame_idx_start = motion_data['Frame_index'].max() + 1
else:
    frame_idx_start = 0
    with open(output_csv_path, 'w') as f:
        f.write("Frame_index,totalMotion,avgMotion\n")

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    out.release()
    sys.exit(1)

# Apply Gaussian filter to the first frame and preprocess it
filtered_prev_frame = cv2.GaussianBlur(prev_frame, (3, 3), 0)
prev_frame_tensor = preprocess_frame(cv2.cvtColor(filtered_prev_frame, cv2.COLOR_BGR2RGB))

# Process the video frame-by-frame
frame_idx = frame_idx_start
motion_records = []

while True:
    # Read the next frame
    ret, curr_frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    # Apply Gaussian filter to reduce noise
    filtered_curr_frame = cv2.GaussianBlur(curr_frame, (3, 3), 0)

    # Preprocess the current frame for the model
    curr_frame_tensor = preprocess_frame(cv2.cvtColor(filtered_curr_frame, cv2.COLOR_BGR2RGB))

    # Convert tensors back to NumPy arrays for NeuFlowV2
    prev_frame_np = prev_frame_tensor.permute(1, 2, 0).numpy() * 255.0
    curr_frame_np = curr_frame_tensor.permute(1, 2, 0).numpy() * 255.0
    prev_frame_np = prev_frame_np.astype(np.uint8)
    curr_frame_np = curr_frame_np.astype(np.uint8)

    # Estimate motion using NeuFlowV2
    result = estimator(prev_frame_np, curr_frame_np)
    u = result[..., 0]
    v = result[..., 1]

    # Compute motion magnitude
    magnitude = np.sqrt(u**2 + v**2)
    average_motion = np.mean(magnitude)
    total_motion = np.sum(magnitude)

    # Save motion data
    motion_records.append((frame_idx, total_motion, average_motion))

    # Resize the original frame (without Gaussian filtering) for output video
    display_frame = cv2.resize(curr_frame, (target_video_width, target_video_height))

    # Overlay motion information onto the resized frame
    info_text = f"Frame: {frame_idx} | Total Motion: {total_motion:.2f} | Avg Motion: {average_motion:.2f}"
    cv2.putText(display_frame, info_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

    # Write the frame to the output video
    out.write(display_frame)

    # Update the previous frame
    prev_frame_tensor = curr_frame_tensor

    frame_idx += 1

# Write motion data to CSV
with open(output_csv_path, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(motion_records)

# Normalize and save motion data
motion_data = pd.DataFrame(motion_records, columns=["Frame_index", "totalMotion", "avgMotion"])
scaled_avg_motion = motion_data["avgMotion"]**2
min_val = scaled_avg_motion.min()
max_val = scaled_avg_motion.max()
motion_data["Normalized Motion"] = (scaled_avg_motion - min_val) / (max_val - min_val)
motion_data.to_csv(output_normalized_csv_path, index=False)

print(f"Motion data saved to {output_csv_path}")
print(f"Normalized motion data saved to {output_normalized_csv_path}")

# Release resources
cap.release()
out.release()
print(f"Output video saved to {output_video_path}")

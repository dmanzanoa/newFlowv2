import cv2
from imread_from_url import imread_from_url
from neuflowv2 import NeuFlowV2, draw_flow
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Image as IPImage, display
import numpy as np
import os
import sys
import glob
import numpy as np
import cv2
import csv
import pandas as pd
import subprocess


# Initialize model
model_path = "models/neuflow_sintel.onnx"
estimator = NeuFlowV2(model_path)

# Load images
#img1 = imread_from_url("https://github.com/princeton-vl/RAFT/blob/master/demo-frames/frame_0016.png?raw=true")
#img2 = imread_from_url("https://github.com/princeton-vl/RAFT/blob/master/demo-frames/frame_0025.png?raw=true")

# Estimate optical flow
#flow = estimator(img1, img2)

# Draw Flow
#flow_img = draw_flow(flow, img1)


# Convert the NumPy array (flow_img) to a PIL Image
#flow_img_pil = Image.fromarray(flow_img)

# Save the PIL Image
#flow_img_pil.save("flow_output.png")

# Ensure the script receives a video filename as an argument
if len(sys.argv) != 2:
    print("Usage: python process_video.py <video_filename>")
    sys.exit(1)

# Input video filename
filename = sys.argv[1]

# Derived paths and settings
folder_name = filename.replace(".mp4", "")
os.makedirs(folder_name, exist_ok=True)

video_path = filename
output_frames_path = os.path.join(folder_name, "frames")
os.makedirs(output_frames_path, exist_ok=True)

resize_width = 224
resize_height = 224

# Function to find the last processed frame

def last_frame_processed(csvpath):
    if not os.path.exists(csvpath):
        return -1
    else:
        df = pd.read_csv(csvpath)
        return int(df.iloc[-1]["Frame_index"])

# Step 1: Extract frames using ffmpeg
subprocess.run([
    "ffmpeg", "-i", video_path,
    "-vf", f"scale={resize_width}:{resize_height}",
    os.path.join(output_frames_path, "frame_%04d.png")
])

# Step 2: Initialize the model
#config_file = 'pwcnet_ft_4x1_300k_sintel_final_384x768.py'  # Update this path
#checkpoint_file = 'pwcnet_ft_4x1_300k_sintel_final_384x768.pth'  # Update this path
device = 'cuda'

# Step 3: Process frames
frame_files = sorted(glob.glob(os.path.join(output_frames_path, "*.png")))
output_csv_path = os.path.join(folder_name, "motion_data.csv")
magnitude_list = []
x_data = []

frame_idx = last_frame_processed(output_csv_path)

with open(output_csv_path, 'a', newline='') as file:
    writer = csv.writer(file)
    if frame_idx == -1:
        print("Creating new file...")
        writer.writerow(["Frame_index", "totalMotion", "avgMotion"])
        frame_idx = 0
    else:
        frame_idx += 1

    for i in range(frame_idx, len(frame_files) - 1):
        frame = cv2.imread(frame_files[i])
        next_frame = cv2.imread(frame_files[i + 1])
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        next_frame = cv2.GaussianBlur(next_frame, (3, 3), 0)
        result = estimator(frame, next_frame)
        u = result[..., 0]
        v = result[..., 1]
        magnitude = np.sqrt(u**2 + v**2)
        average_motion = np.mean(magnitude)
        total_motion = np.sum(magnitude)
        writer.writerow([i, total_motion, average_motion])

# Step 4: Plot normalized motion
motion_data = pd.read_csv(output_csv_path)
x_data = motion_data['Frame_index'].tolist()
avg_motion = motion_data['avgMotion'].tolist()

# Normalize motion data
scaled_avg_motion = [val**2 for val in avg_motion]
min_val = min(scaled_avg_motion)
max_val = max(scaled_avg_motion)
normalized_motion = [(val - min_val) / (max_val - min_val) if max_val != min_val else 0 for val in scaled_avg_motion]

# Save normalized data
plot_data = pd.DataFrame({
    'Frame Index': x_data,
    'Normalized Motion': normalized_motion
})

output_normalized_csv_path = os.path.join(folder_name, "normalized_data.csv")
plot_data.to_csv(output_normalized_csv_path, index=False)

print(f"Motion data saved to {output_csv_path}")

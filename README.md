#NeuFlow v2: Dockerized Video Processing Pipeline

This repository provides a Dockerized solution for running the NeuFlow v2 video processing pipeline. Follow the instructions below to set up and run the pipeline.

##Getting Started
###Step 1: Download the Dockerfile

Download the Dockerfile directly from the repository:

https://github.com/dmanzanoa/newFlowv2/blob/main/Dockerfile

The Dockerfile includes all dependencies and integrates the codebase required to execute the code.

###Step 2: Build the Docker Image
Navigate to the directory where the Dockerfile is stored and execute the following command to build the Docker image:

```bash
docker build -t neuflowv2 .
```
###Step 3: Run the Docker Container

Once the image is built, run the container using:

```bash
docker run --gpus all -it neuflowv2
```
###Step 4: Copy Video Files into the Container

Before running the code, copy your video file into the Docker container. Use the following command to achieve this:

```bash
docker cp /local/video_path.mp4 /container-id:/container-path/
```
You can get the container path from the console e.g:
![image](https://github.com/user-attachments/assets/1e325a97-9b6d-439e-8b19-f7471396fc27)

Example:
If the video is located at /home/video1.mp4 and the container ID is 5b4630f83f3a, execute:

```bash
docker cp /home/video1.mp4 5b4630f83f3a:/workspace/NeuFlow_v2-Pytorch-Inference/  
```
###Step 5: Run the Python Program
Inside the container, run the following command to process your video:
```bash
python3 inference_video.py video1.mp4  
```
###Step 6: Copy Output Files to Local System
Once the code completes, it generates a folder named after the video file (e.g., video1) inside the container. This folder contains the following outputs:

    normalized_data.csv: Raw outputs from the flow model.
    processed_video.mp4: The processed video file with additional frame count.

To copy these files to your local machine, use the following commands:

```bash
   docker cp 5b4630f83f3a:/workspace/NeuFlow_v2-Pytorch-Inference/video1/normalized_data.csv /home/ 
```

```bash
   docker cp 5b4630f83f3a:/workspace/NeuFlow_v2-Pytorch-Inference/video1/processed_video.mp4 /home/
```
#Output Details

    Processed Video (processed_video.mp4):
        The video includes frame numbers displayed in the upper-left corner, synchronized with the CSV output.

    Normalized Data (normalized_data.csv):
        Total Motion: Sum of magnitudes per frame.
        Average Motion: Average magnitude per frame.
        Maximum Motion: Maximum magnitude per frame.

These outputs provide a comprehensive analysis of motion dynamics within the video.

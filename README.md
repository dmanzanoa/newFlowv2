# NeuFlow v2: Dockerized Video Processing Code

This repository provides a Dockerized solution for running the NeuFlow v2 video processing pipeline. Follow the instructions below to set up and run the pipeline.

## Getting Started
### Step 1: Clone the repository
```bash
git clone https://github.com/dmanzanoa/newFlowv2.git
```
### Step 2: Build the Docker Image inside the repository
Execute the following command to build the Docker image:

```bash
docker build -t neuflowv2 .
```
### Step 3: Run the Docker Container

Once the image is built, run the container using:

```bash
docker run --device=nvidia.com/gpu=all -it neuflowv2
```
### Step 4: Move to vendor/optical_flow_measure/ folder

```bash
cd vendor/optical_flow_measure/
```

### Step 5: Set API environment variables


### Step 6: Make sure Highlighter account has object class


### Step 7: Excute the agent

```bash
hl agent run agents/OpticalFlowAgent.json -f inputs/test.mp4
```

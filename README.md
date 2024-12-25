## Instructions
Download the dockerfile from the following link:

https://github.com/dmanzanoa/newFlowv2/blob/main/Dockerfile

The dockerfile will integrate everything needed to run the code (including this repository)

## Docker commands
Step1: After downloading the docker file, execute the following command in the console withing the folder where the dockerfile is stored:

```bash
docker build -t neuflowv2 .
```
Then after the image is created, excecute:

```bash
docker run --gpus all -it neuflowv2
```
Inside the container, we need copy the video file inside the container. In order to realize this, we need to apply this command

```bash
docker cp /video_path/ /container-path/
```
You can get the container path from the console
![image](https://github.com/user-attachments/assets/39bc29eb-616f-4b34-ab5c-a98972a9434e)

Go to the dockerfile folder and excute

```bash
   docker build -t neuflowv2 .
```
After the dockerfile is built, run the image created

```bash
  docker run --gpus all -it neuflowv2
```

## Inside the container


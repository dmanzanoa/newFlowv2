## Instructions
Download the dockfile from the following link:

https://github.com/dmanzanoa/neuflowv2df.git](https://github.com/dmanzanoa/newFlowv2/blob/main/Dockerfile


Go to the dockerfile folder and excute

```bash
   docker build -t neuflowv2 .
```
After the dockerfile is built, run the image created

```bash
  docker run --gpus all -it neuflowv2
```

## Inside the container


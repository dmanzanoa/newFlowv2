## Instructions
Download the dockfile in the following link
```bash
git clone https://github.com/dmanzanoa/neuflowv2df.git
```

Go to the dockerfile folder and excute

```bash
   docker build -t neuflowv2 .
```
After the dockerfile is built, run the image created

```bash
  docker run --gpus all -it neuflowv2
```

## Inside the container


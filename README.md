# Mujoco-Headless-Tutorial


### Intro
Rendering - the visualization of the robot simulation state - is a crucial part of robotics development. Mujoco renders using OpenGL - a graphics library that is mainly used to (understandably) render visuals that users interact with. Hence, the robotics workflow of 1. Simulate thousands of times faster than reality, 2. Render some of these simulations in retrospect for debugging, save them to a file, then watch the file, can be non-trivial, especially in the common case that we want both the simulation and rendering to happen on a big GPU someone on the cloud, in a *headless* metal box without a screen (A CLI-only Linux instance).

https://github.com/user-attachments/assets/05bc32b4-8bbb-4c5f-ab64-f5fbad47b207

Fortunately, it turns out to not be too hard to get this working!

**My Biases**
Of course, this approach might not work verbatim for everyone. It works on my setup with Ubuntu 22.04. I've gotten it working on both a 4060Ti and 4090 GPU. 

### Building the Docker Container
**Prerequisites**

1. **Install Docker**. Copy and paste the commands from [here](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)
2. **Install [Nvidia's Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**
    - Copy and paste the commands from the *Installing with Apt* and *Configuring Docker* sections.
    - Essentially, this library modifies your docker installation. One important feature is it enables the `--gpus all` flag in your `docker run` command, allowing your docker container to use your Nvidia GPU.
  Installing with Apt

**Build the cudaGL image**

The cudaGL Docker image extends the base cuda image with the libraries required for rendering with OpenGL. At the time of writing, this container is not downloadable from the Nvidia Cloud Containers repository, so we'll have to build it ourselves. Fortunately, Nvidia was nice enough to give us a script to make this very easy.

1. Download the script and its supporting files: `git clone https://gitlab.com/nvidia/container-images/cuda.git`
2. Use the script to build cudagl with your choice of cuda and Ubuntu version. These don't need to match the versions on your system so you're free to choose pretty much any. Replace andrewluo101 with any string you'd like. `sudo ./build.sh -d --image-name andrewluo101/cudagl --cuda-version 12.4.1 --os ubuntu --os-version 22.04 --arch x86_64 --cudagl`. This script builds three different versions of the CudaGL on top of one-another: base, runtime and devel. We'll use the latter going forward, which contains the most libraries and configuration.
3. Run the docker container. We'll run it interactively, so it takes over the terminal session. We'll also configure `run` to remove the container (but not the image) upon `exit`. `sudo docker run --gpus all -it --rm andrewluo101/cudagl:12.4.1-devel-ubuntu22.04`

**Configure the container**

This is a set of commands to benchmark the rendering speed on your container. I find that rendering is about 10% slower in the container than on the host: roughly 3k fps versus 3.3k fps, on the non-trivial Franka Emika Panda robot.

First, the container is built on a minimal linux installation, so we'll have to install some supporting tools.
```
apt update
apt install python3
apt install python3-pip
pip install mujoco
apt install git
```

Now, let's get a robot we can render:
```
git clone https://github.com/google-deepmind/mujoco_menagerie.git
cd mujoco_menagerie/franka_emika_panda/
```

And let's benchmark:
```
cat << EOF > bench.py
import mujoco
import time

mj_model = mujoco.MjModel.from_xml_path("mjx_scene.xml")
renderer = mujoco.Renderer(mj_model)
mj_data = mujoco.MjData(mj_model)

t0 = time.time()
for i in range(10000):
    mujoco.mj_step(mj_model, mj_data)
    renderer.update_scene(mj_data)
    renderer.render()

print("Rendering at {:.2f} FPS".format(10000 / (time.time() - t0)))
renderer.close()
EOF

export MUJOCO_GL=egl # tell Mujoco to use the headless rendering backend
python3 bench.py 
```

If it works, congrats! You can double-check the rendering produces reasonable outputs by `apt install ffmpeg; pip install mediapy` and using `mediapy.write_video(path, list[renderered frames])` followed by `docker cp <container_id>:/path/to/out.mp4 /path/on/host/` where `container_id` can be seen from `sudo docker container list`.  

But this is extra. At this point, all we need to do is put the container on the cloud!

### Deploying on a Cloud Container
**Overview**
What is the cheapest way of accessing a GPU for some serious computation?

*Buy one*. You'd probably be limited to gamer GPU's. As of writing, probably the best gamer GPU is the RTX 4090. This GPU has 24 GB of VRAM, making it a standard for robotics research and a starting point for playing with foundation models. A desktop with one costs around $5000 CAD - almost $4k USD. There's rarely 50% off discounts since the GPU sets a hard bottom limit on the price.

*Use Google Colab*. For $10 USD a month, you can get Google Colab Premium, which gives you 100 compute credits. This gives you access to the A100 GPU with 40 GB of VRAM for about 9 hours a month. There's two weaknesses I find with Colab:
1. It's really slow for some reason. Training the exact same Reinforcement Learning agent took 13 minutes on a Runpod RTX4090 but 26 on Colab's A100. The default T4 GPU renders the [Sapien simulator](https://github.com/haosulab/SAPIEN/tree/master)  at half the speed of my entry-level 4060 Ti GPU.
2. Since it's a jupyter notebook, it's great for quickly visualizing data but it's a pain to do complicated projects, both in terms of installing required packages and importing custom Python modules.

*Use a cloud container service*. I first became aware of this option when Andrej Karpathy [trained GPT2](https://www.youtube.com/watch?v=l8pRSuU81PU) in 90 minutes by harnessing the power of eight A100 GPU's (~$160,000 of compute), paying $20 in rental fees ($0, minus sponsorship). While Karpathy used Lambda Labs, there's lots of other options: Runpod, Vast.ai to name two. Let's talk more about this interesting option.

**Cloud Docker Hosts**
I ended up going with Runpod since I was having payment issues with Lambda Labs. Runpod is a *Cloud Docker Host*. How it works is thousands of GPU owners across the world download Docker + Nvidia Container Toolkit on their machines. For a modest price, you can ship them any docker image you want and access their GPU through it. Note that for some reason, if you book their GPUs through their command line tools, it can be over 30% cheaper than through their website. Here are some prices at time of writing:

| GPU | Price/h (USD) |
| --- | --- |
| 4070 | $0.11 |
| 4090 | $0.44 |
| A100 | $1.1 |

Unlike Google Colab, I haven't found a weird slow-down in the run-times; a 4090 on Runpod runs Mujoco + Jax neural network training about as fast as my local one. This option makes financial sense. Using a 4090 for 4 hours every day for the year would cost around $633 USD, so it would take over 6 years to get to the cost of buying one. But the real beauty is flexibility. Don't want to wait for 10 random seeds? Programatically spawn ten 4090 instances rather than wait overnight. Need more VRAM? Plop your container onto an A100. And no need to worry about the 5090 making your 4090 obselete.

Of course, if you really want to go down this road, it would greatly help to be a Docker ninja. And there's occasional issues with Runpod not having enough GPU's of the requested type.

*Using Cloud Docker Hosts using VSCode*
TODO; consider sending me a DM if this is interesting to you. Basically, you need your docker image to be running the SSH service. Runpod would expose this SSH service to the internet, which vscode can then connect to. The result is that you can write and run code on the cloud container as if it was the desktop under your desk (heh)

This is how my DockerFile looks.
```
FROM andrewluo101/cudagl:12.4.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y openssh-server
# Get a weird error w/o below line.
RUN mkdir /var/run/sshd
COPY id_rsa.pub /root/.ssh/authorized_keys
#COPY requirements.txt /
#RUN pip install --trusted-host pypi.python.org -r /requirements.txt

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
```

**Appendix: Port Forwarding**
If you're the fan of doing all of your development on a local docker container for ease of environment management, it can helpful to allow the docker container to directly output to your monitor, for example a GUI program such as Mujoco's viewer.

This is super simple to pull off.  Just run a config command and modify the docker run arguments:

```
xhost +local:docker
sudo docker run -it --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix --device /dev/dri:/dev/dri --rm andrewluo101/cudagl:12.5.1-devel-ubuntu22.04
```

Now, after setting up your container, if you run, for instance, python -m mujoco.viewer --mjcf mjx_scene.xml (following the tutorial above), the real-time simulation visualiser should spin up and allow for interaction as if it had escaped its containment!

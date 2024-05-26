# DRL Homework 3

### Topics
- Policy Gradient
  - REINFORCE
  - A2C
  - PPO

### Structure

Follow the "hw.ipynb" ipython notebook for instructions:

### Installation

To start your homework, you need to install requirements. We recommend that you use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment for this homework.

```
conda create -n rlhw3 python=3.7
conda activate rlhw3
```

If you are going to use GPU, install [Pytorch](https://pytorch.org/get-started/locally/) using the link and remove it from requirements.

You can install the requirements with the following commands in the homework directory:

```
conda install -c conda-forge swig
conda install nodejs
pip install -r requirements.txt
python -m ipykernel install --user --name=rlhw3
```
Then you need to install the homework package. You can install the package with the following command: (Make sure that you are at the homework directory.)

```
pip install -e .
```

This command will install the homework package in development mode so that the installation location will be the current directory.

### Docker

You can also use docker to work on your homework. Simply build a docker image from the homework directory using the following command:


```
docker build -t rlhw3 .
```

You may need to install docker first if you don't have it already.

After building a container we need to mount the homework directory at your local computer to the container we want to run. Note that the container will install necessary python packages in build.

You can run the container using the command below as long as your current directory is the homework directory:

```
sudo docker run -it --rm -p 8889:8889 -v $PWD:/hw3 rlhw3
```

This way you can connect the container at ```localhost:8889``` in your browser. Note that, although we are using docker, changes are made in your local directory since we mounted it.

You can also use it interactively by simply running:

```
sudo docker run -it --rm -p 8889:8889 -v $PWD:/hw3 rlhw3 /bin/bash
```

> Note: Running docker with cuda requires additional steps!

### Submission

You need to submit this repository after you filled it (and additional files that you used if there happens to be any). You also need to fill "logs" directory with the results of your experiments as instructed in the ipython notebook (progress.csv files that you used in your plots). Submissions are done via Ninova until the submission deadline. For the atari model parameters, you should put a google drive link in the ipython notebook.

### Evaluation

- Experiments 60%
  - LunarLander
    - A2C (7.5%)
    - PPO (7.5%)
  - CartPole (5%)
  - BipedalWalker (15%)
  - Pong (25%) 
- Implementations 50%
  - REINFORCE (10%)
  - A2C (20%)
  - PPO (20%)
- Bonus Comments 5%

Homework grade will be clipped to the maximum 100 points.

### Related Readings (**Must Read**)

- Reinforcement Learning: An Introduction (2nd), Richard S. Sutton and Andrew G. Barto Chapter 12 & 13 
- [A3C](https://arxiv.org/abs/1602.01783)
- [GAE](https://arxiv.org/pdf/1506.02438.pdf)
- [PPO](https://arxiv.org/pdf/1707.06347.pdf)

### Contact
TA: Kubilay Kağan Kömürcü
kubilaykagankomurcu@gmail.com

TA: Caner Özer
ozerc@itu.edu.tr

### Author
TA: Tolga Ok
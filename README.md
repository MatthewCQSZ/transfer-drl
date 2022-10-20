# transfer-drl

Here are the common instructions for setting up robosuite and robosuite-benchmarking repositories. I have tested this setup on Ubuntu 18.04, WSL1/Ubuntu20.04, and WSL2/Ubuntu20.04.

## Installing Mujoco

1. First install MuJoCo 2.10 from https://github.com/deepmind/mujoco/releases/tag/2.1.0. Feel free to test out other versions of MuJoCo, but this is the version that has been vetted.
2. You can follow MuJoCo's instructions to test out simulating one of the preloaded MuJoCo models such as humanoid-100.xml. Follow the Getting Started section of the instructions to find
the command, https://mujoco.readthedocs.io/en/latest/programming.html#getting-started. 
3. Follow the remaining instructions to install and setup MuJoCo-py, https://github.com/openai/mujoco-py. 

## Install linux dependencies
```shell
  sudo apt install curl git libgl1-mesa-dev libgl1-mesa-glx libglew-dev \
         libosmesa6-dev software-properties-common net-tools unzip vim \
         virtualenv wget xpra xserver-xorg-dev libglfw3-dev patchelf
 ```
 
 ## Setting up and installing robosuite
 
 We have found that setting robosuite up using virtualenv instead of a conda environment seems to work better and not have bugs running training. Feel free to follow robosuite 
 instructions to use conda if that is what you prefer. The following setup will assume though that you used virtualenv.
 
 ```shell
   cd ~
   mkvirtualenv robosuite -p python3
  ```
   To load this environment use:
   
  ```shell
    workon robosuite
  ```
    
 In order to set your Python path home, add the following to ~/virtualenv/robosuite/bin/activate.
    
   ```shell
     export PYTHONPATH=.
   ```
   
 Next, update pip.
 ```shell
   pip install --upgrade pip setuptools wheel
 ```
 
 Now, install robosuite using the following.
 ```shell
  pip install robosuite
 ```
 
 ## Cloning the robosuite and robosuite-benchmark repositories. 
 
 Clone the following repositories into a local directory. 
 ```shell
   mkdir ~/git-workspace/
   cd ~/git-workspace/
   git clone https://github.com/ARISE-Initiative/robosuite.git robosuite-git
   git clone https://github.com/ARISE-Initiative/robosuite-benchmark.git
   
 ```
 
 You can now test the robosuite environment by running the demo_random_action.py file. You can follow robosuite's quickstart instructions to test to make sure robosuite is
 working properly. https://robosuite.ai/docs/quickstart.html. 
 You can also run the following to test our the random demo.
 ```shell
   python -m robosuite.demos.demo_random_action
 ```
 
 ## Setting up robosuite-benchmarking repository
 Follow the instructions from robosuite-benchmarking repository in order to install viskit and rlkit that are used for training and visualization. https://github.com/ARISE-Initiative/robosuite-benchmark
   
   
 

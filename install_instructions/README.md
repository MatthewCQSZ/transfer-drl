## Installing Mujoco

Here are the common instructions for setting up robosuite and robosuite-benchmarking repositories. I have tested this setup on Ubuntu 18.04, WSL1/Ubuntu20.04, and WSL2/Ubuntu20.04 and Ubuntu22.04.

1. First install MuJoCo 2.10 from https://github.com/deepmind/mujoco/releases/tag/2.1.0. Feel free to test out other versions of MuJoCo, but this is the version that has been vetted.
2. Download the license key from the following:https://www.roboti.us/license.html. 
3. Unzip the downloaded  mujoco210-linux-x86_64.tar.gz into ~/.mujoco/mjpro210, and place your license key (the mjkey.txt file downloaded from step 2) at ~/.mujoco/mjkey.txt.
4. Run pip3 install -U 'mujoco-py<2.2,>=2.1'

## Install linux dependencies
```shell
  sudo apt install curl git libgl1-mesa-dev libgl1-mesa-glx libglew-dev \
         libosmesa6-dev software-properties-common net-tools unzip vim \
         virtualenv wget xpra xserver-xorg-dev libglfw3-dev patchelf
 ```

## Method to test out your MuJoCo software and download mujoco-py
1. You can follow MuJoCo's instructions to test out simulating one of the preloaded MuJoCo models such as humanoid-100.xml. Follow the Getting Started section of the instructions to find
the command, https://mujoco.readthedocs.io/en/latest/programming.html#getting-started. 
2. Follow the remaining instructions to install and setup MuJoCo-py, https://github.com/openai/mujoco-py. 


 ## Setting up and installing robosuite
 
 We have found that setting robosuite up using virtualenv instead of a conda environment seems to work better and not have bugs running training. Feel free to follow robosuite 
 instructions to use conda if that is what you prefer. The following setup will assume though that you used virtualenv.
 
 ```shell
   cd ~
   mkvirtualenv --python=/usr/bin/python3 robosuite
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
 
 Now, install required packages using the following.
 ```shell
  pip install -r requirements.txt
 ```
   
   
 

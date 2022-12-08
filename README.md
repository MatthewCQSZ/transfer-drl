# transfer-drl

Here are the common instructions for setting up robosuite and robosuite-benchmarking repositories. I have tested this setup on Ubuntu 18.04, WSL1/Ubuntu20.04, and WSL2/Ubuntu20.04 and Ubuntu22.04.

## Installing Mujoco

1. First install MuJoCo 2.10 from https://github.com/deepmind/mujoco/releases/tag/2.1.0. Feel free to test out other versions of MuJoCo, but this is the version that has been vetted.
2. Download the license key from the following:https://www.roboti.us/license.html. 
3. Unzip the downloaded  mujoco210-linux-x86_64.tar.gz into ~/.mujoco/mjpro210, and place your license key (the mjkey.txt file downloaded from step 2) at ~/.mujoco/mjkey.txt.
4. Run pip3 install -U 'mujoco-py<2.2,>=2.1'
5. $ python3
- import mujoco_py
- While runnig above step if there is error "mujoco py install error - fatal error: GL/osmesa.h: No such file or directory"
- Try $ sudo apt-get install libosmesa6-dev
- Or this "No such file or directory: 'patchelf' on mujoco-py installation"
- $ sudo apt-get install patchelf
7. If everything is fine, run the following in the python3 env:
- import os
- mj_path = mujoco_py.utils.discover_mujoco()
- xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
- model = mujoco_py.load_model_from_path(xml_path)
- sim = mujoco_py.MjSim(model)
9. After that print following:
- print(sim.data.qpos)
```
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 ```
- sim.step()
- print(sim.data.qpos)
```
[-2.09531783e-19  2.72130735e-05  6.14480786e-22 -3.45474715e-06
  7.42993721e-06 -1.40711141e-04 -3.04253586e-04 -2.07559344e-04
  8.50646247e-05 -3.45474715e-06  7.42993721e-06 -1.40711141e-04
 -3.04253586e-04 -2.07559344e-04 -8.50646247e-05  1.11317030e-04
 -7.03465386e-05 -2.22862221e-05 -1.11317030e-04  7.03465386e-05
 -2.22862221e-05]
 ```

##Mujoco is installed successfully

11. You can follow MuJoCo's instructions to test out simulating one of the preloaded MuJoCo models such as humanoid-100.xml. Follow the Getting Started section of the instructions to find
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
   
   
 

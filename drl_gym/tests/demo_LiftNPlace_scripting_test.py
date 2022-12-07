"""
Scripted demo to test the lift and place environment that we created. This version tests running the arm into the wall.
Uncomment actions under count < 200 to see a lift and place scripted demo.
"""

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *

if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    options["env_name"] = "LiftAndPlaceBarrier"
    options["robots"] = "Panda"

    # Choose controller
    controller_name = "OSC_POSE"

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # Help message to user
    print()
    print('Press "H" to show the viewer control panel.')

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        reward_shaping=True,
        use_camera_obs=False,
        render_camera="frontview",
        control_freq=20,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec
    #print(low)
    #print(high)

    # do visualization
    for eps in range(10):
        print(f"Episode number is {eps+1}")
        env.reset()
        count = 0
        gripper_action = 0.0
        while True:
            #action = np.random.uniform(low, high)
            if count < 200:
                action = np.array((0, 0, -0.1, 0, 0, 0, 0))
            # elif 130 < count < 190:
            #     action = np.array((0.13, 0, 0, 0, 0, 0, -0.8))
            # elif 190 < count < 220:
            #     action = np.array((0, 0, 0, 0, 0, 0, 0.8))
            # elif 220 < count < 255:
            #     action = np.array((0, 0, 0.15, 0, 0, 0, 0))
            # elif 255 < count < 440:
            #     action = np.array((0, -0.1, 0, 0, 0, 0, 0))
            # elif 440 < count < 450:
            #     action = np.array((0, 0, 0, 0, 0, 0, -0.8))
            else:
                action = np.array((0, -1.0, 0, 0, 0, 0, 0))
            #action = np.array((0, 0, 0, 0, 0, 0, 0))
            print("action is: ", action)
            obs, reward, done, info = env.step(action)
            count = count + 1
            print("count: ", count)
            print("done: ", done)
            print("reward: ", reward)
            print("observation: ", obs)
            #print("info: ", info)
            env.render()
            if done:
                break


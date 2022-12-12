"""
Demo created to test random actions within our created environments.
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
    options["env_name"] = "LiftWithTerminals"
    options["robots"] = "Panda"

    # Choose controller
    controller_name = choose_controller()

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
        ignore_done=False,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec

    # do visualization
    for eps in range(10):
        print(f"Episode number is {eps+1}")
        env.reset()
        while True:
            action = np.random.uniform(low, high)
            #print("action is: ", action)
            obs, reward, done, info = env.step(action)
            print("done: ", done)
            print("reward: ", reward)
            #print("info: ", info)
            env.render()
            if done:
                break


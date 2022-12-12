## Running Demo Files

!! These instructions assume that you have a local copy of robosuite github (https://github.com/ARISE-Initiative/robosuite) downloaded into your virtualenv.
1. In order to run the demo files to test the environments that we created in robosuite,
it is necessary to move the demo files to robosuite's demo directory (https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/demos). 
2. You will also need to move our environment files to robosuite's manipulation directory under 
their environments directory (https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/environments/manipulation).
3. Lastly, you will need to register these environments with robosuite. This can be done by following the format within this file
   https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/__init__.py to add your new environments.

## Running Transfer Metric Plotter Unit Test

1. This file can be run from an IDE or from a terminal by running the following from this directory:
````
python transfer_metric_plotter_test.py
````

2. The results of the unit test should be 2 passed tests.
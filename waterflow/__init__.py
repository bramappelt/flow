"""
Global variables that need to be accessible over the complete
flow package.

.. data:: ROOT_DIR

    Absolute path to the location of the flow package.

.. data:: DATA_DIR

    Absolute path to the directory for model related data.

.. data:: OUTPUT_DIR

    Absolute path to a save location for the model output data.
    The name of this directory will be **flowoutput** and will
    be created under the documents directory of your platform.

    Windows
        C:/user/Documents/flowoutput
    Linux
        /home/usr/Documents/flowoutput

"""
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT_DIR, 'data')

OUTPUT_DIR = os.path.join(os.path.expanduser('~'), 'Documents/flowoutput/')
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

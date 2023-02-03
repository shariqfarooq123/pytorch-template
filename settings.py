import os
import pathlib

ROOT = pathlib.Path(__file__).parent.resolve()
HOME_DIR = os.path.expanduser("~")
SAVE_DIR = ROOT/"checkpoints"
PROJECT_NAME = "MyPytorchProject"
# main code for the agent
import sys
# from utils import utils_capture
import threading
from time import time, sleep
from agents import usis
import keyboard 
# from app import client
import os
import config
config = config.Config()


def train_model():
    #training
    agent = usis.USIS()
    agent.build_model()
    print(agent.get_model_memory_usage())
    agent.train_model()

    return


if __name__ == '__main__':
    train_model()
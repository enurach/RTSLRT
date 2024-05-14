import csv
import os
import cv2
import random

import pandas as pd
import numpy as np
import tensorflow as tf

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from .handLandmarks import HandLandmarksDataset


RANDOM_SEED = 42


def build_dataset(annotation, train = False):
    dataset = HandLandmarksDataset(annotation = annotation, train=train)
    return dataset

def build_dataloader(dataset, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader


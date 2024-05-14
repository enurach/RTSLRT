import torch
import torch.nn as nn

from model import SLDense

def build_model(input_size, output_size, hidden_size):
    """Builds and returns an instance of SLDense."""
    model = SLDense(input_size, output_size, hidden_size)
    return model


def load_checkpoint(model, checkpoint_path):
    """Loads weights from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def save_checkpoint(model, checkpoint_path):
    """Saves model weights to a checkpoint file."""
    torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)



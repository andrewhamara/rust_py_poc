#!/usr/bin/env python3
import os
from pathlib import Path
import torch
from torch.backends import cuda, cudnn
from models.detection.yolox.utils.boxes import postprocess
from data.utils.representations import StackedHistogram
import pickle

import numpy as np

# Test
import time

# env settings
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# faster matmul
cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')

import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from modules.utils.fetch import fetch_model_module


# yolo prediction classes
classes = {
    0: "pedestrian",
    1: "vehicle",
    2: "micromobility"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# serialize hidden LSTM state for rust
def serialize_states(hidden_states):
    return pickle.dumps(hidden_states)

# deserialize hidden LSTM state from rust
def deserialize_states(serialized_states):
    return pickle.loads(serialized_states)

# decode bytes to tensors
def decode_event_bytes(event_bytes):
    dtype = torch.int64
    #print(len(event_bytes))
    num_events = len(event_bytes) // 32

    events = torch.frombuffer(event_bytes, dtype=dtype).reshape(num_events, 4)

    t = events[:, 0]
    x = events[:, 1]
    y = events[:, 2]
    p = events[:, 3]

    #print('events converted to tensors')
    return x.to(device), y.to(device), p.to(device), t.to(device)

def init_model():
    if model is None:
        print('initializing model...')
        #initialize(config_path="eTraM/rvt_eTram/config/model/maxvit_yolox", version_base=None)
        initialize(config_path="config", version_base=None)
        config = compose(config_name="val")
        print(OmegaConf.to_yaml(config))
        ckpt_path = Path(config.checkpoint)
        model_module = fetch_model_module(config=config)
        model = type(model_module).load_from_checkpoint(str(ckpt_path), **{'full_config': config})
        model = model.to(device)
        model.eval()
        print('model initialized!')
    return model

import matplotlib.pyplot as plt
import matplotlib.patches as patches
def plot_events_with_bboxes(histogram_rep, predictions, save_path='event_plot.png', num_bins=3):
    """
    Saves events from the histogram with bounding boxes overlaid.

    Args:
        histogram_rep: torch.Tensor containing the stacked histogram representation of the events.
        predictions: List of bounding boxes from the YOLO model output.
        save_path: The file path to save the plot as an image.
        num_bins: Number of stacked histogram bins to include in the plot.
    """
    # Sum across the first `num_bins` bins for visualization
    event_image = histogram_rep[0, :2*num_bins].sum(dim=0).cpu().numpy()  # Sum across the temporal bins

    # Normalize and increase brightness
    event_image = np.clip(event_image * 128, 0, 255)

    # Create a plot
    fig, ax = plt.subplots(1)
    ax.imshow(event_image, cmap='jet', vmin=0, vmax=255)


    # Plot each bounding box
    if predictions is not None:
        for pred in predictions:
            if pred is not None:
                for box in pred:
                    x1, y1, x2, y2, conf, class_score = box[:6]  # Extracting bbox (x1, y1, x2, y2)

                    # Define width and height for the rectangle
                    width = x2 - x1
                    height = y2 - y1

                    # Create a Rectangle patch
                    rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')

                    # Add the patch to the Axes
                    ax.add_patch(rect)

    # Save the figure to the given path
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # Close the plot to avoid displaying it

def main(event_bytes, hidden_states=None):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # setup
    model = init_model()

    # bytes to tensors
    x, y, p, t = decode_event_bytes(event_bytes)

    # sort events by time
    i = torch.argsort(t)
    x, y, p, t = x[i], y[i], p[i], t[i]

    t0, tn = t[0], t[-1]

    # tensors to histogram
    histogram = StackedHistogram(bins=10, height=720, width=1280)
    histogram_rep = histogram.construct(x=x, y=y, pol=p, time=t)
    histogram_rep = histogram_rep.to(torch.float32)
    histogram_rep = histogram_rep.unsqueeze(0)

    # model stuff
    num_classes = config.model.head.num_classes
    confidence_threshold = config.model.postprocess.confidence_threshold
    nms_threshold = config.model.postprocess.nms_threshold

    # init variables
    output = None
    predictions = None
    serialized_hidden_states = None

    if hidden_states:
        hidden_states = deserialize_states(hidden_states)

    # forward pass
    with torch.inference_mode():
        output, _, hidden_states = model(
            event_tensor=histogram_rep,
            previous_states=hidden_states,
            retrieve_detections=True
        )

        serialized_hidden_states = serialize_states(hidden_states)

        class_logits = output[:, :, 5:]

        class_probabilities = torch.softmax(class_logits, dim=-1)
        predicted_classes = torch.argmax(class_probabilities, dim=-1)


        predictions = postprocess(prediction=output,
                                  num_classes=num_classes,
                                  conf_thre=confidence_threshold/50.0,
                                  #conf_thre=.001,
                                  nms_thre=nms_threshold)

    predictions_list = []
    for batch_idx, pred in enumerate(predictions):
        if pred is not None:
            for i, box in enumerate(pred): 
                class_idx = predicted_classes[batch_idx][i].item()
                predictions_list.append(class_idx)
        else:
            print("No predictions found.")

    return predictions_list, serialized_hidden_states

if __name__ == '__main__':
    dummy_event_bytes = b'\x00' * 2400
    predictions, hidden_states = main(dummy_event_bytes)
    print(predictions)

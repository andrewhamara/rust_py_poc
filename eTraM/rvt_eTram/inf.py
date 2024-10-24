#!/usr/bin/env python3
import os
from pathlib import Path
import torch
from torch.backends import cuda, cudnn
import pickle
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelSummary
import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from modules.utils.fetch import fetch_model_module, fetch_data_module
from config.modifier import dynamically_modify_train_config
from data.utils.representations import StackedHistogram
from models.detection.yolox.utils.boxes import postprocess
from config.modifier import dynamically_modify_train_config
from scripts.genx.preprocess_dataset import downsample_ev_repr


# Event byte processing and utilities from your previous script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels = {
    0 : "pedestrian",
    1 : "vehicle",
    2 : "micromobility"
}

def print_predictions(predictions, class_labels):
    for idx, pred in enumerate(predictions):
        if pred is not None:
            for i, box in enumerate(pred):
                class_idx = int(box[5])
                class_label = class_labels.get(class_idx, "Unknown")
                print(f"prediction {i}: class index = {class_idx}, label = {class_label}")

def serialize_states(hidden_states):
    return pickle.dumps(hidden_states)

def deserialize_states(serialized_states):
    return pickle.loads(serialized_states)

def decode_event_bytes(event_bytes):
    dtype = torch.int64
    num_events = len(event_bytes) // 32
    events = torch.frombuffer(event_bytes, dtype=dtype).reshape(num_events, 4)
    t = events[:, 0]
    x = events[:, 1]
    y = events[:, 2]
    p = events[:, 3]
    return x.to(device), y.to(device), p.to(device), t.to(device)

# Integration of PyTorch Lightning and the previous script logic
@hydra.main(config_path='config', config_name='val', version_base='1.2')
def main(config: DictConfig):
    # Dynamically modify config if needed
    dynamically_modify_train_config(config)
    # Resolve and validate the config
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    # ---------------------
    # GPU options
    # ---------------------
    gpus = config.hardware.gpus
    assert isinstance(gpus, int), 'no more than 1 GPU supported'
    gpus = [gpus]

    # ---------------------
    # Model setup
    # ---------------------
    logger = CSVLogger(save_dir='./validation_logs')
    ckpt_path = Path(config.checkpoint)
    print('-------')
    module = fetch_model_module(config=config)
    model = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config})

    model.eval()
    return model.to(device)  # Return model for inference usage

# forward pass
def inference_pipeline(event_bytes, hidden_states=None):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # init model
    initialize(config_path="config", version_base="1.2")
    config = compose(config_name="val")
    dynamically_modify_train_config(config)
    model = main(config)

    # decode event bytes
    x, y, p, t = decode_event_bytes(event_bytes)
    i = torch.argsort(t)
    x, y, p, t = x[i], y[i], p[i], t[i]

    t0, tn = t[0], t[-1]

    # stacked histogram
    histogram = StackedHistogram(bins=10, height=720, width=1280)
    histogram_rep = histogram.construct(x=x, y=y, pol=p, time=t)
    histogram_rep = histogram_rep.to(torch.float32).unsqueeze(0).to(device)

    # downsample by 2
    downsampled_events = downsample_ev_repr(histogram_rep, scale_factor=0.5)
    
    input_padder = model.input_padder
    padded_event_tensor = input_padder.pad_tensor_ev_repr(downsampled_events)

    # If hidden states are provided, deserialize them
    if hidden_states:
        hidden_states = deserialize_states(hidden_states)

    # Forward pass through the model
    with torch.inference_mode():
        output, _, hidden_states = model(
            event_tensor=padded_event_tensor,
            previous_states=hidden_states,
            retrieve_detections=True
        )

    # Serialize hidden states for future use
    serialized_hidden_states = serialize_states(hidden_states)

    # Process predictions (modify this section as needed for post-processing)
    predictions = postprocess(prediction=output.clone(), num_classes=config.model.head.num_classes,
                              conf_thre=config.model.postprocess.confidence_threshold / 50.0,
                              nms_thre=config.model.postprocess.nms_threshold)


    # Return predictions and serialized hidden states
    return predictions, serialized_hidden_states

if __name__ == '__main__':
    dummy_event_bytes = b'\x00' * 2400
    predictions, hidden_states = inference_pipeline(dummy_event_bytes, None)
    print_predictions(predictions, labels)
    predictions, hidden_states = inference_pipeline(dummy_event_bytes, hidden_states)
    print_predictions(predictions, labels)

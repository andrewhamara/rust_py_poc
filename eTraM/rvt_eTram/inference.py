#!/usr/bin/env python3
import os
from pathlib import Path
import torch
from torch.backends import cuda, cudnn
from models.detection.yolox.utils.boxes import postprocess
from data.utils.representations import StackedHistogram
import pickle


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
    dtype = torch.int32
    num_events = len(event_bytes) // 16

    events = torch.frombuffer(event_bytes, dtype=dtype).reshape(num_events, 4)

    x = events[:, 0]
    y = events[:, 1]
    p = events[:, 2]
    t = events[:, 3]

    return x.to(device), y.to(device), p.to(device), t.to(device)

model, config = None, None

def init_model():
    global model, config
    if model is None:
        print('initializing model...')
        initialize(config_path="eTraM/rvt_eTram/config/model", version_base=None)
        config = compose(config_name="rnndet")
        ckpt_path = Path(config.checkpoint)
        model_module = fetch_model_module(config=config)
        model = type(model_module).load_from_checkpoint(str(ckpt_path), **{'full_config': config})
        model = model.to(device)
        model.eval()
        print('model initialized!')

def main(event_bytes, hidden_states=None):
    global model
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()

    # setup
    init_model()

    # bytes to tensors
    x, y, p, t = decode_event_bytes(event_bytes)

    # sort events by time
    i = torch.argsort(t)
    x, y, p, t = x[i], y[i], p[i], t[i]

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
                                  #conf_thre=confidence_threshold,
                                  conf_thre=.001,
                                  nms_thre=nms_threshold)

    predictions_list = []
    for batch_idx, pred in enumerate(predictions):
        print(f"batch {batch_idx} predictions:")
        if pred is not None:
            for i, box in enumerate(pred): 
                class_idx = predicted_classes[batch_idx][i].item()
                predictions_list.append(class_idx)
        else:
            print("No predictions found.")

    return predictions_list, serialized_hidden_states

if __name__ == '__main__':
    dummy_event_bytes = b'\x00' * 2000
    predictions, hidden_states = main(dummy_event_bytes)
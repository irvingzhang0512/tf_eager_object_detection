import pickle
import numpy as np


def pytorch_to_tf_np(v):
    if v.ndim == 4:
        # OUT, IN, H, W --> H, W, IN, OUT
        return np.ascontiguousarray(v.transpose(2, 3, 1, 0))
    if v.ndim == 2:
        return np.ascontiguousarray(v.transpose())
    return v


def convert_pth_to_dict(pth_dir, dict_path):
    import torch
    torch_file = torch.load(pth_dir)

    tf_dict = {}
    for key in torch_file['model'].keys():
        tf_dict[key] = pytorch_to_tf_np(torch_file['model'][key].cpu().numpy())

    with open(dict_path, 'wb') as f:
        pickle.dump(tf_dict, f)

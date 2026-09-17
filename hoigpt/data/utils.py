import torch
import rich
import pickle
import numpy as np


def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


# padding to max length in one batch
def collate_tensors(batch):
    if isinstance(batch[0], np.ndarray):
        batch = [torch.from_numpy(b) for b in batch]

    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate_motion_field(batch):
    first = batch[0]
    if isinstance(first, dict):
        return {
            key: collate_tensors([torch.as_tensor(sample[key]).long() for sample in batch]).long()
            for key in first.keys()
        }

    tensor_batch = [torch.as_tensor(sample) for sample in batch]
    if tensor_batch[0].is_floating_point():
        tensor_batch = [sample.float() for sample in tensor_batch]
    else:
        tensor_batch = [sample.long() for sample in tensor_batch]
    return collate_tensors(tensor_batch)

def humanml3d_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    EvalFlag = False if notnone_batches[0][5] is None else True

    # Sort by text length
    if EvalFlag:
        notnone_batches.sort(key=lambda x: x[5], reverse=True)

    # Motion only
    adapted_batch = {
        "motion":
        collate_motion_field([b[1] for b in notnone_batches]),
        "length": [b[2] for b in notnone_batches],
    }

    # Text and motion
    if notnone_batches[0][0] is not None:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "all_captions": [b[7] for b in notnone_batches],
        })

    # Evaluation related
    if EvalFlag:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "word_embs":
            collate_tensors(
                [torch.tensor(b[3]).float() for b in notnone_batches]),
            "pos_ohot":
            collate_tensors(
                [torch.tensor(b[4]).float() for b in notnone_batches]),
            "text_len":
            collate_tensors([torch.tensor(b[5]) for b in notnone_batches]),
            "tokens": [b[6] for b in notnone_batches],
        })

    # Tasks
    if len(notnone_batches[0]) >= 9:
        adapted_batch.update({"tasks": [b[8] for b in notnone_batches]})
    if len(notnone_batches[0]) >= 10:
        adapted_batch.update({"name": [b[9] for b in notnone_batches]})
    if len(notnone_batches[0]) >= 11:
        adapted_batch.update({"pc": collate_tensors([torch.tensor(b[10][0]).float() for b in notnone_batches])})
        adapted_batch.update({"pc_full": collate_tensors([torch.tensor(b[10][1]).float() for b in notnone_batches])})

    if len(notnone_batches[0]) >= 12:
        adapted_batch.update({"subset": [b[11] for b in notnone_batches]})

    # print(len(notnone_batches[0]))

    # import pdb; pdb.set_trace()
    return adapted_batch


def load_pkl(path, description=None, progressBar=False):
    if progressBar:
        with rich.progress.open(path, 'rb', description=description) as file:
            data = pickle.load(file)
    else:
        with open(path, 'rb') as file:
            data = pickle.load(file)
    return data

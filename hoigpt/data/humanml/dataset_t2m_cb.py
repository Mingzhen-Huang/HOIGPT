import codecs as cs
import json
import os
import random
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import spacy
from rich.progress import track
from torch.utils import data


obj_class = [
    "box",
    "capsulemachine",
    "espressomachine",
    "ketchup",
    "laptop",
    "microwave",
    "mixer",
    "notebook",
    "phone",
    "scissors",
    "waffleiron",
]


def _is_triplet_tokens(tokens):
    return isinstance(tokens, dict)


def _token_length(tokens):
    if _is_triplet_tokens(tokens):
        return int(tokens["left"].shape[-1])
    return int(tokens.shape[-1])


def _token_stream_count(tokens):
    if _is_triplet_tokens(tokens):
        return int(tokens["left"].shape[0])
    return int(tokens.shape[0])


def _slice_tokens(tokens, start_idx, end_idx):
    if _is_triplet_tokens(tokens):
        return {
            key: value[..., start_idx:end_idx]
            for key, value in tokens.items()
        }
    return tokens[..., start_idx:end_idx]


def _select_stream(tokens, stream_idx):
    if _is_triplet_tokens(tokens):
        return {
            key: value[stream_idx].astype(np.int64)
            for key, value in tokens.items()
        }
    return tokens[stream_idx].astype(np.int64)


def _drop_boundary_token(tokens, drop_head):
    if _is_triplet_tokens(tokens):
        return {
            key: value[1:] if drop_head else value[:-1]
            for key, value in tokens.items()
        }
    return tokens[1:] if drop_head else tokens[:-1]


class Text2MotionDatasetCB(data.Dataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length=196,
        min_motion_length=20,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        stage="lm_pretrain",
        code_path="VQVAE",
        task_path=None,
        std_text=False,
        code_format="auto",
        **kwargs,
    ):
        self.tiny = tiny
        self.unit_length = unit_length
        self.mean = mean
        self.std = std
        self.code_format = code_format

        split_file = pjoin(data_root, split + ".txt")
        motion_dir = pjoin(data_root, code_path)
        text_dir = pjoin(data_root, "texts")

        if task_path:
            instructions = task_path
        elif stage == "lm_pretrain":
            instructions = pjoin(data_root, "template_pretrain.json")
        elif stage in ["lm_instruct", "lm_rl"]:
            instructions = pjoin(data_root, "template_instructions.json")
        else:
            raise NotImplementedError(f"stage {stage} not implemented")

        if not task_path and not os.path.isfile(instructions):
            instructions = str(Path(__file__).resolve().parents[3] / "assets" / Path(instructions).name)

        self.id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(line.strip())

        if tiny or debug:
            enumerator = enumerate(self.id_list)
            maxdata = 100
        else:
            enumerator = enumerate(track(self.id_list, f"Loading HumanML3D {split}"))
            maxdata = 1e10

        new_name_list = []
        data_dict = {}

        for _, name in enumerator:
            if len(new_name_list) > maxdata:
                break
            try:
                m_token_list = self._load_motion_tokens(motion_dir, name)

                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()

                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split("#")
                            caption = line_split[0]
                            t_tokens = line_split[1].split(" ")
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict["caption"] = caption
                            text_dict["tokens"] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                start_idx = int(f_tag * fps / unit_length)
                                end_idx = int(to_tag * fps / unit_length)
                                if start_idx >= end_idx:
                                    continue

                                m_token_list_new = []
                                for stream_idx in range(_token_stream_count(m_token_list)):
                                    sliced = _slice_tokens(
                                        _select_stream(m_token_list, stream_idx),
                                        start_idx,
                                        end_idx,
                                    )
                                    if _token_length(sliced) > 0:
                                        m_token_list_new.append(sliced)

                                if len(m_token_list_new) == 0:
                                    continue
                                new_name = f"{name}_{f_tag}_{to_tag}"
                                data_dict[new_name] = {
                                    "m_token_list": m_token_list_new,
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                        except Exception:
                            pass

                if flag:
                    data_dict[name] = {
                        "m_token_list": [
                            _select_stream(m_token_list, idx)
                            for idx in range(_token_stream_count(m_token_list))
                        ],
                        "text": text_data,
                    }
                    new_name_list.append(name)
            except Exception:
                pass

        self.data_dict = data_dict
        self.name_list = new_name_list
        if not self.name_list:
            raise ValueError(
                f"No matching motion tokens/captions loaded from {motion_dir}. "
                "Run scripts/get_motion_code.py for this split before LM training."
            )
        self.nlp = spacy.load("en_core_web_sm") if std_text else None
        self.std_text = std_text
        with open(instructions, "r", encoding="utf-8") as handle:
            self.instructions = json.load(handle)
        self.tasks = []
        for task in self.instructions.keys():
            for subtask in self.instructions[task].keys():
                self.tasks.append(self.instructions[task][subtask])
        print("total length of the dataset is " + str(len(self.name_list)))

    def _load_motion_tokens(self, motion_dir, name):
        npz_path = pjoin(motion_dir, f"{name}.npz")
        npy_path = pjoin(motion_dir, f"{name}.npy")

        if self.code_format == "hoi_triplet":
            if not os.path.exists(npz_path):
                raise FileNotFoundError(npz_path)
            return self._load_triplet_tokens(npz_path)

        if self.code_format == "flat_motion":
            if not os.path.exists(npy_path):
                raise FileNotFoundError(npy_path)
            return self._load_flat_tokens(npy_path)

        if os.path.exists(npz_path):
            return self._load_triplet_tokens(npz_path)
        if os.path.exists(npy_path):
            return self._load_flat_tokens(npy_path)
        raise FileNotFoundError(f"No token file found for {name} in {motion_dir}")

    def _load_flat_tokens(self, token_path):
        tokens = np.load(token_path)
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        return tokens.astype(np.int64)

    def _load_triplet_tokens(self, token_path):
        tokens = np.load(token_path)
        left = tokens["left"]
        right = tokens["right"]
        obj = tokens["obj"]
        if left.ndim == 1:
            left = left[None, :]
            right = right[None, :]
            obj = obj[None, :]
        return {
            "left": left.astype(np.int64),
            "right": right.astype(np.int64),
            "obj": obj.astype(np.int64),
        }

    def __len__(self):
        return len(self.name_list) * len(self.tasks)

    def __getitem__(self, item):
        data_idx = item % len(self.name_list)
        task_idx = item // len(self.name_list)

        data = self.data_dict[self.name_list[data_idx]]
        m_token_list, text_list = data["m_token_list"], data["text"]

        m_tokens = random.choice(m_token_list)
        text_data = random.choice(text_list)
        caption = text_data["caption"]
        if self.std_text:
            doc = self.nlp(caption)
            word_list = []
            for token in doc:
                word = token.text
                if not word.isalpha():
                    continue
                if (token.pos_ == "NOUN" or token.pos_ == "VERB") and (word != "left"):
                    word_list.append(token.lemma_)
                else:
                    word_list.append(word)
            caption = " ".join(word_list)

        all_captions = [
            " ".join([token.split("/")[0] for token in text_dic["tokens"]])
            for text_dic in text_list
        ]

        coin = np.random.choice([False, False, True])
        if coin and _token_length(m_tokens) > 1:
            m_tokens = _drop_boundary_token(m_tokens, drop_head=np.random.choice([True, False]))

        m_tokens_len = _token_length(m_tokens)
        tasks = self.tasks[task_idx]
        return caption, m_tokens, m_tokens_len, None, None, None, None, all_captions, tasks

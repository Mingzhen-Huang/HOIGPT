import math
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    T5ForConditionalGeneration,
)

from .tools.token_emb import NewTokenEmb


class MLM(nn.Module):
    def __init__(
        self,
        model_path: str,
        model_type: str = "t5",
        stage: str = "lm_pretrain",
        new_token_type: str = "insert",
        motion_codebook_size: int = 512,
        framerate: float = 20.0,
        down_t: int = 4,
        predict_ratio: float = 0.2,
        inbetween_ratio: float = 0.25,
        max_length: int = 1024,
        lora: bool = False,
        quota_ratio: float = 0.5,
        noise_density: float = 0.15,
        mean_noise_span_length: int = 3,
        token_format: str = "flat_motion",
        **kwargs,
    ) -> None:
        super().__init__()

        self.m_codebook_size = motion_codebook_size
        self.max_length = max_length
        self.framerate = framerate
        self.down_t = down_t
        self.predict_ratio = predict_ratio
        self.inbetween_ratio = inbetween_ratio
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.quota_ratio = quota_ratio
        self.stage = stage
        self.token_format = token_format

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
        if model_type == "t5":
            self.language_model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.lm_type = "encdec"
        elif model_type == "gpt2":
            self.language_model = GPT2LMHeadModel.from_pretrained(model_path)
            self.lm_type = "dec"
        else:
            raise ValueError("type must be either seq2seq or conditional")

        if self.lm_type == "dec":
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.motion_start_token = f"<motion_id_{self.m_codebook_size}>"
        self.motion_end_token = f"<motion_id_{self.m_codebook_size + 1}>"
        self.motion_mask_token = f"<motion_id_{self.m_codebook_size + 2}>"

        added_tokens = self._build_added_tokens()
        self.num_added_motion_tokens = len(added_tokens)
        self.tokenizer.add_tokens(added_tokens)

        if new_token_type == "insert":
            self.language_model.resize_token_embeddings(len(self.tokenizer))
        elif new_token_type == "mlp":
            shared = NewTokenEmb(self.language_model.shared, self.num_added_motion_tokens)
            self.language_model.resize_token_embeddings(len(self.tokenizer))
            self.language_model.shared = shared

        if lora:
            from peft import LoraConfig, get_peft_model

            peft_config = LoraConfig(
                bias="none",
                task_type="CAUSAL_LM",
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
            )
            self.language_model = get_peft_model(self.language_model, peft_config)

    def _build_added_tokens(self):
        if self.token_format == "flat_motion":
            return [f"<motion_id_{i}>" for i in range(self.m_codebook_size + 3)]

        if self.token_format != "hoi_triplet":
            raise ValueError(f"Unsupported token format: {self.token_format}")

        self.hoi_start_token = "<hoi_start>"
        self.hoi_end_token = "<hoi_end>"
        self.hoi_left_token = "<left_hand>"
        self.hoi_right_token = "<right_hand>"
        self.hoi_object_token = "<object>"
        self.hoi_mask_token = "<hoi_mask>"
        return (
            [
                self.hoi_start_token,
                self.hoi_end_token,
                self.hoi_left_token,
                self.hoi_right_token,
                self.hoi_object_token,
                self.hoi_mask_token,
            ]
            + [f"<hand_id_{i}>" for i in range(self.m_codebook_size)]
            + [f"<obj_id_{i}>" for i in range(self.m_codebook_size)]
        )

    def _motion_device(self, motion_tokens):
        if isinstance(motion_tokens, dict):
            return next(iter(motion_tokens.values())).device
        if isinstance(motion_tokens, list) and len(motion_tokens) > 0 and isinstance(motion_tokens[0], dict):
            return motion_tokens[0]["left"].device
        if isinstance(motion_tokens, list) and len(motion_tokens) > 0:
            return motion_tokens[0].device
        return self.language_model.device

    def _paper_pretrain_condition(self):
        if self.token_format == "hoi_triplet" and self.stage == "lm_pretrain":
            return random.choice(["supervised", "supervised", "supervised", "text", "motion"])
        return random.choice(["supervised", "supervised", "supervised"])

    def forward(self, texts: List[str], motion_tokens, lengths: List[int], tasks: dict):
        if self.lm_type == "encdec":
            return self.forward_encdec(texts, motion_tokens, lengths, tasks)
        if self.lm_type == "dec":
            return self.forward_dec(texts, motion_tokens, lengths, tasks)
        raise NotImplementedError("Only conditional_multitask supported")

    def forward_encdec(self, texts: List[str], motion_tokens, lengths: List[int], tasks: dict):
        motion_strings = self.motion_token_to_string(motion_tokens, lengths)
        condition = self._paper_pretrain_condition()

        if condition == "text":
            inputs = texts
            outputs = texts
        elif condition == "motion":
            inputs = motion_strings
            outputs = motion_strings
        else:
            inputs, outputs = self.template_fulfill(tasks, lengths, motion_strings, texts)

        source_encoding = self.tokenizer(
            inputs,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        device = self._motion_device(motion_tokens)
        source_attention_mask = source_encoding.attention_mask.to(device)
        source_input_ids = source_encoding.input_ids.to(device)

        if condition in ["text", "motion"]:
            batch_size, expanded_input_length = source_input_ids.shape
            mask_indices = np.asarray(
                [self.random_spans_noise_mask(expanded_input_length) for _ in range(batch_size)]
            )
            target_mask = ~mask_indices
            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
            target_sentinel = self.create_sentinel_ids(target_mask.astype(np.int8))

            labels_input_ids = self.filter_input_ids(source_input_ids, target_sentinel)
            source_input_ids = self.filter_input_ids(source_input_ids, input_ids_sentinel)
            labels_attention_mask = None
        else:
            target_inputs = self.tokenizer(
                outputs,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            labels_input_ids = target_inputs.input_ids.to(device)
            labels_attention_mask = target_inputs.attention_mask.to(device)

        labels_input_ids[labels_input_ids == 0] = -100
        return self.language_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask if condition == "supervised" else None,
            labels=labels_input_ids,
            decoder_attention_mask=labels_attention_mask if condition == "supervised" else None,
        )

    def forward_dec(self, texts: List[str], motion_tokens, lengths: List[int], tasks: dict):
        self.tokenizer.padding_side = "right"
        motion_strings = self.motion_token_to_string(motion_tokens, lengths)
        condition = self._paper_pretrain_condition()

        if condition == "text":
            labels = texts
        elif condition == "motion":
            labels = motion_strings
        else:
            inputs, outputs = self.template_fulfill(tasks, lengths, motion_strings, texts)
            labels = [inputs[i] + " \n " + outputs[i] + self.tokenizer.eos_token for i in range(len(inputs))]

        inputs = self.tokenizer(
            labels,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        device = self._motion_device(motion_tokens)
        labels_input_ids = inputs.input_ids.to(device)
        labels_attention_mask = inputs.attention_mask.to(device)
        return self.language_model(
            input_ids=labels_input_ids,
            attention_mask=labels_attention_mask,
            labels=inputs["input_ids"],
        )

    def generate_direct(
        self,
        texts: List[str],
        max_length: int = 256,
        num_beams: int = 1,
        do_sample: bool = True,
        bad_words_ids: List[int] = None,
    ):
        self.device = self.language_model.device

        if self.lm_type == "dec":
            texts = [text + " \n " for text in texts]

        source_encoding = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        source_input_ids = source_encoding.input_ids.to(self.device)
        source_attention_mask = source_encoding.attention_mask.to(self.device)

        if self.lm_type == "encdec":
            outputs = self.language_model.generate(
                source_input_ids,
                attention_mask=source_attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                bad_words_ids=bad_words_ids,
            )
        else:
            outputs = self.language_model.generate(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=do_sample,
                max_new_tokens=max_length,
            )
            self.tokenizer.padding_side = "left"

        outputs_string = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs_tokens, cleaned_text = self.motion_string_to_token(outputs_string)
        return outputs_tokens, cleaned_text

    def generate_conditional(
        self,
        texts: Optional[List[str]] = None,
        motion_tokens=None,
        lengths: Optional[List[int]] = None,
        task: str = "t2m",
        with_len: bool = False,
        stage: str = "train",
        tasks: dict = None,
        obj_name: str = None,
    ):
        self.device = self.language_model.device

        if task in ["t2m", "m2m", "pred", "inbetween"]:
            if task == "t2m":
                assert texts is not None
                motion_strings = [""] * len(texts)
                if not with_len:
                    if tasks is None:
                        tasks = [{"input": ["Generate motion: <Caption_Placeholder>"], "output": [""]}] * len(texts)
                    lengths = [0] * len(texts)
                else:
                    tasks = [{
                        "input": ["Generate motion with <Frame_Placeholder> frames: <Caption_Placeholder>"],
                        "output": [""],
                    }] * len(texts)
            elif task == "pred":
                assert motion_tokens is not None and lengths is not None
                texts = [""] * len(lengths)
                tasks = [{"input": ["Predict motion: <Motion_Placeholder_s1>"], "output": [""]}] * len(lengths)
                motion_strings = self.motion_token_to_string(motion_tokens, lengths)
            elif task == "inbetween":
                assert motion_tokens is not None and lengths is not None
                tasks = [{
                    "input": ["Complete the masked motion: <Motion_Placeholder_Masked> for <Caption_Placeholder>"],
                    "output": [""],
                }] * len(lengths)
                motion_strings = self.motion_token_to_string(motion_tokens, lengths)
            else:
                motion_strings = self.motion_token_to_string(motion_tokens, lengths)

            inputs, outputs = self.template_fulfill(
                tasks,
                lengths,
                motion_strings,
                texts,
                stage,
                obj_name=obj_name,
            )
            outputs_tokens, _ = self.generate_direct(
                inputs,
                max_length=self.max_length,
                num_beams=1,
                do_sample=True,
            )
            return outputs_tokens

        if task == "m2t":
            assert motion_tokens is not None and lengths is not None
            motion_strings = self.motion_token_to_string(motion_tokens, lengths)
            if not with_len:
                tasks = [{"input": ["Generate text: <Motion_Placeholder>"], "output": [""]}] * len(lengths)
            else:
                tasks = [{
                    "input": ["Generate text with <Frame_Placeholder> frames: <Motion_Placeholder>"],
                    "output": [""],
                }] * len(lengths)
            texts = [""] * len(lengths)
            inputs, outputs = self.template_fulfill(tasks, lengths, motion_strings, texts, obj_name=obj_name)
            _, cleaned_text = self.generate_direct(
                inputs,
                max_length=40,
                num_beams=1,
                do_sample=False,
            )
            return cleaned_text

        raise ValueError(f"Unsupported task: {task}")

    def _flatten_triplet_batch(self, motion_token):
        if isinstance(motion_token, dict):
            return [{key: motion_token[key][i] for key in motion_token} for i in range(motion_token["left"].shape[0])]
        return motion_token

    def _infer_token_length(self, motion_token_i, fallback_length):
        if isinstance(motion_token_i, dict):
            return min(int(fallback_length), int(motion_token_i["left"].shape[0]))
        return min(int(fallback_length), int(motion_token_i.shape[0]))

    def motion_token_to_string(self, motion_token, lengths: List[int]):
        if self.token_format == "hoi_triplet":
            motion_string = []
            for i, motion_i in enumerate(self._flatten_triplet_batch(motion_token)):
                left = motion_i["left"].detach().cpu().tolist()
                right = motion_i["right"].detach().cpu().tolist()
                obj = motion_i["obj"].detach().cpu().tolist()
                valid_len = self._infer_token_length(motion_i, lengths[i])
                motion_string.append(self._triplet_ids_to_string(list(zip(left[:valid_len], right[:valid_len], obj[:valid_len]))))
            return motion_string

        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu() if motion_token[i].device.type == "cuda" else motion_token[i]
            motion_list = motion_i.tolist()[: lengths[i]]
            motion_string.append(
                self.motion_start_token
                + "".join([f"<motion_id_{int(token)}>" for token in motion_list])
                + self.motion_end_token
            )
        return motion_string

    def motion_token_list_to_string(self, motion_token):
        if self.token_format == "hoi_triplet":
            lengths = [token["left"].shape[0] for token in self._flatten_triplet_batch(motion_token)]
            return self.motion_token_to_string(motion_token, lengths)

        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu() if motion_token[i].device.type == "cuda" else motion_token[i]
            motion_list = motion_i.tolist()
            motion_string.append(
                self.motion_start_token
                + "".join([f"<motion_id_{int(token)}>" for token in motion_list])
                + self.motion_end_token
            )
        return motion_string

    def _triplet_ids_to_string(self, triplets: List[Tuple[int, int, int]]):
        pieces = [self.hoi_start_token]
        for left_id, right_id, obj_id in triplets:
            pieces.extend(
                [
                    self.hoi_left_token,
                    f"<hand_id_{int(left_id)}>",
                    self.hoi_right_token,
                    f"<hand_id_{int(right_id)}>",
                    self.hoi_object_token,
                    f"<obj_id_{int(obj_id)}>",
                ]
            )
        pieces.append(self.hoi_end_token)
        return "".join(pieces)

    def _triplet_string_to_ids(self, motion_string: str):
        tags = []
        cursor = 0
        while True:
            start = motion_string.find("<", cursor)
            if start < 0:
                break
            end = motion_string.find(">", start)
            if end < 0:
                break
            tags.append(motion_string[start : end + 1])
            cursor = end + 1

        try:
            start_idx = tags.index(self.hoi_start_token) + 1
            end_idx = tags.index(self.hoi_end_token, start_idx)
        except ValueError:
            return [(0, 0, 0)], motion_string

        payload = tags[start_idx:end_idx]
        triplets = []
        chunk_size = 6
        for idx in range(0, len(payload) - chunk_size + 1, chunk_size):
            chunk = payload[idx : idx + chunk_size]
            if chunk[0] != self.hoi_left_token or chunk[2] != self.hoi_right_token or chunk[4] != self.hoi_object_token:
                continue
            if not chunk[1].startswith("<hand_id_") or not chunk[3].startswith("<hand_id_") or not chunk[5].startswith("<obj_id_"):
                continue
            try:
                triplets.append(
                    (
                        int(chunk[1][9:-1]),
                        int(chunk[3][9:-1]),
                        int(chunk[5][8:-1]),
                    )
                )
            except ValueError:
                continue

        if not triplets:
            triplets = [(0, 0, 0)]

        clean_string = motion_string.replace("".join(tags[start_idx:end_idx]), "<Motion_Placeholder>")
        return triplets, clean_string

    def motion_string_to_token(self, motion_string: List[str]):
        motion_tokens = []
        output_string = []
        if self.token_format == "hoi_triplet":
            for string in motion_string:
                triplets, clean_string = self._triplet_string_to_ids(string)
                motion_tokens.append(
                    {
                        "left": torch.tensor([triplet[0] for triplet in triplets], dtype=torch.long, device=self.device),
                        "right": torch.tensor([triplet[1] for triplet in triplets], dtype=torch.long, device=self.device),
                        "obj": torch.tensor([triplet[2] for triplet in triplets], dtype=torch.long, device=self.device),
                    }
                )
                output_string.append(clean_string)
            return motion_tokens, output_string

        for string in motion_string:
            middle = self.get_middle_str(string, self.motion_start_token, self.motion_end_token)
            string_list = middle.split("><")
            token_list = [int(token.split("_")[-1].replace(">", "")) for token in string_list[1:-1]]
            if len(token_list) == 0:
                token_list = [0]
            motion_tokens.append(torch.tensor(token_list, dtype=torch.long, device=self.device))
            output_string.append(string.replace(middle, "<Motion_Placeholder>"))
        return motion_tokens, output_string

    def _split_triplet_string(self, motion_string: str):
        triplets, _ = self._triplet_string_to_ids(motion_string)
        return triplets

    def placeholder_fulfill(self, prompt: str, length: int, motion_string: str, text: str):
        seconds = math.floor(length / self.framerate)

        if self.token_format == "hoi_triplet" and motion_string:
            triplets = self._split_triplet_string(motion_string)
            token_length = len(triplets)
            predict_head = int(token_length * self.predict_ratio + 1)
            masked_head = int(token_length * self.inbetween_ratio + 1)
            masked_tail = int(token_length * (1 - self.inbetween_ratio) + 1)

            motion_predict_head = self._triplet_ids_to_string(triplets[:predict_head])
            motion_predict_last = self._triplet_ids_to_string(triplets[predict_head:])
            masked_triplets = (
                triplets[:masked_head]
                + [(0, 0, 0)] * max(masked_tail - masked_head, 0)
                + triplets[masked_tail:]
            )
            motion_masked = self._triplet_ids_to_string(masked_triplets)
        else:
            motion_splited = motion_string.split(">")
            token_length = length / self.down_t
            predict_head = int(token_length * self.predict_ratio + 1)
            masked_head = int(token_length * self.inbetween_ratio + 1)
            masked_tail = int(token_length * (1 - self.inbetween_ratio) + 1)

            motion_predict_head = ">".join(motion_splited[:predict_head]) + ">" + self.motion_end_token
            motion_predict_last = self.motion_start_token + ">".join(motion_splited[predict_head:])
            motion_masked = (
                ">".join(motion_splited[:masked_head])
                + ">"
                + self.motion_mask_token * max(masked_tail - masked_head, 0)
                + ">".join(motion_splited[masked_tail:])
            )

        if random.random() < self.quota_ratio:
            text = f"\"{text}\""

        return (
            prompt.replace("<Caption_Placeholder>", text)
            .replace("<Motion_Placeholder>", motion_string)
            .replace("<Frame_Placeholder>", f"{length}")
            .replace("<Second_Placeholder>", "%.1f" % seconds)
            .replace("<Motion_Placeholder_s1>", motion_predict_head)
            .replace("<Motion_Placeholder_s2>", motion_predict_last)
            .replace("<Motion_Placeholder_Masked>", motion_masked)
        )

    def template_fulfill(self, tasks, lengths, motion_strings, texts, stage="test", obj_name=None):
        inputs = []
        outputs = []
        for i in range(len(lengths)):
            input_template = random.choice(tasks[i]["input"])
            output_template = random.choice(tasks[i]["output"])
            length = lengths[i]
            try:
                obj = texts[i].split(" ")[1]
            except Exception:
                obj = obj_name
            input_template = "Given an object of " + obj + " " + input_template
            inputs.append(self.placeholder_fulfill(input_template, length, motion_strings[i], texts[i]))
            outputs.append(self.placeholder_fulfill(output_template, length, motion_strings[i], texts[i]))
        return inputs, outputs

    def get_middle_str(self, content, startStr, endStr):
        try:
            startIndex = content.index(startStr)
            if startIndex >= 0:
                startIndex += len(startStr)
            endIndex = content.index(endStr)
        except Exception:
            return self.motion_start_token + "<motion_id_0>" + self.motion_end_token

        return self.motion_start_token + content[startIndex:endIndex] + self.motion_end_token

    def random_spans_noise_mask(self, length):
        orig_length = length
        num_noise_tokens = int(np.round(length * self.noise_density))
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        def _random_segmentation(num_items, num_segments):
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)
        return is_noise[:orig_length]

    def create_sentinel_ids(self, mask_indices):
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(
            sentinel_ids != 0,
            (len(self.tokenizer) - sentinel_ids - self.num_added_motion_tokens),
            0,
        )
        sentinel_ids -= mask_indices - start_indices
        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        batch_size = input_ids.shape[0]
        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids.to("cpu"))
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32),
            ],
            axis=-1,
        )
        return torch.tensor(input_ids, device=next(self.language_model.parameters()).device)

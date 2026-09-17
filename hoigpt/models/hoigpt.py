import numpy as np
import os
import random
import torch
import time
from hoigpt.config import instantiate_from_config
from os.path import join as pjoin
from hoigpt.losses.hoigpt import GPTLosses
from .base import BaseModel
import json


class HOIGPT(BaseModel):
    """
    Stage 1 Motion Tokenizer
    Stage 2 Motion-language pretrian
    Stage 3 Motion-language instruction tuning
    """

    def __init__(self,
                 cfg,
                 datamodule,
                 lm,
                 motion_vae,
                 codebook_size=512,
                 stage='vae',
                 debug=True,
                 condition='text',
                 task='t2m',
                 metrics_dict=['TM2TMetrics'],
                 **kwargs):

        self.save_hyperparameters(ignore='datamodule', logger=False)
        self.datamodule = datamodule
        super().__init__()

        # Instantiate motion tokenizer
        if motion_vae != None:
            self.vae = instantiate_from_config(motion_vae)

        # Tokenizer training/export does not need pretrained language weights.
        self.lm = instantiate_from_config(lm) if 'lm' in stage else None

        # Freeze the motion tokenizer for lm training
        if 'lm' in self.hparams.stage:
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False

        # Instantiate the losses
        self._losses = torch.nn.ModuleDict({
            split: GPTLosses(cfg, self.hparams.stage, self.datamodule.njoints)
            for split in ["losses_train", "losses_test", "losses_val"]
        })

        # self.lhand_layer = build_mano_aa(is_rhand=False, create_transl=True, flat_hand=False)
        # self.rhand_layer = build_mano_aa(is_rhand=True, create_transl=True, flat_hand=False)

        # Data transform
        self.feats2joints = datamodule.feats2joints

        # Count codebook frequency
        self.codePred = []
        self.codeFrequency = torch.zeros((self.hparams.codebook_size, ))
        self.cfg = cfg
        if cfg.LOSS.LAMBDA_GEO > 0 and stage == "vae":
            from hoigpt.lib.models.mano import build_mano_aa

            self.lhand_layer = build_mano_aa(is_rhand=False, create_transl=True, flat_hand=False)
            self.rhand_layer = build_mano_aa(is_rhand=True, create_transl=True, flat_hand=False)
            self.lhand_layer.eval()
            self.rhand_layer.eval()
        self.loc_index = [96,97,98,195,196,197,205,206,207]

    def train(self, mode=True):
        super().train(mode)
        if 'lm' in self.hparams.stage:
            self.vae.eval()
        return self

    def _is_triplet_tokens(self, tokens):
        return isinstance(tokens, dict)

    def _token_length(self, tokens):
        if self._is_triplet_tokens(tokens):
            return int(tokens['left'].shape[-1])
        return int(tokens.shape[-1])

    def _token_at(self, tokens, idx):
        if self._is_triplet_tokens(tokens):
            return {key: value[idx] for key, value in tokens.items()}
        return tokens[idx]

    def _squeeze_encoded_token(self, tokens):
        if self._is_triplet_tokens(tokens):
            return {key: value[0] for key, value in tokens.items()}
        return tokens[0]

    def _clamp_motion_tokens(self, tokens):
        if self._is_triplet_tokens(tokens):
            return {
                key: torch.clamp(value.long(), 0, self.hparams.codebook_size - 1)
                for key, value in tokens.items()
            }
        return torch.clamp(tokens.long(), 0, self.hparams.codebook_size - 1)

    def _concat_motion_tokens(self, prefix, suffix):
        if self._is_triplet_tokens(prefix):
            return {key: torch.cat([prefix[key], suffix[key]], dim=-1) for key in prefix.keys()}
        return torch.cat((prefix, suffix))

    def _select_pc(self, pc_conds, idx):
        if pc_conds is None:
            return None
        return pc_conds[idx:idx + 1]

    def forward(self, batch, task="t2m"):
        texts = batch["text"]
        lengths_ref = batch["length"]
        pc_conds = batch.get("pc")

        # Forward
        # texts = ['Generate motion: ' + text for text in texts]
        outputs, output_texts = self.lm.generate_direct(texts, do_sample=True)

        # Motion Decode
        feats_rst_lst = []
        lengths = []
        max_len = 0

        for i in range(len(texts)):
            if task == "pred":
                motion = self.vae.decode(
                    self._concat_motion_tokens(self._token_at(batch["motion"], i), outputs[i]),
                    self._select_pc(pc_conds, i),
                )
            elif task in ["t2m", "m2t", "inbetween"]:
                motion = self.vae.decode(outputs[i], self._select_pc(pc_conds, i))
                # motion = self.datamodule.denormalize(motion)
                lengths.append(motion.shape[1])
            else:
                raise NotImplementedError

            if motion.shape[1] > max_len:
                max_len = motion.shape[1]

            if task in ["t2m", "m2t", "pred"]:
                feats_rst_lst.append(motion)

            elif task == "inbetween":
                motion = torch.cat(
                    (batch["motion_heading"][i][None],
                     motion[:, lengths_ref[i] // 4:lengths_ref[i] // 4 * 3,
                            ...], batch["motion_tailing"][i][None]),
                    dim=1)
                feats_rst_lst.append(motion)

        feats_rst = torch.zeros(
            (len(feats_rst_lst), max_len, motion.shape[-1])).to(self.device)

        # padding and concat
        for i in range(len(feats_rst_lst)):
            feats_rst[i, :feats_rst_lst[i].shape[1], ...] = feats_rst_lst[i]

        # Recover joints for evaluation
        joints_rst = feats_rst if feats_rst.shape[-1] == 208 else self.feats2joints(feats_rst)

        # return set
        outputs = {
            "texts": output_texts,
            "feats": feats_rst,
            "joints": joints_rst,
            "length": lengths
        }

        return outputs

    def train_lm_forward(self, batch):
        tokens_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = batch["tasks"]
        all_captions = batch['all_captions']
        if self.hparams.condition == 'caption':
            texts = [random.choice(all_captions[i]) for i in range(len(texts))]

        # LLM Forward
        outputs = self.lm(texts, tokens_ref, lengths, tasks)
        # outputs = self.t2m_gpt.generate(texts)
        return {'outputs': outputs}

    @torch.no_grad()
    def val_t2m_forward(self, batch):
        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        pc_conds = batch['pc'] if 'pc' in batch else None
        tasks = None
        # import sys; sys.exit(1)
        if self.trainer.datamodule.is_mm:
            texts = texts * self.hparams.cfg.METRIC.MM_NUM_REPEATS
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            if pc_conds is not None:
                pc_conds = pc_conds.repeat_interleave(self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS
            from pathlib import Path

            instructions = self.hparams.cfg.DATASET.TASK_PATH
            if not instructions:
                instructions = Path(self.datamodule.hparams.data_root) / 'template_instructions.json'
                if not instructions.is_file():
                    instructions = Path(__file__).resolve().parents[2] / 'assets' / 'template_instructions.json'
            with open(instructions, 'r') as stream:
                text_tasks = json.load(stream)["Text-to-Motion"]
            tasks = [text_tasks.get("t2m", text_tasks.get("caption"))] * len(texts)

        if self.hparams.condition == 'caption':
            tasks = [{
                'input': ['<Caption_Placeholder>'],
                'output': ['']
            }] * len(texts)

        if self.hparams.cfg.DATASET.TASK_PATH:
            instructions = pjoin(self.hparams.cfg.DATASET.TASK_PATH)
            instructions = json.load(open(instructions, 'r'))
            text_tasks = instructions["Text-to-Motion"]
            tasks = [text_tasks.get("t2m", text_tasks.get("caption"))] * len(texts)

        min_len = lengths.copy()
        # Forward
        outputs = self.lm.generate_conditional(texts,
                                               lengths=lengths,
                                               stage='test',
                                               tasks=tasks)

        # Motion Decode
        feats_rst = torch.zeros_like(feats_ref)

        for i in range(len(texts)):
            outputs[i] = self._clamp_motion_tokens(outputs[i])
            pc_cond = self._select_pc(pc_conds, i)
            # import pdb; pdb.set_trace() 

            ## problem with the length
            if self._token_length(outputs[i]) > 1:
                motion = self.vae.decode(outputs[i], pc_cond)
            else:
                motion = torch.zeros_like(feats_ref[i:i + 1, ...])

            min_len[i] = min(motion.shape[1], lengths[i])

            # Cut Motion
            feats_rst[i:i + 1, :min_len[i], ...] = motion[:, :lengths[i]]
            # import pdb; pdb.set_trace()

        # Recover joints for evaluation
        # import pdb; pdb.set_trace()
        # joints_ref = self.feats2joints(feats_ref)
        # joints_rst = self.feats2joints(feats_rst)

        joints_ref = feats_ref
        joints_rst = feats_rst

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": min_len
            # "length": lengths
        }

        return rs_set

    @torch.no_grad()
    def val_m2t_forward(self, batch):
        self.hparams.metrics_dict = []

        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        # import pdb; pdb.set_trace()
        obj_name = batch['name'][0].split("_")[1]
        all_captions = batch['all_captions']
        pc_conds = batch['pc'] if 'pc' in batch else None

        # Motion Encode
        motion_tokens = []
        lengths_tokens = []
        for i in range(len(feats_ref)):
            pc_cond = pc_conds[i:i+1] if pc_conds is not None else pc_conds
            motion_token, _ = self.vae.encode(feats_ref[i:i + 1], pc_cond)
            motion_tokens.append(self._squeeze_encoded_token(motion_token))
            lengths_tokens.append(self._token_length(motion_tokens[-1]))

        # Forward
        # import pdb; pdb.set_trace()
        outputs = self.lm.generate_conditional(motion_tokens=motion_tokens,
                                               lengths=lengths_tokens,
                                               task="m2t",
                                               stage='test', obj_name=obj_name)

        rs_set = {
            "m_ref": feats_ref,
            "t_ref": all_captions,
            "t_ref": texts,
            "t_pred": outputs,
            "length": lengths
        }

        return rs_set

    @torch.no_grad()
    def val_m2m_forward(self, batch, task="pred"):
        feats_ref = batch["motion"]
        lengths = batch["length"]
        texts = batch["text"]
        pc_conds = batch['pc'] if 'pc' in batch else None
        obj_name = batch['name'][0].split("_")[1]

        # Motion Encode
        motion_tokens = []
        lengths_tokens = []
        for i in range(len(feats_ref)):
            pc_cond = pc_conds[i:i+1] if pc_conds is not None else pc_conds
            motion_token, _ = self.vae.encode(feats_ref[i:i + 1], pc_cond)
            motion_tokens.append(self._squeeze_encoded_token(motion_token))
            lengths_tokens.append(self._token_length(motion_tokens[-1]))

        # Forward
        # import pdb; pdb.set_trace()
        outputs = self.lm.generate_conditional(motion_tokens=motion_tokens,
                                               lengths=lengths,
                                               task=task,
                                               texts=texts,
                                               stage='test', obj_name=obj_name)

        # Motion Decode
        feats_rst = torch.zeros_like(feats_ref)
        min_len = lengths.copy()

        for i in range(len(lengths)):
            outputs[i] = self._clamp_motion_tokens(outputs[i])
            pc_cond = self._select_pc(pc_conds, i)

            if self._token_length(outputs[i]) > 1:
                # motion = self.vae.decode(outputs[i])
                motion = self.vae.decode(outputs[i], pc_cond)
            else:
                motion = torch.zeros_like(feats_ref[i:i + 1, ...])

            min_len[i] = min(motion.shape[1], lengths[i])

            # Cut Motion
            feats_rst[i:i + 1, :min_len[i], ...] = motion[:, :lengths[i]]

        # Recover joints for evaluation
        joints_ref = feats_ref
        joints_rst = feats_rst

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        # import pdb; pdb.set_trace()
        # return set
        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": min_len
            # "length": lengths
        }

        return rs_set

    def train_vae_forward(self, batch):
        # batch detach
        # print(ggg)
        feats_ref = batch["motion"]

        pc_cond = batch['pc'] if 'pc' in batch else None
        # import pdb; pdb.set_trace()
        feats_rst, loss_commit, perplexity = self.vae(feats_ref, pc_cond)

        # import pdb; pdb.set_trace()
        if self.cfg.LOSS.LAMBDA_GEO > 0:
            from hoigpt.utils.m2m import mo2mesh

            joints_rst = mo2mesh(self.datamodule, feats_rst, batch['pc_full'], self.lhand_layer, self.rhand_layer)
            joints_ref = mo2mesh(self.datamodule, feats_ref, batch['pc_full'], self.lhand_layer, self.rhand_layer)
        else:
            joints_rst = joints_ref = None

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "joints_ref": joints_ref,
            "m_rst": feats_rst,
            "joints_rst": joints_rst,
            "loss_commit": loss_commit,
            "perplexity": perplexity,
            "subset": batch['subset'][0],
        }
        # import pdb; pdb.set_trace()
        # rs_set['m_rst'][...,self.loc_index] = rs_set['m_ref'][...,self.loc_index]
        # rs_set['joints_rst'][..., self.loc_index] = rs_set['m_ref'][..., self.loc_index]
        return rs_set

    @torch.no_grad()
    def val_vae_forward(self, batch, split="train"):
        # Detach batch
        feats_ref = batch["motion"]
        lengths = batch["length"]
        pc_conds = batch['pc'] if 'pc' in batch else None

        # Repeat for multimodal evaluation
        if self.trainer.datamodule.is_mm:
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS

        # Motion encode & decode
        feats_rst = torch.zeros_like(feats_ref)

        for i in range(len(feats_ref)):
            if lengths[i] == 0:
                continue
            # import pdb; pdb.set_trace()
            pc_cond = pc_conds[i:i+1] if pc_conds is not None else pc_conds
            # lengths = 
            feats_pred, _, _ = self.vae(feats_ref[i:i + 1, :lengths[i]], pc_cond)
            feats_rst[i:i + 1, :feats_pred.shape[1], :] = feats_pred
            

            # code_pred, _ = self.vae.encode(feats_ref[i:i + 1, :lengths[i]])

            # codeFre_pred = torch.bincount(code_pred[0],
            #                               minlength=self.hparams.codebook_size).to(
            #                                   self.codeFrequency.device)
            # self.codePred.append(code_pred[0])
            # self.codeFrequency += codeFre_pred


        # Recover joints for evaluation
        joints_ref = feats_ref
        joints_rst = feats_rst

        # Renorm to the evaluator feature space before TM2T/FID scoring.
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # Return set
        rs_set = {
            "m_ref": feats_ref,
            "joints_ref": joints_ref,
            "m_rst": feats_rst,
            "joints_rst": joints_rst,
            "length": lengths,
        }

        # rs_set['m_rst'][...,self.loc_index] = rs_set['m_ref'][...,self.loc_index]
        # rs_set['joints_rst'][..., self.loc_index] = rs_set['m_ref'][..., self.loc_index]


        return rs_set


    def allsplit_step(self, split: str, batch, batch_idx):
        # Compute the losses
        loss = None
        # import pdb; pdb.set_trace()
        if self.hparams.stage == "vae" and split in ["train", "val"]:
            rs_set = self.train_vae_forward(batch)
            loss, loss_dict = self._losses['losses_' + split].update(rs_set, return_dict=True)
            # import pdb; pdb.set_trace()
            for k, v in loss_dict.items():
                self.log(k, v)
        elif self.hparams.stage in ["lm_instruct", "lm_pretrain"
                                    ] and split in ["train"]:
            rs_set = self.train_lm_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
        elif self.hparams.stage == 'lm_rl' and split in ['train']:
            rs_set = self.train_rl_forward(batch)
            loss = None

        # Compute the metrics
        if split in ["val", "test"]:
            if self.hparams.stage == "vae":
                rs_set = self.val_vae_forward(batch, split)
            elif self.hparams.stage in ["lm_instruct", "lm_pretrain", "lm_rl"]:
                if self.hparams.task == "t2m":
                    rs_set = self.val_t2m_forward(batch)
                elif self.hparams.task == "m2t":
                    rs_set = self.val_m2t_forward(batch)
                elif self.hparams.task in ["m2m", "pred", "inbetween"]:
                    rs_set = self.val_m2m_forward(batch, self.hparams.task)

            if self.hparams.task not in ["m2t"]:
                # MultiModality evaluation sperately
                if self.trainer.datamodule.is_mm:
                    metrics_dicts = ['MMMetrics']
                else:
                    metrics_dicts = self.hparams.metrics_dict
                    
                if self.hparams.task not in ['pred', 'inbetween'] and 'PredMetrics' in metrics_dicts:
                    metrics_dicts.remove('PredMetrics')
                # else:
                #     metrics_dicts.append('PredMetrics')
                # import pdb; pdb.set_trace()
                for metric in metrics_dicts:
                    lengths = batch['length']
                    if metric == "TemosMetric":
                        getattr(self.metrics,
                                metric).update(rs_set["joints_rst"],
                                               rs_set["joints_ref"], lengths)
                    elif metric == "TM2TMetrics":
                        if self.hparams.stage in [
                                "lm_instruct", "lm_pretrain", "lm_rl"
                        ]:
                            word_embs = batch['word_embs']
                            pos_ohot = batch['pos_ohot']
                            text_lengths = batch['text_len']
                            if self.trainer.datamodule.is_mm:
                                word_embs = word_embs.repeat_interleave(
                                    self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                                    dim=0)
                                pos_ohot = pos_ohot.repeat_interleave(
                                    self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                                    dim=0)
                                text_lengths = text_lengths.repeat_interleave(
                                    self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                                    dim=0)
                        else:
                            word_embs = None
                            pos_ohot = None
                            text_lengths = None

                        getattr(self.metrics, metric).update(
                            feats_ref=rs_set["m_ref"],
                            feats_rst=rs_set["m_rst"],
                            lengths_ref=lengths,
                            lengths_rst=rs_set['length'],
                            word_embs=word_embs,
                            pos_ohot=pos_ohot,
                            text_lengths=text_lengths,
                        )
                    elif metric == "UncondMetrics":
                        getattr(self.metrics, metric).update(
                            recmotion_embeddings=rs_set["lat_rm"],
                            gtmotion_embeddings=rs_set["lat_m"],
                            lengths=lengths,
                        )
                    elif metric == "MRMetrics":
                        getattr(self.metrics,
                                metric).update(rs_set["joints_rst"],
                                               rs_set["joints_ref"], lengths)
                    elif metric == "PredMetrics":
                        getattr(self.metrics,
                                metric).update(rs_set["joints_rst"],
                                               rs_set["joints_ref"], lengths)
                    elif metric == "MMMetrics":
                        # pass
                        getattr(self.metrics,
                                metric).update(rs_set["m_rst"],
                                               rs_set['length'])
                    else:
                        raise TypeError(f"Not support this metric {metric}")

            elif self.hparams.task == "m2t" and self.hparams.stage in [
                    "lm_instruct", "lm_pretrain", "lm_rl"
            ]:
                self.hparams.metrics_dict = metrics_dicts = ['M2TMetrics']
                for metric in metrics_dicts:
                    if metric == "M2TMetrics":
                        getattr(self.metrics, metric).update(
                            feats_ref=rs_set["m_ref"],
                            pred_texts=rs_set["t_pred"],
                            gt_texts=batch["all_captions"],
                            lengths=rs_set['length'],
                            word_embs=batch["word_embs"],
                            pos_ohot=batch["pos_ohot"],
                            text_lengths=batch["text_len"],
                        )
                    

            # if True:
            #     import pdb; pdb.set_trace()

        # return forward output rather than loss during test
        if split in ["test"]:
            if self.hparams.task != "m2t":
                return rs_set["joints_rst"], rs_set["length"], rs_set[
                    "joints_ref"], batch['name']
                # pass
            elif self.hparams.task == "m2t":
                return rs_set["t_pred"], batch["length"]
                # return batch["length"]

        # import pdb; pdb.set_trace()
        return loss

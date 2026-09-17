import os
import numpy as np
import torch, pdb
import logging
from pathlib import Path
from pytorch_lightning import LightningModule
from os.path import join as pjoin
from collections import OrderedDict
from hoigpt.metrics import BaseMetrics
from hoigpt.config import get_obj_from_str

class BaseModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.configure_metrics()

        # Ablation
        self.test_step_outputs = []
        self.times = []
        self.rep_i = 0

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        outputs = self.allsplit_step("test", batch, batch_idx)
        self.test_step_outputs.append(outputs)
        return outputs

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)

    def on_train_epoch_end(self):
        # Log steps and losses
        dico = self.step_log_dict()
        # Log losses
        dico.update(self.loss_log_dict('train'))
        # Write to log only if not sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)

    def on_validation_epoch_end(self):
        # Log steps and losses
        dico = self.step_log_dict()
        # Log losses
        dico.update(self.loss_log_dict('train'))
        dico.update(self.loss_log_dict('val'))
        # Log metrics
        dico.update(self.metrics_log_dict())
        # Write to log only if not sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)

    def on_test_epoch_end(self):
        # Log metrics
        dico = self.metrics_log_dict()
        # Write to log only if not sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)
        self.save_npy(self.test_step_outputs)
        self.rep_i = self.rep_i + 1
        # Free up the memory
        self.test_step_outputs.clear()

    def preprocess_state_dict(self, state_dict):
        new_state_dict = OrderedDict()
        
        metric_state_dict = self.metrics.state_dict()
        loss_state_dict = self._losses.state_dict()

        for k, v in metric_state_dict.items():
            new_state_dict['metrics.' + k] = v

        for k, v in loss_state_dict.items():
            new_state_dict['_losses.' + k] = v

        for k, v in state_dict.items():
            if self.lm is None and k.startswith('lm.'):
                continue
            if not k.startswith(('_losses.', 'metrics.')):
                new_state_dict[k] = v

        return new_state_dict

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = self.preprocess_state_dict(state_dict)
        return super().load_state_dict(new_state_dict, strict=strict)

    def step_log_dict(self):
        return {
            "epoch": float(self.trainer.current_epoch),
            "step": float(self.trainer.current_epoch)
        }

    def loss_log_dict(self, split: str):
        losses = self._losses['losses_' + split]
        loss_dict = losses.compute(split)
        return loss_dict

    def metrics_log_dict(self):
        # For TM2TMetrics MM
        if self.trainer.datamodule.is_mm and "TM2TMetrics" in self.hparams.metrics_dict:
            metrics_dicts = ['MMMetrics']
        else:
            metrics_dicts = self.hparams.metrics_dict

        # Compute all metrics
        metrics_log_dict = {}
        for metric in metrics_dicts:
            metrics_dict = getattr(
                self.metrics,
                metric).compute(sanity_flag=self.trainer.sanity_checking)
            metrics_log_dict.update({
                f"Metrics/{metric}": value.item()
                for metric, value in metrics_dict.items()
            })

        return metrics_log_dict
    
    def configure_optimizers(self):
        # Optimizer
        optim_target = self.hparams.cfg.TRAIN.OPTIM.target
        if len(optim_target.split('.')) == 1:
            optim_target = 'torch.optim.' + optim_target
        optimizer = get_obj_from_str(optim_target)(
            params=self.parameters(), **self.hparams.cfg.TRAIN.OPTIM.params)

        # Scheduler
        scheduler_target = self.hparams.cfg.TRAIN.LR_SCHEDULER.target
        if len(scheduler_target.split('.')) == 1:
            scheduler_target = 'torch.optim.lr_scheduler.' + scheduler_target
        lr_scheduler = get_obj_from_str(scheduler_target)(
            optimizer=optimizer, **self.hparams.cfg.TRAIN.LR_SCHEDULER.params)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def configure_metrics(self):
        self.metrics = BaseMetrics(datamodule=self.datamodule, **self.hparams)

    def save_npy(self, outputs):
        cfg = self.hparams.cfg
        if not cfg.TEST.SAVE_PREDICTIONS:
            return
        output_dir = Path(cfg.FOLDER) / cfg.model.target.split('.')[-2] / cfg.NAME / ("samples_" + cfg.TIME)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.hparams.task == "m2t":
            import json
            captions = [caption for result in outputs for caption in result[0]]
            with open(output_dir / "captions.json", "w", encoding="utf-8") as handle:
                json.dump(captions, handle, ensure_ascii=False, indent=2)
            return

        render = cfg.TEST.get("RENDER_PREDICTIONS", False)
        if render:
            from hoigpt.lib.models.mano import build_mano_aa
            from hoigpt.lib.utils.renderer import Renderer
            lhand_layer = build_mano_aa(is_rhand=False, create_transl=True, flat_hand=False).to(self.device)
            rhand_layer = build_mano_aa(is_rhand=True, create_transl=True, flat_hand=False).to(self.device)
            renderer = Renderer(device=str(self.device), camera="arctic_front")

        for predictions, lengths, references, names in outputs:
            for index, name in enumerate(names):
                features = self.datamodule.denormalize(predictions[index, :lengths[index]])
                path = output_dir / f"{name}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, features.detach().cpu().numpy())
                if render:
                    self.vis_hoi(features, lhand_layer, rhand_layer, renderer,
                                 str(path.with_suffix("")), name.split('_')[1])


    def vis_hoi(self, pred_denorm, lhand_layer, rhand_layer, renderer, path, obj_name):
        from hoigpt.lib.models.object import build_object_model
        from hoigpt.utils.vishoi import rendering_hoi

        dataname = self.datamodule.name
        data_root = self.datamodule.hparams.data_root
        object_model = build_object_model(pjoin(data_root, dataname + '.pkl'))
        pred_denorm = pred_denorm
        x_lhand, x_rhand, obj_6d = pred_denorm[:,:99], pred_denorm[:,99:198], pred_denorm[:, 198:208]
        rendering_hoi(x_lhand, x_rhand, obj_6d, lhand_layer, rhand_layer, object_model, renderer, obj_name, path, dataname, data_root+'/meshes')

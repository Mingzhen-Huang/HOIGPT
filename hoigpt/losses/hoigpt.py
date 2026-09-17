import torch
import torch.nn as nn
from .base import BaseLosses

class CommitLoss(nn.Module):
    """
    Useless Wrapper
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, commit, commit2, **kwargs):
        return commit


class GPTLosses(BaseLosses):
    
    def __init__(self, cfg, stage, num_joints, **kwargs):
        # Save parameters
        self.stage = stage
        recons_loss = cfg.LOSS.ABLATION.RECONS_LOSS
        self.is_articulated = cfg.DATASET.Arc

        # Define losses
        losses = []
        params = {}
        if stage == "vae":
            losses.append("recons_feature")
            params['recons_feature'] = cfg.LOSS.LAMBDA_FEATURE

            losses.append("recons_velocity")
            params['recons_velocity'] = cfg.LOSS.LAMBDA_VELOCITY

            losses.append("vq_commit")
            params['vq_commit'] = cfg.LOSS.LAMBDA_COMMIT
        elif stage in ["lm_pretrain", "lm_instruct"]:
            losses.append("gpt_loss")
            params['gpt_loss'] = cfg.LOSS.LAMBDA_CLS

        # if cfg.LOSS.LAMBDA_GEO > 0 and stage == "vae":
        self.geoloss_weight = cfg.LOSS.LAMBDA_GEO
        self.geo_weights = {
            'penetration': cfg.LOSS.GEO_WEIGHTS.PENETRATION,
            'approach': cfg.LOSS.GEO_WEIGHTS.APPROACH,
            'region': cfg.LOSS.GEO_WEIGHTS.REGION,
        }
        # import pdb;pdb.set_trace()
            

        # Define loss functions & weights
        losses_func = {}
        for loss in losses:
            if loss.split('_')[0] == 'recons':
                if recons_loss == "l1":
                    losses_func[loss] = nn.L1Loss
                elif recons_loss == "l2":
                    losses_func[loss] = nn.MSELoss
                elif recons_loss == "l1_smooth":
                    losses_func[loss] = nn.SmoothL1Loss
            elif loss.split('_')[1] in [
                    'commit', 'loss', 'gpt', 'm2t2m', 't2m2t'
            ]:
                losses_func[loss] = CommitLoss
            elif loss.split('_')[1] in ['cls', 'lm']:
                losses_func[loss] = nn.CrossEntropyLoss
            else:
                raise NotImplementedError(f"Loss {loss} not implemented.")

        super().__init__(cfg, losses, params, losses_func, num_joints,
                         **kwargs)

    def update(self, rs_set, return_dict=False):
        '''Update the losses'''
        total: float = 0.0
        loss_dict = {}

        if self.stage in ["vae"]:
            nfeats = rs_set['m_rst'].shape[-1]
            if nfeats in [263, 135 + 263]:
                if nfeats == 135 + 263:
                    vel_start = 135 + 4
                elif nfeats == 263:
                    vel_start = 4
                total += self._update_loss(
                    "recons_velocity",
                    rs_set['m_rst'][..., vel_start:(self.num_joints - 1) * 3 +
                                    vel_start],
                    rs_set['m_ref'][..., vel_start:(self.num_joints - 1) * 3 +
                                    vel_start])
            elif nfeats == 208:
                # import pdb; pdb.set_trace()
                if rs_set['subset'] == "arctic":
                    feat_loss = self._update_loss(
                    "recons_feature",
                    rs_set['m_rst'],
                    rs_set['m_ref'])
                else:
                    rs_set['m_rst'][...,-10] = rs_set['m_ref'][...,-10].clone()
                    feat_loss = self._update_loss(
                        "recons_feature",
                        torch.cat([rs_set['m_rst'][...,:-10], rs_set['m_rst'][...,-9:]], -1),
                        torch.cat([rs_set['m_ref'][...,:-10], rs_set['m_ref'][...,-9:]], -1))

                total += feat_loss
                # rs_set['m_rst'][..., -3:] = rs_set['m_ref'][..., :]
                # rs_set['m_rst'][..., 99:102] = rs_set['m_ref'][..., 99:102]
                # rs_set['m_rst'][..., -3:] = rs_set['m_ref'][..., -3:]

                loss_loc = self._update_loss(
                    "recons_feature",
                    torch.cat([rs_set['m_rst'][...,:96], rs_set['m_rst'][...,99:195], rs_set['m_rst'][...,-9:-3]], -1),
                    torch.cat([rs_set['m_ref'][...,:96], rs_set['m_ref'][...,99:195], rs_set['m_ref'][...,-9:-3]], -1))
                # feat_loss = self._update_loss(
                #     "recons_feature",
                #     rs_set['m_rst'],
                #     rs_set['m_ref'])
  
                # lhand_loc = rs_set['m_rst'][..., :3]
                # rhand_loc = rs_set['m_rst'][..., 99:102]
                # obj_loc = rs_set['m_rst'][..., -3:]

                # lhand_loc_gt = rs_set['m_ref'][..., :3]
                # rhand_loc_gt = rs_set['m_ref'][..., 99:102]
                # obj_loc_gt = rs_set['m_ref'][..., -3:]

                loss_loc = self._update_loss(
                    "recons_velocity",
                    rs_set['m_rst'][...,-10:], rs_set['m_ref'][...,-10:])

                total += 1 * loss_loc

                # loss_loc = self._update_loss(
                #     "recons_velocity",
                #     [obj_loc, lhand_loc, rhand_loc], [obj_loc_gt, lhand_loc_gt, obj_loc_gt])
                # loss_hoiloc = self._update_loss(
                #     "recons_velocity",
                #     [obj_loc-lhand_loc, obj_loc-rhand_loc],
                #     [obj_loc_gt-lhand_loc_gt, obj_loc_gt-rhand_loc_gt])
                
                # loss_obj = self._update_loss(
                #     "recons_velocity",
                #     rs_set['m_rst'][...,-9:],
                #     rs_set['m_ref'][...,-9:])
                
                loss_dict = {
                    "loc_loss": loss_loc,
                    "feat_loss": feat_loss,
                }
                # total += (loss_loc+loss_obj+loss_hoiloc)
                # import pdb; pdb.set_trace()
                if self.geoloss_weight > 0 and rs_set['joints_rst'] is not None and rs_set['joints_ref'] is not None:
                    from .geometric_loss import TTT_loss

                    lpene_loss, lapproach_loss, lregion_loss = TTT_loss(
                        rs_set['joints_rst']['lhand_verts'],
                        rs_set['joints_rst']['lhand_faces'],
                        rs_set['joints_rst']['obj_verts_tf'],
                        rs_set['joints_ref']['lhand_verts'],
                        rs_set['joints_ref']['obj_verts_tf'],
                        hand_joint=rs_set['joints_rst']['lhand_joint'],
                    )
                    rpene_loss, rapproach_loss, rregion_loss = TTT_loss(
                        rs_set['joints_rst']['rhand_verts'],
                        rs_set['joints_rst']['rhand_faces'],
                        rs_set['joints_rst']['obj_verts_tf'],
                        rs_set['joints_ref']['rhand_verts'],
                        rs_set['joints_ref']['obj_verts_tf'],
                        hand_joint=rs_set['joints_rst']['rhand_joint'],
                    )

                    penetration_loss = lpene_loss + rpene_loss
                    approach_loss = lapproach_loss + rapproach_loss
                    region_loss = lregion_loss + rregion_loss
                    geo_loss = (
                        self.geo_weights['penetration'] * penetration_loss
                        + self.geo_weights['approach'] * approach_loss
                        + self.geo_weights['region'] * region_loss
                    )

                    total += self.geoloss_weight * geo_loss
                    loss_dict.update({
                        "geo_penetration": penetration_loss,
                        "geo_approach": approach_loss,
                        "geo_region": region_loss,
                    })
                # import pdb; pdb.set_trace()
            total += self._update_loss("vq_commit", rs_set['loss_commit'],
                                       rs_set['loss_commit'])

        if self.stage in ["lm_pretrain", "lm_instruct"]:
            total += self._update_loss("gpt_loss", rs_set['outputs'].loss,
                                       rs_set['outputs'].loss)

        # Update the total loss
        self.total += total.detach()
        self.count += 1
        if return_dict:
            return total, loss_dict
        return total

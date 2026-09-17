from hoigpt.lib.models.mano import build_mano_aa
from hoigpt.lib.models.object import build_object_model
from hoigpt.lib.utils.data import(
    process_obj_result, 
    process_hand_result, 
)

import torch


def mo2mesh(datamodule, motion, obj_verts, lhand_layer, rhand_layer, obj_top_idx=None):
    motion_denorm = datamodule.denormalize(motion)
    bs, ws, dim = motion.shape
    
    pred_motion_denorm = motion_denorm.reshape(-1, dim)
    lhand_verts, lhand_faces, lhand_joint = process_hand_result(lhand_layer, pred_motion_denorm[:,:99])
    rhand_verts, rhand_faces, rhand_joint = process_hand_result(rhand_layer, pred_motion_denorm[:,99:198])

    # lhand_verts_gt, lhand_faces_gt, lhand_joint_gt = process_hand_result(lhand_layer, gt_motion_denorm[:,:99])
    # rhand_verts_gt, rhand_faces_gt, rhand_joint_gt = process_hand_result(rhand_layer, gt_motion_denorm[:,99:198])

    
    obj_verts_tfs = []
    for i in range(obj_verts.shape[0]):
        obj_verts_tf = process_obj_result(obj_verts[i].float(), motion_denorm[i, :,-10:], datamodule.name, obj_top_idx).float()
        obj_verts_tfs.append(obj_verts_tf)
    # import pdb; pdb.set_trace()
    obj_verts_tfs = torch.cat(obj_verts_tfs, 0)
    return {
        "lhand_verts": lhand_verts, "lhand_faces": lhand_faces, "lhand_joint": lhand_joint,
        "rhand_verts": rhand_verts, "rhand_faces": rhand_faces, "rhand_joint":rhand_joint, "obj_verts_tf": obj_verts_tfs
    }
    

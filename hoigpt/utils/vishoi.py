import torch
import numpy as np
from hoigpt.lib.models.object import build_object_model
from hoigpt.lib.utils.rot import axis_angle_to_rot6d
from hoigpt.lib.utils.demo_utils import (
    get_obj_top_file,
    proc_results,
    get_object_hand_info,
    get_hoi_info,
)
from hoigpt.lib.utils.proc import (
    proc_long_torch_cuda, 
    proc_torch_cuda, 
    pc_normalize,
)
from hoigpt.lib.utils.file import (
    make_save_folder, 
    save_video, 
    save_mesh_obj, 
)
from hoigpt.lib.models.object import build_object_model

from hoigpt.lib.utils.visualize import render_videos
from hoigpt.lib.models.mano import build_mano_aa
from hoigpt.lib.utils.renderer import Renderer

@torch.no_grad()   
def rendering_hoi(x_lhand, x_rhand, obj_6d, lhand_layer, rhand_layer, object_model, renderer, object_name, path, dataname, data_root):
    # if obj_top_idx != None:
    obj_pc, obj_pc_normal, normalized_obj_pc, _, obj_verts, obj_faces, \
        obj_top_idx, obj_pc_top_idx = get_hoi_info(data_root, dataname, object_name, object_model)

    obj_verts = obj_verts.float().to(obj_6d.device)


    obj_verts, lhand_vertices, lhand_faces, \
    rhand_vertices, rhand_faces = \
    proc_results(
        x_lhand, x_rhand, obj_6d, 
        obj_verts, lhand_layer, rhand_layer, 
        1, 1, 
        dataname, obj_top_idx
    )


    motion_video = render_videos(
        renderer, lhand_vertices, lhand_faces, 
        rhand_vertices, rhand_faces, 
        obj_verts, obj_faces, 
        1, 1, 
    )

    # Delete unused variables to save memory
    # del lhand_vertices, lhand_faces, rhand_vertices, rhand_faces, obj_verts, obj_faces

    frames = save_video(
        motion_video, fps=10, format="gif",
        save_path=path
    )

    # Delete unused variables to save memory
    lhand_layer.eval()
    rhand_layer.eval()
    return np.transpose(frames, (0,3,1,2))

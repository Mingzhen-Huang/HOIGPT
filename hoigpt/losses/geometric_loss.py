import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points


CONTACT_THRESHOLD_SQ = 0.02 ** 2
PENETRATION_BATCH_CHUNK = 8
PENETRATION_POINT_CHUNK = 128
CONTACT_FINGERTIP_PRIOR = [
    697, 698, 699, 700, 712, 713, 714, 715, 737, 738, 739, 740, 741, 743, 744, 745,
    746, 748, 749, 750, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764,
    765, 766, 767, 768, 46, 47, 48, 49, 164, 165, 166, 167, 194, 195, 223, 237, 238,
    280, 281, 298, 301, 317, 320, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332,
    333, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354,
    355, 356, 357, 358, 359, 375, 376, 386, 387, 396, 397, 402, 403, 413, 429, 433,
    434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 452, 453, 454, 455, 456,
    459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 484, 485, 486,
    496, 497, 506, 507, 513, 514, 524, 545, 546, 547, 548, 549, 550, 551, 552, 553,
    555, 563, 564, 565, 566, 567, 570, 572, 573, 574, 575, 576, 577, 578, 580, 581,
    582, 583, 600, 601, 602, 614, 615, 624, 625, 630, 631, 641, 663, 664, 665, 666,
    667, 668, 670, 672, 680, 681, 682, 683, 684, 686, 687, 688, 689, 690, 691, 692,
    693, 694, 695, 73, 96, 98, 99, 772, 774, 775, 777,
]


def get_NN(src_xyz, trg_xyz, k=1):
    src_lengths = torch.full(
        (src_xyz.shape[0],),
        src_xyz.shape[1],
        dtype=torch.int64,
        device=src_xyz.device,
    )
    trg_lengths = torch.full(
        (trg_xyz.shape[0],),
        trg_xyz.shape[1],
        dtype=torch.int64,
        device=trg_xyz.device,
    )
    src_nn = knn_points(src_xyz, trg_xyz, lengths1=src_lengths, lengths2=trg_lengths, K=k)
    return src_nn.dists[..., 0], src_nn.idx[..., 0]


def get_faces_xyz(faces_idx, xyz):
    if faces_idx.dim() == 2:
        faces_idx = faces_idx.unsqueeze(0).expand(xyz.shape[0], -1, -1)
    faces_idx = faces_idx.long()
    batch_idx = torch.arange(xyz.shape[0], device=xyz.device)[:, None, None]
    return xyz[batch_idx, faces_idx]


def batch_mesh_contains_points(
    ray_origins,
    obj_triangles,
    direction=torch.tensor([0.4395064455, 0.617598629942, 0.652231566745]),
):
    tol_thresh = 1e-7
    batch_size = obj_triangles.shape[0]
    triangle_nb = obj_triangles.shape[1]
    point_nb = ray_origins.shape[1]

    direction = direction.to(device=ray_origins.device, dtype=ray_origins.dtype)
    exterior = torch.empty(batch_size, point_nb, dtype=torch.bool, device=ray_origins.device)

    for batch_start in range(0, batch_size, PENETRATION_BATCH_CHUNK):
        batch_end = min(batch_start + PENETRATION_BATCH_CHUNK, batch_size)
        triangles = obj_triangles[batch_start:batch_end]
        points = ray_origins[batch_start:batch_end]

        v0 = triangles[:, :, 0]
        v1 = triangles[:, :, 1]
        v2 = triangles[:, :, 2]
        v0v1 = v1 - v0
        v0v2 = v2 - v0

        chunk_batch = triangles.shape[0]
        batch_direction = direction.view(1, 1, 3).expand(chunk_batch, triangle_nb, 3)
        pvec = torch.cross(batch_direction, v0v2, dim=2)
        dets = (v0v1 * pvec).sum(dim=2)
        parallel = dets.abs() < tol_thresh
        invdet = 1 / (dets + 0.1 * tol_thresh)

        for point_start in range(0, point_nb, PENETRATION_POINT_CHUNK):
            point_end = min(point_start + PENETRATION_POINT_CHUNK, point_nb)
            point_chunk = points[:, point_start:point_end]

            tvec = point_chunk[:, :, None, :] - v0[:, None, :, :]
            u_val = (tvec * pvec[:, None, :, :]).sum(dim=-1) * invdet[:, None, :]
            u_correct = (u_val > 0) & (u_val < 1)

            qvec = torch.cross(tvec, v0v1[:, None, :, :], dim=-1)
            v_val = (qvec * batch_direction[:, None, :, :]).sum(dim=-1) * invdet[:, None, :]
            v_correct = (v_val > 0) & (u_val + v_val < 1)

            t = (qvec * v0v2[:, None, :, :]).sum(dim=-1) * invdet[:, None, :]
            t_pos = t >= tol_thresh
            final_inter = v_correct & u_correct & (~parallel[:, None, :]) & t_pos
            exterior[batch_start:batch_end, point_start:point_end] = final_inter.sum(dim=2) % 2 == 0

    return exterior


def _zero_like(reference):
    return reference.new_zeros(())


def _safe_contact_vertices(hand_xyz):
    valid_prior = [idx for idx in CONTACT_FINGERTIP_PRIOR if idx < hand_xyz.shape[1]]
    if not valid_prior:
        return hand_xyz
    return hand_xyz[:, valid_prior, :]


def _masked_mean(values, mask):
    mask = mask.bool()
    if not mask.any():
        return values.mean() * 0.0
    return values[mask].mean()


def Contact_loss(obj_xyz, hand_xyz, cmap):
    hand_xyz_prior = _safe_contact_vertices(hand_xyz)
    obj_cd, _ = get_NN(obj_xyz, hand_xyz_prior)
    if cmap.bool().any():
        return 3000.0 * obj_cd[cmap.bool()].mean()

    k = min(8, obj_cd.shape[1])
    return 3000.0 * obj_cd.topk(k, dim=1, largest=False).values.mean()


def TTT_loss(
    hand_xyz,
    hand_face,
    obj_xyz,
    hand_xyz_gt=None,
    obj_xyz_gt=None,
    cmap_affordance=None,
    cmap_pointnet=None,
    hand_joint=None,
):
    if hand_xyz.numel() == 0 or obj_xyz.numel() == 0:
        zero = _zero_like(hand_xyz)
        return zero, zero, zero

    hand_nn_dist, _ = get_NN(obj_xyz, hand_xyz)
    pred_contact = hand_nn_dist < CONTACT_THRESHOLD_SQ

    penetration_loss = _zero_like(hand_xyz)
    if hand_face is not None:
        interior_sum = _zero_like(hand_xyz)
        interior_count = 0
        with torch.no_grad():
            for batch_start in range(0, hand_xyz.shape[0], PENETRATION_BATCH_CHUNK):
                batch_end = min(batch_start + PENETRATION_BATCH_CHUNK, hand_xyz.shape[0])
                hand_triangles = get_faces_xyz(
                    hand_face[batch_start:batch_end] if hand_face.dim() == 3 else hand_face,
                    hand_xyz[batch_start:batch_end],
                )
                exterior = batch_mesh_contains_points(
                    obj_xyz[batch_start:batch_end],
                    hand_triangles,
                )
                interior = ~exterior
                if interior.any():
                    interior_sum = interior_sum + hand_nn_dist[batch_start:batch_end][interior].sum()
                    interior_count += int(interior.sum().item())
        if interior_count > 0:
            penetration_loss = 120.0 * (interior_sum / interior_count)

    approach_loss = 2.5 * Contact_loss(obj_xyz, hand_xyz, cmap=pred_contact)
    if hand_joint is not None:
        joint_nn_dist, _ = get_NN(obj_xyz, hand_joint)
        joint_k = min(8, joint_nn_dist.shape[1])
        approach_loss = approach_loss + 250.0 * joint_nn_dist.topk(
            joint_k,
            dim=1,
            largest=False,
        ).values.mean()

    region_terms = []
    if hand_xyz_gt is not None and obj_xyz_gt is not None:
        gt_nn_dist, _ = get_NN(obj_xyz_gt, hand_xyz_gt)
        gt_contact = gt_nn_dist < CONTACT_THRESHOLD_SQ
        region_terms.append(F.mse_loss(pred_contact.float(), gt_contact.float()))
    if cmap_affordance is not None and cmap_pointnet is not None:
        region_terms.append(1e-4 * F.mse_loss(cmap_affordance, cmap_pointnet))

    if region_terms:
        region_loss = sum(region_terms) / len(region_terms)
    else:
        region_loss = _zero_like(hand_xyz)

    return penetration_loss, approach_loss, region_loss


"""Feature conversions extracted from the original ARCTIC/GRAB scripts.

No MANO model, renderer, checkpoint, or CUDA device is needed to construct the
208 pose/translation features. Rotations retain the original column-based 6D
convention from hoigpt.lib.utils.rot.
"""

import numpy as np
import torch

from hoigpt.lib.utils.rot import axis_angle_to_rot6d


def array(value, width, label):
    value = np.asarray(value)
    if value.ndim != 2 or value.shape[1] != width or not len(value):
        raise ValueError(f'{label}: expected nonempty [T, {width}], got {value.shape}')
    if not np.issubdtype(value.dtype, np.floating):
        value = value.astype(np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f'{label}: non-finite values')
    return value


def rotation6d(value):
    shape = value.shape
    with torch.no_grad():
        return axis_angle_to_rot6d(torch.from_numpy(np.ascontiguousarray(value)).reshape(-1, 3)).reshape(shape[0], -1).numpy()


def hand_features(hand, pose_key, rotation_key, translation_key, origin, frames):
    pose = array(hand[pose_key], 45, pose_key)
    orient = array(hand[rotation_key], 3, rotation_key)
    translation = array(hand[translation_key], 3, translation_key)
    if not all(len(x) == frames for x in (pose, orient, translation)):
        raise ValueError('Hand and object frame counts differ')
    pose6d = rotation6d(np.concatenate((orient, pose), axis=1))
    return np.concatenate((pose6d, translation - origin), axis=1)


def bounds(start, end, count):
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= count:
        raise ValueError(f'Expected 0 <= start < end <= {count}; got [{start}, {end})')


def arctic_features(mano, objects, start, end):
    """ARCTIC: hand translations in meters, object translations in millimeters.

    Rebase all translations to the object at the clip's first frame; end is
    exclusive, matching the original ARCTIC slicing convention.
    """
    objects = array(objects, 7, 'ARCTIC object: articulation, axis-angle, translation')
    bounds(start, end, len(objects))
    origin = objects[start:start + 1, 4:] / 1000.0
    left = hand_features(mano['left'], 'pose', 'rot', 'trans', origin, len(objects))
    right = hand_features(mano['right'], 'pose', 'rot', 'trans', origin, len(objects))
    obj = np.zeros((len(objects), 10), dtype=np.float64)
    obj[:, :1] = objects[:, :1]
    obj[:, 1:7] = rotation6d(objects[:, 1:4])
    obj[:, 7:] = objects[:, 4:] / 1000.0 - origin
    return np.concatenate((left, right, obj), axis=1)[start:end].copy()


def grab_features(mano, objects, start, end):
    """GRAB: translations in meters; rigid articulation feature is zero.

    Preserve the original GRAB reference at source frame 0, NOT clip start.
    The manifest uses an exclusive end; legacy inclusive ends must be incremented.
    """
    objects = array(objects, 6, 'GRAB object: axis-angle, translation')
    bounds(start, end, len(objects))
    origin = objects[:1, 3:]
    left = hand_features(mano['left'], 'fullpose', 'global_orient', 'transl', origin, len(objects))
    right = hand_features(mano['right'], 'fullpose', 'global_orient', 'transl', origin, len(objects))
    obj = np.zeros((len(objects), 10), dtype=np.float64)
    obj[:, 1:7] = rotation6d(objects[:, :3])
    obj[:, 7:] = objects[:, 3:] - origin
    return np.concatenate((left, right, obj), axis=1)[start:end].copy()

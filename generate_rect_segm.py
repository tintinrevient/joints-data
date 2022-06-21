import glob

from detectron2.utils.logger import setup_logger
setup_logger()

import cv2, os, re
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from densepose.config import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor
import torch
import argparse
from scipy.spatial import ConvexHull
import pandas as pd


# files of config
openpose_keypoints_dir = os.path.join('keypoints')

# coarse segmentation:
# 0 = Background
# 1 = Torso,
# 2 = Right Hand, 3 = Left Hand, 4 = Left Foot, 5 = Right Foot,
# 6 = Upper Leg Right, 7 = Upper Leg Left, 8 = Lower Leg Right, 9 = Lower Leg Left,
# 10 = Upper Arm Left, 11 = Upper Arm Right, 12 = Lower Arm Left, 13 = Lower Arm Right,
# 14 = Head

COARSE_ID = [
    'Background',
    'Torso',
    'RHand', 'LHand', 'LFoot', 'RFoot',
    'RThigh', 'LThigh', 'RCalf', 'LCalf',
    'LUpperArm', 'RUpperArm', 'LLowerArm', 'RLowerArm',
    'Head'
]

# BGRA -> alpha channel: 0 = transparent, 255 = non-transparent
COARSE_TO_COLOR = {
    'Background': [255, 255, 255, 255],
    'Torso': [191, 78, 22, 255],
    'RThigh': [167, 181, 44, 255],
    'LThigh': [141, 187, 91, 255],
    'RCalf': [114, 191, 147, 255],
    'LCalf': [96, 188, 192, 255],
    'LUpperArm': [87, 207, 112, 255],
    'RUpperArm': [55, 218, 162, 255],
    'LLowerArm': [25, 226, 216, 255],
    'RLowerArm': [37, 231, 253, 255],
    'Head': [14, 251, 249, 255]
}

# fine segmentation:
# 0 = Background
# 1, 2 = Torso,
# 3 = Right Hand, 4 = Left Hand, 5 = Left Foot, 6 = Right Foot,
# 7, 9 = Upper Leg Right, 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right, 12, 14 = Lower Leg Left,
# 15, 17 = Upper Arm Left, 16, 18 = Upper Arm Right, 19, 21 = Lower Arm Left, 20, 22 = Lower Arm Right,
# 23, 24 = Head

FINE_TO_COARSE_SEGMENTATION = {
    1: 1,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 6,
    10: 7,
    11: 8,
    12: 9,
    13: 8,
    14: 9,
    15: 10,
    16: 11,
    17: 10,
    18: 11,
    19: 12,
    20: 13,
    21: 12,
    22: 13,
    23: 14,
    24: 14
}

# Body 25 Keypoints
JOINT_ID = [
    'Nose', 'Neck',
    'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
    'MidHip',
    'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'REye', 'LEye', 'REar', 'LEar',
    'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel',
    'Background'
]


def _is_valid(keypoints):

    # check the scores for each main keypoint, which MUST exist!
    # main_keypoints = BODY BOX
    main_keypoints = ['Nose', 'Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'MidHip']

    keypoints = dict(zip(JOINT_ID, keypoints))

    # filter the main keypoints by score > 0
    filtered_keypoints = [key for key, value in keypoints.items() if key in main_keypoints and value[2] > 0]
    if len(filtered_keypoints) < 6:
        print('Number of valid keypoints (must be >= 6):', len(filtered_keypoints))
        return False
    else:
        return True


def _extract_i_from_iuvarr(iuv_arr):
    return iuv_arr[0, :, :]


def _extract_u_from_iuvarr(iuv_arr):
    return iuv_arr[1, :, :]


def _extract_v_from_iuvarr(iuv_arr):
    return iuv_arr[2, :, :]


def _extract_segm(result_densepose, is_coarse=True):

    iuv_array = torch.cat(
        (result_densepose.labels[None].type(torch.float32), result_densepose.uv * 255.0)
    ).type(torch.uint8)

    iuv_array = iuv_array.cpu().numpy()

    segm = _extract_i_from_iuvarr(iuv_array)

    if is_coarse:
        for fine_idx, coarse_idx in FINE_TO_COARSE_SEGMENTATION.items():
            segm[segm == fine_idx] = coarse_idx

    mask = np.zeros(segm.shape, dtype=np.uint8)
    mask[segm > 0] = 1

    return mask, segm


def _segm_xy(segm, segm_id, box_xywh):

    # bbox
    box_x, box_y, box_w, box_h = np.array(box_xywh).astype(int)

    y, x = np.where(segm == segm_id)

    # translate from the bbox coordinate to the original image coordinate
    return list(zip(x+box_x, y+box_y))


def _get_dict_of_segm_and_keypoints(segm, keypoints, box_xywh):

    segm_xy_list = []

    bg_xy = [] # 0
    segm_xy_list.append(bg_xy)

    torso_xy = _segm_xy(segm=segm, segm_id=1, box_xywh=box_xywh)
    segm_xy_list.append(torso_xy)

    r_hand_xy = [] # 2
    l_hand_xy = [] # 3
    l_foot_xy = [] # 4
    r_foot_xy = [] # 5
    segm_xy_list.append(r_hand_xy)
    segm_xy_list.append(l_hand_xy)
    segm_xy_list.append(l_foot_xy)
    segm_xy_list.append(r_foot_xy)

    r_thigh_xy = _segm_xy(segm=segm, segm_id=6, box_xywh=box_xywh)
    l_thigh_xy = _segm_xy(segm=segm, segm_id=7, box_xywh=box_xywh)
    r_calf_xy = _segm_xy(segm=segm, segm_id=8, box_xywh=box_xywh)
    l_calf_xy = _segm_xy(segm=segm, segm_id=9, box_xywh=box_xywh)
    segm_xy_list.append(r_thigh_xy)
    segm_xy_list.append(l_thigh_xy)
    segm_xy_list.append(r_calf_xy)
    segm_xy_list.append(l_calf_xy)

    l_upper_arm_xy = _segm_xy(segm=segm, segm_id=10, box_xywh=box_xywh)
    r_upper_arm_xy = _segm_xy(segm=segm, segm_id=11, box_xywh=box_xywh)
    l_lower_arm_xy = _segm_xy(segm=segm, segm_id=12, box_xywh=box_xywh)
    r_lower_arm_xy = _segm_xy(segm=segm, segm_id=13, box_xywh=box_xywh)
    segm_xy_list.append(l_upper_arm_xy)
    segm_xy_list.append(r_upper_arm_xy)
    segm_xy_list.append(l_lower_arm_xy)
    segm_xy_list.append(r_lower_arm_xy)

    head_xy = _segm_xy(segm=segm, segm_id=14, box_xywh=box_xywh)
    segm_xy_list.append(head_xy)

    # segments dictionary
    segm_xy_dict = dict(zip(COARSE_ID, segm_xy_list))

    # keypoints dictionary
    keypoints = np.array(keypoints).astype(int)
    keypoints_dict = dict(zip(JOINT_ID, keypoints))

    return segm_xy_dict, keypoints_dict


def _segm_xy_centroid(segm_xy):

    size = len(segm_xy)

    x = [x for x, y in segm_xy if not np.isnan(x)]
    y = [y for x, y in segm_xy if not np.isnan(y)]
    centroid = (sum(x) / size, sum(y) / size)

    return centroid


def _get_dict_of_midpoints(segm_xy_dict, keypoints_dict):

    midpoints_dict = {}

    # head centroid
    head_centroid_x, head_centroid_y = _segm_xy_centroid(segm_xy_dict['Head'])
    midpoints_dict['Head'] = np.array([head_centroid_x, head_centroid_y])

    # torso midpoint
    midpoints_dict['Torso'] = (keypoints_dict['Neck'] + keypoints_dict['MidHip']) / 2

    # upper limbs
    midpoints_dict['RUpperArm'] = (keypoints_dict['RShoulder'] + keypoints_dict['RElbow']) / 2
    midpoints_dict['RLowerArm'] = (keypoints_dict['RElbow'] + keypoints_dict['RWrist']) / 2
    midpoints_dict['LUpperArm'] = (keypoints_dict['LShoulder'] + keypoints_dict['LElbow']) / 2
    midpoints_dict['LLowerArm'] = (keypoints_dict['LElbow'] + keypoints_dict['LWrist']) / 2

    # lower limbs
    midpoints_dict['RThigh'] = (keypoints_dict['RHip'] + keypoints_dict['RKnee']) / 2
    midpoints_dict['RCalf'] = (keypoints_dict['RKnee'] + keypoints_dict['RAnkle']) / 2
    midpoints_dict['LThigh'] = (keypoints_dict['LHip'] + keypoints_dict['LKnee']) / 2
    midpoints_dict['LCalf'] = (keypoints_dict['LKnee'] + keypoints_dict['LAnkle']) / 2

    return midpoints_dict


def _calc_angle(point1, center, point2):

    try:
        a = np.array(point1)[0:2] - np.array(center)[0:2]
        b = np.array(point2)[0:2] - np.array(center)[0:2]

        cos_theta = np.dot(a, b)
        sin_theta = np.cross(a, b)

        rad = np.arctan2(sin_theta, cos_theta)
        deg = np.rad2deg(rad)

        if np.isnan(rad):
            return 0, 0

        return rad, deg

    except Exception as ex:
        print(ex)
        return 0, 0


def _get_dict_of_rotated_angles(keypoints_dict, midpoints_dict):

    rotated_angles_dict = {}

    # head
    reference_point = np.array(keypoints_dict['Neck']) + np.array((0, -100, 0))
    rad, deg = _calc_angle(point1=midpoints_dict['Head'], center=keypoints_dict['Neck'], point2=reference_point)
    rotated_angles_dict['Head'] = rad

    # torso
    reference_point = np.array(keypoints_dict['MidHip']) + np.array((0, -100, 0))
    rad, deg = _calc_angle(point1=keypoints_dict['Neck'], center=keypoints_dict['MidHip'], point2=reference_point)
    rotated_angles_dict['Torso'] = rad

    # upper limbs
    reference_point = np.array(keypoints_dict['RShoulder']) + np.array((-100, 0, 0))
    rad, deg = _calc_angle(point1=keypoints_dict['RElbow'], center=keypoints_dict['RShoulder'], point2=reference_point)
    rotated_angles_dict['RUpperArm'] = rad

    reference_point = np.array(keypoints_dict['RElbow']) + np.array((-100, 0, 0))
    rad, deg = _calc_angle(point1=keypoints_dict['RWrist'], center=keypoints_dict['RElbow'], point2=reference_point)
    rotated_angles_dict['RLowerArm'] = rad

    reference_point = np.array(keypoints_dict['LShoulder']) + np.array((100, 0, 0))
    rad, deg = _calc_angle(point1=keypoints_dict['LElbow'], center=keypoints_dict['LShoulder'], point2=reference_point)
    rotated_angles_dict['LUpperArm'] = rad

    reference_point = np.array(keypoints_dict['LElbow']) + np.array((100, 0, 0))
    rad, deg = _calc_angle(point1=keypoints_dict['LWrist'], center=keypoints_dict['LElbow'], point2=reference_point)
    rotated_angles_dict['LLowerArm'] = rad

    # lower limbs
    reference_point = np.array(keypoints_dict['RHip']) + np.array((0, 100, 0))
    rad, deg = _calc_angle(point1=keypoints_dict['RKnee'], center=keypoints_dict['RHip'], point2=reference_point)
    rotated_angles_dict['RThigh'] = rad

    reference_point = np.array(keypoints_dict['RKnee']) + np.array((0, 100, 0))
    rad, deg = _calc_angle(point1=keypoints_dict['RAnkle'], center=keypoints_dict['RKnee'], point2=reference_point)
    rotated_angles_dict['RCalf'] = rad

    reference_point = np.array(keypoints_dict['LHip']) + np.array((0, 100, 0))
    rad, deg = _calc_angle(point1=keypoints_dict['LKnee'], center=keypoints_dict['LHip'], point2=reference_point)
    rotated_angles_dict['LThigh'] = rad

    reference_point = np.array(keypoints_dict['LKnee']) + np.array((0, 100, 0))
    rad, deg = _calc_angle(point1=keypoints_dict['LAnkle'], center=keypoints_dict['LKnee'], point2=reference_point)
    rotated_angles_dict['LCalf'] = rad

    return rotated_angles_dict


def _draw_segm_and_keypoints(image, segm_xy_dict, keypoints_dict):

    # head
    for x, y in segm_xy_dict['Head']:
        cv2.circle(image, (x, y), 1, COARSE_TO_COLOR['Head'], -1)
    cv2.circle(image, (keypoints_dict['Nose'][0], keypoints_dict['Nose'][1]), 5, (255, 0, 255), -1)

    # torso
    for x, y in segm_xy_dict['Torso']:
        cv2.circle(image, (x, y), 1, COARSE_TO_COLOR['Torso'], -1)
    cv2.circle(image, (keypoints_dict['Neck'][0], keypoints_dict['Neck'][1]), 5, (255, 0, 255), -1)

    # upper limbs
    for x, y in segm_xy_dict['RUpperArm']:
        cv2.circle(image, (x, y), 1, COARSE_TO_COLOR['RUpperArm'], -1)
    for x, y in segm_xy_dict['RLowerArm']:
        cv2.circle(image, (x, y), 1, COARSE_TO_COLOR['RLowerArm'], -1)
    for x, y in segm_xy_dict['LUpperArm']:
        cv2.circle(image, (x, y), 1, COARSE_TO_COLOR['LUpperArm'], -1)
    for x, y in segm_xy_dict['LLowerArm']:
        cv2.circle(image, (x, y), 1, COARSE_TO_COLOR['LLowerArm'], -1)
    cv2.circle(image, (keypoints_dict['RShoulder'][0], keypoints_dict['RShoulder'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['RElbow'][0], keypoints_dict['RElbow'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['RWrist'][0], keypoints_dict['RWrist'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['LShoulder'][0], keypoints_dict['LShoulder'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['LElbow'][0], keypoints_dict['LElbow'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['LWrist'][0], keypoints_dict['LWrist'][1]), 5, (255, 0, 255), -1)

    # lower limbs
    for x, y in segm_xy_dict['RThigh']:
        cv2.circle(image, (x, y), 1, COARSE_TO_COLOR['RThigh'], -1)
    for x, y in segm_xy_dict['RCalf']:
        cv2.circle(image, (x, y), 1, COARSE_TO_COLOR['RCalf'], -1)
    for x, y in segm_xy_dict['LThigh']:
        cv2.circle(image, (x, y), 1, COARSE_TO_COLOR['LThigh'], -1)
    for x, y in segm_xy_dict['LCalf']:
        cv2.circle(image, (x, y), 1, COARSE_TO_COLOR['LCalf'], -1)
    cv2.circle(image, (keypoints_dict['MidHip'][0], keypoints_dict['MidHip'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['RHip'][0], keypoints_dict['RHip'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['RKnee'][0], keypoints_dict['RKnee'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['RAnkle'][0], keypoints_dict['RAnkle'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['LHip'][0], keypoints_dict['LHip'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['LKnee'][0], keypoints_dict['LKnee'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['LAnkle'][0], keypoints_dict['LAnkle'][1]), 5, (255, 0, 255), -1)

    cv2.imshow('original image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _euclidian(point1, point2):

    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def _remove_outlier(segm_xy):

    # outlier factor
    factor = 2

    # mean of [x, y]
    xy_mean = np.mean(segm_xy, axis=0)

    # mean distance between [x, y] and mean of [x, y]
    distance_mean = np.mean([_euclidian(xy, xy_mean) for xy in segm_xy])

    # remove outliers from segm_xy
    segm_xy_without_outliers = [xy for xy in segm_xy if _euclidian(xy, xy_mean) <= distance_mean * factor]

    return segm_xy_without_outliers


def _get_min_bounding_rect(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.
    """

    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval.astype(int)


def _draw_one_segm_bbox(image, segm_id, segm_xy):

    global bbox_segm_dict

    # remove outliers
    # print('Before removing outliers:', len(segm_xy))
    segm_xy = np.array(_remove_outlier(segm_xy=segm_xy)).astype(int)
    # print('After removing outliers:', len(segm_xy))

    # get the minimum bounding rectangle of segm_xy
    try:
        rect_xy = _get_min_bounding_rect(segm_xy)
    except Exception as ex:
        print(ex)
        return

    cv2.fillPoly(image, [rect_xy], COARSE_TO_COLOR[segm_id])

    dist1 = _euclidian(rect_xy[0], rect_xy[1])
    dist2 = _euclidian(rect_xy[1], rect_xy[2])
    if 'Arm' in segm_id:
        w = max(dist1, dist2)
        h = min(dist1, dist2)
    elif segm_id != 'Head':
        w = min(dist1, dist2)
        h = max(dist1, dist2)
    else:
        w = 0
        h = 0

    bbox_segm_dict[segm_id]  = {}
    bbox_segm_dict[segm_id]['half_w'] = int(w / 2)
    bbox_segm_dict[segm_id]['half_h'] = int(h / 2)


def _draw_segm_bbox(image, segm_xy_dict, keypoints_dict):

    # head
    _draw_one_segm_bbox(image, segm_id='Head', segm_xy=segm_xy_dict['Head'])

    # torso
    _draw_one_segm_bbox(image, segm_id='Torso', segm_xy=segm_xy_dict['Torso'])

    # upper limbs
    _draw_one_segm_bbox(image, segm_id='RUpperArm', segm_xy=segm_xy_dict['RUpperArm'])
    _draw_one_segm_bbox(image, segm_id='RLowerArm', segm_xy=segm_xy_dict['RLowerArm'])
    _draw_one_segm_bbox(image, segm_id='LUpperArm', segm_xy=segm_xy_dict['LUpperArm'])
    _draw_one_segm_bbox(image, segm_id='LLowerArm', segm_xy=segm_xy_dict['LLowerArm'])

    # lower limbs
    _draw_one_segm_bbox(image, segm_id='RThigh', segm_xy=segm_xy_dict['RThigh'])
    _draw_one_segm_bbox(image, segm_id='RCalf', segm_xy=segm_xy_dict['RCalf'])
    _draw_one_segm_bbox(image, segm_id='LThigh', segm_xy=segm_xy_dict['LThigh'])
    _draw_one_segm_bbox(image, segm_id='LCalf', segm_xy=segm_xy_dict['LCalf'])


def _rotate(point, center, rad):

    x = ((point[0] - center[0]) * np.cos(rad)) - ((point[1] - center[1]) * np.sin(rad)) + center[0]
    y = ((point[0] - center[0]) * np.sin(rad)) + ((point[1] - center[1]) * np.cos(rad)) + center[1]

    if len(point) == 3:
        return [int(x), int(y), point[2]] # for keypoints with score
    elif len(point) == 2:
        return (int(x), int(y)) # for segments (x, y) without score


def _draw_one_rotated_and_scaled_segm(segm_id, segm_xy, midpoint, scaler):

    global temp_segm_dict

    # for the segments not inferred by DensePose
    if len(segm_xy) < 1:
        temp_segm_dict[segm_id] = {}
        temp_segm_dict[segm_id]['half_w'] = 0
        temp_segm_dict[segm_id]['half_h'] = 0
        return

    min_x, min_y = np.min(segm_xy, axis=0).astype(int)
    max_x, max_y = np.max(segm_xy, axis=0).astype(int)

    # translate and scale w + h
    w = int((max_x - min_x) * scaler)
    h = int((max_y - min_y) * scaler)

    img_bg = np.empty((h, w, 4), np.uint8)
    img_bg.fill(255)
    img_bg[:, :] = COARSE_TO_COLOR[segm_id]

    # translate and scale midpoint
    translated_midpoint = ((midpoint[0:2] - np.array([min_x, min_y])) * scaler).astype(int)
    cv2.circle(img_bg, (translated_midpoint[0], translated_midpoint[1]), 5, (255, 0, 255), -1)

    temp_segm_dict[segm_id] = {}
    if segm_id != 'Head':
        temp_segm_dict[segm_id]['half_w'] = min(max((w - translated_midpoint[0]), translated_midpoint[0]),
                                                int(bbox_segm_dict[segm_id]['half_w'] * scaler))
        temp_segm_dict[segm_id]['half_h'] = min(max((h - translated_midpoint[1]), translated_midpoint[1]),
                                                int(bbox_segm_dict[segm_id]['half_h'] * scaler))
    else:
        temp_segm_dict[segm_id]['half_w'] = max((w - translated_midpoint[0]), translated_midpoint[0])
        temp_segm_dict[segm_id]['half_h'] = max((h - translated_midpoint[1]), translated_midpoint[1])


def _draw_one_norm_segm(image, segm_id, norm_midpoint):

    global temp_segm_dict
    global norm_segm_dict

    if segm_id == 'RUpperArm' or segm_id == 'LUpperArm':
        half_h = max(temp_segm_dict['RUpperArm']['half_h'], temp_segm_dict['LUpperArm']['half_h'])
        half_w = max(temp_segm_dict['RUpperArm']['half_w'], temp_segm_dict['LUpperArm']['half_w'])

    elif segm_id == 'RLowerArm' or segm_id == 'LLowerArm':
        half_h = max(temp_segm_dict['RLowerArm']['half_h'], temp_segm_dict['LLowerArm']['half_h'])
        half_w = max(temp_segm_dict['RLowerArm']['half_w'], temp_segm_dict['LLowerArm']['half_w'])

    elif segm_id == 'RThigh' or segm_id == 'LThigh':
        half_h = max(temp_segm_dict['RThigh']['half_h'], temp_segm_dict['LThigh']['half_h'])
        half_w = max(temp_segm_dict['RThigh']['half_w'], temp_segm_dict['LThigh']['half_w'])

    elif segm_id == 'RCalf' or segm_id == 'LCalf':
        half_h = max(temp_segm_dict['RCalf']['half_h'], temp_segm_dict['LCalf']['half_h'])
        half_w = max(temp_segm_dict['RCalf']['half_w'], temp_segm_dict['LCalf']['half_w'])

    else:
        half_h = temp_segm_dict[segm_id]['half_h']
        half_w = temp_segm_dict[segm_id]['half_w']

    # if the segment does not exist on both left and right side, assign 0 as default
    norm_segm_dict[segm_id + '_w'] = int(half_w * 2) if half_w >= 1 and half_h >= 1 else 0
    norm_segm_dict[segm_id + '_h'] = int(half_h * 2) if half_w >= 1 and half_h >= 1 else 0

    img_bg = np.empty((int(half_h*2), int(half_w*2), 4), np.uint8)
    img_bg.fill(255)
    img_bg[:, :] = COARSE_TO_COLOR[segm_id]

    norm_midpoint_x, norm_midpoint_y = norm_midpoint
    min_x = int(norm_midpoint_x - half_w)
    max_x = int(norm_midpoint_x + half_w)
    min_y = int(norm_midpoint_y - half_h)
    max_y = int(norm_midpoint_y + half_h)

    # print(image[min_y:max_y, min_x:max_x, :].shape)
    # print(img_bg.shape)

    # draw the normalized segment
    if img_bg.shape == image[min_y:max_y, min_x:max_x, :].shape:
        added_image = cv2.addWeighted(image[min_y:max_y, min_x:max_x, :], 0.1, img_bg, 0.9, 0)
        try:
            image[min_y:max_y, min_x:max_x, :] = added_image
        except TypeError as ex:
            print(ex)

    # draw the normalized midpoint
    cv2.circle(image, tuple(norm_midpoint), radius=2, color=(255, 0, 255), thickness=-1)


def _draw_norm_segm(segm_xy_dict, keypoints_dict, midpoints_dict, rotated_angles_dict):

    global temp_segm_dict
    global norm_segm_dict

    # white image
    image = np.empty((624, 624, 4), np.uint8)
    image.fill(255)

    # common settings
    # coordinates [x, y] coming from distribution_segm.extract_contour_on_vitruve()
    # nose_y 146
    # torso_y 281
    # rupper_arm_x 218
    # rlower_arm_x 149
    # lupper_arm_x 405
    # llower_arm_x 474
    # thigh_y 427
    # calf_y 544

    # [x, y]
    mid_x = 312
    arm_line_y = 217
    right_leg_x = 288
    left_leg_x = 336

    norm_nose_xy = [mid_x, 146]
    norm_mid_torso_xy = [mid_x, 281]

    norm_mid_rupper_arm_xy = [218, arm_line_y]
    norm_mid_rlower_arm_xy = [149, arm_line_y]
    norm_mid_lupper_arm_xy = [405, arm_line_y]
    norm_mid_llower_arm_xy = [474, arm_line_y]

    norm_mid_rthigh_xy = [right_leg_x, 427]
    norm_mid_rcalf_xy = [right_leg_x, 544]
    norm_mid_lthigh_xy = [left_leg_x, 427]
    norm_mid_lcalf_xy = [left_leg_x, 544]

    # rotated segments
    rotated_segm_xy_dict = {}
    # rotated midpoints
    rotated_midpoints_dict = {}

    # head
    rotated_segm_xy_dict['Head'] = np.array([_rotate((x, y), keypoints_dict['Neck'], rotated_angles_dict['Head']) for (x, y) in segm_xy_dict['Head']])
    rotated_midpoints_dict['Head'] = np.array(_rotate(midpoints_dict['Head'], keypoints_dict['Neck'], rotated_angles_dict['Head']))

    # calculate scaler based on the segment 'head'
    min_x, min_y = np.min(rotated_segm_xy_dict['Head'], axis=0).astype(int)
    max_x, max_y = np.max(rotated_segm_xy_dict['Head'], axis=0).astype(int)
    h = max((rotated_midpoints_dict['Head'][1] - min_y) * 2, (max_y - rotated_midpoints_dict['Head'][1]) * 2)
    if h > 0:
        scaler = 60 / h
    norm_segm_dict['scaler'] = scaler

    _draw_one_rotated_and_scaled_segm('Head', rotated_segm_xy_dict['Head'], rotated_midpoints_dict['Head'], scaler)

    # torso
    rotated_segm_xy_dict['Torso'] = np.array([_rotate((x, y), keypoints_dict['MidHip'], rotated_angles_dict['Torso']) for (x, y) in segm_xy_dict['Torso']])
    rotated_midpoints_dict['Torso'] = np.array(_rotate(midpoints_dict['Torso'], keypoints_dict['MidHip'], rotated_angles_dict['Torso']))
    _draw_one_rotated_and_scaled_segm('Torso', rotated_segm_xy_dict['Torso'], rotated_midpoints_dict['Torso'], scaler)

    # upper limbs
    rotated_segm_xy_dict['RUpperArm'] = np.array([_rotate((x, y), keypoints_dict['RShoulder'], rotated_angles_dict['RUpperArm']) for (x, y) in segm_xy_dict['RUpperArm']])
    rotated_midpoints_dict['RUpperArm'] = np.array(_rotate(midpoints_dict['RUpperArm'], keypoints_dict['RShoulder'], rotated_angles_dict['RUpperArm']))
    _draw_one_rotated_and_scaled_segm('RUpperArm', rotated_segm_xy_dict['RUpperArm'], rotated_midpoints_dict['RUpperArm'], scaler)

    rotated_segm_xy_dict['RLowerArm'] = np.array([_rotate((x, y), keypoints_dict['RElbow'], rotated_angles_dict['RLowerArm']) for (x, y) in segm_xy_dict['RLowerArm']])
    rotated_midpoints_dict['RLowerArm'] = np.array(_rotate(midpoints_dict['RLowerArm'], keypoints_dict['RElbow'], rotated_angles_dict['RLowerArm']))
    _draw_one_rotated_and_scaled_segm('RLowerArm', rotated_segm_xy_dict['RLowerArm'], rotated_midpoints_dict['RLowerArm'], scaler)

    rotated_segm_xy_dict['LUpperArm'] = np.array([_rotate((x, y), keypoints_dict['LShoulder'], rotated_angles_dict['LUpperArm']) for (x, y) in segm_xy_dict['LUpperArm']])
    rotated_midpoints_dict['LUpperArm'] = np.array(_rotate(midpoints_dict['LUpperArm'], keypoints_dict['LShoulder'], rotated_angles_dict['LUpperArm']))
    _draw_one_rotated_and_scaled_segm('LUpperArm', rotated_segm_xy_dict['LUpperArm'], rotated_midpoints_dict['LUpperArm'], scaler)

    rotated_segm_xy_dict['LLowerArm'] = np.array([_rotate((x, y), keypoints_dict['LElbow'], rotated_angles_dict['LLowerArm']) for (x, y) in segm_xy_dict['LLowerArm']])
    rotated_midpoints_dict['LLowerArm'] = np.array(_rotate(midpoints_dict['LLowerArm'], keypoints_dict['LElbow'], rotated_angles_dict['LLowerArm']))
    _draw_one_rotated_and_scaled_segm('LLowerArm', rotated_segm_xy_dict['LLowerArm'], rotated_midpoints_dict['LLowerArm'], scaler)

    # lower limbs
    rotated_segm_xy_dict['RThigh'] = np.array([_rotate((x, y), keypoints_dict['RHip'], rotated_angles_dict['RThigh']) for (x, y) in segm_xy_dict['RThigh']])
    rotated_midpoints_dict['RThigh'] = np.array(_rotate(midpoints_dict['RThigh'], keypoints_dict['RHip'], rotated_angles_dict['RThigh']))
    _draw_one_rotated_and_scaled_segm('RThigh', rotated_segm_xy_dict['RThigh'], rotated_midpoints_dict['RThigh'], scaler)

    rotated_segm_xy_dict['RCalf'] = np.array([_rotate((x, y), keypoints_dict['RKnee'], rotated_angles_dict['RCalf']) for (x, y) in segm_xy_dict['RCalf']])
    rotated_midpoints_dict['RCalf'] = np.array(_rotate(midpoints_dict['RCalf'], keypoints_dict['RKnee'], rotated_angles_dict['RCalf']))
    _draw_one_rotated_and_scaled_segm('RCalf', rotated_segm_xy_dict['RCalf'], rotated_midpoints_dict['RCalf'], scaler)

    rotated_segm_xy_dict['LThigh'] = np.array([_rotate((x, y), keypoints_dict['LHip'], rotated_angles_dict['LThigh']) for (x, y) in segm_xy_dict['LThigh']])
    rotated_midpoints_dict['LThigh'] = np.array(_rotate(midpoints_dict['LThigh'], keypoints_dict['LHip'], rotated_angles_dict['LThigh']))
    _draw_one_rotated_and_scaled_segm('LThigh', rotated_segm_xy_dict['LThigh'], rotated_midpoints_dict['LThigh'], scaler)

    rotated_segm_xy_dict['LCalf'] = np.array([_rotate((x, y), keypoints_dict['LKnee'], rotated_angles_dict['LCalf']) for (x, y) in segm_xy_dict['LCalf']])
    rotated_midpoints_dict['LCalf'] = np.array(_rotate(midpoints_dict['LCalf'], keypoints_dict['LKnee'], rotated_angles_dict['LCalf']))
    _draw_one_rotated_and_scaled_segm('LCalf', rotated_segm_xy_dict['LCalf'], rotated_midpoints_dict['LCalf'], scaler)

    # head
    _draw_one_norm_segm(image, 'Head', norm_nose_xy)

    # torso
    _draw_one_norm_segm(image, 'Torso', norm_mid_torso_xy)

    # upper limbs
    _draw_one_norm_segm(image, 'RUpperArm', norm_mid_rupper_arm_xy)
    _draw_one_norm_segm(image, 'RLowerArm', norm_mid_rlower_arm_xy)
    _draw_one_norm_segm(image, 'LUpperArm', norm_mid_lupper_arm_xy)
    _draw_one_norm_segm(image, 'LLowerArm', norm_mid_llower_arm_xy)

    # lower limbs
    _draw_one_norm_segm(image, 'RThigh', norm_mid_rthigh_xy)
    _draw_one_norm_segm(image, 'RCalf', norm_mid_rcalf_xy)
    _draw_one_norm_segm(image, 'LThigh', norm_mid_lthigh_xy)
    _draw_one_norm_segm(image, 'LCalf', norm_mid_lcalf_xy)

    # debug - draw the normalized segments
    # cv2.imshow('norm image', image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return image


def _match(people_densepose, people_box_xywh, people_keypoints):

    matched_densepose, matched_box_xywh, matched_keypoints = [], [], []

    # Condition 1: mean_x and mean_y of the keypoints within bbox!!!
    # Condition 1: range_x and range_y of the keypoints > 0.5 * bbox!!!
    for keypoints in people_keypoints:
        positive_score_keypoints = [[x, y, score] for x, y, score in keypoints if score != 0]
        mean_x, mean_y, _ = np.mean(positive_score_keypoints, axis=0)
        min_x, min_y, _ = np.min(positive_score_keypoints, axis=0)
        max_x, max_y, _ = np.max(positive_score_keypoints, axis=0)
        range_x = max_x - min_x
        range_y = max_y - min_y

        for idx, box_xywh in enumerate(people_box_xywh):
            x, y, w, h = box_xywh
            if mean_x > x and mean_x < (x + w) and mean_y > y and mean_y < (y + h) and range_x > 0.5 * w and range_y > 0.5 * h:

                # updated matched data
                matched_densepose.append(people_densepose[idx])
                matched_box_xywh.append(people_box_xywh[idx])
                matched_keypoints.append(keypoints)

    return matched_densepose, matched_box_xywh, matched_keypoints


def visualize(infile, score_cutoff):
    print("Generate the normalized segment for:", infile)

    global norm_segm_dict

    image = cv2.imread(infile)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = np.tile(img_gray[:, :, np.newaxis], [1, 1, 3])

    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.MODEL.DEVICE = 'cpu'

    cfg.merge_from_file('./configs/densepose_rcnn_R_50_FPN_s1x.yaml')
    cfg.MODEL.WEIGHTS = './models/densepose_rcnn_R_50_FPN_s1x.pkl'

    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)

    # filter the probabilities of scores for each bbox > score_cutoff
    instances = outputs['instances']
    confident_detections = instances[instances.scores > score_cutoff]

    # extractor
    extractor = DensePoseResultExtractor()
    people_densepose, people_box_xywh = extractor(confident_detections)

    # boxes_xywh: tensor -> numpy array
    try:
        people_box_xywh = people_box_xywh.numpy()
    except AttributeError as ex:
        print(ex)
        return

    # load keypoints
    file_keypoints = os.path.join(openpose_keypoints_dir, '{}_keypoints.npy'.format(infile[infile.find('/') + 1:infile.rfind('.')]))
    try:
        people_keypoints = np.load(file_keypoints, allow_pickle='TRUE').item()['keypoints']
    except FileNotFoundError as ex:
        print(ex)
        return

    if len(people_keypoints) < 1 or len(people_densepose) < 1:
        print('Size of keypoints:', len(people_keypoints), 'Size of densepose:', len(people_densepose))
        return

    matched_densepose, matched_box_xywh, matched_keypoints = _match(people_densepose, people_box_xywh, people_keypoints)

    person_index = 0
    for person_densepose, person_box_xywh, person_keypoints in zip(matched_densepose, matched_box_xywh, matched_keypoints):

        # condition: valid body box!!!
        if _is_valid(person_keypoints):

            # increase the number of valid people
            person_index += 1

            # extract segm + mask
            mask, segm = _extract_segm(result_densepose=person_densepose)

            # get segm_xy + keypoints
            segm_xy_dict, keypoints_dict = _get_dict_of_segm_and_keypoints(segm, person_keypoints, person_box_xywh)

            # if the head does not exist, continue to the next person!!!
            if len(segm_xy_dict['Head']) < 1:
                continue

            # get midpoints
            midpoints_dict = _get_dict_of_midpoints(segm_xy_dict, keypoints_dict)

            # get rotated angles
            rotated_angles_dict = _get_dict_of_rotated_angles(keypoints_dict, midpoints_dict)

            # debug - draw the original segments
            # _draw_segm_and_keypoints(img_gray.copy(), segm_xy_dict, keypoints_dict)

            # draw the bbox of segments
            _draw_segm_bbox(img_gray.copy(), segm_xy_dict, keypoints_dict)

            # draw the normalized segments
            image = _draw_norm_segm(segm_xy_dict, keypoints_dict, midpoints_dict, rotated_angles_dict)

            # save the normalized data
            index_name = _generate_index_name(infile, person_index)
            df = pd.DataFrame(data=norm_segm_dict, index=[index_name])
            with open(os.path.join('output', 'norm_segm.csv'), 'a') as csv_file:
                df.to_csv(csv_file, index=True, header=False)
            # empty the data
            norm_segm_dict = {}


def _generate_index_name(infile, person_index):

    iter_list = [iter.start() for iter in re.finditer(r"/", infile)]
    artist = infile[iter_list[1] + 1:iter_list[2]]
    painting_number = infile[iter_list[2] + 1:infile.rfind('.')]
    index_name = '{}_{}_{}'.format(artist, painting_number, person_index)

    # impressionism
    # artist = 'Impressionism'
    # painting_number = infile[infile.rfind('/')+1:infile.rfind('.')]
    # index_name = '{}_{}_{}'.format(artist, painting_number, person_index)

    return index_name


if __name__ == '__main__':

    # modern
    # python generate_rect_segm.py --input datasets/modern/Paul\ Delvaux/90551.jpg
    # python generate_rect_segm.py --input datasets/modern/Paul\ Gauguin/30963.jpg

    # classical
    # python generate_rect_segm.py --input datasets/classical/Michelangelo/12758.jpg
    # python generate_rect_segm.py --input datasets/classical/Artemisia\ Gentileschi/45093.jpg

    parser = argparse.ArgumentParser(description='DensePose - Infer the segments')
    parser.add_argument('--input', help='Path to image file')
    args = parser.parse_args()

    bbox_segm_dict = {}
    temp_segm_dict = {}
    norm_segm_dict = {}

    # generate the normalized segment for one image
    if os.path.isfile(args.input):
        visualize(infile=args.input, score_cutoff=0.95)

    elif os.path.isdir(args.input):
        for infile in glob.glob(f'{args.input}/*.jpg'):
            visualize(infile, score_cutoff=0.95)

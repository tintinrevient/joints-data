import os, cv2
import numpy as np
import argparse
import pandas as pd
from pycocotools.coco import COCO
from densepose.structures import DensePoseDataRelative
from generate_rect_segm_coco import (
    COARSE_TO_COLOR,
    _calc_angle, _get_dp_mask, _segm_xy_centroid, _get_dict_of_midpoints, _get_dict_of_segm_and_keypoints
)


# the path to the data of norm_segm.csv
fname_norm_segm_coco_man = os.path.join('output', 'norm_segm_coco_man.csv')
fname_norm_segm_coco_woman = os.path.join('output', 'norm_segm_coco_woman.csv')

# the path to the data of contour.csv
fname_contour = os.path.join('output', 'contour.csv')

# dataset setting
coco_folder = os.path.join('datasets', 'coco')

# dense_pose annotation
dp_coco = COCO(os.path.join(coco_folder, 'annotations', 'densepose_minival2014.json'))

# caption annotation
caption_coco = COCO(os.path.join(coco_folder, 'annotations', 'captions_val2014.json'))


def _get_head_segm_centroid(head_xy):

    # get the centroid of the head segment
    head_centroid_x, head_centroid_y = _segm_xy_centroid(head_xy)

    return head_centroid_x, head_centroid_y


def _get_rotated_angles(keypoints, midpoints):

    rotated_angles = {}

    # head
    reference_point = np.array(keypoints['Neck']) + np.array((0, -100, 0))
    rad, deg = _calc_angle(point1=midpoints['Head'], center=keypoints['Neck'], point2=reference_point)
    # rotate back to original in reverse direction
    rotated_angles['Head'] = -deg

    # torso
    reference_point = np.array(keypoints['MidHip']) + np.array((0, -100, 0))
    rad, deg = _calc_angle(point1=keypoints['Neck'], center=keypoints['MidHip'], point2=reference_point)
    rotated_angles['Torso'] = -deg

    # upper limbs
    reference_point = np.array(keypoints['RShoulder']) + np.array((-100, 0, 0))
    rad, deg = _calc_angle(point1=keypoints['RElbow'], center=keypoints['RShoulder'], point2=reference_point)
    rotated_angles['RUpperArm'] = -deg

    reference_point = np.array(keypoints['RElbow']) + np.array((-100, 0, 0))
    rad, deg = _calc_angle(point1=keypoints['RWrist'], center=keypoints['RElbow'], point2=reference_point)
    rotated_angles['RLowerArm'] = -deg

    reference_point = np.array(keypoints['LShoulder']) + np.array((100, 0, 0))
    rad, deg = _calc_angle(point1=keypoints['LElbow'], center=keypoints['LShoulder'], point2=reference_point)
    rotated_angles['LUpperArm'] = -deg

    reference_point = np.array(keypoints['LElbow']) + np.array((100, 0, 0))
    rad, deg = _calc_angle(point1=keypoints['LWrist'], center=keypoints['LElbow'], point2=reference_point)
    rotated_angles['LLowerArm'] = -deg

    # lower limbs
    reference_point = np.array(keypoints['RHip']) + np.array((0, 100, 0))
    rad, deg = _calc_angle(point1=keypoints['RKnee'], center=keypoints['RHip'], point2=reference_point)
    rotated_angles['RThigh'] = -deg

    reference_point = np.array(keypoints['RKnee']) + np.array((0, 100, 0))
    rad, deg = _calc_angle(point1=keypoints['RAnkle'], center=keypoints['RKnee'], point2=reference_point)
    rotated_angles['RCalf'] = -deg

    reference_point = np.array(keypoints['LHip']) + np.array((0, 100, 0))
    rad, deg = _calc_angle(point1=keypoints['LKnee'], center=keypoints['LHip'], point2=reference_point)
    rotated_angles['LThigh'] = -deg

    reference_point = np.array(keypoints['LKnee']) + np.array((0, 100, 0))
    rad, deg = _calc_angle(point1=keypoints['LAnkle'], center=keypoints['LKnee'], point2=reference_point)
    rotated_angles['LCalf'] = -deg

    return rotated_angles


def _draw_norm_segm(image, midpoints, rotated_angles, dict_norm_segm, is_contour):

    # scaler
    scaler = 1 / dict_norm_segm['scaler']

    # head
    rect = ((midpoints['Head'][0], midpoints['Head'][1]),
            (dict_norm_segm['Head_w'] * scaler, dict_norm_segm['Head_h'] * scaler),
            rotated_angles['Head'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['Head'] if not is_contour else color, thickness=thickness)

    # torso
    rect = ((midpoints['Torso'][0], midpoints['Torso'][1]),
            (dict_norm_segm['Torso_w'] * scaler, dict_norm_segm['Torso_h'] * scaler),
            rotated_angles['Torso'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['Torso'] if not is_contour else color, thickness=thickness)

    # upper limbs
    rect = ((midpoints['RUpperArm'][0], midpoints['RUpperArm'][1]),
            (dict_norm_segm['RUpperArm_w'] * scaler, dict_norm_segm['RUpperArm_h'] * scaler),
            rotated_angles['RUpperArm'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['RUpperArm'] if not is_contour else color, thickness=thickness)

    rect = ((midpoints['RLowerArm'][0], midpoints['RLowerArm'][1]),
            (dict_norm_segm['RLowerArm_w'] * scaler, dict_norm_segm['RLowerArm_h'] * scaler),
            rotated_angles['RLowerArm'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['RLowerArm'] if not is_contour else color, thickness=thickness)

    rect = ((midpoints['LUpperArm'][0], midpoints['LUpperArm'][1]),
            (dict_norm_segm['LUpperArm_w'] * scaler, dict_norm_segm['LUpperArm_h'] * scaler),
            rotated_angles['LUpperArm'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['LUpperArm'] if not is_contour else color, thickness=thickness)

    rect = ((midpoints['LLowerArm'][0], midpoints['LLowerArm'][1]),
            (dict_norm_segm['LLowerArm_w'] * scaler, dict_norm_segm['LLowerArm_h'] * scaler),
            rotated_angles['LLowerArm'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['LLowerArm'] if not is_contour else color, thickness=thickness)

    # lower limbs
    rect = ((midpoints['RThigh'][0], midpoints['RThigh'][1]),
            (dict_norm_segm['RThigh_w'] * scaler, dict_norm_segm['RThigh_h'] * scaler),
            rotated_angles['RThigh'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['RThigh'] if not is_contour else color, thickness=thickness)

    rect = ((midpoints['RCalf'][0], midpoints['RCalf'][1]),
            (dict_norm_segm['RCalf_w'] * scaler, dict_norm_segm['RCalf_h'] * scaler),
            rotated_angles['RCalf'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['RCalf'] if not is_contour else color, thickness=thickness)

    rect = ((midpoints['LThigh'][0], midpoints['LThigh'][1]),
            (dict_norm_segm['LThigh_w'] * scaler, dict_norm_segm['LThigh_h'] * scaler),
            rotated_angles['LThigh'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['LThigh'] if not is_contour else color, thickness=thickness)

    rect = ((midpoints['LCalf'][0], midpoints['LCalf'][1]),
            (dict_norm_segm['LCalf_w'] * scaler, dict_norm_segm['LCalf_h'] * scaler),
            rotated_angles['LCalf'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['LCalf'] if not is_contour else color, thickness=thickness)


def visualize(image_id, person_index, gender, artist):

    entry = dp_coco.loadImgs(image_id)[0]

    dataset_name = entry['file_name'][entry['file_name'].find('_') + 1:entry['file_name'].rfind('_')]
    image_fpath = os.path.join(coco_folder, dataset_name, entry['file_name'])

    print('image_fpath:', image_fpath)

    im_gray = cv2.imread(image_fpath, cv2.IMREAD_GRAYSCALE)
    im_gray = np.tile(im_gray[:, :, np.newaxis], [1, 1, 3])

    dp_annotation_ids = dp_coco.getAnnIds(imgIds=entry['id'])
    dp_annotations = dp_coco.loadAnns(dp_annotation_ids)

    # iterate through all the people in one image
    for dp_annotation in dp_annotations:

        # check the validity of annotation
        is_valid, _ = DensePoseDataRelative.validate_annotation(dp_annotation)

        if not is_valid:
            continue

        # 1. keypoints
        keypoints = dp_annotation['keypoints']

        # 2. bbox
        bbox_xywh = np.array(dp_annotation["bbox"]).astype(int)

        # 3. segments of dense_pose
        if ('dp_masks' in dp_annotation.keys()):
            mask = _get_dp_mask(dp_annotation['dp_masks'])

            x1, y1, x2, y2 = bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]

            x2 = min([x2, im_gray.shape[1]])
            y2 = min([y2, im_gray.shape[0]])

            segm = cv2.resize(mask, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_NEAREST)

        # step 1. get segm_xy + keypoints dict
        segm_xy_dict, keypoints_dict = _get_dict_of_segm_and_keypoints(segm, keypoints, bbox_xywh)

        # step 2: get all the midpoints
        midpoints_dict = _get_dict_of_midpoints(segm_xy_dict, keypoints_dict)

        # step 3: get the rotation angles
        rotated_angles = _get_rotated_angles(keypoints_dict, midpoints_dict)

        # step 4: load the data of norm_segm
        if gender == 'man':
            df_norm_segm = pd.read_csv(fname_norm_segm_coco_man, index_col=0)
        elif gender == 'woman':
            df_norm_segm = pd.read_csv(fname_norm_segm_coco_woman, index_col=0)

        index_name = generate_index_name(image_id, person_index)
        dict_norm_segm = df_norm_segm.loc[index_name]

        # step 5: draw the norm_segm on the original image
        _draw_norm_segm(im_gray, midpoints_dict, rotated_angles, dict_norm_segm, False)

        # step 6: draw the specified painter's average contour on the original image
        if artist:
            df_contour = pd.read_csv(fname_contour, index_col=0)
            dict_contour = df_contour.loc[artist]
            dict_contour['scaler'] = dict_norm_segm['scaler']
            _draw_norm_segm(im_gray, midpoints_dict, rotated_angles, dict_contour, True)

        # save the final image
        if artist:
            fnorm = '{}_{}_norm_with_{}_contour.jpg'.format(image_id, gender, artist)
        else:
            fnorm = '{}_{}_norm.jpg'.format(image_id, gender)
        cv2.imwrite(os.path.join('pix', fnorm), im_gray)

        # show the final image
        image_window = 'image'
        cv2.imshow(image_window, im_gray)
        cv2.setWindowProperty(image_window, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def generate_index_name(image_id, person_index):

    index_name = '{}_{}'.format(image_id, person_index)

    return index_name


if __name__ == '__main__':

    # settings
    thickness = 2
    color = (255, 0, 255)

    parser = argparse.ArgumentParser(description='DensePose - Visualize the dilated and symmetrical segment')
    parser.add_argument('--image', help='Image ID')
    parser.add_argument('--gender', help='Gender - man or woman')
    parser.add_argument('--artist', help='Superimpose the average contour of a specified artist')
    args = parser.parse_args()

    # python visualize_rect_segm_coco.py --image 25057 --gender man
    # python visualize_rect_segm_coco.py --image 54931 --gender woman
    # python visualize_rect_segm_coco.py --image 54931 --gender woman --artist "Paul Gauguin"
    visualize(image_id=int(args.image), person_index=1, gender=args.gender, artist=args.artist)
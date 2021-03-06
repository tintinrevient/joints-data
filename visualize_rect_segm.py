import os, cv2, re
import numpy as np
import argparse
import pandas as pd
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from densepose.config import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor
from generate_rect_segm import (
    openpose_keypoints_dir, JOINT_ID, COARSE_TO_COLOR,
    _extract_segm, _segm_xy, _segm_xy_centroid, _calc_angle
)


# the path to the data of norm_segm.csv
fname_norm_segm = os.path.join('output', 'norm_segm.csv')


def _get_head_segm_centroid(infile, densepose_idx):

    image = cv2.imread(infile)

    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.MODEL.DEVICE = 'cpu'

    cfg.merge_from_file('./configs/densepose_rcnn_R_50_FPN_s1x.yaml')
    cfg.MODEL.WEIGHTS = './models/densepose_rcnn_R_50_FPN_s1x.pkl'

    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)

    # filter the probabilities of scores for each bbox > score_cutoff
    instances = outputs['instances']
    confident_detections = instances[instances.scores > 0.95]

    # extractor
    extractor = DensePoseResultExtractor()
    results_densepose, boxes_xywh = extractor(confident_detections)

    # extract the segment
    mask, segm = _extract_segm(result_densepose=results_densepose[densepose_idx])

    # get x and y of the head segment
    head_xy = _segm_xy(segm=segm, segm_id=14, box_xywh=boxes_xywh[densepose_idx])

    # get the centroid of the head segment
    head_centroid_x, head_centroid_y = _segm_xy_centroid(head_xy)

    return head_centroid_x, head_centroid_y


def _get_midpoints(infile, densepose_idx, keypoints):

    midpoints = {}

    # head centroid
    head_centroid_x, head_centroid_y = _get_head_segm_centroid(infile, densepose_idx)
    midpoints['Head'] = np.array([head_centroid_x, head_centroid_y])

    # torso midpoint
    midpoints['Torso'] = (keypoints['Neck'] + keypoints['MidHip']) / 2

    # upper limbs
    midpoints['RUpperArm'] = (keypoints['RShoulder'] + keypoints['RElbow']) / 2
    midpoints['RLowerArm'] = (keypoints['RElbow'] + keypoints['RWrist']) / 2
    midpoints['LUpperArm'] = (keypoints['LShoulder'] + keypoints['LElbow']) / 2
    midpoints['LLowerArm'] = (keypoints['LElbow'] + keypoints['LWrist']) / 2

    # lower limbs
    midpoints['RThigh'] = (keypoints['RHip'] + keypoints['RKnee']) / 2
    midpoints['RCalf'] = (keypoints['RKnee'] + keypoints['RAnkle']) / 2
    midpoints['LThigh'] = (keypoints['LHip'] + keypoints['LKnee']) / 2
    midpoints['LCalf'] = (keypoints['LKnee'] + keypoints['LAnkle']) / 2

    return midpoints


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


def _draw_norm_segm(image, midpoints, rotated_angles, dict_norm_segm):

    # scaler
    scaler = 1 / dict_norm_segm['scaler']

    # head
    rect = ((midpoints['Head'][0], midpoints['Head'][1]),
            (dict_norm_segm['Head_w'] * scaler, dict_norm_segm['Head_h'] * scaler),
            rotated_angles['Head'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['Head'], thickness=thickness)

    # torso
    rect = ((midpoints['Torso'][0], midpoints['Torso'][1]),
            (dict_norm_segm['Torso_w'] * scaler, dict_norm_segm['Torso_h'] * scaler),
            rotated_angles['Torso'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['Torso'], thickness=thickness)

    # upper limbs
    rect = ((midpoints['RUpperArm'][0], midpoints['RUpperArm'][1]),
            (dict_norm_segm['RUpperArm_w'] * scaler, dict_norm_segm['RUpperArm_h'] * scaler),
            rotated_angles['RUpperArm'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['RUpperArm'], thickness=thickness)

    rect = ((midpoints['RLowerArm'][0], midpoints['RLowerArm'][1]),
            (dict_norm_segm['RLowerArm_w'] * scaler, dict_norm_segm['RLowerArm_h'] * scaler),
            rotated_angles['RLowerArm'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['RLowerArm'], thickness=thickness)

    rect = ((midpoints['LUpperArm'][0], midpoints['LUpperArm'][1]),
            (dict_norm_segm['LUpperArm_w'] * scaler, dict_norm_segm['LUpperArm_h'] * scaler),
            rotated_angles['LUpperArm'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['LUpperArm'], thickness=thickness)

    rect = ((midpoints['LLowerArm'][0], midpoints['LLowerArm'][1]),
            (dict_norm_segm['LLowerArm_w'] * scaler, dict_norm_segm['LLowerArm_h'] * scaler),
            rotated_angles['LLowerArm'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['LLowerArm'], thickness=thickness)

    # lower limbs
    rect = ((midpoints['RThigh'][0], midpoints['RThigh'][1]),
            (dict_norm_segm['RThigh_w'] * scaler, dict_norm_segm['RThigh_h'] * scaler),
            rotated_angles['RThigh'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['RThigh'], thickness=thickness)

    rect = ((midpoints['RCalf'][0], midpoints['RCalf'][1]),
            (dict_norm_segm['RCalf_w'] * scaler, dict_norm_segm['RCalf_h'] * scaler),
            rotated_angles['RCalf'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['RCalf'], thickness=thickness)

    rect = ((midpoints['LThigh'][0], midpoints['LThigh'][1]),
            (dict_norm_segm['LThigh_w'] * scaler, dict_norm_segm['LThigh_h'] * scaler),
            rotated_angles['LThigh'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['LThigh'], thickness=thickness)

    rect = ((midpoints['LCalf'][0], midpoints['LCalf'][1]),
            (dict_norm_segm['LCalf_w'] * scaler, dict_norm_segm['LCalf_h'] * scaler),
            rotated_angles['LCalf'])
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=COARSE_TO_COLOR['LCalf'], thickness=thickness)


def visualize(infile, openpose_idx, densepose_idx):

    # step 1: load keypoints
    file_keypoints = os.path.join(openpose_keypoints_dir,
                                  '{}_keypoints.npy'.format(infile[infile.find('/') + 1:infile.rfind('.')]))
    data_keypoints = np.load(file_keypoints, allow_pickle='TRUE').item()['keypoints']

    # zip the joint ID with the data_keypoints
    # data_keypoints contains the list of the keypoints for all the people in one image
    keypoints = dict(zip(JOINT_ID, data_keypoints[openpose_idx-1]))
    print('keypoints:', keypoints)

    # step 2.1: get all the midpoints
    midpoints = _get_midpoints(infile, densepose_idx-1, keypoints)
    print('midpoints:', midpoints)

    # step 2.2: get the rotation angles
    rotated_angles = _get_rotated_angles(keypoints, midpoints)
    print('rotated_angles:', rotated_angles)

    # step 3.1: load the data of norm_segm
    df_norm_segm = pd.read_csv(fname_norm_segm, index_col=0)
    index_name = generate_index_name(infile, openpose_idx)
    dict_norm_segm = df_norm_segm.loc[index_name]
    print(dict_norm_segm)

    # step 3.2: draw the norm_segm on the original image
    # load the original image
    image = cv2.imread(infile)
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_gray = np.tile(im_gray[:, :, np.newaxis], [1, 1, 3])
    # draw norm_segm
    _draw_norm_segm(im_gray, midpoints, rotated_angles, dict_norm_segm)

    # debug
    # cv2.circle(im_gray, (int(midpoints['Torso'][0]), int(midpoints['Torso'][1])), radius=10, color=(0, 255, 0), thickness=-1)
    # cv2.circle(im_gray, (int(keypoints['Neck'][0]), int(keypoints['Neck'][1])), radius=10, color=(0, 255, 255), thickness=-1)
    # cv2.circle(im_gray, (int(keypoints['MidHip'][0]), int(keypoints['MidHip'][1])), radius=10, color=(255, 0, 255), thickness=-1)

    # save the final image
    fpath = infile.split('/')
    artist = fpath[2]
    fname = fpath[3][:fpath[3].rfind('.')]
    fnorm = '{}_{}_norm.jpg'.format(artist, fname)
    cv2.imwrite(os.path.join('pix', fnorm), im_gray)

    # show the final image
    image_window = 'image'
    cv2.imshow(image_window, im_gray)
    cv2.setWindowProperty(image_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_index_name(infile, openpose_idx):

    # each artist
    iter_list = [iter.start() for iter in re.finditer(r"/", infile)]
    artist = infile[iter_list[1] + 1:iter_list[2]]
    painting_number = infile[iter_list[2] + 1:infile.rfind('.')]

    # impressionism
    # artist = 'Impressionism'
    # painting_number = int(infile[infile.rfind('/')+1:infile.rfind('.')])

    index_name = '{}_{}_{}'.format(artist, painting_number, openpose_idx)

    return index_name


if __name__ == '__main__':

    # settings
    thickness = 2

    # modern
    # python visualize_rect_segm.py --input datasets/modern/Paul\ Delvaux/90551.jpg
    # python visualize_rect_segm.py --input datasets/modern/Paul\ Gauguin/30963.jpg

    # classical
    # python visualize_rect_segm.py --input datasets/classical/Michelangelo/12758.jpg
    # python visualize_rect_segm.py --input datasets/classical/Artemisia\ Gentileschi/45093.jpg

    parser = argparse.ArgumentParser(description='DensePose - Visualize the dilated and symmetrical segment')
    parser.add_argument('--input', help='Path to image file')
    args = parser.parse_args()

    # for a single person in one image: openpose_idx=1, densepose_idx=1
    # for multiple people in one image: openpose_idx might or might not be equal to densepose_idx
    # Hint 1: the head will match if openpose_idx is matched with densepose_idx!
    # Hint 2: Paul Delvaux_69696_3 -> 3 = openpose_idx!
    visualize(infile=args.input, openpose_idx=1, densepose_idx=1)
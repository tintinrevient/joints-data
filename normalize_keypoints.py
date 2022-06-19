import numpy as np
import os
import cv2
import argparse
from pathlib import Path
import pandas as pd


# data type
# keypoints = {key: (x, y, score)}
# pixel = (x, y)

# 1. Body 25 keypoints
JOINT_ID = [
    'Nose', 'Neck',
    'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
    'MidHip',
    'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'REye', 'LEye', 'REar', 'LEar',
    'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel',
    'Background'
]

JOINT_PAIR = [
    # ('REar', 'REye'), ('LEar', 'LEye'), ('REye', 'Nose'), ('LEye', 'Nose'),
    ('Nose', 'Neck'), ('Neck', 'MidHip'),
    ('Neck', 'RShoulder'), ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'),
    ('Neck', 'LShoulder'), ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
    ('MidHip', 'RHip'), ('MidHip', 'LHip'),
    ('RHip', 'RKnee'), ('RKnee', 'RAnkle'), ('LHip', 'LKnee'), ('LKnee', 'LAnkle')
]

JOINT_TRIPLE = [
    ('Nose', 'Neck', 'MidHip'),
    ('RShoulder','Neck', 'MidHip'), ('LShoulder', 'Neck', 'MidHip'),
    ('RElbow', 'RShoulder','Neck'), ('LElbow', 'LShoulder', 'Neck'),
    ('RWrist', 'RElbow', 'RShoulder'), ('LWrist', 'LElbow', 'LShoulder'),
    ('RHip', 'MidHip', 'Neck'), ('LHip', 'MidHip', 'Neck'),
    ('RKnee', 'RHip', 'MidHip'), ('LKnee', 'LHip', 'MidHip'),
    ('RAnkle', 'RKnee', 'RHip'), ('LAnkle', 'LKnee', 'LHip')
]

# 'zero' and 'nan' will result in errors in hierarchical clustering
minimum_positive_above_zero = np.nextafter(0, 1)


def _euclidian(point1, point2):

    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def _calc_angle(point1, center, point2):

    try:
        a = np.array(point1)[0:2] - np.array(center)[0:2]
        b = np.array(point2)[0:2] - np.array(center)[0:2]

        cos_theta = np.dot(a, b)
        sin_theta = np.cross(a, b)

        rad = np.arctan2(sin_theta, cos_theta)
        deg = np.rad2deg(rad)

        if np.isnan(rad):
            return minimum_positive_above_zero, minimum_positive_above_zero

        return rad, deg

    except:
        return minimum_positive_above_zero, minimum_positive_above_zero


def _rotate(point, center, rad):

    x = ((point[0] - center[0]) * np.cos(rad)) - ((point[1] - center[1]) * np.sin(rad)) + center[0];
    y = ((point[0] - center[0]) * np.sin(rad)) + ((point[1] - center[1]) * np.cos(rad)) + center[1];

    if len(point) == 3:
        return (int(x), int(y), point[2])  # for keypoints with score
    elif len(point) == 2:
        return (int(x), int(y))  # for pixel (x, y) without score


def is_valid(keypoints):

    # check the scores for each main keypoint, which MUST exist!
    # main_keypoints = BODY BOX
    main_keypoints = ['Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'MidHip']

    # filter the main keypoints by score = 0
    filtered_keypoints = [key for key, value in keypoints.items() if key in main_keypoints and value[2] == 0]

    if len(filtered_keypoints) != 0:
        return False
    else:
        return True


def clip_bbox(image, keypoints, dimension):
    '''
    for keypoints of one person
    '''

    min_x = dimension[1]
    max_x = 0
    min_y = dimension[0]
    max_y = 0

    for key, value in keypoints.items():
        x, y, score = value

        if score == 0.0:
            continue

        if x < min_x and x >=0:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y and y >=0:
            min_y = y
        if y > max_y:
            max_y = y

    x = int(min_x)
    y = int(min_y)
    w = int(max_x - min_x)
    h = int(max_y - min_y)

    image_bbox = image[y:y + h, x:x + w]

    return image_bbox


def calc_joint_angle(output_dict, keypoints):
    '''
    for keypoints of one person
    '''

    for index, triple in enumerate(JOINT_TRIPLE):

        point1 = keypoints.get(triple[0])
        center = keypoints.get(triple[1])
        point2 = keypoints.get(triple[2])

        col_name = '{}_{}_{}'.format(triple[0], triple[1], triple[2])

        if col_name not in output_dict:
            output_dict[col_name] = []

        if point1[2] != 0 and center[2] != 0 and point2[2] != 0:
            rad, deg = _calc_angle(point1=point1, center=center, point2=point2)
            output_dict[col_name].append(rad)
        else:
            output_dict[col_name].append(minimum_positive_above_zero)


def rotate_to_vertical_pose(keypoints):

    rotated_keypoints = {}

    reference_point = np.array(keypoints['MidHip']) + np.array((0, -100, 0))
    rad, deg = _calc_angle(point1=keypoints['Neck'], center=keypoints['MidHip'], point2=reference_point)

    rotated_keypoints = {key: _rotate(value, keypoints['MidHip'], rad) for key, value in keypoints.items()}

    return rotated_keypoints


def normalize_pose(keypoints):
    '''
    for keypoints of one person
    '''

    # default dimension + length
    width = height = 300
    center = int(width / 2)

    # white background image
    image = np.empty((width, height, 3), np.uint8)
    image.fill(255) # fill with white

    # drawing settings
    line_color = (0, 0, 255) # bgr -> red
    line_thickness = 3

    # normalized joint locations (x, y, score)
    neck_xy = (center, 100)
    midhip_xy = (center, 170)  # length of body = 70 -> (170 - 100) = midhip_y - neck_y
    upper_xy = (center, 70)  # length of upper limbs incl. neck! = 30 -> (100 - 70) = neck_y - upper_y
    lower_xy = (center, 140)  # length of lower limbs = 30 -> (170 - 140) = midhip_y - lower_y

    # Neck to MidHip as base
    cv2.line(image, neck_xy, midhip_xy, color=line_color, thickness=line_thickness)

    # Neck to Nose
    # reference virtual line to locate the nose!!! = a line from neck_xy to upper_xy
    nose_ref = np.array([0, -30, 0])

    if keypoints.get('Nose')[2] != 0:
        rad, deg = _calc_angle(np.array(keypoints.get('Neck')) + nose_ref, keypoints.get('Neck'), keypoints.get('Nose'))
        nose_xy = _rotate(upper_xy, neck_xy, rad)
        cv2.line(image, neck_xy, nose_xy, color=line_color, thickness=line_thickness)

    # RIGHT
    # Neck to RShoulder
    rad, deg = _calc_angle(np.array(keypoints.get('Neck')) + nose_ref, keypoints.get('Neck'), keypoints.get('RShoulder'))
    rsho_xy = _rotate(upper_xy, neck_xy, rad)
    cv2.line(image, neck_xy, rsho_xy, color=line_color, thickness=line_thickness)

    # RShoulder to RElbow
    if keypoints.get('RElbow')[2] != 0:
        rad, deg = _calc_angle(keypoints.get('Neck'), keypoints.get('RShoulder'), keypoints.get('RElbow'))
        relb_xy = _rotate(neck_xy, rsho_xy, rad)
        cv2.line(image, rsho_xy, relb_xy, color=line_color, thickness=line_thickness)

    # RElbow to RWrist
    if keypoints.get('RElbow')[2] != 0 and keypoints.get('RWrist')[2] != 0:
        rad, deg = _calc_angle(keypoints.get('RShoulder'), keypoints.get('RElbow'), keypoints.get('RWrist'))
        rwrist_xy = _rotate(rsho_xy, relb_xy, rad)
        cv2.line(image, relb_xy, rwrist_xy, color=line_color, thickness=line_thickness)

    # MidHip to RHip
    rad, deg = _calc_angle(keypoints.get('Neck'), keypoints.get('MidHip'), keypoints.get('RHip'))
    rhip_xy = _rotate(lower_xy, midhip_xy, rad)
    cv2.line(image, midhip_xy, rhip_xy, color=line_color, thickness=line_thickness)

    # RHip to RKnee
    if keypoints.get('RKnee')[2] != 0:
        rad, deg = _calc_angle(keypoints.get('MidHip'), keypoints.get('RHip'), keypoints.get('RKnee'))
        rknee_xy = _rotate(midhip_xy, rhip_xy, rad)
        cv2.line(image, rhip_xy, rknee_xy, color=line_color, thickness=line_thickness)

    # RKnee to RAnkle
    if keypoints.get('RKnee')[2] != 0 and keypoints.get('RAnkle')[2] != 0:
        rad, deg = _calc_angle(keypoints.get('RHip'), keypoints.get('RKnee'), keypoints.get('RAnkle'))
        rankle_xy = _rotate(rhip_xy, rknee_xy, rad)
        cv2.line(image, rknee_xy, rankle_xy, color=line_color, thickness=line_thickness)

    # LEFT
    # Neck to LShoulder
    rad, deg = _calc_angle(np.array(keypoints.get('Neck')) + nose_ref, keypoints.get('Neck'), keypoints.get('LShoulder'))
    lsho_xy = _rotate(upper_xy, neck_xy, rad)
    cv2.line(image, neck_xy, lsho_xy, color=line_color, thickness=line_thickness)

    # LShoulder to LElbow
    if keypoints.get('LElbow')[2] != 0:
        rad, deg = _calc_angle(keypoints.get('Neck'), keypoints.get('LShoulder'), keypoints.get('LElbow'))
        lelb_xy = _rotate(neck_xy, lsho_xy, rad)
        cv2.line(image, lsho_xy, lelb_xy, color=line_color, thickness=line_thickness)

    # LElbow to LWrist
    if keypoints.get('LElbow')[2] != 0 and keypoints.get('LWrist')[2] != 0:
        rad, deg = _calc_angle(keypoints.get('LShoulder'), keypoints.get('LElbow'), keypoints.get('LWrist'))
        lwrist_xy = _rotate(lsho_xy, lelb_xy, rad)
        cv2.line(image, lelb_xy, lwrist_xy, color=line_color, thickness=line_thickness)

    # MidHip to LHip
    rad, deg = _calc_angle(keypoints.get('Neck'), keypoints.get('MidHip'), keypoints.get('LHip'))
    lhip_xy = _rotate(lower_xy, midhip_xy, rad)
    cv2.line(image, midhip_xy, lhip_xy, color=line_color, thickness=line_thickness)

    # LHip to LKnee
    if keypoints.get('LKnee')[2] != 0:
        rad, deg = _calc_angle(keypoints.get('MidHip'), keypoints.get('LHip'), keypoints.get('LKnee'))
        lknee_xy = _rotate(midhip_xy, lhip_xy, rad)
        cv2.line(image, lhip_xy, lknee_xy, color=line_color, thickness=line_thickness)

    # LKnee to LAnkle
    if keypoints.get('LKnee')[2] != 0 and keypoints.get('LAnkle')[2] != 0:
        rad, deg = _calc_angle(keypoints.get('LHip'), keypoints.get('LKnee'), keypoints.get('LAnkle'))
        lankle_xy = _rotate(lhip_xy, lknee_xy, rad)
        cv2.line(image, lknee_xy, lankle_xy, color=line_color, thickness=line_thickness)

    return image


def process_keypoints(infile, output_dict={}, output_index=[]):

    data = np.load(infile, allow_pickle='TRUE').item()

    print('input:', infile)
    print('number of people:', data['keypoints'].shape[0])

    person_index = 0
    index_fname = '{}_{}'.format(infile.split('/')[3], infile[infile.rfind('/')+1:infile.rfind('_')])
    artist = infile.split('/')[2]
    image_fname = infile.replace('/data/', '/pix/').replace('_keypoints.npy', '_rendered.png')
    image = cv2.imread(image_fname)

    # iterate through all people
    for keypoints in data['keypoints']:

        # process one person!!!

        # create keypoints dictionary
        keypoints = dict(zip(JOINT_ID, keypoints))

        # if not valid, skip to the next person
        if not is_valid(keypoints=keypoints):
            continue

        # generate person index
        person_index += 1
        output_index.append('{}_{}_{}'.format(artist, index_fname, person_index))

        #################################
        # Output 1 - Angles of 3 Joints #
        #################################
        # To generate the dendrogram!!!
        calc_joint_angle(output_dict, keypoints)

        ###########################
        # Output 2 - Bounding Box #
        ###########################
        image_bbox = clip_bbox(image, keypoints, data['dimension'])

        person_fname = image_fname.replace('_rendered', '_' + str(person_index))
        cv2.imwrite(person_fname, image_bbox)

        #############################
        # Output 3 - Normalize pose #
        #############################
        # rotation: transform any poses to neck-midhip-straight poses, i.e., stand up, sit up, etc...
        rotated_keypoints = rotate_to_vertical_pose(keypoints=keypoints)

        # normalize the length of limbs
        image_norm = normalize_pose(keypoints=rotated_keypoints)

        # crop the image!!!
        image_norm = image_norm[50:250, 50:250]

        norm_fname = image_fname.replace('_rendered', '_norm_' + str(person_index))
        cv2.imwrite(norm_fname, image_norm)
        print('output', norm_fname)

    return output_dict, output_index


if __name__ == '__main__':

    # python normalize_keypoints.py --input keypoints/
    # python normalize_keypoints.py --input keypoints/classical/Michelangelo/
    # python normalize_keypoints.py --input keypoints/classical/Michelangelo/1304_keypoints.npy

    parser = argparse.ArgumentParser(description='Extract the angles of keypoints')
    parser.add_argument("--input", help="a directory or a single npy keypoints data")
    args = parser.parse_args()

    output_dict = {}
    output_index = []

    if os.path.isfile(args.input):
        process_keypoints(infile=args.input)

    elif os.path.isdir(args.input):
        for path in Path(args.input).rglob('*.npy'):
            output_dict, output_index = process_keypoints(infile=str(path), output_dict=output_dict, output_index=output_index)

        df = pd.DataFrame(data=output_dict, index=output_index)
        df.to_csv('output/joint_angles.csv', index=True)

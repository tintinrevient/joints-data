from PIL import Image
import cv2
import math
import numpy as np
import argparse


def _euclidian(point1, point2):
  return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


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

    except:
      return 0, 0


def vector_length(vector):
  vector_len = math.sqrt(vector[0] ** 2 + vector[1] ** 2)

  if vector_len == 0.0:
    return 0.0001

  return vector_len


def points_distance(point1, point2):
  return vector_length((point1[0] - point2[0],point1[1] - point2[1]))


def clamp(value, minimum, maximum):
  return max(min(value,maximum),minimum)


# Warp an image according to the given points and shift vectors
# @param image input image
# @param points list of (x, y, dx, dy) tuples
# @return warped image
def warp(image, points):
  print("Warping...")

  result = Image.new("RGB",image.size,"black")

  image_pixels = image.load()
  result_pixels = result.load()

  for y in range(image.size[1]):
    for x in range(image.size[0]):

      offset = [0,0]

      for point in points:
        point_position = (point[0] + point[2],point[1] + point[3])
        shift_vector = (point[2], point[3])

        helper = 1.0 / (3 * (points_distance((x,y),point_position) / vector_length(shift_vector)) ** 4 + 1)

        offset[0] -= helper * shift_vector[0]
        offset[1] -= helper * shift_vector[1]

      coords = (clamp(x + int(offset[0]),0,image.size[0] - 1),clamp(y + int(offset[1]),0,image.size[1] - 1))

      result_pixels[x,y] = image_pixels[coords[0],coords[1]]

  return result


def get_points(image_id, artist, image_path):

  image = cv2.imread(image_path)

  src_boxes = np.load('warp/{}_on_{}_src_boxes.npy'.format(image_id, artist), allow_pickle='TRUE').item()
  dst_boxes = np.load('warp/{}_on_{}_dst_boxes.npy'.format(image_id, artist), allow_pickle='TRUE').item()

  points = []
  segments = ['Head', 'Torso', 'RUpperArm', 'RLowerArm', 'LUpperArm', 'LLowerArm', 'RThigh', 'RCalf', 'LThigh', 'LCalf']

  # head
  for segment in segments:
    print("Processing", segment)

    for point in zip(src_boxes[segment], dst_boxes[segment]):
      # np.array -> list -> unpack list with *
      # the tuple is (src_x, src_y, dx, dy)
      points.append((*list(point[0]), *list(point[1] - point[0])))

      cv2.circle(image, tuple(point[0]), radius=3, color=src_color, thickness=-1)
      cv2.circle(image, tuple(point[1]), radius=3, color=dst_color, thickness=-1)

  # save the warpingn points
  fname = image_path.split('.')[0]
  suffix = image_path.split('.')[1]
  cv2.imwrite(fname + '_' + args.artist + '_points.' + suffix, image)

  # show the warping points
  cv2.imshow('debug', image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  return points


if __name__ == '__main__':

  # settings
  src_color = (255, 0, 255)
  dst_color = (0, 255, 0)

  # image path
  image_path = 'warp/COCO_val2014_000000054931.jpg'

  # python warp.py --artist "Amedeo Modigliani"

  parser = argparse.ArgumentParser(description='DensePose - Visualize the dilated and symmetrical segment')
  parser.add_argument('--artist', help='Superimpose the average contour of a specified artist')
  args = parser.parse_args()

  # step 1 - get points to guide warping
  points = get_points(54931, args.artist, image_path)

  # step 2 - warp the image
  image = Image.open(image_path)
  image = warp(image, points)

  # save the warped image
  fname = image_path.split('.')[0]
  image.save(fname + '_' + args.artist + '_warped.png', "PNG")
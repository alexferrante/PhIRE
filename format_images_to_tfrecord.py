import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image

HR_DATA_DIR = 'data_processing_util/prepared_data/tmp/HR'
MR_DATA_DIR = 'data_processing_util/prepared_data/tmp/MR'
LR_DATA_DIR = 'data_processing_util/prepared_data/tmp/LR'

BORDER_REGION_IDS = ['1', '2', '3', '4', '5', '6', '7', '9', '10', '11', '13', '15', '16', '18', \
                    '19', '20', '22', '23', '25', '26', '28', '30', '31', '33', '34', '36', '37', \
                    '39', '40', '41', '43', '45', '46', '48', '49', '50', '52', '53', '55', '61', \
                    '64', '65', '67', '69', '73', '76', '79', '82', '84', '88', '90', '91', '93', \
                    '96', '103', '107', '111', '114', '117', '121', '123', '126', '127', '128', \
                    '131', '132', '133', '134', '135', '136', '137', '139', '140', '143', '145', \
                    '147', '149', '151', '153', '155', '156', '157', '158', '159', '160', '162', \
                    '163', '164', '165', '167', '168', '170', '171', '172', '173', '174', '175', \
                    '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', \
                    '187', '188', '189', '190', '191', '192', '193', '194', '195']

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def remove_transparency(filename, bg_colour=(255, 255, 255)):
    im = Image.open(filename)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        alpha = im.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        bg = bg.convert("RGB")
        bg.save(filename)
    else:
        return im

def get_image_ndarrays(resolution_step_id, selected_dates, single_channel=True):
  if resolution_step_id == 'LR-MR':
    HR_HT, HR_WD, C = 64, 64, 1 #3
    LR_HT, LR_WD = 32, 32
    hr_label = 'county'
    hr_dir = MR_DATA_DIR
    lr_label = 'state'
    lr_dir = LR_DATA_DIR
  elif resolution_step_id == 'MR-HR':
    HR_HT, HR_WD, C = 128, 128, 1 #3
    LR_HT, LR_WD = 64, 64
    hr_label = 'ct'
    hr_dir = HR_DATA_DIR
    lr_label = 'county'
    lr_dir = MR_DATA_DIR
  if selected_dates is not None:
    hr_imgs = []
    lr_imgs = []
    for date in selected_dates:
      matching_hr = glob.glob(f'{hr_dir}/*{date}_{hr_label}.png')
      hr_imgs.extend(matching_hr)
      matching_lr = glob.glob(f'{lr_dir}/*{date}_{lr_label}.png')
      lr_imgs.extend(matching_lr)
    hr_imgs = sorted(hr_imgs)
    lr_imgs = sorted(lr_imgs)
  else:
    hr_imgs = sorted(glob.glob(f'{hr_dir}/*{hr_label}.png'))
    lr_imgs = sorted(glob.glob(f'{lr_dir}/*{lr_label}.png'))

  data_hr = np.ndarray(shape=(len(hr_imgs), HR_HT, HR_WD, C))
  np_idx = 0
  for hr_img in hr_imgs:
      # img_h = remove_transparency(hr_img)
      if single_channel:
        img_h = cv2.imread(hr_img, cv2.IMREAD_GRAYSCALE)
      else:
        img_h = Image.open(hr_img)
      img_h = np.expand_dims(img_h, axis=-1)
      data_hr[np_idx] = img_h
      np_idx += 1

  data_lr = np.ndarray(shape=(len(lr_imgs), LR_HT, LR_WD, C))
  np_idx = 0
  for lr_img in lr_imgs:
      # img_l = remove_transparency(lr_img)
      if single_channel:
        img_l = cv2.imread(lr_img, cv2.IMREAD_GRAYSCALE)
      else:
        img_l = Image.open(lr_img)
      img_l = np.expand_dims(img_l, axis=-1)
      data_lr[np_idx] = img_l
      np_idx += 1

  return data_hr, data_lr, HR_HT, HR_WD, LR_HT, LR_WD, C


def convert_prepared_images_to_tfrecord(resolution_step_id, out_filename, mode='train', selected_dates=None):
  data_hr, data_lr, HR_HT, HR_WD, LR_HT, LR_WD, C = get_image_ndarrays(resolution_step_id, selected_dates)
  with tf.io.TFRecordWriter(f'data/{out_filename}.tfrecord') as writer:
    for j in range(data_hr.shape[0]):
      if mode == 'train':
        h_HR, w_HR, c = HR_HT,HR_WD, C
        h_LR, w_LR, c = LR_HT, LR_WD, C
        features = tf.train.Features(feature={
                                      'index': _int64_feature(j),
                                    'data_LR': _bytes_feature(data_lr[j,...].tostring()),
                                        'h_LR': _int64_feature(h_LR),
                                        'w_LR': _int64_feature(w_LR),
                                    'data_HR': _bytes_feature(data_hr[j,...].tostring()),
                                        'h_HR': _int64_feature(h_HR),
                                        'w_HR': _int64_feature(w_HR),
                                          'c': _int64_feature(c)})
      elif mode == 'test':
        h_LR, w_LR, c = LR_HT, LR_WD, C
        features = tf.train.Features(feature={
                                      'index': _int64_feature(j),
                                    'data_LR': _bytes_feature(data_lr[j,...].tostring()),
                                        'h_LR': _int64_feature(h_LR),
                                        'w_LR': _int64_feature(w_LR),
                                          'c': _int64_feature(c)})
      example = tf.train.Example(features=features)
      writer.write(example.SerializeToString()) 

convert_prepared_images_to_tfrecord('MR-HR', 'bw_jan01_14_regions', mode='train')
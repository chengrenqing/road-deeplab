# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Removes the color map from segmentation annotations.

Removes the color map from the ground truth segmentation annotations and save
the results to output_dir.
"""
import glob
import os.path
import numpy as np

from PIL import Image

import tensorflow as tf
import PIL.ImageDraw
import json

FLAGS = tf.app.flags.FLAGS

#np.set_printoptions(threshold='nan')
tf.app.flags.DEFINE_string('original_gt_folder',
                           './pascal_voc_seg/VOCdevkit/VOC2012-ORI/SegmentationClass',
                           'Original ground truth annotations.')

tf.app.flags.DEFINE_string('segmentation_format', 'json', 'Segmentation format.')

tf.app.flags.DEFINE_string('output_dir',
                           './pascal_voc_seg/VOCdevkit/VOC2012-ORI/SegmentationClassRaw',
                           'folder to save modified ground truth annotations.')
def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = Image.fromarray(mask)
    # mask.show() 
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def trans_label(json_file):
    with open(json_file) as f:
        data = json.load(f)
        img_file = os.path.join('./pascal_voc_seg/VOCdevkit/VOC2012-ORI/JPEGImages/', data['imagePath'])
        img = np.asarray(Image.open(img_file))
        img_shape = img.shape
        shapes = data['shapes']
        class_name_to_id = {'repair':1}
        print img_shape,img_shape[:2]
        cls = np.zeros(img_shape[:2],dtype=np.uint8)
        #cls += 20
        for shape in shapes:
            polygons = shape['points']
            label = shape['label']
            cls_id = class_name_to_id[label]
            mask = polygons_to_mask(img_shape[:2], polygons)
            cls[mask] = cls_id
        pil_image = Image.fromarray(cls)
        #print cls.shape,type(cls)
        #print cls
        return cls

            
def _save_annotation(annotation, filename):
    print filename
    print type(annotation)
    pil_image = Image.fromarray(np.array(annotation).astype(dtype=np.uint8))
    with tf.gfile.Open(filename, mode='w') as f:
        pil_image.save(f, 'PNG')        
def main(unused_argv):
  # Create the output directory if not exists.
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
      tf.gfile.MakeDirs(FLAGS.output_dir)
    annotations = glob.glob(os.path.join(FLAGS.original_gt_folder,
                                       '*.' + FLAGS.segmentation_format))
    #print annotations
    for annotation in annotations:
        raw_annotation = trans_label(annotation)
        filename = os.path.basename(annotation)[:-5]
        print annotation,filename,type(raw_annotation)
        _save_annotation(raw_annotation,
                     os.path.join(
                         FLAGS.output_dir,
                         filename + '.png'))
        #break


if __name__ == '__main__':
  tf.app.run()

#this code is used to generate *.tfrecord file

import tensorflow as tf
import numpy as np
import os
from PIL import Image

os.chdir('F:\yuda\BOSS_256_8')

def gen_tfrecords():
	img_names = [('%d.pgm'%i) for i in range(0, 49999)]
	writer = tf.python_io.TFRecordWriter("train.tfrecords")
	count = 0
	for img_name in img_names:
		img = Image.open(img_name)
		#if(count <= 5) :
		#	print(np.array(img, dtype = np.int32))
		img_raw = img.tobytes()
		img_feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
			"index": tf.train.Feature(int64_list=tf.train.Int64List(value=[count]))
        }
		example = tf.train.Example(features=tf.train.Features(feature = img_feature))
		writer.write(example.SerializeToString())
		count += 1
	writer.close()

	
gen_tfrecords()

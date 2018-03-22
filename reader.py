import tensorflow as tf
import numpy as np
import os
from PIL import Image

def read_and_decode(filename): #return tensor,
	#filename = 'train.tfrecords'
	
	filename_queue = tf.train.string_input_producer([filename])
	
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	
	features = tf.parse_single_example(serialized_example,
		features = {
			'index' : tf.FixedLenFeature([], tf.int64),
			'img_raw' : tf.FixedLenFeature([], tf.string)
		}
	)
	
	img = tf.decode_raw(features['img_raw'], tf.uint8)
	img = tf.reshape(img, [256, 256, 1])
	index = tf.cast(features['index'], tf.int32)
	
	return img
	
def input_batch():
	filename = 'E:\\Workspace\\GAN\\BOSSbase_1.01'
	return tf.train.shuffle_batch(read_and_decode(filename), batch_size = 10, capacity=10000,min_after_dequeue=10)
"""testing use:
init = tf.global_variables_initializer()
	
with tf.Session() as sess:
	sess.run(init)
	img, index = read_and_decode('')
	
	img_batch, index_batch = tf.train.shuffle_batch([img, index],
                                                batch_size=200, capacity=2000,
                                                min_after_dequeue=10)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord = coord, sess=sess)
	
	dict = {}
	count = 0
	
	for i in range(50):
		img, iindex = sess.run([img_batch, index_batch])
		for j in iindex :
			if '%d'%j not in dict :
				dict['%d'%j] = True
			else :
				count +=1
	print(count)
				
		#print(img.shape)
	
	coord.request_stop()
	coord.join(threads)
"""
	
												
	
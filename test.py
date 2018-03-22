import tensorflow as tf
import numpy as np

class test(object):
	a=tf.constant(2)

with tf.Session() as sess:
	#sess.run(init)
	T = test()
	print (sess.run(T.a))

#print(res.shape)
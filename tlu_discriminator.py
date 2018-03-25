import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages
import six
from SRMfilters import *
from ops import *


class TluDiscriminator(object):
	def __init__(self, hps , mode='train'):
		self.hps=hps
		self.mode=mode
		self.bn = []
		self.bn_cnt=0
		
		self.step=tf.contrib.framework.get_or_create_global_step()
		
		self._init=tf.global_variables_initializer()
	def _build_model(self, input_x , pro, reuse=False):
	
		with tf.variable_scope('discriminator') as scope:
			if(reuse):
				scope.reuse_variables()
			
			image_size=self.hps['image_size']
			stride_1 = self._stride_array(1)
			stride_2 = self._stride_array(2)
			btnk_units = 4
			
			with tf.variable_scope('d_init'):#5x5x1 -> 5x5x30
				x=input_x
				x=self._dis_initconv('d_init_conv', x)
				#x=tf.nn.sigmoid(x)
				#x+=pro
				#x=self._batch_norm('d_init_bn', x)
				#x=self._relu('d_init_relu', x)
				
			with tf.variable_scope('d_res0'):
				x=self._res_unit(x, 30, 32, stride_1,5)
			with tf.variable_scope('d_res1'):
				x=self._res_unit(x, 32, 32, stride_1,5)#,channels 30-> 32
			
			with tf.variable_scope('d_res2'):
				x=self._res_unit_bottleneck(x, 32, 64, stride_1, 5)#128
			with tf.variable_scope('d_res3'):				
				x=self._res_unit(x, 64, 64, stride_1,5)
				
			with tf.variable_scope('d_res4'):
				x=self._res_unit_bottleneck(x, 64, 128, stride_1, 5)#64

			with tf.variable_scope('d_res6'):
				x=self._res_unit_bottleneck(x, 128, 256, stride_1, 5)#32

			with tf.variable_scope('d_res8'):
				x=self._res_unit_bottleneck(x, 256, 512, stride_1, 5)#16
	
			with tf.variable_scope('d_res10'):
				x=self._res_unit_bottleneck(x, 512, 1024, stride_1, 5)#8	
				
			with tf.variable_scope('d_fc'):
				logits=self._fully_connected(x, 1)
				predictions = tf.nn.sigmoid(logits)
				return logits, predictions

		
	def _fully_connected(self, x, out_dim):
		
		x=tf.reshape(x, shape=[-1, int(x.get_shape()[1]*x.get_shape()[2]*x.get_shape()[3])])#self.images.get_shape()[0]

		w = tf.get_variable('DW', [x.get_shape()[1], out_dim],
						initializer=tf.random_normal_initializer(stddev=0.01))
		# 参数b，0值初始化
		b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())		
		return tf.nn.xw_plus_b(x, w, b)

		
	def _res_unit(self, x, in_filter, out_filter, stride,filter_size=3):
	
		bn1=batch_norm(name='bn1')
		bn2=batch_norm(name='bn2')
		bn3=batch_norm(name='bn3')

		orig_x = x
		with tf.variable_scope('sub1'):
			x=self._conv('conv', x, filter_size, in_filter, out_filter, stride)
			
			x=bn1(x, train=self.hps['mode'])
			
			x=self._relu('relu', x)	
		with tf.variable_scope('sub2'):
			x=self._conv('conv', x, filter_size, out_filter, out_filter, stride)
			
			x=bn2(x, train=self.hps['mode'])

		
		if(in_filter!=out_filter):
			with tf.variable_scope('add'):
				orig_x=self._conv('conv', orig_x, 1, 30, 32, stride)

				orig_x=bn3(orig_x, train=self.hps['mode'])
				
		x+=orig_x
		x=self._relu('relu', x)
		return x

	def _res_unit_bottleneck(self, x, in_filter, out_filter, stride, filter_size=3):
		rd_stride=self._stride_array(2)#reduce the size of input,havle it
		btnk_dim=int(out_filter/4)
		orig_x = x
		
		bn1=batch_norm(name='bn1')
		bn2=batch_norm(name='bn2')
		bn3=batch_norm(name='bn3')
		bn4=batch_norm(name='bn4')
		
		with tf.variable_scope('bottlenect_sub1'):
			x=self._conv('rd_conv', x, 1, in_filter, btnk_dim, stride)
			x=bn1(x, train=self.hps['mode'])
			x=self._relu('rd_relu', x)	
			
			x=self._conv('conv', x, filter_size, btnk_dim, btnk_dim, rd_stride)
			x=bn2(x, train=self.hps['mode'])
			x=self._relu('relu', x)		

			x=self._conv('rcv_conv', x, 1, btnk_dim, out_filter, stride)
			x=bn3(x, train=self.hps['mode'])
			

		with tf.variable_scope('bottlenect_add'):
			orig_x=tf.nn.avg_pool(orig_x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
			orig_x = self._conv('add_conv', orig_x, 1, in_filter, out_filter, stride)
			
			orig_x=bn4(orig_x, train=self.hps['mode'])
			
		x+=orig_x
		x=self._relu('relu', x)
		return x
	
		
	def _stride_array(self, stride):
		return [1, stride, stride, 1]
		
	def _relu(slef, name, x):
		with tf.variable_scope(name) as scope:
			return tf.nn.relu(x)

	def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
		with tf.variable_scope(name):
			n = filter_size * filter_size * out_filters
			# 获取或新建卷积核，正态随机初始化
			kernel = tf.get_variable(
				'DW', 
				[filter_size, filter_size, in_filters, out_filters],
				tf.float32, 
				initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
			# 计算卷积
			return tf.nn.conv2d(x, kernel, strides, padding='SAME')
			
	def _dis_initconv(self, name, x):
		with tf.variable_scope(name):
			n = 5*5*30
			SRMfilters = SRM()
			kernel = tf.get_variable('DW',initializer=SRMfilters.get_filters(),trainable=False)
			#kernel = tf.get_variable(
			#	'DW', 
			#	[5, 5, 1, 30],
			#	tf.float32, 
			#	initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
			return tf.nn.conv2d(x, kernel, self._stride_array(1), padding='SAME')


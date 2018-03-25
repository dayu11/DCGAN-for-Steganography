#this code is used to train a network which can simulate +-1 embedding

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]= '-1'

def fc(input, output_size, name, stddev = 1.0,reuse=False):
	with tf.variable_scope(name) as scope:
		w_size = [int(input.shape[-1]), output_size]
		b_size = [1, output_size]
		weights = tf.get_variable(name='t_weights', shape=w_size, initializer=tf.random_normal_initializer(mean=0, stddev=stddev))
		bias = tf.get_variable(name='t_bias', shape=b_size, initializer=tf.random_normal_initializer(mean=0, stddev=stddev))
		return tf.matmul(input, weights) + bias
	
def nn(x, num_nerous):
	num_nerous = 10	
	
	#第一个子网络，共四层，激活函数均使用sigmoid
	n1_out1 = tf.nn.sigmoid(fc(x, num_nerous, 'n1_layer1'), name = 'n1_out1')
	n1_out1 -= 0.5

	n1_out2 = tf.nn.sigmoid(fc(n1_out1, num_nerous, 'n1_layer2'), name = 'n1_out2')
	n1_out2 -= 0.5
	
	n1_out3 = tf.nn.sigmoid(fc(n1_out2, 1, 'n1_layer3'), name = 'n1_out3')

	#第二个子网络，共四层，激活函数均使用sigmoid
	n2_out1 = tf.nn.sigmoid(fc(x, num_nerous, 'n2_layer1'), name = 'n2_out1')
	n2_out1 -= 0.5
	
	n2_out2 = tf.nn.sigmoid(fc(n2_out1, num_nerous, 'n2_layer2'), name = 'n2_out2')
	n2_out2 -= 0.5
	
	n2_out3 = tf.nn.sigmoid(fc(n2_out2, 1, 'n2_layer3'), name = 'n2_out3')
	n2_out3 -= 1

	return n1_out3 + n2_out3
	
def get_y(x):
	y=np.zeros([x.shape[0],1])
	len=x.shape[0]
	for i in range(len):
		if(x[i,1]<x[i,0]/2):
			y[i]=-1.0
		elif(x[i,1]>1-x[i,0]/2):
			y[i]=1.0
	return y

def tes_train(input=None, mode='train'):
	x=tf.placeholder(tf.float32, shape=[None, 2], name = 'input')#输入
	y=tf.placeholder(tf.float32, shape=[None, 1], name = 'ground_truth')#输入
		
	#最终预测将两个子网络的输出加起来
	pred_y = nn(x, 10)
	if(mode == 'test'):	
		saver = tf.train.Saver()
		init = tf.global_variables_initializer()
		
		with tf.Session() as sess:
			sess.run(init)
			saver.restore(sess, os.getcwd() + '\\tes_ckpt\\tes.ckpt')
			
			x_trn = input#np.random.uniform(size=[1000,2])
			y_trn =get_y(x_trn)
			
			pred=sess.run(pred_y, feed_dict={x:x_trn, y:y_trn})
			print(pred)
		return 
			
	#构建损失
	loss = tf.reduce_mean(tf.square(pred_y - y))
	
	#优化器
	optimizer = tf.train.GradientDescentOptimizer(0.01)
	train = optimizer.minimize(loss)	
		
	saver = tf.train.Saver()
		
	init = tf.global_variables_initializer()
	if(mode == 'train'):
		with tf.Session() as sess:
			sess.run(init)
			saver.restore(sess, os.getcwd() + '\\tes_ckpt\\tes.ckpt')
			Loss = 0
			for step in range(6000000):
				x_trn = np.random.uniform(size=[1000,2])
				#计算真实值，:,0 is p :,1 is rand
				#0.1 0.9
				y_trn=get_y(x_trn)
				"""
				y_neg1 = np.sign(x_trn[:,1]-x_trn[:,0]/2)  -1
				y_pos1 = np.sign(x_trn[:,1]-(1-x_trn[:,0]/2)) 
				y_trn = y_pos1 + y_neg1
				y_trn/=2
				y_trn = y_trn[:,None]
				"""
				LOSS,_,pred = sess.run([loss, train, pred_y], feed_dict = {x:x_trn, y:y_trn})#每次迭代的数据都是全部随机的[1000,2]
				Loss += LOSS
				if(step % 1000 == 0):
					print('step%d'%step, Loss/1000)
					Loss = 0
			save_path = saver.save(sess, os.getcwd() + '\\tes_ckpt\\tes.ckpt')
	else :
		with tf.Session() as sess:
			sess.run(init)
			saver.restore(sess, os.getcwd() + '\\tes_ckpt\\tes.ckpt')
			
			dict={}
			for i in range(1,3):
				for j in range(1,4):
					with tf.variable_scope('n%d_layer%d'%(i,j),reuse=True):
						weights = tf.get_variable(name='t_weights')
						bias = tf.get_variable(name='t_bias')
						dict['n%d_layer%d'%(i,j)] = {}
						dict['n%d_layer%d'%(i,j)]['w'] = (sess.run(weights)).tolist()
						dict['n%d_layer%d'%(i,j)]['b'] = (sess.run(bias)).tolist()
						
			f=open('tes_ckpt/weights.txt', 'w')
			f.write(str(dict))
			f.close()
			print('weight saved')
			#with tf.variable_scope('n1_layer1',reuse=True) as scope:
			#	weights = tf.get_variable(name='t_weights')
			#	print(sess.run(weights))
			#print(sess.run(pred_y, feed_dict={x:input}))
			

#tes_train(None, '1')

test = np.random.uniform(size=[100,2])
tes_train(test,'s')
"""
		
y=get_y(test)
print(y, np.sum(y))
"""	

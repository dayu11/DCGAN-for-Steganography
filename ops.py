import numpy as np
import tensorflow as tf
from PIL import Image
import scipy.misc
import os

def saveImg(array, name):
	scipy.misc.imsave(name + '.pgm', array)
	
def norm(a):
	min=257
	max=-1
	for i in range(256):
		for j in range(256):
			if(a[i][j]<min):
				min=a[i][j]
			if(a[i][j]>max):
				max=a[i][j]
	
	for i in range(256):
		for j in range(256):
			a[i][j]=(a[i][j]-min)/max
			a[i][j]= a[i][j] * 256
	return a

def cross_entropy(x, y):
	return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
	
def get_sparsity(pro):
	p=0.0001
	p_=tf.reduce_mean(pro, axis=0)
	spar=p*tf.log(p/(p_+1e-10)) + (1-p)*tf.log((1-p)/(1-p_))
	return tf.reduce_mean(spar)

def get_label(type):
	if(type):
		return np.array([[1], [1], [1], [1], [1],[1], [1], [1], [1], [1]], dtype=np.float)
	else:
		return np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]], dtype=np.float)

def get_test_data(path):
	os.chdir(path)
	img_names=[('%d.pgm'%i) for i in range(50000, 56000, 3)]
	list=[]
	for img_name in img_names:
		img=Image.open(img_name)
		a=np.array(img)
		list.append(a.tolist())
	return np.array(list, dtype=np.float)
def get_trn_data(path):
	tmp=os.getcwd()
	os.chdir(path)
	img_names=[('%d.pgm'%i) for i in range(50000)]
	list=[]
	cnt = 0
	for img_name in img_names:
		cnt+=1
		if(cnt%5000==0):
			print(cnt)
		img=Image.open(img_name)
		a=np.array(img)
		list.append(a.tolist())
	ary=np.array(list, dtype=np.float)
	ary=ary.reshape([-1,256,256,1])
	os.chdir(tmp)
	return ary
class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)
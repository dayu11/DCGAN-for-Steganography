#0:2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]= '3'
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages
import six
from reader import *
from ops import *
from tlu_discriminator import *
from wrf_generator import *

hps={} #Hyperparameters
hps['image_size']=256
hps['g_alph']=10**7
hps['g_beta']=1
hps['capacity']=0.1
hps['g_lrn_rate']=2e-3
hps['d_lrn_rate']=2e-3
hps['step_num']=5000
hps['batch_size']=10
hps['dis_extra_step']=0
hps['data_path']='F:\\yuda\\record\\train.tfrecords'
hps['mode']=True#true for training
hps['outer_iter']=50


data_path='F:\\yuda\\record\\train.tfrecords'
raw_image= read_and_decode(data_path)#from tfrecord file
img_batch = tf.train.shuffle_batch([raw_image],
				batch_size=hps['batch_size'], capacity=10000,
				min_after_dequeue=10)
coord = tf.train.Coordinator()
img_batch=tf.cast(img_batch, dtype=tf.float32)

Dis=TluDiscriminator(hps)#create discriminator/generator object
Gen=WrfGenerator(hps)#generator's model is built in its constructor
	
#start to build the discriminator model
real_logits,real_pred =Dis._build_model(Gen.images, Gen.pro)
real_label=tf.placeholder(tf.float32, shape=[hps['batch_size'], 1])

fake_logits,fake_pred =Dis._build_model(Gen.stego, Gen.pro, True)
fake_label=tf.placeholder(tf.float32, shape=[hps['batch_size'], 1])

#build loss terms
dis_loss_real=tf.reduce_mean(cross_entropy(real_logits, real_label))
dis_loss_fake=tf.reduce_mean(cross_entropy(fake_logits, fake_label))
dis_loss=dis_loss_real+dis_loss_fake

gen_loss_fake=tf.reduce_mean(cross_entropy(fake_logits, real_label))#real/fake loss
gen_loss_p2=tf.reduce_mean(tf.square(Gen.capacity-tf.constant(hps['image_size']*hps['image_size']*hps['capacity'])))#capacity loss

gen_loss_p1=hps['g_alph']*gen_loss_fake+gen_loss_p2#+1e6*gen_loss_p3

t_vars = tf.trainable_variables()

d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

#build optimizer
dis_optimizer = tf.train.AdamOptimizer(hps['d_lrn_rate'], beta1=0.5)
min_dis_loss= dis_optimizer.minimize(dis_loss, var_list = d_vars)
dis_train_op=[min_dis_loss]
	
gen_optimizer_1 = tf.train.AdamOptimizer(hps['g_lrn_rate'], beta1=0.5)
min_gen_loss_1= gen_optimizer_1.minimize(gen_loss_p1, var_list = g_vars, global_step=Gen.step)

gen_train_op=[min_gen_loss_1]
saver= tf.train.Saver()
init = tf.global_variables_initializer()
try:
	os.mkdir(os.getcwd()+'\\result')
except:
	useless=0

#control the usage of GPU memory
sess_config=tf.ConfigProto()
sess_config.gpu_options.per_process_gpu_memory_fraction=0.45
with tf.Session(config=sess_config) as sess:
	
	if(hps['mode']==True):
		threads = tf.train.start_queue_runners(coord = coord, sess=sess)
	
	sess.run([init, Dis._init, Gen._init])
	try:
		saver.restore(sess, os.getcwd() + '\\gan_ckpt\\gan.ckpt')
	except:
		useless=0
	real_truth=get_label(True)
	fake_truth=get_label(False)
	if(hps['mode']==True):
		#training process
		for i in range(hps['outer_iter']):
			for j in range(hps['step_num']):
				step=sess.run(Gen.step)
				data=sess.run(img_batch)
				
				rand=np.random.uniform(0.0, 1.0, Gen.rand_shape)

				for i in range(hps['dis_extra_step']):	
					sess.run(dis_train_op, feed_dict={Gen.images:data, Gen.rand:rand, real_label:real_truth, fake_label:fake_truth})
				#run discriminator train_op
				fake, d_loss,_ = sess.run([fake_pred, dis_loss, dis_train_op], feed_dict={Gen.images:data, Gen.rand:rand, real_label:real_truth, fake_label:fake_truth})

				if(step!=0 and step%1000 == 0):
					#save ckpt
					pro, l1,l2,_ = sess.run([Gen.pro, gen_loss_fake, gen_loss_p2, gen_train_op], feed_dict={Gen.images:data, Gen.rand:rand, real_label:real_truth})
					save_path = saver.save(sess, os.getcwd() + '\\gan_ckpt\\gan.ckpt')
					saver.save(sess, os.getcwd() + '\\gan_ckpt_all\\gan.ckpt%d'%step)
				else :
					#run generator train_op
					l1,l2,_ = sess.run([gen_loss_fake, gen_loss_p2, gen_train_op], feed_dict={Gen.images:data, Gen.rand:rand, real_label:real_truth})
					print('%d step: '%step, d_loss, ' ', l1, l2)

	else :
		#testing process
		test_data=get_test_data('F:\\yuda\\BOSS_256_8')
		test_data=np.reshape(test_data, [int(test_data.shape[0]), 256, 256, 1])
		count=0
		final_cap=0.0
		for i in range(200):
			rand=np.random.uniform(0.0, 1.0, Gen.rand_shape)
			images, pros, cap= sess.run([Gen.images, Gen.pro, Gen.capacity],feed_dict={Gen.images:test_data[10*i:10*i+10], Gen.rand:rand})
			cap=1.0*cap/hps['image_size']/hps['image_size']
			cap=np.mean(cap)
			final_cap+=cap
			for j in range(hps['batch_size']):
				a=np.reshape(pros[j], [hps['image_size'],hps['image_size']])
				b=np.reshape(images[j], [hps['image_size'],hps['image_size']])
				np.save('F:\\yuda\\12_GAN\\pros'+'\\'+'%d.npy'%count, a)
				#saveImg(a,'F:\\yuda\\7_GAN\\pros'+'\\%d'%count)
				saveImg(b,'F:\\yuda\\12_GAN\\covers'+'\\%d'%count)
				count+=1
			print('process %d cap:%r'%(i,cap))
		print('final_cap:%r'%(final_cap/200))
	if(hps['mode']=='train'):
		coord.request_stop()
		coord.join(threads)
	

	

	
	

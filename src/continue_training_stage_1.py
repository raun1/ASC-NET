
import keras
from keras import optimizers
#from keras.utils import multi_gpu_model
import scipy as sp
import scipy.misc, scipy.ndimage.interpolation
from medpy import metric
import numpy as np
import os
from keras import losses
import tensorflow as tf
from keras.models import Model,Sequential
from keras.layers import Input,merge, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D,Dropout,Conv2DTranspose,add,multiply,Dense,Flatten
from keras.layers.normalization import BatchNormalization as bn
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from keras import regularizers 
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
#from keras.applications import Xception
from keras.utils import multi_gpu_model
import random
import numpy as np 
from keras.callbacks import EarlyStopping, ModelCheckpoint
import nibabel as nib
import cv2
CUDA_VISIBLE_DEVICES = [0,1,2,3]
os.environ['CUDA_VISIBLE_DEVICES']=','.join([str(x) for x in CUDA_VISIBLE_DEVICES])
smooth=1.
input_shape=160,160,1
########################################Losses#################################################
def special_loss_disjoint(y_true,y_pred):
	
	y_true,y_pred=tf.split(y_pred, 2,axis=-1)
	
	thresholded_pred = tf.where( tf.greater( 0.0000000000000001, y_pred ), 1 * tf.ones_like( y_pred ), y_pred )#where(cond : take true values : take false values)
	
	thresholded_true=tf.where( tf.greater( 0.0000000000000001, y_true ), 1 * tf.ones_like( y_true ), y_true )
	
	return dice_coef(thresholded_true,thresholded_pred)

def dice_coef(y_true, y_pred):

	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
	return dice_coef(y_true, y_pred)

################################################################################################



def build_discriminator(input_shape,learn_rate=1e-3):
	l2_lambda = 0.0002
	DropP = 0.3
	kernel_size=3

	inputs = Input(input_shape,name="disc_ip")

	conv0a = Conv2D( 32, (kernel_size, kernel_size), activation='relu', padding='same', 
	               kernel_regularizer=regularizers.l2(l2_lambda),name='disc_l2_conc15' )(inputs)


	conv0a = bn(name='disc_l2_bn1')(conv0a)

	conv0b = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda),name='disc_l2_conc16' )(conv0a)

	conv0b = bn(name='disc_l2_bn2')(conv0b)




	pool0 = MaxPooling2D(pool_size=(2, 2),name='disc_l2_mp1')(conv0b)

	pool0 = Dropout(DropP,name='disc_l2_d1')(pool0)






	conv2a = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda),name='disc_l2_conc17' )(pool0)

	conv2a = bn(name='disc_l2_bn3')(conv2a)

	conv2b = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda) ,name='disc_l2_conc18')(conv2a)

	conv2b = bn(name='disc_l2_bn4')(conv2b)

	pool2 = MaxPooling2D(pool_size=(2, 2),name='disc_l2_mp2')(conv2b)

	pool2 = Dropout(DropP,name='disc_l2_d2')(pool2)







	conv3a = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda),name='disc_l2_conc19' )(pool2)

	conv3a = bn(name='disc_l2_bn5')(conv3a)

	conv3b = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda) ,name='disc_l2_conc20')(conv3a)

	conv3b = bn(name='disc_l2_bn6')(conv3b)



	pool3 = MaxPooling2D(pool_size=(2, 2),name='disc_l2_mp3')(conv3b)

	pool3 = Dropout(DropP,name='disc_l2_d3')(pool3)


	conv4a = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda),name='disc_l2_conc21' )(pool3)

	conv4a = bn(name='disc_l2_bn7')(conv4a)

	conv4b = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda),name='disc_l2_conc22' )(conv4a)

	conv4b = bn(name='disc_l2_bn8')(conv4b)

	pool4 = MaxPooling2D(pool_size=(2, 2),name='disc_l2_mp4')(conv4b)

	pool4 = Dropout(DropP,name='disc_l2_d4')(pool4)





	conv5a = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda) ,name='disc_l2_conc23')(pool4)

	conv5a = bn(name='disc_l2_bn9')(conv5a)

	conv5b = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda) ,name='disc_l2_conc24')(conv5a)

	conv5b = bn(name='disc_l2_bn10')(conv5b)

	flat=Flatten()(conv5b)

	output_disc=Dense(1,activation='tanh',name='disc_output')(flat)#placeholder

	model=Model(inputs=[inputs],outputs=[output_disc])
	model.compile(loss='mae',
	        optimizer=keras.optimizers.Adam(lr=5e-5),
	        metrics=['accuracy'])
	#model.summary()
	return model

input_shape=160,160,1

from keras.models import load_model
generator=load_model('disjoint_un_sup_mse_generator.h5', custom_objects={'dice_coef_loss':dice_coef_loss,'special_loss_disjoint':special_loss_disjoint})
discriminator=load_model('disjoint_un_sup_mse_discriminator.h5', custom_objects={'dice_coef_loss':dice_coef_loss,'special_loss_disjoint':special_loss_disjoint})

for layer in discriminator.layers: layer.trainable = False
generator.compile(optimizer=keras.optimizers.Adam(lr=5e-5),loss={
                                            
                                            'new_res_1_final_opa':'mse',
                                            'x_u_net_opsp':special_loss_disjoint
                                            
                                            })

discriminator.compile(loss='mae',
        optimizer=keras.optimizers.Adam(lr=5e-5),
        metrics=['accuracy'])

final_input=generator.input



x_u_net_opsp=(generator.get_layer('x_u_net_opsp').output)
final_output_gans=discriminator(generator.get_layer('new_final_op').output)
final_output_seg=(generator.get_layer('new_xfinal_op').output)
final_output_res=(generator.get_layer('new_res_1_final_opa').output)

#final_model.add(generator)
#final_model.add(discriminator)
final_model=Model(inputs=[final_input],outputs=[final_output_gans,final_output_seg,final_output_res,x_u_net_opsp])

final_model.compile(optimizer=keras.optimizers.Adam(lr=5e-5),metrics=['mae'],loss={'model_2':'mae',
																						
																						'new_res_1_final_opa':'mse',
																						'x_u_net_opsp':special_loss_disjoint})


print("full gans")
final_model.summary()
print(final_model.input)
print(final_model.output)
print("============================================================================================================================================================")
print("generator")
generator.summary()
print(generator.input)
print(generator.output)
print("============================================================================================================================================================")

print("discriminator")
discriminator.summary()
print(discriminator.get_input_at(0))
print(discriminator.get_input_at(1))
#print(discriminator.output)
print("============================================================================================================================================================")
#print(discriminator.get_input_at(2))
#print(discriminator.input[2])
#X_train=np.ones((1,160,160,1))
#final_model.fit([X_train],[1],batch_size=1,nb_epoch=1,shuffle=False)
#print ("hi",final_model.predict([X_train],batch_size=1))


def train_disc(real_data,fake_data,true_label,ep,loss_ch):

	discriminator=build_discriminator(input_shape)
	discriminator.name='model_2'
	for layer in discriminator.layers: layer.trainable = False
	generator.compile(optimizer=keras.optimizers.Adam(lr=5e-5),loss={
	                                            
	                                            'new_res_1_final_opa':'mse',
	                                            'x_u_net_opsp':special_loss_disjoint
	                                            
	                                            })

	discriminator.compile(loss='mae',
	        optimizer=keras.optimizers.Adam(lr=5e-5),
	        metrics=['accuracy'])

	final_input=generator.input
	

	
	x_u_net_opsp=(generator.get_layer('x_u_net_opsp').output)
	final_output_gans=discriminator(generator.get_layer('new_final_op').output)
	final_output_seg=(generator.get_layer('new_xfinal_op').output)
	final_output_res=(generator.get_layer('new_res_1_final_opa').output)

	final_model=Model(inputs=[final_input],outputs=[final_output_gans,final_output_seg,final_output_res,x_u_net_opsp])

	final_model.compile(optimizer=keras.optimizers.Adam(lr=5e-5),metrics=['mae'],loss={'model_2':'mae',
																							
																							'new_res_1_final_opa':'mse',
																							'x_u_net_opsp':special_loss_disjoint})
		
	for layer in discriminator.layers: layer.trainable = True
	

	generator.compile(optimizer=keras.optimizers.Adam(lr=5e-5),loss={
                                            
                                            'new_res_1_final_opa':'mse',
                                            
                                            })

	discriminator.compile(loss='mae',
	        optimizer=keras.optimizers.Adam(lr=5e-5),
	        metrics=['accuracy'])
	multi_discriminator=multi_gpu_model(discriminator,gpus=4)
	multi_discriminator.compile(loss='mae',
        optimizer=keras.optimizers.Adam(lr=5e-5),
        metrics=['accuracy'])
	final_model.compile(optimizer=keras.optimizers.Adam(lr=5e-5),metrics=['accuracy'],loss={'model_2':'mae',
																							
																							'new_res_1_final_opa':'mse',
																							'x_u_net_opsp':special_loss_disjoint})

	discriminator.summary()
	
	
	y_train_true=-np.ones(shape=len(real_data))
	y_train_true=y_train_true#-0.1
	print(y_train_true.shape)

	



	y_train_fake=np.ones(shape=len(fake_data))
	y_train_fake=y_train_fake#-0.1
	
	real_data=(list)(real_data)
	fake_data=(list)(fake_data)
	y_train_true=(list)(y_train_true)

	y_train_fake=(list)(y_train_fake)
	merged_inputs=[real_data+fake_data]
	real_data=[]
	fake_data=[]
	merged_gt=[y_train_true+y_train_fake]
	print('hi')

	y_train_fake=[]
	y_train_true=[]
	from sklearn.utils import shuffle
	merged_inputs,merged_gt=shuffle(merged_inputs,merged_gt)

	merged_inputs=np.array(merged_inputs)
	merged_gt=np.array(merged_gt)
	merged_inputs=np.squeeze(merged_inputs,axis=(0,))
	merged_gt=np.squeeze(merged_gt,axis=(0,))

	print("training_discriminator===============================================================================")
	while(True):
		xx=(int)((raw_input)("press 1 to keep training"))
		ep=(int)((raw_input)("enter updated number of epochs"))
		if(xx!=1):
			break
		#multi_discriminator.summary()
		multi_discriminator.fit([merged_inputs],[merged_gt],batch_size=72*4,nb_epoch=ep,shuffle=True)

	return



def train_generator(true_label,ep,loss_ch):
	for layer in discriminator.layers: layer.trainable = False
	

	generator.compile(optimizer=keras.optimizers.Adam(lr=5e-5),loss={
                                            
                                            'new_res_1_final_opa':'mse',
                                            'x_u_net_opsp':special_loss_disjoint
                                            
                                            })

	discriminator.compile(loss='mae',
	        optimizer=keras.optimizers.Adam(lr=5e-5),
	        metrics=['accuracy'])
	final_model.compile(optimizer=keras.optimizers.Adam(lr=5e-5),metrics=['accuracy'],loss={'model_2':'mae',
																							
																							'new_res_1_final_opa':'mse',
																							'x_u_net_opsp':special_loss_disjoint})

	multi_final_model=multi_gpu_model(final_model,gpus=4)
	multi_final_model.compile(optimizer=keras.optimizers.Adam(lr=5e-5),metrics=['accuracy'],loss={'model_2':'mae',
																							
																							'new_res_1_final_opa':'mse',
																							'x_u_net_opsp':special_loss_disjoint})

	#discriminator.summary()
	X_train=np.load("input_for_generator.npy")
	#X_train=np.load("input_for_generator.npy")

	
	
	
	y_train=[]

	for j in range(0,len(X_train)):
		
		y_train.append(-1)
	y_train=np.array(y_train)
	#print(multi_final_model.summary())
	y_empty=np.zeros(shape=(X_train.shape))
	while(True):
		xx=(int)((raw_input)("press 1 to keep training"))
		ep=(int)((raw_input)("enter updated number of epochs"))
		if(xx!=1):
			break
		
		multi_final_model.fit([X_train],[y_train,X_train,y_empty],batch_size=16*4,nb_epoch=ep,shuffle=True)
		result=generator.predict([X_train[0:1000]],batch_size=16)
		result[1]=(result[1]-np.amin(result[1]))/((np.amax(result[1]))-(np.amin(result[1])))
		result[0]=(result[0]-np.amin(result[0]))/((np.amax(result[0]))-(np.amin(result[0])))
		result[2]=(result[2]-np.amin(result[2]))/((np.amax(result[2]))-(np.amin(result[2])))
		for i in range(0,1000):
			cv2.imwrite("outputs/norm/id1/"+str(i)+".png",(result[0][i])*255)
			cv2.imwrite("outputs/norm/id2/"+str(i)+".png",(result[1][i])*255)
			cv2.imwrite("outputs/norm/id3/"+str(i)+".png",(result[2][i])*255)
			cv2.imwrite("outputs/norm/in/"+str(i)+".png",(X_train[i])*255)
		#final_model.fit([X_train],[y_train,X_train,y_empty],batch_size=16,nb_epoch=ep,shuffle=True)
	return

while(True):
	i_p=(int)(raw_input("press 0 to train disc and 1 to train gen 2 to save models 3 to check outputs anything else to quit"))

	if(i_p==0):
		#'''
		discriminator=build_discriminator(input_shape)
		discriminator.name='model_2'
		for layer in discriminator.layers: layer.trainable = False
		generator.compile(optimizer=keras.optimizers.Adam(lr=5e-5),loss={
		                                            
		                                            'new_res_1_final_opa':'mse',
		                                            'x_u_net_opsp':special_loss_disjoint
		                                            
		                                            })

		discriminator.compile(loss='mae',
		        optimizer=keras.optimizers.Adam(lr=5e-5),
		        metrics=['accuracy'])

		#discriminator.trainable=False
		final_input=generator.input
		#final_input_1=discriminator.input
		#connect the two

		#discriminator.input=generator.get_layer('output_gen').output
		x_u_net_opsp=(generator.get_layer('x_u_net_opsp').output)
		final_output_gans=discriminator(generator.get_layer('new_final_op').output)
		final_output_seg=(generator.get_layer('new_xfinal_op').output)
		final_output_res=(generator.get_layer('new_res_1_final_opa').output)

		#final_model.add(generator)
		#final_model.add(discriminator)
		final_model=Model(inputs=[final_input],outputs=[final_output_gans,final_output_seg,final_output_res,x_u_net_opsp])

		final_model.compile(optimizer=keras.optimizers.Adam(lr=5e-5),metrics=['mae'],loss={'model_2':'mae',
																								
																								'new_res_1_final_opa':'mse',
																								'x_u_net_opsp':special_loss_disjoint})
		loss_ch=0
		#'''
		print ("training disc")

		ep=(int)(raw_input("enter number of epochs"))
		real_data=np.load("good_dic_to_train_disc.npy")
		
		
		
		X_train_tumors=np.load("input_for_generator.npy")
		
		
		
		fake_data=generator.predict([X_train_tumors])[0]
		
		print("fake_data_shape",fake_data.shape)
		

		true_label=1
	
		print((real_data.shape),(fake_data.shape),true_label,ep)
		proceed=(int)((raw_input)("proceed press 1"))
		if(proceed==1):
			train_disc(real_data,fake_data,true_label,ep,loss_ch)
		else:
			continue

	elif(i_p==1):
		print("training gen")
		loss_ch=0
		ep=(int)(raw_input("enter number of epochs"))
		true_label=1
		
		proceed=(int)((raw_input)("proceed press 1"))
		if(proceed==1):
			train_generator(true_label,ep,loss_ch)
		else:
			continue
	elif(i_p==2):
		import h5py

		final_model.save('disjoint_un_sup_mse_complete_gans.h5')
		generator.save("disjoint_un_sup_mse_generator.h5")
		discriminator.save("disjoint_un_sup_mse_discriminator.h5")

		
	elif(i_p==3):
		X_train=np.load("input_for_generator.npy")
		y_train=np.load("tumor_mask_for_generator.npy")
		result=generator.predict([X_train[0:1000]],batch_size=16)
		#result=np.array(result)
		#print (result.shape)

		print(np.amax(result[0]),np.amax(result[1]),np.amax(result[2]),np.amax(result[3]))

		print(np.amin(result[0]),np.amin(result[1]),np.amin(result[2]),np.amin(result[3]))
		




		for i in range(0,1000):
			cv2.imwrite("outputs/id1/"+str(i)+".png",(result[0][i])*255)
			cv2.imwrite("outputs/id2/"+str(i)+".png",(result[1][i])*255)
			cv2.imwrite("outputs/id3/"+str(i)+".png",(result[2][i])*255)
			
			cv2.imwrite("outputs/norm/in/"+str(i)+".png",X_train[i]*255)
			cv2.imwrite("outputs/norm/op/"+str(i)+".png",y_train[i]*255)


		result[1]=(result[1]-np.amin(result[1]))/((np.amax(result[1]))-(np.amin(result[1])))
		result[0]=(result[0]-np.amin(result[0]))/((np.amax(result[0]))-(np.amin(result[0])))
		result[2]=(result[2]-np.amin(result[2]))/((np.amax(result[2]))-(np.amin(result[2])))
		for i in range(0,1000):
			cv2.imwrite("outputs/norm/id1/"+str(i)+".png",(result[0][i])*255)
			cv2.imwrite("outputs/norm/id2/"+str(i)+".png",(result[1][i])*255)
			cv2.imwrite("outputs/norm/id3/"+str(i)+".png",(result[2][i])*255)
		
	else:
		break


















#ps aux --sort=-%mem | awk 'NR<=10{print $0}'

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
CUDA_VISIBLE_DEVICES = [1]
os.environ['CUDA_VISIBLE_DEVICES']=','.join([str(x) for x in CUDA_VISIBLE_DEVICES])
smooth=1.
def special_loss_disjoint(y_true,y_pred):

	y_true,y_pred=tf.split(y_pred, 2,axis=-1)
	thresholded_pred = tf.where( tf.greater( 0.0, y_pred ), 1 * tf.ones_like( y_pred ), y_pred )
	thresholded_true=tf.where( tf.greater( 0.0, y_true ), 1 * tf.ones_like( y_true ), y_true )
	#tf.keras.backend.print_tensor(first)
	return dice_coef(thresholded_true,thresholded_pred)

def dice_coef(y_true, y_pred):

	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
	return dice_coef(y_true, y_pred)


def build_generator(input_shape,learn_rate=1e-3):



	l2_lambda = 0.0002
	DropP = 0.3
	kernel_size=3

	inputs = Input(input_shape)








	conv0a = Conv2D( 32, (kernel_size, kernel_size), activation='relu', padding='same', 
	               kernel_regularizer=regularizers.l2(l2_lambda),name='l2_conc15' )(inputs)


	conv0a = bn(name='l2_bn1')(conv0a)

	conv0b = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda),name='l2_conc16' )(conv0a)

	conv0b = bn(name='l2_bn2')(conv0b)




	pool0 = MaxPooling2D(pool_size=(2, 2),name='l2_mp1')(conv0b)

	pool0 = Dropout(DropP,name='l2_d1')(pool0)






	conv2a = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda),name='l2_conc17' )(pool0)

	conv2a = bn(name='l2_bn3')(conv2a)

	conv2b = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda) ,name='l2_conc18')(conv2a)

	conv2b = bn(name='l2_bn4')(conv2b)

	pool2 = MaxPooling2D(pool_size=(2, 2),name='l2_mp2')(conv2b)

	pool2 = Dropout(DropP,name='l2_d2')(pool2)







	conv3a = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda),name='l2_conc19' )(pool2)

	conv3a = bn(name='l2_bn5')(conv3a)

	conv3b = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda) ,name='l2_conc20')(conv3a)

	conv3b = bn(name='l2_bn6')(conv3b)
	


	pool3 = MaxPooling2D(pool_size=(2, 2),name='l2_mp3')(conv3b)

	pool3 = Dropout(DropP,name='l2_d3')(pool3)


	conv4a = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda),name='l2_conc21' )(pool3)

	conv4a = bn(name='l2_bn7')(conv4a)

	conv4b = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda),name='l2_conc22' )(conv4a)

	conv4b = bn(name='l2_bn8')(conv4b)
	
	pool4 = MaxPooling2D(pool_size=(2, 2),name='l2_mp4')(conv4b)

	pool4 = Dropout(DropP,name='l2_d4')(pool4)





	conv5a = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda) ,name='l2_conc23')(pool4)

	conv5a = bn(name='l2_bn9')(conv5a)

	conv5b = Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda) ,name='l2_conc24')(conv5a)

	conv5b = bn(name='l2_bn10')(conv5b)

	



	up6 = concatenate([Conv2DTranspose(256,(2, 2), strides=(2, 2), padding='same',name='l2_conc25')(conv5b), (conv4b)], axis=3,name='l2_conc1')


	up6 = Dropout(DropP,name='l2_d5')(up6)

	conv6a = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda) ,name='l2_conc26')(up6)

	conv6a = bn(name='l2_bn11')(conv6a)

	conv6b = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda),name='l2_conc27' )(conv6a)

	conv6b = bn(name='l2_bn12')(conv6b)

	



	up7 = concatenate([Conv2DTranspose(128,(2, 2), strides=(2, 2), padding='same',name='l2_conc28')(conv6b),(conv3b)], axis=3,name='l2_conc2')

	up7 = Dropout(DropP,name='l2_d6')(up7)
	#add second output here

	conv7a = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda) ,name='l2_conc29')(up7)

	conv7a = bn(name='l2_bn13')(conv7a)



	conv7b = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda) ,name='l2_conc30')(conv7a)

	conv7b = bn(name='l2_bn14')(conv7b)



	


	up8 = concatenate([Conv2DTranspose(64,(2, 2), strides=(2, 2), padding='same',name='l2_conc31')(conv7b), (conv2b)], axis=3,name='l2_conc3')

	up8 = Dropout(DropP,name='l2_d7')(up8)

	conv8a = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda) ,name='l2_conc32')(up8)

	conv8a = bn(name='l2_bn15')(conv8a)


	conv8b = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda),name='l2_conc33' )(conv8a)

	conv8b = bn(name='l2_bn16')(conv8b)



	up10 = concatenate([Conv2DTranspose(32,(2, 2), strides=(2, 2), padding='same',name='l2_conc34')(conv8b),(conv0b)],axis=3,name='l2_conc4')

	conv10a = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda) ,name='l2_conc35')(up10)

	conv10a = bn(name='l2_bn17')(conv10a)



	conv10b = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda),name='l2_conc36' )(conv10a)

	conv10b = bn(name='l2_bn18')(conv10b)



	new_final_op=Conv2D(1, (1, 1), activation='sigmoid',name='new_final_op')(conv10b)



	#--------------------------------------------------------------------------------------

	
	xup6 = concatenate([Conv2DTranspose(256,(2, 2), strides=(2, 2), padding='same',name='l2_conc38')(conv5b), (conv4b)], axis=3,name='l2_conc5')



	xup6 = Dropout(DropP,name='l2_d8')(xup6)

	xconv6a = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda),name='l2_conc39' )(xup6)

	xconv6a = bn(name='l2_bn19')(xconv6a)



	xconv6b = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda),name='l2_conc40' )(xconv6a)

	xconv6b = bn(name='l2_bn20')(xconv6b)


	


	xup7 = concatenate([Conv2DTranspose(128,(2, 2), strides=(2, 2), padding='same',name='l2_conc41')(xconv6b),(conv3b)], axis=3,name='l2_conc6')#xconv6b

	xup7 = Dropout(DropP,name='l2_d9')(xup7)

	xconv7a = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda) ,name='l2_conc42')(xup7)

	xconv7a = bn(name='l2_bn21')(xconv7a)


	xconv7b = Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda) ,name='l2_conc43')(xconv7a)

	xconv7b = bn(name='l2_bn22')(xconv7b)
	

	xup8 = concatenate([Conv2DTranspose(64,(2, 2), strides=(2, 2), padding='same',name='l2_conc44')(xconv7b),(conv2b)], axis=3,name='l2_conc7')

	xup8 = Dropout(DropP,name='l2_d10')(xup8)
	#add third xoutxout here

	xconv8a = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda) ,name='l2_conc45')(xup8)

	xconv8a = bn(name='l2_bn23')(xconv8a)


	xconv8b = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda),name='l2_conc46' )(xconv8a)

	xconv8b = bn(name='l2_bn24')(xconv8b)






	xup10 = concatenate([Conv2DTranspose(32,(2, 2), strides=(2, 2), padding='same',name='l2_conc47')(xconv8b), (conv0b)],axis=3,name='l2_conc8')

	xup10 = Dropout(DropP,name='l2_d11')(xup10)


	xconv10a = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda) ,name='l2_conc48')(xup10)

	xconv10a = bn(name='l2_bn25')(xconv10a)


	xconv10b = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda) ,name='l2_conc49')(xconv10a)

	xconv10b = bn(name='l2_bn26')(xconv10b)






	new_xfinal_op=Conv2D(1, (1, 1), activation='sigmoid',name='new_xfinal_op')(xconv10b)#tan




	#-----------------------------third branch



	#Concatenation fed to the reconstruction layer of all 3

	x_u_net_op0=keras.layers.concatenate([new_final_op,new_xfinal_op],name='l2_conc9')
	x_u_net_opsp=keras.layers.concatenate([new_final_op,new_xfinal_op],name='x_u_net_opsp')

	#res_1_conv0a = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
	#               kernel_regularizer=regularizers.l2(l2_lambda) ,name='mixer_conv')(x_u_net_op0)

	#res_1_conv0a = bn()(res_1_conv0a)

	#res_1_conv0b = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
	#               kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_conv0a)
	#res_1_conv0c = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
	#               kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_conv0b)
	#res_1_conv0d = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
	#               kernel_regularizer=regularizers.l2(l2_lambda) )(res_1_conv0c)                   


	new_res_1_final_opa=Conv2D(1, (1, 1), activation='sigmoid',name='new_res_1_final_opa')(x_u_net_op0)

	model=Model(inputs=[inputs],outputs=[new_final_op,                                    
	                                  new_xfinal_op,                                 
	                                  new_res_1_final_opa,
	                                  x_u_net_opsp
	                                  
	                                  
	                                ])
	model.compile(optimizer=keras.optimizers.Adam(lr=5e-5),loss={
	                                             
                                            'new_res_1_final_opa':'mse',
                                            'x_u_net_opsp':special_loss_disjoint
	                                            
	                                            })
	                                           
	return model



	

	#model.summary()
	#return model

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





	conv5a = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
	               kernel_regularizer=regularizers.l2(l2_lambda) ,name='disc_l2_conc23')(pool4)

	conv5a = bn(name='disc_l2_bn9')(conv5a)

	conv5b = Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
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


input_shape=240,240,1
#final_model = Sequential()

generator=build_generator(input_shape)
discriminator=build_discriminator(input_shape)
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
#X_train=np.ones((1,240,240,1))
#final_model.fit([X_train],[1],batch_size=1,nb_epoch=1,shuffle=False)
#print ("hi",final_model.predict([X_train],batch_size=1))
import h5py

final_model.save('disjoint_un_sup_mse_complete_gans.h5')
generator.save("disjoint_un_sup_mse_generator.h5")
discriminator.save("disjoint_un_sup_mse_discriminator.h5")















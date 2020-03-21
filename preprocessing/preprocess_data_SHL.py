
# coding: utf-8

# ## This file does the below things
# - Loading the raw data from 7 sensors on;y from training folder since we don't have test labels
# - Assumes the raw txt files are available at location: data_path
# - Extract the data for all the sensors from this raw text files and save the extracted data in npz file format
# - The npz files can be loaded easily later which are much faster than the parsing the raw text files.

import numpy as np

def extract_all_sensors(data_path, out_path)

	# Acc data files
	accx_path = data_path+'/Acc_x.txt'
	accy_path = data_path+'/Acc_y.txt'
	accz_path = data_path+'/Acc_z.txt'

	# Gyro data files
	gyrox_path = data_path+'/Gyr_x.txt'
	gyroy_path = data_path+'/Gyr_y.txt'
	gyroz_path = data_path+'/Gyr_z.txt'

	# Mag data files
	magx_path = data_path+'/Mag_x.txt'
	magy_path = data_path+'/Mag_y.txt'
	magz_path = data_path+'/Mag_z.txt'

	# Linear Acc data files
	laccx_path = data_path+'/LAcc_x.txt'
	laccy_path = data_path+'/LAcc_y.txt'
	laccz_path = data_path+'/LAcc_z.txt'

	# Gravity data files
	grax_path = data_path+'/Gyr_x.txt'
	gray_path = data_path+'/Gyr_y.txt'
	graz_path = data_path+'/Gyr_z.txt'

	# Orientation data files
	oriw_path = data_path+'/Ori_w.txt'
	orix_path = data_path+'/Ori_x.txt'
	oriy_path = data_path+'/Ori_y.txt'
	oriz_path = data_path+'/Ori_z.txt'

	# Pressure data file
	press_path= data_path+'/Pressure.txt'

	# Labels
	label_path = data_path+'/Label.txt'

	# Training dataframe order
	order_path = data_path+'/train_order.txt'


	# # Next we load the text files, and convert to np array

	accx =  np.loadtxt(accx_path)
	accy =  np.loadtxt(accy_path)
	accz =  np.loadtxt(accz_path)
	gyrox =  np.loadtxt(gyrox_path)
	gyroy =  np.loadtxt(gyroy_path)
	gyroz =  np.loadtxt(gyroz_path)
	magx =  np.loadtxt(magx_path)
	magy =  np.loadtxt(magy_path)
	magz =  np.loadtxt(magz_path)
	laccx =  np.loadtxt(laccx_path)
	laccy =  np.loadtxt(laccy_path)
	laccz =  np.loadtxt(laccz_path)
	grax =  np.loadtxt(grax_path)
	gray =  np.loadtxt(gray_path)
	graz =  np.loadtxt(graz_path)
	oriw =  np.loadtxt(oriw_path)
	orix =  np.loadtxt(orix_path)
	oriy =  np.loadtxt(oriy_path)
	oriz =  np.loadtxt(oriz_path)
	press =  np.loadtxt(press_path)
	label= np.loadtxt(label_path)
	order = np.loadtxt(order_path)

	# Saving numpy array data format to the disk in npz format
	# Naming is by the number of frames in this data

	np.savez(out_path+'/Data_16310', accx, accy, accz, gyrox, gyroy, gyroz, 
		magx, magy, magz, laccx, laccy, laccz, grax, gray, graz, 
		oriw, orix, oriy, oriz, press, label, order)

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.shape[0]-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L,a.shape[1] ), strides=(S*n,n,a.strides[1]))

def prepare_dataset(window, stride, out_path):

	# Can be modified to extract required data
	'''
	accx= data['arr_0']
	accy= data['arr_1']
	accz= data['arr_2']
	gyrox= data['arr_3']
	gyroy= data['arr_4']
	gyroz= data['arr_5']
	magx= data['arr_6']
	magy= data['arr_7']
	magz= data['arr_8']
	laccx= data['arr_9']
	laccy= data['arr_10']
	laccz= data['arr_11']
	grax= data['arr_12']
	gray= data['arr_13']
	graz= data['arr_14']
	oriw= data['arr_15']
	orix= data['arr_16']
	oriy= data['arr_17']
	oriz= data['arr_18']
	press= data['arr_19']
	label= data['arr_20']
	'''

	# Extracts only acc, gyr and linear acc from the entire dataset

	data_file='/Data_16310.npz'
	data=np.load(out_path+data_file)
	X = np.float32(np.dstack((np.float32(data['arr_0']), np.float32(data['arr_1']), np.float32(data['arr_2']),
                          np.float32(data['arr_3']), np.float32(data['arr_4']), np.float32(data['arr_5']),
                          np.float32(data['arr_9']), np.float32(data['arr_10']), np.float32(data['arr_11']))))
	Y = np.int8(data['arr_20'] - 1) #labels start fromm 0

	X2 = []
	for i in range (0, X.shape[0]):
	    X2.append(strided_app(X[i], window, stride))
	    
	X2 = np.asarray(X2)
	X2 = X2.reshape((-1,window, X.shape[2]))
	del X

	Y = np.expand_dims(Y, axis=2)
	Y2 = []
	for i in range (0, Y.shape[0]):
	    Y2.append(strided_app(Y[i], window, stride))
	    
	Y2 = np.asarray(Y2)
	Y2 = Y2.reshape((-1,500,1))
	m = stats.mode(Y2, axis = 1)
	del Y

	Y2 = m[0][:,0,0]
	z = 280002 
	z1 = 319999
	X_train0 = X2[:z]
	Y_train = Y2[:z].reshape(-1).astype(np.uint8)
	X_val0 = X2[z:z1]
	Y_val = Y2[z:z1].reshape(-1).astype(np.uint8)
	X_test0 = X2[z1:]
	Y_test = Y2[z1:].reshape(-1).astype(np.uint8)
	np.random.seed(42)
	p = np.random.permutation(Y_train.shape[0])
	Y_train = Y_train[p]
	X_train0 = X_train0[p,:,:]
	num_classes = 8
	Y_train = keras.utils.to_categorical(Y_train, num_classes)
	Y_val = keras.utils.to_categorical(Y_val, num_classes)
	Y_test = keras.utils.to_categorical(Y_test, num_classes)

	np.save(out_path + '/X_train_SHL', X_train0)
	np.save(out_path + '/y_train_SHL', Y_train)
	np.save(out_path + '/X_val_SHL', X_val0)
	np.save(out_path + '/y_val_SHL', Y_val)
	np.save(out_path + '/X_test_SHL', X_test0)
	np.save(out_path + '/y_test_SHL', Y_test)


##Main Code
data_path='/home/vikranth/HASCA-Workshop/new_drive/raw-data/train'
out_path='/home/vikranth/HASCA-Workshop/new_drive/extracted_data'
window = 500
stride = 250

extract_all_sensors(data_path, out_path)
prepare_dataset(window, stride, out_path)

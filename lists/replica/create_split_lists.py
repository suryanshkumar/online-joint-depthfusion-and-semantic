import os

## REPLICA
root_left_RGB = # path to data root
filename_train = 'train.txt'
filename_test = 'test.txt'
filename_val = 'val.txt'

scenes_train = ['apartment_1', 'hotel_0', 'room_2', 'office_1', 'room_0', 'office_4', 'frl_apartment_2', 'frl_apartment_4', 'frl_apartment_5']
scenes_test = ['office_0', 'frl_apartment_3', 'office_2', 'room_1']
scenes_val = ['frl_apartment_0', 'frl_apartment_1', 'apartment_2', 'office_3']

# TRAIN
file = open(filename_train,'w')
for scene in scenes_train:
	for trajectory in os.listdir(root_left_RGB + '/' + scene):
		if trajectory.isdigit():
		 
			file.write(scene + '/' + trajectory + '/left_depth_gt')
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_depth_noise_5.0')
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_rgb') # ORACLE DEPTH
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_camera_matrix') # ORACLE DEPTH			
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_class30') # ORACLE DEPTH
			file.write('\n')

file.close() 

# TEST
file = open(filename_test,'w') 
for scene in scenes_test:
	for trajectory in os.listdir(root_left_RGB + '/' + scene):
		if trajectory.isdigit():

			file.write(scene + '/' + trajectory + '/left_depth_gt')
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_depth_noise_5.0')
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_rgb') # ORACLE DEPTH
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_camera_matrix') # ORACLE DEPTH
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_class30') # ORACLE DEPTH
			file.write('\n')

file.close() 

# VAL
file = open(filename_val,'w') 
for scene in scenes_val:
	for trajectory in os.listdir(root_left_RGB + '/' + scene):
		if trajectory.isdigit():

			file.write(scene + '/' + trajectory + '/left_depth_gt')
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_depth_noise_5.0')
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_rgb') # ORACLE DEPTH
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_camera_matrix') # ORACLE DEPTH
			file.write(' ')

			file.write(scene + '/' + trajectory + '/left_class30') # ORACLE DEPTH
			file.write('\n')
			
file.close() 

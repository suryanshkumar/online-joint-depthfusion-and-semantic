import numpy as np
import csv

def replica_color_palette():
		return [
			[31, 119, 180],
			[174, 199, 232],
			[255, 127, 14],
			[255, 187, 120],
			[44, 160, 60],
			[152, 223, 138],
			[214, 39, 40],
			[255, 152, 150],
			[148, 103, 189],
			[197, 176, 213],
			[140, 86, 75],
			[196, 156, 148],
			[227, 119, 194],
			[247, 182, 210],
			[123, 126, 129],
			[195, 200, 205],
			[188, 189, 34],
			[215, 219, 141],
			[23, 190, 207],
			[158, 218, 229],
			[57, 59, 121],
			[82, 84, 163],
			[107, 110, 207],
			[140, 162, 82],
			[181, 207, 107],
			[206, 219, 156],
			[140, 109, 49],
			[189, 158, 57],
			[231, 186, 82],
			[231, 203, 148],
			[132, 60, 57],
			[173, 73, 74],
			[214, 97, 107],
			[99, 121, 57],
			[231, 150, 156],
			[123, 65, 115],
			[165, 81, 148],
			[156, 158, 222],
			[206, 109, 189],
			[222, 158, 214],
		]

def get_mapping():
	table = np.zeros((256,3))
	r = np.linspace(0,255,256, dtype=np.uint8)
	table[:,0] = r
	np.random.seed(10)
	np.random.shuffle(r)
	table[:,1] = r

	np.random.seed(10000)
	np.random.shuffle(r)
	table[:,2] = r

	rgb_map = np.array(replica_color_palette())

	table[0:40,:]    = rgb_map
	table[40:80,:]   = rgb_map[:,[0,2,1]]
	table[80:120,:]  = rgb_map[:,[1,2,0]]
	table[120:160,:] = rgb_map[:,[1,0,2]]
	table[160:200,:] = rgb_map[:,[2,1,0]]
	table[200:240,:] = rgb_map[:,[2,0,1]]
	table[0] = [0,0,0]

	# for i in range(len(table)):
	# 	temp = abs(table - table[i])
	# 	test = np.where(np.sum(temp,axis=1)==0)[0]
	# 	assert len(test) <= 1, 'Two or more colors are equal in the mapping'

	return table

def replica_names(): # Replica 30-label set
	return [
		'undefined',
		'beanbag',
		'bed',
		'bike',
		'book',
		'cabinet',
		'ceiling',
		'chair',
		'clothing',
		'container',
		'curtain',
		'cushion',
		'door',
		'floor',
		'indoor-plant',
		'lamp',
		'refrigerator',
		'rug',
		'shelf',
		'sink' ,
		'sofa',
		'stair',
		'structure',
		'table',
		'tv-screen',
		'tv-stand',
		'wall',
		'wall-cabinet' ,
		'wall-decoration',
		'window',
	]

def scannet_color_palette():
	return [
		(0, 0, 0),           # undefined
		(174, 199, 232),     # wall
		(152, 223, 138),     # floor
		(31, 119, 180),      # cabinet
		(255, 187, 120),     # bed
		(188, 189, 34),      # chair
		(140, 86, 75),       # sofa
		(255, 152, 150),     # table
		(214, 39, 40),       # door
		(197, 176, 213),     # window
		(148, 103, 189),     # bookshelf
		(196, 156, 148),     # picture
		(23, 190, 207),      # counter
		(178, 76, 76),  
		(247, 182, 210),     # desk
		(66, 188, 102), 
		(219, 219, 141),     # curtain
		(140, 57, 197), 
		(202, 185, 52), 
		(51, 176, 203), 
		(200, 54, 131), 
		(92, 193, 61),  
		(78, 71, 183),  
		(172, 114, 82), 
		(255, 127, 14),      # refrigerator
		(91, 163, 138), 
		(153, 98, 156), 
		(140, 153, 101),
		(158, 218, 229),     # shower curtain
		(100, 125, 154),
		(178, 127, 135),
		(120, 185, 128),
		(146, 111, 194),
		(44, 160, 44),       # toilet
		(112, 128, 144),     # sink
		(96, 207, 209), 
		(227, 119, 194),     # bathtub
		(213, 92, 176), 
		(94, 106, 211), 
		(82, 84, 163),       # otherfurn
		(100, 85, 144)
	]


def scannet_nyu40_names(): # NYU-v2 40-label set
	return [
		'undefined',
		'wall',
		'floor',
		'cabinet',
		'bed',
		'chair',
		'sofa',
		'table',
		'door',
		'window',
		'bookshelf',
		'picture',
		'counter',
		'blinds',
		'desk',
		'shelves',
		'curtain',
		'dresser',
		'pillow',
		'mirror',
		'floor mat',
		'clothes',
		'ceiling',
		'books',
		'refridgerator',
		'television',
		'paper',
		'towel',
		'shower curtain',
		'box',
		'whiteboard',
		'person',
		'nightstand',
		'toilet',
		'sink',
		'lamp',
		'bathtub',
		'bag',
		'otherstructure',
		'otherfurniture',
		'otherprop'
	]

def scannet_nyu20_names(): # 20-class subset for evaluation
	return [
		'undefined',
		'wall',
		'floor',
		'cabinet',
		'bed',
		'chair',
		'sofa',
		'table',
		'door',
		'window',
		'bookshelf',
		'picture',
		'counter',
		'desk',
		'curtain',
		'refridgerator',
		'shower curtain',
		'toilet',
		'sink',
		'bathtub',
		'otherfurniture'
	]

def scannet_main_ids():
	return [
		0,
		1,   
		2,   
		3,   
		4,   
		5,   
		6,   
		7,   
		8,   
		9,   
		10,  
		11,  
		12,  
		14,  
		16,  
		24,  
		28,  
		33,  
		34,  
		36,  
		39  
	]

def ids_to_nyu40():

	file = '/srv/beegfs02/scratch/online_semantic_3d/data/data/scannet/scannetv2-labels.combined.tsv'

	with open(file, 'r') as f:
		read_tsv = csv.reader(f, delimiter='\t')
		rows = [row for row in read_tsv][1:]

	mapping = {int(row[0]): int(row[4]) for row in rows}
	mapping.update({0: 0})
			
	return mapping


def ids_to_nyu20():

	map40 = ids_to_nyu40()
	main_ids = scannet_main_ids()

	# original keys to 21 main values in range [0, 40]
	mapping = {k: (v if v in main_ids else 0) for k, v in map40.items()}

	# original keys to 21 main values in range [0, 20]
	mapping = {k: main_ids.index(v) for k, v in mapping.items()}

	return mapping




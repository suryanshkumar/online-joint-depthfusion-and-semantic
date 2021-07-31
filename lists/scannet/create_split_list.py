import os

## ScanNet v2
root = "/srv/beegfs02/scratch/online_semantic_3d/data/data/scannet/"

splits = ['train', 'val', 'test']

for split in splits:

	scan_dir = 'scans_test/' if split == 'test' else 'scans/'

	split_file = root + "tasks/Benchmark/scannetv2_" + split + ".txt"
	with open(split_file, 'r') as fp:
		scenes = [line[:-1] for line in fp]

	save_file = split + ".txt"
	with open(save_file, 'w') as fp:
		for scene in scenes:
			if '_00' in scene or split == 'test':
				fp.write(scan_dir + scene + '/depth') # GT DEPTH			
				fp.write(' ')

				fp.write(scan_dir + scene + '/color') # RGB
				fp.write(' ')

				fp.write(scan_dir + scene + '/label-filt') # NYU40 LABELS
				fp.write(' ')

				fp.write(scan_dir + scene + '/pose') # 4x4 POSES	
				fp.write(' ')

				fp.write(scan_dir + scene + '/intrinsic') # CAMERA PARAMETERS		
				fp.write('\n')
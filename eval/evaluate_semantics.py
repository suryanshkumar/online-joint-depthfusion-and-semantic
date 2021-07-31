#!/path/to/python/executable/in/conda/env

import numpy as np
import h5py
import argparse
import json
from pathlib import Path
from plyfile import PlyData
from scipy import stats

from utils.mesh import Face, Vertex
from utils.mapping import map_labels_to_rgb

parser = argparse.ArgumentParser()
parser.add_argument('--scene', default='office_0')
parser.add_argument('--semantics', default='class30')
parser.add_argument('--gt_mesh', action='store_true', default=False)
args = parser.parse_args()

UNKNOWN_ID = 41
VOXEL_SIZE = 0.01

def translate_points(points, bbox):
    # points: points (=vertices) of the estimated grid
    # bbox: coordinates of the ground truth box
    R = np.diag(np.ones(4))
    R[0:3, -1] = bbox[:, 0] - [np.amin(points[:, 0]), np.amin(points[:, 1]), np.amin(points[:, 2])]
    print("Matrix R (needed in case of manual registration):\n", R)
    points = np.concatenate([points, np.ones((len(points), 1))], axis=1)    # homogeneous coordinates
    points = np.array(np.dot(points, np.transpose(R)))[:, 0:3]              # rotation (only translation in this case)
    return points

def get_names(filepath):
    with open(filepath, "r") as f:
        semantic_info = json.load(f)
    names = [cl['name'] for cl in semantic_info['classes']]
    if not 'undefined' in names:
        names = ['undefined'] + names
    return names

def get_iou(label_id, valid_class_ids, confusion):
    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false negatives
    fn = np.longlong(confusion[label_id, :].sum()) - tp
    # false positives
    not_ignored = [l for l in valid_class_ids if not l == label_id]
    fp = np.longlong(confusion[not_ignored, label_id].sum())

    denom = (tp + fp + fn)
    if denom == 0:
        return (float(0), tp, denom)
    return (float(tp) / denom, tp, denom)

def write_result_file(confusion, valid_class_ids, valid_names, ious, filename):
    with open(filename, 'w') as f:
        f.write('iou scores\n')
        for i in range(len(valid_class_ids)):
            label_id = valid_class_ids[i]
            label_name = valid_names[i]
            iou = ious[label_name][0]
            f.write('{0:<14s}({1:<2d}): {2:>5.3f}\n'.format(label_name, label_id, iou))
        f.write('\nconfusion matrix\n')
        f.write('\t\t\t')
        for i in range(len(valid_class_ids)):
            f.write('\t\t{0:<8d}'.format(valid_class_ids[i]))
        f.write('\n')
        for r in range(len(valid_class_ids)):
            f.write('{0:<14s}({1:<2d})'.format(valid_names[r], valid_class_ids[r]))
            for c in range(len(valid_class_ids)):
                f.write('\t{0:>5.0f}'.format(confusion[valid_class_ids[r],valid_class_ids[c]]))
            f.write('\n')
    print('wrote results to', filename)

def evaluate(est_ids, gt_ids, output_file, name_path):
    names = get_names(name_path)    # contains also label 40 = unknown
    valid_class_ids = np.unique(np.concatenate((est_ids, gt_ids), axis=0))
    valid_names = [names[i] for i in valid_class_ids]

    confusion = np.zeros((len(names), len(names)), dtype=np.ulonglong)
    for (gt, est) in zip(gt_ids, est_ids):
        if gt not in valid_class_ids:
            continue
        if est not in valid_class_ids:
            est = UNKNOWN_ID
        confusion[gt][est] += 1

    class_ious = {}
    for i in range(len(valid_names)):
        label_name = valid_names[i]
        label_id = valid_class_ids[i]
        class_ious[label_name] = get_iou(label_id, valid_class_ids, confusion)

    print("==============================")
    print("semantics result :")
    print("==============================")
    for i in range(len(valid_names)):
        label_name = valid_names[i]
        print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, class_ious[label_name][0], class_ious[label_name][1], class_ious[label_name][2]))
    write_result_file(confusion, valid_class_ids, valid_names, class_ious, output_file)

    
def main():
    # Filenames
    dataset      = Path('/srv/beegfs02/scratch/online_semantic_3d/data/data')
    name_path    = dataset/'replica'/str(args.semantics + '.json')
    gt_grid_in   = dataset/'habitat'/'manual'/args.scene/'gt_semantic_sdf'/args.semantics/str('sdf_' + args.scene + '.hdf')
    est_ids_in   = str(args.scene + '.semantics.hf5')
    if args.gt_mesh:
        scene_in = dataset/'ground_truth_data/gt_sdf_visualizations'/str('meshed_sdf_' + args.scene + '.ply')
        est_out  = str(args.scene + '_gt_mesh_est_semantic.ply')    # est semantic voxelgrid on gt mesh (if 2d semantic gt --> GT3)
        gt_out   = str(args.scene + '_gt_mesh_gt_semantic.ply')     # gt semantic voxelgrid on gt mesh (GT1)
        eval_out = str(args.scene + '_eval_semantics_gt.txt')
    else:
        scene_in = str(args.scene + '.ply')
        est_out  = str(args.scene + '_est_mesh_est_semantic.ply')   # est semantic voxelgrid on est mesh (if 2d semantic gt --> GT4)
        gt_out   = str(args.scene + '_est_mesh_gt_semantic.ply')    # gt semantic voxelgrid on est mesh (GT2)
        eval_out = str(args.scene + '_eval_semantics.txt')

    hf = h5py.File(est_ids_in, 'r')
    est_ids_grid = hf['semantics']

    hf = h5py.File(gt_grid_in, 'r')
    gt_ids_grid = hf['sdf'][1]
    gt_bbox = hf.attrs['bbox']

    if est_ids_grid.shape[0] > gt_ids_grid.shape[0]:
        pad = (est_ids_grid.shape[0] - gt_ids_grid.shape[0]) // 2
        print("Padded gt grid with {} values".format(pad))
        gt_ids_grid = np.pad(gt_ids_grid, pad, 'constant', constant_values=0)
        gt_bbox[:, 0] = gt_bbox[:, 0] - pad * VOXEL_SIZE * np.ones((1,1,1))
        gt_bbox[:, 1] = gt_bbox[:, 1] + pad * VOXEL_SIZE * np.ones((1,1,1))

    est_ids_grid = np.array(est_ids_grid).astype(np.uint8)
    gt_ids_grid = np.array(gt_ids_grid).astype(np.uint8)
    print('Labels in the est semantic grid: ', np.unique(est_ids_grid))
    print('Labels in the gt semantic grid: ', np.unique(gt_ids_grid))

    # Load the est scene on which to paste the voxelgrid labels
    mesh = PlyData.read(scene_in)
    vertices = Vertex(mesh.elements[0])
    faces = Face(mesh.elements[1])
    points = np.array([np.array((vertex['x'], vertex['y'], vertex['z'])) for vertex in vertices])
    vertex_indices = np.array([np.array(face['vertex_indices']) for face in faces])

    vertices.append_property(name='red', dtype='uint8')
    vertices.append_property(name='green', dtype='uint8')
    vertices.append_property(name='blue', dtype='uint8')

    # Translation of the scene points on the grid, using a corner as reference.
    points = translate_points(points, gt_bbox)

    x = (np.round((points[:, 0] - gt_bbox[0, 0]) / VOXEL_SIZE)).astype(np.uint16)
    y = (np.round((points[:, 1] - gt_bbox[1, 0]) / VOXEL_SIZE)).astype(np.uint16)
    z = (np.round((points[:, 2] - gt_bbox[2, 0]) / VOXEL_SIZE)).astype(np.uint16)

    est_ids_vertex = np.array([est_ids_grid[x[i], y[i], z[i]] for i in range(len(x))])
    gt_ids_vertex = np.array([gt_ids_grid[x[i], y[i], z[i]] for i in range(len(x))])

    if args.gt_mesh:
        evaluate(est_ids_vertex, gt_ids_vertex, eval_out, name_path)

    est_ids_per_face = np.array([est_ids_vertex[vertex_indices[i]] for i in range(len(vertex_indices))])
    gt_ids_per_face = np.array([gt_ids_vertex[vertex_indices[i]] for i in range(len(vertex_indices))])

    est_object_id = np.squeeze(np.array(stats.mode(est_ids_per_face, axis=1)[0]))
    gt_object_id = np.squeeze(np.array(stats.mode(gt_ids_per_face, axis=1)[0]))

    faces.append_property(name='object_id', dtype='uint16', data=est_object_id)
    vertices = map_labels_to_rgb(vertices, faces, return_labels=False, new=479)
    PlyData([vertices.element, faces.element]).write(est_out)

    faces.append_property(name='object_id', dtype='uint16', data=gt_object_id)
    vertices = map_labels_to_rgb(vertices, faces, return_labels=False, new=479)
    PlyData([vertices.element, faces.element]).write(gt_out)


if __name__ == '__main__':
    main()
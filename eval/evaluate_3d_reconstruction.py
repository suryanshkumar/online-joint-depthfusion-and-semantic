#!/path/to/python/executable/in/conda/env
import os

# ----------------------------------------------------------------------------
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2017
# Arno Knapitsch <arno.knapitsch@gmail.com >
# Jaesik Park <syncle@gmail.com>
# Qian-Yi Zhou <Qianyi.Zhou@gmail.com>
# Vladlen Koltun <vkoltun@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ----------------------------------------------------------------------------
#
# This python script is for downloading dataset from www.tanksandtemples.org
# The dataset has a different license, please refer to
# https://tanksandtemples.org/license/

# this script requires Open3D python binding
# please follow the intructions in setup.py before running this script.

# This is script is modified by Erik Sandstroem

# Run this script in the folder of the target .ply file by providing the name of the .ply file as
# the input. The evaluation output is stored in the corresponding folder located in the data folder
# of this script.

# example use case: go to the directory where the reconstructed .ply file is located
# then run evaluate_3d_reconstruction.py somename.ply

import numpy as np
import open3d as o3d
from sys import argv
import pathlib
import h5py

from config import *
from registration import (
    trajectory_alignment,
    registration_vol_ds,
    registration_unif,
    read_trajectory,
)
from evaluation import EvaluateHisto
from util import make_dir
from plot import plot_graph


def filter_ply(filename, scene, voxel_size):
    hf = h5py.File(scene + '.tsdf.hf5', 'r')
    tsdf = np.array(hf["TSDF"])
    hf = h5py.File(scene + '.weights.hf5', 'r')
    weights = np.array(hf["weights"])

    resolution = tsdf.shape
    max_resolution = np.array(resolution).max()
    length = max_resolution * voxel_size # the UniformTSDFVolume is always a cube of voxels so we need the largest dimension to be the side length.

    # create the object from which we can call the marching cubes algorithm.
    volume = o3d.integration.UniformTSDFVolume(
                         length=length,
                         resolution=max_resolution,
                         sdf_trunc=0.1,
                         color_type=o3d.integration.TSDFVolumeColorType.NoColor)

    mask = weights > 0 # only copy over tsdf values at indices which we have integrated into.
    indices_x = mask.nonzero()[0]
    indices_y = mask.nonzero()[1]
    indices_z = mask.nonzero()[2]

    for i in range(indices_x.shape[0]):
        volume.set_tsdf_at(tsdf[indices_x[i], indices_y[i], indices_z[i]],
                          indices_x[i] , indices_y[i], indices_z[i])
        volume.set_weight_at(1, indices_x[i], indices_y[i], indices_z[i]) # We can set any non-zero weight here.

    print("Extract a filtered triangle mesh from the volume and visualize it.")
    mesh = volume.extract_triangle_mesh() # runs marching cubes that removes the 2nd artifact surface
    del volume

    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(filename, mesh)


def run_evaluation(scene, ground_truth_data, transformation):

    CURRENT_DIR = str(pathlib.Path().absolute()) 

    out_dir = CURRENT_DIR + '/' + scene + '/'
    make_dir(out_dir)

    pred_ply_path = CURRENT_DIR + '/' + scene + '_filt.ply'
    
    voxel_size = 0.01
    dTau = 0.02 # constant Tau regardless of scene size

    try:
        filter_ply(pred_ply_path, scene, voxel_size)
    except:
        print("Using non filtered mesh")
        pred_ply_path = pred_ply_path.replace('_filt','')

    # below is the new way of handling the 3D evaluation for routedfusion and its derivatives
    gt_trans = np.loadtxt(base_transformation_dir + '/' + transformation)

    if ground_truth_data == 'watertight':
        gt_ply_path = watertight_ground_truth_data_base + '/' + scene + '_processed.ply' # ground truth .ply file
    elif ground_truth_data == 'standard_trunc':
        gt_ply_path = standard_trunc_ground_truth_data_base + '/' + scene + '.ply' 
    elif ground_truth_data == 'artificial_trunc':
        gt_ply_path = artificial_trunc_ground_truth_data_base + '/' + scene + '.ply' 


    print("")
    print("===========================")
    print("Evaluating %s" % scene)
    print("===========================")

    # Load reconstruction and according GT
    # Use these four lines below to also load the normals from the mesh. Note
    # the implementation is not complete - I turned off the normal estimation
    # step in evaluation.py because this caused artifacts. Instead
    # what I need to do is to make sure that the normals which I load here
    # are kept thorugh the cropping and downsampling function of the point cloud.
    # mesh = trimesh.load(pred_ply_path)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    # pcd.normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))
    pcd = o3d.io.read_point_cloud(pred_ply_path)
    mesh = o3d.io.read_triangle_mesh(pred_ply_path)
    gt_pcd = o3d.io.read_point_cloud(gt_ply_path)
    gt_mesh = o3d.io.read_triangle_mesh(gt_ply_path)

    # Registration refinment in 3 iterations
    # r2 = registration_vol_ds(
    #     pcd, gt_pcd, gt_trans, voxel_size, dTau * 80, 20
    # )
    # r3 = registration_vol_ds(
    #     pcd, gt_pcd, r2.transformation, voxel_size, dTau * 20, 20
    # )
    # r = registration_unif(pcd, gt_pcd, r3.transformation, 2 * dTau, 20)

    # Histogramms and P/R/F1
    plot_stretch = 5
    [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
        min1,
        min2,
        max1,
        max2,
        mean1,
        mean2,
        median1,
        median2,
        std1,
        std2,
    ] = EvaluateHisto(
        mesh,
        pcd,
        gt_mesh,
        gt_pcd,
        gt_trans,
        voxel_size,
        dTau,
        out_dir,
        plot_stretch,
        scene,
    )
    eva = [precision, recall, fscore]
    print("==============================")
    print("evaluation result : %s" % scene)
    print("==============================")
    print("distance tau : %.3f" % dTau)
    print("precision : %.4f" % eva[0])
    print("recall : %.4f" % eva[1])
    print("f-score : %.4f" % eva[2])
    print("==============================")
    print("precision statistics")
    print("min: %.4f" % min1)
    print("max: %.4f" % max1)
    print("mean: %.4f" % mean1)
    print("median: %.4f" % median1)
    print("std: %.4f" % std1)
    print("==============================")
    print("recall statistics")
    print("min: %.4f" % min2)
    print("max: %.4f" % max2)
    print("mean: %.4f" % mean2)
    print("median: %.4f" % median2)
    print("std: %.4f" % std2)
    print("==============================")

    with open(out_dir + 'result.txt', 'a+') as f:
        print("==============================", file=f)
        print("evaluation result : %s" % scene, file=f)
        print("==============================", file=f)
        print("distance tau : %.3f" % dTau, file=f)
        print("precision : %.4f" % eva[0], file=f)
        print("recall : %.4f" % eva[1], file=f)
        print("f-score : %.4f" % eva[2], file=f)
        print("==============================", file=f)
        print("precision statistics", file=f)
        print("min: %.4f" % min1, file=f)
        print("max: %.4f" % max1, file=f)
        print("mean: %.4f" % mean1, file=f)
        print("median: %.4f" % median1, file=f)
        print("std: %.4f" % std1, file=f)
        print("==============================", file=f)
        print("recall statistics", file=f)
        print("min: %.4f" % min2, file=f)
        print("max: %.4f" % max2, file=f)
        print("mean: %.4f" % mean2, file=f)
        print("median: %.4f" % median2, file=f)
        print("std: %.4f" % std2, file=f)
        print("==============================", file=f)


    # Plotting
    plot_graph(
        scene,
        fscore,
        dTau,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
        plot_stretch,
        out_dir,
    )


if __name__ == "__main__":
     
    scene = argv[1] # scene
    ground_truth_data = argv[2] # watertight or artificial_trunc or standard_trunc
    transformation = argv[3] 


    run_evaluation(
        scene=scene,
        ground_truth_data=ground_truth_data,
        transformation=transformation
    )

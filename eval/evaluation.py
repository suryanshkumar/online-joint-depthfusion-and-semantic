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

import json
import copy
import os
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font',**{'family':'serif','sans-serif':['Times New Roman']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def read_alignment_transformation(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    return np.asarray(data["transformation"]).reshape((4, 4)).transpose()


def write_color_distances_pcd(path, pcd, distances, max_distance):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    # cmap = plt.get_cmap("afmhot")
    #cmap = plt.get_cmap("hot_r")
    #cmap = plt.get_cmap("winter")
    cmap = plt.get_cmap("hsv")
    distances = np.array(distances)
    #colors = cmap(np.minimum(distances, max_distance) / max_distance)[:, :3] # This is the original line that I replaced below
    # I replaced it because the above line does not give a linear mapping between the color values from minimum distance to maximum distance as the histogram colorization does. Now they are aligned.
    max_dist = 0.05
    # c = distances/np.amax(distances.flatten())
    c = distances/max_dist
    c[c > 0.85] = 0.85
    c += 0.33
    c[c > 1] = c[c > 1] - 1
    # for i in range(len(c)):
    #     if c[i] > 0.85:
    #         c[i] = 0.85
    #     c += 0.33
    #     if c[i] > 1:
    #         c[i] -= 1

    colors = cmap(c)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)


def write_color_distances_mesh(path, mesh, distances, max_distance):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    # cmap = plt.get_cmap("afmhot")
    #cmap = plt.get_cmap("hot_r")
    #cmap = plt.get_cmap("winter")
    cmap = plt.get_cmap("hsv")
    distances = np.array(distances)
    #colors = cmap(np.minimum(distances, max_distance) / max_distance)[:, :3] # This is the original line that I replaced below
    # I replaced it because the above line does not give a linear mapping between the color values from minimum distance to maximum distance as the histogram colorization does. Now they are aligned.
    max_dist = 0.05
    # c = distances/np.amax(distances.flatten())
    c = distances/max_dist
    c[c > 0.85] = 0.85
    c += 0.33
    c[c > 1] = c[c > 1] - 1
    # for i in range(len(c)):
    #     if c[i] > 0.85:
    #         c[i] = 0.85
    #     c += 0.33
    #     if c[i] > 1:
    #         c[i] -= 1

    colors = cmap(c)[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_triangle_mesh(path, mesh)


# Note that this method contained a cropping part that cropped out the points
# that lay within the bounding box of the ground truth point cloud. This is not 
# necessary for me now when I used meshed reconstructions. NOW I INCLUDED IT AGAIN!
def EvaluateHisto(
    source_mesh,
    source,
    target_mesh,
    target,
    trans,
    voxel_size,
    threshold,
    filename_mvs,
    plot_stretch,
    scene_name,
    verbose=True,
):
    print("[EvaluateHisto]")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    s = copy.deepcopy(source)
    s.transform(trans)
    aabb = target.get_axis_aligned_bounding_box()
    #s = s.crop(aabb)

    print('source points before downsampling: ', np.asarray(s.points).shape)
    s = s.voxel_down_sample(voxel_size)
    print('source points after downsampling: ', np.asarray(s.points).shape)
    s.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    print(filename_mvs + "/" + scene_name + ".precision.ply")

    t = copy.deepcopy(target)
    print('target points before downsampling: ', np.asarray(t.points).shape)
    t = t.voxel_down_sample(voxel_size)
    print('target points after downsampling: ', np.asarray(t.points).shape)
    t.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))

    print("[compute_point_cloud_to_point_cloud_distance]")
    distance1 = s.compute_point_cloud_distance(t) # distance from source to target
    print("[compute_point_cloud_to_point_cloud_distance]")
    distance2 = t.compute_point_cloud_distance(s) # distance from target to source

    # plot histograms of the distances
    cm = plt.get_cmap('hsv')
    n, bins, patches = plt.hist(distance1, bins = 1000)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    #print(len(bin_centers))

    # scale values to interval [0,1]
    #print(min(bin_centers))
    max_col = 0.05
    col = bin_centers - min(bin_centers)
    # col /= max(col)
    col /= max_col
    for c, p in zip(col, patches):
        if c > 0.85:
            c = 0.85
        c += 0.33
        if c > 1:
            c -= 1
        plt.setp(p, 'facecolor', cm(c))

    plt.ylabel("$\#$ of points", fontsize=18)
    plt.xlabel("Meters", fontsize=18)
    plt.title("Precision Histogram", fontsize=18)
    plt.grid(True)
    plt.savefig(filename_mvs + "/" + 'histogram_rec_to_gt')
    plt.clf()
    n, bins, patches = plt.hist(distance2, bins = 1000)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max_col
    for c, p in zip(col, patches):
        if c > 0.85:
            c = 0.85
        c += 0.33
        if c > 1:
            c -= 1
        plt.setp(p, 'facecolor', cm(c))

    plt.ylabel("$\#$ of points", fontsize=18)
    plt.xlabel("Meters", fontsize=18)
    plt.title("Recall Histogram", fontsize=18)
    plt.grid(True)
    plt.savefig(filename_mvs + "/" + 'histogram_gt_to_rec')
    


    # write the distances to bin files
    # np.array(distance1).astype("float64").tofile(
    #     filename_mvs + "/" + scene_name + ".precision.bin"
    # )
    # np.array(distance2).astype("float64").tofile(
    #     filename_mvs + "/" + scene_name + ".recall.bin"
    # )

    # Colorize the poincloud files prith the precision and recall values
    # o3d.io.write_point_cloud(
    #     filename_mvs + "/" + scene_name + ".precision.ply", s
    # )
    # o3d.io.write_point_cloud(
    #     filename_mvs + "/" + scene_name + ".precision.ncb.ply", s
    # )
    # o3d.io.write_point_cloud(filename_mvs + "/" + scene_name + ".recall.ply", t)

    source_n_fn = filename_mvs + "/" + scene_name + ".precision.ply"
    target_n_fn = filename_mvs + "/" + scene_name + ".recall.ply"

    print("[ViewDistances] Add color coding to visualize error")
    # eval_str_viewDT = (
    #     OPEN3D_EXPERIMENTAL_BIN_PATH
    #     + "ViewDistances "
    #     + source_n_fn
    #     + " --max_distance "
    #     + str(threshold * 3)
    #     + " --write_color_back --without_gui"
    # )
    # os.system(eval_str_viewDT)
    write_color_distances_mesh(source_n_fn, source_mesh, distance1, 3 * threshold)

    print("[ViewDistances] Add color coding to visualize error")
    # eval_str_viewDT = (
    #     OPEN3D_EXPERIMENTAL_BIN_PATH
    #     + "ViewDistances "
    #     + target_n_fn
    #     + " --max_distance "
    #     + str(threshold * 3)
    #     + " --write_color_back --without_gui"
    # )
    # os.system(eval_str_viewDT)
    write_color_distances_mesh(target_n_fn, target_mesh, distance2, 3 * threshold)

    # get histogram and f-score
    [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ] = get_f1_score_histo2(
        threshold, filename_mvs, plot_stretch, distance1, distance2
    )
    np.savetxt(filename_mvs + "/" + scene_name + ".recall.txt", cum_target)
    np.savetxt(filename_mvs + "/" + scene_name + ".precision.txt", cum_source)
    np.savetxt(
        filename_mvs + "/" + scene_name + ".prf_tau_plotstr.txt",
        np.array([precision, recall, fscore, threshold, plot_stretch]),
    )
    # calculate mean, median, min, max, std of distance1 and distance2
    min1 = np.amin(distance1)
    min2 = np.amin(distance2)
    max1 = np.amax(distance1)
    max2 = np.amax(distance2)
    mean1 = np.mean(distance1)
    mean2 = np.mean(distance2)
    median1 = np.median(distance1)
    median2 = np.median(distance2)
    std1 = np.std(distance1)
    std2 = np.std(distance2)
    np.savetxt(
        filename_mvs + "/" + scene_name + ".min12_max12_mean12_median12_std12.txt",
        np.array([min1, min2, max1, max2, mean1, mean2, median1, median2, std1, std2]),
    )

    return [
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
    ]


def get_f1_score_histo2(
    threshold, filename_mvs, plot_stretch, distance1, distance2, verbose=True
):
    print("[get_f1_score_histo2]")
    dist_threshold = threshold
    if len(distance1) and len(distance2):

        recall = float(sum(d < threshold for d in distance2)) / float(
            len(distance2)
        )
        precision = float(sum(d < threshold for d in distance1)) / float(
            len(distance1)
        )
        fscore = 2 * recall * precision / (recall + precision)
        num = len(distance1)
        bins = np.arange(0, dist_threshold * plot_stretch, dist_threshold / 100)
        hist, edges_source = np.histogram(distance1, bins)
        cum_source = np.cumsum(hist).astype(float) / num

        num = len(distance2)
        bins = np.arange(0, dist_threshold * plot_stretch, dist_threshold / 100)
        hist, edges_target = np.histogram(distance2, bins)
        cum_target = np.cumsum(hist).astype(float) / num

    else:
        precision = 0
        recall = 0
        fscore = 0
        edges_source = np.array([0])
        cum_source = np.array([0])
        edges_target = np.array([0])
        cum_target = np.array([0])

    return [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ]

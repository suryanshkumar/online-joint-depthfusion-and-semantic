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

from trajectory_io import read_trajectory, convert_trajectory_to_pointcloud
import copy
import numpy as np
import open3d as o3d
MAX_POINT_NUMBER = 4e6


def read_mapping(filename):
    mapping = []
    with open(filename, "r") as f:
        n_sampled_frames = int(f.readline())
        n_total_frames = int(f.readline())
        mapping = np.zeros(shape=(n_sampled_frames, 2))
        metastr = f.readline()
        for iter in range(n_sampled_frames):
            metadata = list(map(int, metastr.split()))
            mapping[iter, :] = metadata
            metastr = f.readline()
    return [n_sampled_frames, n_total_frames, mapping]


def gen_sparse_trajectory(mapping, f_trajectory):
    sparse_traj = []
    for m in mapping:
        sparse_traj.append(f_trajectory[int(m[1] - 1)])
    return sparse_traj


def trajectory_alignment(
    map_file, traj_to_register, gt_traj_col, gt_trans, scene
): # the scene data is not used
    traj_pcd_col = convert_trajectory_to_pointcloud(gt_traj_col) # this creates a point in 3D for each camera center.
    traj_pcd_col.transform(gt_trans) # this applies the transformation to this point cloud
    corres = o3d.utility.Vector2iVector(
        np.asarray(list(map(lambda x: [x, x], range(len(gt_traj_col)))))
    )
    # the corres list is just full of rows from [0, 0] to [263, 263] This makes sense
    rr = o3d.registration.RANSACConvergenceCriteria()
    rr.max_iteration = 100000
    rr.max_validation = 100000

    # in this case a log file was used which contains
    # every movie frame (see tutorial for details)
    if len(traj_to_register) > 1600:
        # The map_file is not even used if the number of camera poses don't exceed 1600. I have 400
        # For now, I don't need to care about it. According to the tutorial, this is only central for video
        # reconstruction
        n_sampled_frames, n_total_frames, mapping = read_mapping(map_file)
        traj_col2 = gen_sparse_trajectory(mapping, traj_to_register)
        traj_to_register_pcd = convert_trajectory_to_pointcloud(traj_col2)
    else:
        traj_to_register_pcd = convert_trajectory_to_pointcloud(
            traj_to_register
        )

    randomvar = 0.0
    nr_of_cam_pos = len(traj_to_register_pcd.points)
    rand_number_added = np.asanyarray(traj_to_register_pcd.points) * (
        np.random.rand(nr_of_cam_pos, 3) * randomvar - randomvar / 2.0 + 1
    )
    # Since the randomvar is 0, list_rand = traj_to_register_pcd
    list_rand = list(rand_number_added)
    traj_to_register_pcd_rand = o3d.geometry.PointCloud()
    # print(len(traj_to_register_pcd.points))
    for elem in list_rand:
        traj_to_register_pcd_rand.points.append(elem)

    # At this point, traj_to_register_pcd_rand = traj_to_register_pcd

    # Rough registration based on aligned colmap SfM data
    print(traj_to_register_pcd_rand.points[0])
    print(traj_pcd_col.points[0])
    reg = o3d.registration.registration_ransac_based_on_correspondence(
        traj_to_register_pcd_rand,
        traj_pcd_col,
        corres,
        0.2,
        o3d.registration.TransformationEstimationPointToPoint(True),
        6,
        rr,
    )
    print(reg.transformation)
    print(gt_trans)
    return reg.transformation


# This function is called "crop_and_downsample" because it used to also crop out the
# point cloud that was only true to the bounding box of the ground truth point cloud, 
# but since I am not working on non-meshed colmap, this is not a problem so I removed the 
# cropping. The name remains to remind me that I can add cropping here at some point. 
# Check the tanks and temples original code for this. I can use the cropping function
# open3d.geometry.crop_point_cloud(input, min_bound, max_bound). NOW I INCLUDED IT AGAIN!
def crop_and_downsample(
    pcd,
    down_sample_method="voxel",
    voxel_size=0.01,
    trans=np.identity(4),
    target_pcd=None,
):
    pcd_copy = copy.deepcopy(pcd)
    pcd_copy.transform(trans)

    aabb = target_pcd.get_axis_aligned_bounding_box()
    # aabb.color = (1,0,0)
    obb = target_pcd.get_oriented_bounding_box()
    # obb.color = (0, 1,0)
    pcd_crop = pcd_copy.crop(aabb)
    # o3d.visualization.draw_geometries([pcd_crop, aabb, obb])

    # pcd_crop = crop_point_cloud(pcd_copy, target_pcd.get_min_bound(), target_pcd.get_max_bound())
    # vol = o3d.visualization.SelectionPolygonVolume()
    # min_ = [pcd_copy.get_min_bound()[0], 0, pcd_copy.get_min_bound()[-1]]
    # max_ = [pcd_copy.get_max_bound()[0], 0, pcd_copy.get_max_bound()[-1]]
    # vol.bounding_polygon = o3d.utility.Vector3dVector(np.array([min_, max_]))
    # print(np.array((target_pcd.get_min_bound(), target_pcd.get_max_bound())).shape)
    # vol.axis_max =target_pcd.get_max_bound()[-2]
    # vol.axis_min = target_pcd.get_min_bound()[-2]
    # vol.orthogonal_axis = 'Y'
    # print(vol.bounding_polygon)
    # pcd_crop = vol.crop_point_cloud(pcd_copy)

    if down_sample_method == "voxel":
        # return voxel_down_sample(pcd_crop, voxel_size)
        return pcd_crop.voxel_down_sample(voxel_size)
    elif down_sample_method == "uniform":
        n_points = len(pcd_crop.points)
        if n_points > MAX_POINT_NUMBER:
            ds_rate = int(round(n_points / float(MAX_POINT_NUMBER)))
            return pcd_crop.uniform_down_sample(ds_rate)
    return pcd_crop


def registration_unif(
    source,
    gt_target,
    init_trans,
    threshold,
    max_itr,
    max_size=4 * MAX_POINT_NUMBER, # this is not used
    verbose=True,
):
    if verbose:
        print("[Registration] threshold: %f" % threshold)
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    s = crop_and_downsample(
        source, down_sample_method="uniform", trans=init_trans, target_pcd=gt_target
    )
    t = crop_and_downsample(
        gt_target, down_sample_method="uniform", target_pcd=gt_target
    )
    reg = o3d.registration.registration_icp(
        s,
        t,
        threshold,
        np.identity(4),
        o3d.registration.TransformationEstimationPointToPoint(True),
        o3d.registration.ICPConvergenceCriteria(1e-6, max_itr),
    )
    reg.transformation = np.matmul(reg.transformation, init_trans)
    return reg


def registration_vol_ds(
    source,
    gt_target,
    init_trans,
    voxel_size,
    threshold,
    max_itr,
    verbose=True,
):
    if verbose:
        print(
            "[Registration] voxel_size: %f, threshold: %f"
            % (voxel_size, threshold)
        )
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    s = crop_and_downsample(
        source,
        down_sample_method="voxel",
        voxel_size=voxel_size,
        trans=init_trans,
        target_pcd=gt_target,
    )
    t = crop_and_downsample(
        gt_target,
        down_sample_method="voxel",
        voxel_size=voxel_size,
        target_pcd=gt_target,
    )
    reg = o3d.registration.registration_icp(
        s,
        t,
        threshold,
        np.identity(4),
        o3d.registration.TransformationEstimationPointToPoint(True),
        o3d.registration.ICPConvergenceCriteria(1e-6, max_itr),
    )
    reg.transformation = np.matmul(reg.transformation, init_trans)
    return reg

import numpy as np
cimport numpy as np

from camera.transform import pixel_to_world_coord
from copy import copy


cdef np.ndarray depth_rendering(np.float32_t[:, :] extrinsics,
                                np.ndarray intrinsics,
                                tuple shape,
                                np.float32_t[:, :, :, ::1] volume,
                                float resolution, np.ndarray offset):

    cdef int width = shape[1]
    cdef int height = shape[0]

    cdef np.ndarray upper_left_p = np.asarray([0, 0]).astype(np.float32)
    cdef np.ndarray upper_right_p = np.asarray([width, 0]).astype(np.float32)
    cdef np.ndarray lower_left_p = np.asarray([0, height]).astype(np.float32)
    cdef np.ndarray lower_right_p = np.asarray([width, height]).astype(np.float32)

    cdef float image_plane = 1

    upper_left_w = pixel_to_world_coord(upper_left_p, intrinsics, extrinsics, image_plane)
    upper_right_w = pixel_to_world_coord(upper_right_p, intrinsics, extrinsics, image_plane)
    lower_left_w = pixel_to_world_coord(lower_left_p, intrinsics, extrinsics, image_plane)
    lower_right_w = pixel_to_world_coord(lower_right_p, intrinsics, extrinsics, image_plane)

    cdef np.ndarray eye = np.zeros(4)
    eye[3] = 1
    eye = np.dot(extrinsics[:3], eye)

    # get direction vectors for all 4 image corners
    cdef np.ndarray upper_left_normal = (upper_left_w - eye)/np.linalg.norm(upper_left_w - eye)
    cdef np.ndarray upper_right_normal = (upper_right_w - eye)/np.linalg.norm(upper_right_w - eye)
    cdef np.ndarray lower_left_normal = (lower_left_w - eye)/np.linalg.norm(lower_left_w - eye)
    cdef np.ndarray lower_right_normal = (lower_right_w - eye)/np.linalg.norm(lower_right_w - eye)

    cdef float inv_height = 1./height
    cdef float inv_width = 1./width

    # delta vectors
    cdef np.ndarray left_normal_y_delta = (lower_left_normal - upper_left_normal) * inv_height
    cdef np.ndarray right_normal_y_delta = (lower_right_normal - upper_right_normal) * inv_height

    # starting point
    cdef np.ndarray left_normal = upper_left_normal
    cdef np.ndarray right_normal = upper_right_normal

    cdef np.ndarray depth
    depth = np.zeros(shape)

    cdef np.ndarray ray_direction
    cdef np.ndarray normal_x_delta

    for y in range(0, height):

        # initialize ray direction
        ray_direction = left_normal
        normal_x_delta = (right_normal - left_normal) * inv_width

        for x in range(0, width):

            # normalize ray direction
            ray_direction = ray_direction/np.linalg.norm(ray_direction)

            # get depth value
            depth[y, x] = trace_ray(eye, ray_direction, volume, resolution, offset)

            # update ray direction
            ray_direction += normal_x_delta

        # update line endpoints
        left_normal += left_normal_y_delta
        right_normal += right_normal_y_delta

    return depth


def trace_ray(np.ndarray eye, np.ndarray direction, np.float32_t[:, :, :, ::1] volume,
              float resolution, np.ndarray offset):

    cdef int coord_origin_x = <int>round(offset[0]/resolution)
    cdef int coord_origin_y = <int>round(offset[1]/resolution)
    cdef int coord_origin_z = <int>round(offset[2]/resolution)

    cdef np.ndarray eye_coord_v = (copy(eye) - offset)/0.05

    cdef np.ndarray eye_idx_v = (copy(eye)/0.05).astype(int)
    eye_idx_v[0] -= coord_origin_x
    eye_idx_v[1] -= coord_origin_y
    eye_idx_v[2] -= coord_origin_z

    cdef np.ndarray current_position = copy(eye)
    cdef np.ndarray current_idx = copy(eye_idx_v)

    cdef np.ndarray delta_dist = np.zeros((3, ))

    cdef np.ndarray next = np.zeros((3, ))
    cdef np.ndarray nextN = np.zeros((3, ))
    cdef np.ndarray step = np.zeros((3, ))
    cdef np.ndarray tDelta = np.zeros((3, ))

    # for each dim
    for i in range(0, 3):

        delta = direction[i]
        delta = 0.05*1./delta

        if delta < 0.:
            # step in negative direction
            step[i] = -1
            next[i] = (current_idx[i] - eye_coord_v[i]) * delta
            nextN[i] = (eye_coord_v[i] - current_idx[i] - 1.) * delta
            tDelta[i] = -delta
        else:
            step[i] = 1
            next[i] = (current_idx[i] + 1. - eye_coord_v[i]) * delta
            nextN[i] = delta*(eye_coord_v[i] - current_idx[i])
            tDelta[i] = delta

        #print(i, step[i], delta, next[i], current_idx[i], eye_coord_v[i])

    cdef int previous_voxel
    cdef float current_voxel
    cdef int side

    previous_voxel = 0

    while True:

        side = 0

        for i in range(0, 3):
            if next[side] > next[i]:
                side = i

        next[side] += tDelta[side]
        current_idx[side] += step[side]
        current_position[side] += step[side]*resolution

        if (current_idx[side] < 0) or (current_idx[side] >= volume.shape[side]):
            break

        current_voxel = volume[current_idx[0], current_idx[1], current_idx[2], 0]

        if current_voxel == 1:
            return np.linalg.norm(current_position - eye)

        #if previous_voxel == 1 and current_voxel > 0.:
            #return np.linalg.norm(current_position - eye)

        #if current_voxel < 0. or volume[current_coord[0], current_coord[1], current_coord[2], -1] < 0:
            #previous_voxel = 1

    return 0.



def depth_rendering(np.float32_t[:, :] extrinsics, np.ndarray intrinsics,
                    shape, np.float32_t[:, :, :, ::1] volume, float resolution, np.ndarray offset):

    cdef int width = shape[1]
    cdef int height = shape[0]

    cdef np.ndarray upper_left_p = np.asarray([0, 0]).astype(np.float32)
    cdef np.ndarray upper_right_p = np.asarray([width, 0]).astype(np.float32)
    cdef np.ndarray lower_left_p = np.asarray([0, height]).astype(np.float32)
    cdef np.ndarray lower_right_p = np.asarray([width, height]).astype(np.float32)

    cdef float image_plane = 1

    upper_left_w = pixel_to_world_coord(upper_left_p, intrinsics, extrinsics, image_plane)
    upper_right_w = pixel_to_world_coord(upper_right_p, intrinsics, extrinsics, image_plane)
    lower_left_w = pixel_to_world_coord(lower_left_p, intrinsics, extrinsics, image_plane)
    lower_right_w = pixel_to_world_coord(lower_right_p, intrinsics, extrinsics, image_plane)

    cdef np.ndarray eye = np.zeros(4)
    eye[3] = 1
    eye = np.dot(extrinsics[:3], eye)

    # get direction vectors for all 4 image corners
    cdef np.ndarray upper_left_normal = (upper_left_w - eye)/np.linalg.norm(upper_left_w - eye)
    cdef np.ndarray upper_right_normal = (upper_right_w - eye)/np.linalg.norm(upper_right_w - eye)
    cdef np.ndarray lower_left_normal = (lower_left_w - eye)/np.linalg.norm(lower_left_w - eye)
    cdef np.ndarray lower_right_normal = (lower_right_w - eye)/np.linalg.norm(lower_right_w - eye)

    cdef float inv_height = 1./height
    cdef float inv_width = 1./width

    # delta vectors
    cdef np.ndarray left_normal_y_delta = (lower_left_normal - upper_left_normal) * inv_height
    cdef np.ndarray right_normal_y_delta = (lower_right_normal - upper_right_normal) * inv_height

    # starting point
    cdef np.ndarray left_normal = upper_left_normal
    cdef np.ndarray right_normal = upper_right_normal

    cdef np.ndarray depth
    depth = np.zeros(shape)

    cdef np.ndarray ray_direction
    cdef np.ndarray normal_x_delta

    for y in range(0, height):

        # initialize ray direction
        ray_direction = left_normal
        normal_x_delta = (right_normal - left_normal) * inv_width

        for x in range(0, width):

            # normalize ray direction
            ray_direction = ray_direction/np.linalg.norm(ray_direction)

            # get depth value
            depth[y, x] = trace_ray(eye, ray_direction, volume, resolution, offset)

            # update ray direction
            ray_direction += normal_x_delta

        # update line endpoints
        left_normal += left_normal_y_delta
        right_normal += right_normal_y_delta

    return depth


def trace_ray(np.ndarray eye, np.ndarray direction, np.float32_t[:, :, :, ::1] volume,
              float resolution, np.ndarray offset):

    cdef int coord_origin_x = <int>round(offset[0]/resolution)
    cdef int coord_origin_y = <int>round(offset[1]/resolution)
    cdef int coord_origin_z = <int>round(offset[2]/resolution)

    cdef np.ndarray eye_coord_v = (copy(eye) - offset)/0.05

    cdef np.ndarray eye_idx_v = (copy(eye)/0.05).astype(int)
    eye_idx_v[0] -= coord_origin_x
    eye_idx_v[1] -= coord_origin_y
    eye_idx_v[2] -= coord_origin_z

    cdef np.ndarray current_position = copy(eye)
    cdef np.ndarray current_idx = copy(eye_idx_v)

    cdef np.ndarray delta_dist = np.zeros((3, ))

    cdef np.ndarray next = np.zeros((3, ))
    cdef np.ndarray nextN = np.zeros((3, ))
    cdef np.ndarray step = np.zeros((3, ))
    cdef np.ndarray tDelta = np.zeros((3, ))

    # for each dim
    for i in range(0, 3):

        delta = direction[i]
        delta = 0.05*1./delta

        if delta < 0.:
            # step in negative direction
            step[i] = -1
            next[i] = (current_idx[i] - eye_coord_v[i]) * delta
            nextN[i] = (eye_coord_v[i] - current_idx[i] - 1.) * delta
            tDelta[i] = -delta
        else:
            step[i] = 1
            next[i] = (current_idx[i] + 1. - eye_coord_v[i]) * delta
            nextN[i] = delta*(eye_coord_v[i] - current_idx[i])
            tDelta[i] = delta

        #print(i, step[i], delta, next[i], current_idx[i], eye_coord_v[i])

    cdef int previous_voxel
    cdef float current_voxel
    cdef int side

    previous_voxel = 0

    while True:

        side = 0

        for i in range(0, 3):
            if next[side] > next[i]:
                side = i

        next[side] += tDelta[side]
        current_idx[side] += step[side]
        current_position[side] += step[side]*resolution

        if (current_idx[side] < 0) or (current_idx[side] >= volume.shape[side]):
            break

        current_voxel = volume[current_idx[0], current_idx[1], current_idx[2], 0]

        if current_voxel == 1:
            return np.linalg.norm(current_position - eye)

        #if previous_voxel == 1 and current_voxel > 0.:
            #return np.linalg.norm(current_position - eye)

        #if current_voxel < 0. or volume[current_coord[0], current_coord[1], current_coord[2], -1] < 0:
            #previous_voxel = 1

    return 0.


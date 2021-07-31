#lib: boundscheck=False
#lib: initializedcheck=False
#lib: cdivision=True
from copy import copy
import numpy as np
cimport numpy as np
from libc.math cimport round, sqrt

cdef class Volume:

    cdef float[:, ::1] bbox
    cdef float resolution
    cdef float[:, :, :] volume

    cdef float resolutionx
    cdef float resolutiony
    cdef float resolutionz

    def __init__(self, bbox, resolution, volume_shape=None):

        assert resolution > 0


        self.bbox = bbox.astype(np.float32)
        self.resolution = resolution

        if volume_shape is None:
            volume_size = np.diff(bbox, axis=1)
            volume_shape = volume_size.ravel() / self.resolution
            volume_shape = np.ceil(volume_shape).astype(np.int32).tolist()
        else:
            volume_shape = volume_shape

        self.resolutionx = np.diff(bbox)[0]/volume_shape[0]
        self.resolutiony = np.diff(bbox)[1]/volume_shape[1]
        self.resolutionz = np.diff(bbox)[2]/volume_shape[2]


        self.volume = np.zeros(volume_shape, dtype=np.float32)

    def get_volume(self):
        return np.array(self.volume)

    def get_mask(self):
        return self.update_mask

    def fuse(self,
             np.float32_t[:, :] depth_proj_matrix,
             np.float32_t[:, :] depth_map):

        cdef int i, j, k
        cdef float x, y, z

        cdef float depth_proj_x, depth_proj_y, depth_proj_z
        cdef int depth_image_proj_x, depth_image_proj_y
        cdef float depth, signed_distance
        cdef int label
        cdef float label_prob

        cdef int h = depth_map.shape[0]
        cdef int w = depth_map.shape[1]

        for i in range(self.volume.shape[0]):
            x = self.bbox[0, 0] + i * self.resolutionx
            for j in range(self.volume.shape[1]):
                y = self.bbox[1, 0] + j * self.resolutiony
                for k in range(self.volume.shape[2]):
                    z = self.bbox[2, 0] + k * self.resolutiony

                    # Compute the depth of the current voxel wrt. the camera.
                    depth_proj_z = depth_proj_matrix[2, 0] * x + \
                                   depth_proj_matrix[2, 1] * y + \
                                   depth_proj_matrix[2, 2] * z + \
                                   depth_proj_matrix[2, 3]


                    # Check if voxel behind camera.
                    if depth_proj_z <= 0:
                        continue

                    # Compute pixel location of the current voxel in the image.
                    depth_proj_x = depth_proj_matrix[0, 0] * x + \
                                   depth_proj_matrix[0, 1] * y + \
                                   depth_proj_matrix[0, 2] * z + \
                                   depth_proj_matrix[0, 3]
                    depth_proj_y = depth_proj_matrix[1, 0] * x + \
                                   depth_proj_matrix[1, 1] * y + \
                                   depth_proj_matrix[1, 2] * z + \
                                   depth_proj_matrix[1, 3]

                    depth_image_proj_x = <int>round(depth_proj_x / depth_proj_z)
                    depth_image_proj_y = <int>round(depth_proj_y / depth_proj_z)

                    # Check if projection is inside image.
                    if (depth_image_proj_x < 0 or depth_image_proj_y < 0 or
                        depth_image_proj_x >= depth_map.shape[1] or
                        depth_image_proj_y >= depth_map.shape[0]):
                        continue

                    self.volume[i, j, k] += 1


cdef class TSDFVolume:

    cdef float[:, ::1] bbox
    cdef float free_space_vote
    cdef float occupied_space_vote
    cdef float resolution
    cdef float max_distance
    cdef float[:, :, :, ::1] volume
    cdef int[:, :, :] update_mask
    cdef float[:, :, :] weights


    cdef float resolutionx
    cdef float resolutiony
    cdef float resolutionz


    def __init__(self, num_labels, bbox, resolution, resolution_factor,
                 free_space_vote=0.5, occupied_space_vote=1, volume_shape=None):

        assert num_labels > 0
        assert resolution > 0
        assert resolution_factor > 0
        assert free_space_vote >= 0
        assert occupied_space_vote >= 0


        self.bbox = bbox.astype(np.float32)
        self.resolution = resolution

        self.max_distance = resolution_factor * self.resolution
        self.free_space_vote = free_space_vote
        self.occupied_space_vote = occupied_space_vote


        if volume_shape is None:
            volume_size = np.diff(bbox, axis=1)
            volume_shape = volume_size.ravel() / self.resolution
            volume_shape = np.ceil(volume_shape).astype(np.int32).tolist()
        else:
            volume_shape = volume_shape

        self.resolutionx = np.diff(bbox)[0]/volume_shape[0]
        self.resolutiony = np.diff(bbox)[1]/volume_shape[1]
        self.resolutionz = np.diff(bbox)[2]/volume_shape[2]


        self.volume = np.zeros(volume_shape + [2], dtype=np.float32)
        self.weights = np.zeros(volume_shape, dtype=np.float32)

        self.update_mask = np.zeros(volume_shape).astype(np.int32)

    def get_volume(self):
        return np.array(self.volume)

    def get_depth(self, np.float32_t[:, :] extrinsics, np.ndarray intrinsics, shape):
        cdef np.ndarray resolution = np.asarray([self.resolutionx, self.resolutiony, self.resolutionz])
        cdef np.ndarray offset = np.asarray([self.bbox[0, 0], self.bbox[1, 0], self.bbox[2, 0]], dtype=np.float32)
        return depth_rendering(extrinsics, intrinsics, shape, self.volume, resolution, offset)

    def get_mask(self):
        return self.update_mask

    def fuse(self,
             np.float32_t[:, :] depth_proj_matrix,
             np.float32_t[:, :] depth_map,
             np.float32_t[:, :] weight_map):

        cdef int i, j, k
        cdef float x, y, z
        cdef float depth_proj_x, depth_proj_y, depth_proj_z
        cdef int depth_image_proj_x, depth_image_proj_y
        cdef int label_image_proj_x, label_image_proj_y
        cdef float depth, signed_distance
        cdef int label
        cdef float label_prob

        cdef int h = depth_map.shape[0]
        cdef int w = depth_map.shape[1]

        for i in range(self.volume.shape[0]):
            x = self.bbox[0, 0] + i * self.resolution
            for j in range(self.volume.shape[1]):
                y = self.bbox[1, 0] + j * self.resolution
                for k in range(self.volume.shape[2]):
                    z = self.bbox[2, 0] + k * self.resolution

                    # Compute the depth of the current voxel wrt. the camera.
                    depth_proj_z = depth_proj_matrix[2, 0] * x + \
                                   depth_proj_matrix[2, 1] * y + \
                                   depth_proj_matrix[2, 2] * z + \
                                   depth_proj_matrix[2, 3]

                    # Check if voxel behind camera.
                    if depth_proj_z <= 0:
                        continue

                    # Compute pixel location of the current voxel in the image.
                    depth_proj_x = depth_proj_matrix[0, 0] * x + \
                                   depth_proj_matrix[0, 1] * y + \
                                   depth_proj_matrix[0, 2] * z + \
                                   depth_proj_matrix[0, 3]
                    depth_proj_y = depth_proj_matrix[1, 0] * x + \
                                   depth_proj_matrix[1, 1] * y + \
                                   depth_proj_matrix[1, 2] * z + \
                                   depth_proj_matrix[1, 3]

                    depth_image_proj_x = <int>round(depth_proj_x / depth_proj_z)
                    depth_image_proj_y = <int>round(depth_proj_y / depth_proj_z)

                    # Check if projection is inside image.
                    if (depth_image_proj_x < 0 or depth_image_proj_y < 0 or
                        depth_image_proj_x >= depth_map.shape[1] or
                        depth_image_proj_y >= depth_map.shape[0]):
                        continue

                    # Extract measured depth at projection.
                    depth = depth_map[depth_image_proj_y, depth_image_proj_x]

                    if depth == 0.:
                        continue

                    weight = weight_map[depth_image_proj_y, depth_image_proj_x]

                    # Check if voxel is inside the truncated distance field.
                    signed_distance = depth - depth_proj_z

                    if abs(signed_distance) > self.max_distance:
                        # Check if voxel is between observed depth and camera.
                        if signed_distance > 0:

                            if self.volume[i, j, k, -1] == 10.e7:
                                self.volume[i, j, k, -1] = -self.free_space_vote
                            else:
                                # Vote for free space.
                                self.volume[i, j, k, -1] -= self.free_space_vote
                        continue

                    self.update_mask[i, j, k] += 1


                    weight_old = self.weights[i, j, k]
                    value_old = self.volume[i, j, k, 0]
                    new_value = weight_old*value_old + weight*signed_distance
                    new_value = new_value/(weight_old + weight)
                    weight_new = weight_old + weight

                    self.volume[i, j, k, 0] = copy(new_value)
                    self.weights[i, j, k] = copy(weight_new)

    def sanity_fuse(self,
                    np.float32_t[:, :] depth_proj_matrix,
                    np.float32_t[:, :] depth_map,
                    np.float32_t[:, :] weight_map):

        cdef int i, j, k
        cdef float x, y, z
        cdef float depth_proj_x, depth_proj_y, depth_proj_z
        cdef int depth_image_proj_x, depth_image_proj_y
        cdef int label_image_proj_x, label_image_proj_y
        cdef float depth, signed_distance
        cdef int label
        cdef float label_prob

        cdef int h = depth_map.shape[0]
        cdef int w = depth_map.shape[1]

        for i in range(self.volume.shape[0]):
            x = self.bbox[0, 0] + i * self.resolutionx
            for j in range(self.volume.shape[1]):
                y = self.bbox[1, 0] + j * self.resolutiony
                for k in range(self.volume.shape[2]):
                    z = self.bbox[2, 0] + k * self.resolutionz

                    # Compute the depth of the current voxel wrt. the camera.
                    depth_proj_z = depth_proj_matrix[2, 0] * x + \
                                   depth_proj_matrix[2, 1] * y + \
                                   depth_proj_matrix[2, 2] * z + \
                                   depth_proj_matrix[2, 3]


                    # Check if voxel behind camera.
                    if depth_proj_z <= 0:
                        continue

                    # Compute pixel location of the current voxel in the image.
                    depth_proj_x = depth_proj_matrix[0, 0] * x + \
                                   depth_proj_matrix[0, 1] * y + \
                                   depth_proj_matrix[0, 2] * z + \
                                   depth_proj_matrix[0, 3]
                    depth_proj_y = depth_proj_matrix[1, 0] * x + \
                                   depth_proj_matrix[1, 1] * y + \
                                   depth_proj_matrix[1, 2] * z + \
                                   depth_proj_matrix[1, 3]

                    depth_image_proj_x = <int>round(depth_proj_x / depth_proj_z)
                    depth_image_proj_y = <int>round(depth_proj_y / depth_proj_z)

                    # Check if projection is inside image.
                    if (depth_image_proj_x < 0 or depth_image_proj_y < 0 or
                        depth_image_proj_x >= depth_map.shape[1] or
                        depth_image_proj_y >= depth_map.shape[0]):
                        continue

                    # Extract measured depth at projection.
                    depth = depth_map[depth_image_proj_y, depth_image_proj_x]

                    if depth > depth_proj_z + self.resolutionz or depth < depth_proj_z - self.resolutionz:
                        continue

                    self.volume[i, j, k, 0] = 1


cdef class MulticlassTSDFVolume:

    cdef float[:, ::1] bbox
    cdef float free_space_vote
    cdef float occupied_space_vote
    cdef float resolution
    cdef float max_distance
    cdef float[:, :, :, ::1] volume

    def __init__(self, num_labels, bbox, resolution, resolution_factor,
                 free_space_vote=0.5, occupied_space_vote=1):
        assert num_labels > 0
        assert resolution > 0
        assert resolution_factor > 0
        assert free_space_vote >= 0
        assert occupied_space_vote >= 0

        self.bbox = bbox.astype(np.float32)
        self.resolution = resolution
        self.max_distance = resolution_factor * self.resolution
        self.free_space_vote = free_space_vote
        self.occupied_space_vote = occupied_space_vote

        volume_size = np.diff(bbox, axis=1)
        volume_shape = volume_size.ravel() / self.resolution
        volume_shape = np.ceil(volume_shape).astype(np.int32).tolist()
        self.volume = np.zeros(volume_shape + [num_labels + 1],
                               dtype=np.float32)

    def get_volume(self):
        return np.array(self.volume)

    def fuse(self,
             np.float32_t[:, ::1] depth_proj_matrix,
             np.float32_t[:, ::1] label_proj_matrix,
             np.float32_t[:, ::1] depth_map,
             np.float32_t[:, :, ::1] label_map):
        assert label_map.shape[2] == self.volume.shape[3] - 1

        cdef int i, j, k
        cdef float x, y, z
        cdef float depth_proj_x, depth_proj_y, depth_proj_z
        cdef float label_proj_x, label_proj_y, label_proj_z
        cdef int depth_image_proj_x, depth_image_proj_y
        cdef int label_image_proj_x, label_image_proj_y
        cdef float depth, signed_distance
        cdef int label
        cdef float label_prob

        for i in range(self.volume.shape[0]):
            x = self.bbox[0, 0] + i * self.resolution
            for j in range(self.volume.shape[1]):
                y = self.bbox[1, 0] + j * self.resolution
                for k in range(self.volume.shape[2]):
                    z = self.bbox[2, 0] + k * self.resolution

                    # Compute the depth of the current voxel wrt. the camera.
                    depth_proj_z = depth_proj_matrix[2, 0] * x + \
                                   depth_proj_matrix[2, 1] * y + \
                                   depth_proj_matrix[2, 2] * z + \
                                   depth_proj_matrix[2, 3]
                    label_proj_z = label_proj_matrix[2, 0] * x + \
                                   label_proj_matrix[2, 1] * y + \
                                   label_proj_matrix[2, 2] * z + \
                                   label_proj_matrix[2, 3]

                    # Check if voxel behind camera.
                    if depth_proj_z <= 0 or label_proj_z <= 0:
                        continue

                    # Compute pixel location of the current voxel in the image.
                    depth_proj_x = depth_proj_matrix[0, 0] * x + \
                                   depth_proj_matrix[0, 1] * y + \
                                   depth_proj_matrix[0, 2] * z + \
                                   depth_proj_matrix[0, 3]
                    depth_proj_y = depth_proj_matrix[1, 0] * x + \
                                   depth_proj_matrix[1, 1] * y + \
                                   depth_proj_matrix[1, 2] * z + \
                                   depth_proj_matrix[1, 3]
                    label_proj_x = label_proj_matrix[0, 0] * x + \
                                   label_proj_matrix[0, 1] * y + \
                                   label_proj_matrix[0, 2] * z + \
                                   label_proj_matrix[0, 3]
                    label_proj_y = label_proj_matrix[1, 0] * x + \
                                   label_proj_matrix[1, 1] * y + \
                                   label_proj_matrix[1, 2] * z + \
                                   label_proj_matrix[1, 3]
                    depth_image_proj_x = <int>round(depth_proj_x / depth_proj_z)
                    depth_image_proj_y = <int>round(depth_proj_y / depth_proj_z)
                    label_image_proj_x = <int>round(label_proj_x / label_proj_z)
                    label_image_proj_y = <int>round(label_proj_y / label_proj_z)

                    # Check if projection is inside image.
                    if (depth_image_proj_x < 0 or depth_image_proj_y < 0 or
                        depth_image_proj_x >= depth_map.shape[1] or
                        depth_image_proj_y >= depth_map.shape[0] or
                        label_image_proj_x < 0 or label_image_proj_y < 0 or
                        label_image_proj_x >= label_map.shape[1] or
                        label_image_proj_y >= label_map.shape[0]):
                        continue

                    # Extract measured depth at projection.
                    depth = depth_map[depth_image_proj_y, depth_image_proj_x]

                    if depth == 0.:
                        continue

                    # Check if voxel is inside the truncated distance field.
                    signed_distance = depth - depth_proj_z
                    if abs(signed_distance) > self.max_distance:
                        # Check if voxel is between observed depth and camera.
                        if signed_distance > 0:
                            # Vote for free space.
                            self.volume[i, j, k, -1] -= self.free_space_vote
                        continue

                    # Accumulate the votes for each label.
                    for label in range(label_map.shape[2]):
                        label_prob = label_map[label_image_proj_y,
                                               label_image_proj_x,
                                               label]
                        if signed_distance < 0:
                            self.volume[i, j, k, label] -= \
                                label_prob * self.occupied_space_vote
                        else:
                            self.volume[i, j, k, label] += \
                                label_prob * self.occupied_space_vote

def depth_rendering(np.float32_t[:, :] extrinsics,
                    np.ndarray intrinsics,
                    shape,
                    np.float32_t[:, :, :, ::1] volume,
                    double resolution,
                    np.ndarray offset,
                    np.ndarray frame):

    cdef int width = shape[1]
    cdef int height = shape[0]

    cdef float image_plane = 1.

    cdef np.ndarray point_to
    cdef np.ndarray coords_p

    cdef np.ndarray eye = np.zeros(4)
    eye[3] = 1
    eye = np.dot(extrinsics[:3], eye)

    cdef np.ndarray ray_direction
    cdef np.ndarray depth
    depth = np.zeros((height, width))

    cdef float px = intrinsics[0, 2]
    cdef float py = intrinsics[1, 2]

    coords_eye = np.asarray([py, px]).astype(np.float32)
    eye = pixel_to_world_coord(coords_eye, intrinsics, extrinsics, 0.)

    cdef np.ndarray proj = np.linalg.inv(extrinsics)
    cdef p_homogenuous = np.ones((4, ))



    for x in range(0, width):
        for y in range(0, height):

            coords_p = np.asarray([x, y]).astype(np.float32)
            point_to = pixel_to_world_coord(coords_p, intrinsics,
                                            extrinsics, frame[y, x])

            point_from = (eye - offset)/resolution
            point_to = (point_to - offset)/resolution

            if np.linalg.norm(point_from - point_to) == 0.:
                continue

            p = trace_ray(point_from, point_to, volume, resolution)
            p = np.asarray(p)

            p = p*resolution + offset

            p_homogenuous[0] = p[0]
            p_homogenuous[1] = p[1]
            p_homogenuous[2] = p[2]

            pc = np.dot(proj[:3], p_homogenuous)

            depth[y, x] = pc[2]
            # update ray direction

    return depth


def trace_ray_by_point(np.ndarray point_from, np.ndarray point_to,
                       np.float32_t[:, :, :, ::1] volume,
                       float resolution, np.ndarray offset):

    cdef np.ndarray current_idx = np.floor(point_from).astype(int)
    cdef np.ndarray current_pos = np.copy(point_from)
    cdef np.ndarray start = np.copy(point_from)

    cdef np.ndarray direction = np.zeros((3, ))
    cdef np.ndarray next = np.zeros((3, ))
    cdef np.ndarray nextN = np.zeros((3, ))
    cdef np.ndarray step = np.zeros((3, ))
    cdef np.ndarray tDelta = np.zeros((3, ))

    direction[0] = point_to[0] - point_from[0]
    direction[1] = point_to[1] - point_from[1]
    direction[2] = point_to[2] - point_from[2]
    direction = direction/np.linalg.norm(direction)

    # for each dim
    for i in range(0, 3):

        delta = 1./(direction[i] + 10.e-8)

        if delta < 0.:
            # step in negative direction
            step[i] = -1
            next[i] = (current_idx[i] - point_from[i]) * delta
            nextN[i] = (point_from[i] - current_idx[i] - 1.) * delta
            tDelta[i] = -copy(delta)
        else:
            step[i] = 1
            next[i] = (current_idx[i] + 1. - point_from[i]) * delta
            nextN[i] = delta*(point_from[i] - current_idx[i])
            tDelta[i] = copy(delta)

        #print(i, step[i], delta, next[i], current_idx[i], eye_coord_v[i])

    cdef int previous_voxel
    cdef float current_voxel
    cdef int side

    previous_voxel = 0
    depth = 0.

    while True:

        side = 0

        for i in range(0, 3):
            if next[side] > next[i]:
                side = i

        t = copy(next[side])
        next[side] += copy(tDelta[side])
        current_idx[side] += copy(step[side])

        # TODO: correct this line of code
        current_pos = start + t*direction
        current_pos[side] += 0.001

        # if np.linalg.norm(current_pos - point_to) < 0.001:
        #     return current_pos

        #print(side)
        if (current_idx[side] < 0) or (current_idx[side] >= volume.shape[side]):
            break

        current_voxel = volume[current_idx[0], current_idx[1], current_idx[2], 0]

        # volume_ref[current_idx[0], current_idx[1], current_idx[2]] = 1.

        if current_voxel != 0.:
            return current_pos
            #return depth

        #if previous_voxel == 1 and current_voxel > 0.:
            #return np.linalg.norm(current_position - eye)

        #if current_voxel < 0. or volume[current_coord[0], current_coord[1], current_coord[2], -1] < 0:
            #previous_voxel = 1

    return point_from


def trace_ray(np.float64_t[:] point_from, np.float64_t[:] point_to,
              np.float32_t[:, :, :, ::1] volume, float resolution):

    cdef double current_idx[3]

    current_idx[0] = <double>round(point_from[0])
    current_idx[1] = <double>round(point_from[1])
    current_idx[2] = <double>round(point_from[2])

    cdef double current_pos[3]
    current_pos[0] = copy(point_from[0])
    current_pos[1] = copy(point_from[1])
    current_pos[2] = copy(point_from[2])

    cdef double start[3]
    start = copy(current_pos)

    cdef double direction[3]
    cdef double next[3]
    cdef double nextN[3]
    cdef double step[3]
    cdef double tDelta[3]

    direction[0] = point_to[0] - point_from[0]
    direction[1] = point_to[1] - point_from[1]
    direction[2] = point_to[2] - point_from[2]

    cdef double norm = 0
    for i in range(0, 3):
        norm += direction[i]**2
    norm = sqrt(norm)

    if norm < 0.0001:
        print(direction)
        print(norm)

    for i in range(0, 3):
        direction[i] = direction[i]/norm

    cdef double delta

    # for each dim
    for i in range(0, 3):

        delta = 1./(direction[i] + 1.e-08)

        if delta < 0.:
            # step in negative direction
            step[i] = -1.
            next[i] = (current_idx[i] - point_from[i]) * delta
            nextN[i] = (point_from[i] - current_idx[i] - 1.) * delta
            tDelta[i] = -copy(delta)

        else:
            step[i] = 1.
            next[i] = (current_idx[i] + 1. - point_from[i]) * delta
            nextN[i] = delta*(point_from[i] - current_idx[i])
            tDelta[i] = copy(delta)

        #print(i, step[i], delta, next[i], current_idx[i], eye_coord_v[i])

    cdef int previous_voxel
    cdef float current_voxel
    cdef int side

    previous_voxel = 0
    depth = 0.

    while True:

        side = 0

        for i in range(0, 3):
            if next[side] > next[i]:
                side = i

        t = copy(next[side])
        next[side] += copy(tDelta[side])
        current_idx[side] += copy(step[side])

        # TODO: correct this line of code
        current_pos[0] = start[0] + t*direction[0]
        current_pos[1] = start[1] + t*direction[1]
        current_pos[2] = start[2] + t*direction[2]

        current_pos[side] += 0.001

        # if np.linalg.norm(current_pos - point_to) < 0.001:
        #     return current_pos


        #print(side)
        if (<int>round(current_idx[0]) < 0) or \
                (<int>round(current_idx[1]) < 0) or \
                (<int>round(current_idx[2]) < 0) or \
                (<int>round(current_idx[0]) >= volume.shape[0]) or \
                (<int>round(current_idx[1]) >= volume.shape[1]) or \
                (<int>round(current_idx[2]) >= volume.shape[2]):
            break

        current_voxel = volume[<int>round(current_idx[0]),
                               <int>round(current_idx[1]),
                               <int>round(current_idx[2]), 0]

        # volume_ref[current_idx[0], current_idx[1], current_idx[2]] = 1.

        if current_voxel != 0.:
            return current_pos
            #return depth

        #if previous_voxel == 1 and current_voxel > 0.:
            #return np.linalg.norm(current_position - eye)

        #if current_voxel < 0. or volume[current_coord[0], current_coord[1], current_coord[2], -1] < 0:
            #previous_voxel = 1

    return point_from


def pixel_to_camera_coord(np.float32_t[:] pixel_index,
                          np.float32_t[:, :] intrinsics,
                          float plane_distance):

    cdef np.ndarray camera_coord = np.zeros(3, )

    camera_coord[2] = plane_distance
    camera_coord[1] = plane_distance * (pixel_index[1] - intrinsics[1, 2])/intrinsics[1, 1]
    camera_coord[0] = plane_distance * (pixel_index[0] - intrinsics[0, 1]*camera_coord[1] - intrinsics[0, 2])/intrinsics[0, 0]

    return camera_coord


def pixel_to_world_coord(np.float32_t[:] pixel_index,
                         np.float32_t[:, :] intrinsics,
                         np.float32_t[:, :] extrinsics,
                         float plane_distance):

    camera_coord = pixel_to_camera_coord(pixel_index, intrinsics, plane_distance)
    world_coord = np.dot(extrinsics[:3], np.concatenate((camera_coord, np.ones((1, )))))
    return world_coord

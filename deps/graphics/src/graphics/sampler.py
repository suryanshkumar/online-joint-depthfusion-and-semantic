import numpy as np

from tqdm import tqdm

def voxelize(mesh):
    raise NotImplementedError


def mesh_to_pointcloud(vertices, faces, npoints):

    areas = []

    for face in faces:

        p1 = vertices[face[0]]
        p2 = vertices[face[1]]
        p3 = vertices[face[2]]

        v1 = p2 - p1
        v2 = p3 - p1

        areas.append(compute_area(v1, v2))

    areas = np.asarray(areas)
    probabilities = areas/np.sum(areas)

    print(probabilities)

    face_indices = np.argsort(probabilities)[::-1]
    probabilities = probabilities[face_indices]

    print(probabilities)

    points = []

    for i in tqdm(range(0, npoints), total=npoints):

        p_triangle = np.random.uniform(0, 1)
        p_index = find_index(p_triangle, probabilities)
        face_idx = face_indices[p_index]

        p1 = vertices[faces[face_idx][0]]
        p2 = vertices[faces[face_idx][1]]
        p3 = vertices[faces[face_idx][2]]

        points.append(sample_point_from_triangle(p1, p2, p3))

    return np.asarray(points)


def find_index(p, probabilities):

    prob_sum = 0

    for idx, prob in enumerate(probabilities):

        prob_sum += prob

        if p < prob_sum:
            return idx


def compute_area(v1, v2):
    return np.linalg.norm(np.cross(v1, v2))/2


def sample_point_from_triangle(p1, p2, p3):

    a = np.random.uniform(0, 1)
    b = np.random.uniform(0, 1)

    x = p1 + a*(p2 - p1) + b*(p3 - p1)

    return x





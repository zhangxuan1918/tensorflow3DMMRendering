from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def deg2rad(x):
    return 3.1415926535897931 * x / 180


def angle2matrix(angles):
    """
    get rotation matrix from three rotation angles(degree). right-handed.

    :params angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left.
        z: roll. positive for tilting head right.
    :return rotate_matrix: [3, 3]. rotation matrix.
    """

    x, y, z = deg2rad(angles[0]), deg2rad(angles[1]), deg2rad(angles[2])

    # x
    rotate_x = tf.constant(
        [
            [1, 0, 0],
            [0, tf.cos(x), -tf.sin(x)],
            [0, tf.sin(x), tf.cos(x)]
        ],
        shape=(3, 3),
        dtype=tf.float32
    )
    # y
    rotate_y = tf.constant(
        [
            [tf.cos(y), 0, tf.sin(y)],
            [0, 1, 0],
            [-tf.sin(y), 0, tf.cos(y)]
        ],
        shape=(3, 3),
        dtype=tf.float32
    )
    # z
    rotate_z = tf.constant(
        [
            [tf.cos(z), -tf.sin(z), 0],
            [tf.sin(z), tf.cos(z), 0],
            [0, 0, 1]
        ],
        shape=(3, 3),
        dtype=tf.float32
    )

    rotate_matrix = tf.linalg.matmul(rotate_z, tf.linalg.matmul(rotate_y, rotate_x))
    return rotate_matrix


def angle2matrix_3ddfa(angles):
    """
    get rotation matrix from three rotation angles(radian). The same as in 3DDFA.

    :param angles: [3,]. x, y, z angles
        x: pitch.
        y: yaw.
        z: roll.
    :return rotate_matrix: 3x3. rotation matrix.
    """
    # x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    x, y, z = angles[0], angles[1], angles[2]

    # x
    rotate_x = tf.constant(
        [
            [1, 0, 0],
            [0, tf.cos(x), -tf.sin(x)],
            [0, tf.sin(x), tf.cos(x)]
        ],
        shape=(3, 3),
        dtype=tf.float32
    )
    # y
    rotate_y = tf.constant(
        [
            [tf.cos(y), 0, tf.sin(y)],
            [0, 1, 0],
            [-tf.sin(y), 0, tf.cos(y)]
        ],
        shape=(3, 3),
        dtype=tf.float32
    )
    # z
    rotate_z = tf.constant(
        [
            [tf.cos(z), -tf.sin(z), 0],
            [tf.sin(z), tf.cos(z), 0],
            [0, 0, 1]
        ],
        shape=(3, 3),
        dtype=tf.float32
    )

    rotate_matrix = tf.linalg.matmul(rotate_x, tf.linalg.matmul(rotate_y, rotate_z))
    return rotate_matrix


def rotate(vertices, angles):
    """
    rotate vertices.
    X_new = R.dot(X). X: 3 x 1
    :param vertices: [nver, 3]
        rx, ry, rz: degree angles
            rx: pitch. positive for looking down
            ry: yaw. positive for looking left
            rz: roll. positive for tilting head right
    :return rotated vertices: [nver, 3]
    """
    rotate_matrix = angle2matrix(angles)
    rotated_vertices = tf.linalg.matmul(vertices, tf.linalg.matrix_transpose(rotate_matrix))

    return rotated_vertices


def similarity_transform(vertices, scaling, rotate_matrix, t3d):
    """
    similarity transform. dof = 7.
    3D: s*R.dot(X) + t
    Homo: M = [[sR, t],[0^T, 1]].  M.dot(X)

    :param vertices:      [nver, 3]
    :param scaling:       [1,     ] scale factor.
    :param rotate_matrix: [3,    3] rotation matrix.
    :param t3d:           [3,     ] 3d translation vector.

    :return transformed vertices: [nver, 3]
    """
    t3d = tf.squeeze(t3d)
    transformed_vertices = scaling * tf.linalg.matmul(vertices,
                                                      tf.linalg.matrix_transpose(rotate_matrix)) + tf.expand_dims(t3d,
                                                                                                                  0)

    return transformed_vertices


def normalize(x):
    epsilon = 1e-12
    norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=0))
    norm = tf.math.maximum(norm, epsilon)
    return tf.math.divide(x, norm)


def lookat_camera(vertices, eye, at=None, up=None):
    """
    'look at' transformation: from world space to camera space
    standard camera space: 
        camera located at the origin. 
        looking down negative z-axis. 
        vertical vector is y-axis.
    Xcam = R(X - C)
    Homo: [[R, -RC], [0, 1]]
    :param vertices: [nver, 3]
    :param eye:      [3,     ] the XYZ world space position of the camera.
    :param at:       [3,     ] a position along the center of the camera's gaze.
    :param up:       [3,     ] up direction
    :return transformed_vertices: [nver, 3]
    """
    if at is None:
        at = tf.constant([0, 0, 0], tf.float32)
    if up is None:
        up = tf.constant([0, 1, 0], tf.float32)

    z_axis = -normalize(at - eye)  # look forward
    x_axis = normalize(tf.linalg.cross(up, z_axis))  # look right
    y_axis = tf.linalg.cross(z_axis, x_axis)  # look up

    rotate_matrix = tf.stack([x_axis, y_axis, z_axis])  # , axis = 0) # 3 x 3
    transformed_vertices = vertices - eye  # translation
    transformed_vertices = tf.math.multiply(transformed_vertices, tf.linalg.matrix_transpose(rotate_matrix))  # rotation
    return transformed_vertices


def perspective_project(vertices, fovy, aspect_ratio=1., near=0.1, far=1000.):
    """
    perspective projection.

    :param vertices: [nver, 3]
    :param fovy: vertical angular field of view. degree.
    :param aspect_ratio : width / height of field of view
    :param near : depth of near clipping plane
    :param far : depth of far clipping plane
    :return projected_vertices: [nver, 3]
    """
    fovy = deg2rad(fovy)
    top = near * tf.math.tan(fovy)
    right = top * aspect_ratio

    # -- homo
    P = tf.constant(
        [
            [near / right, 0, 0, 0],
            [0, near / top, 0, 0],
            [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, -1, 0]
        ],
        shape=(4, 4),
        dtype=tf.float32
    )
    # create homogeneous vertices: shape [nver, 3] to [nver, 4]
    # last column has value 1
    vertices_homo = tf.concat([vertices, tf.expand_dims(tf.ones(tf.shape(vertices)[0]), 1)], 1)  # [nver, 4]
    projected_vertices = tf.linalg.matmul(vertices_homo, tf.linalg.matrix_transpose(P.T))
    # prospective projection
    projected_vertices = tf.math.divide(projected_vertices, projected_vertices[:, 3:])
    # ignore z coordinate
    projected_vertices = projected_vertices[:, :3]
    # invert y coordinate
    projected_vertices = tf.stack([projected_vertices[:, 0], -projected_vertices[:, 1]], 1)

    # -- non homo. only fovy
    # projected_vertices = vertices.copy()
    # projected_vertices[:,0] = -(near/right)*vertices[:,0]/vertices[:,2]
    # projected_vertices[:,1] = -(near/top)*vertices[:,1]/vertices[:,2]
    return projected_vertices


def to_image(vertices, h, w, is_perspective=False):
    """
    change vertices to image coord system
    3d system: XYZ, center(0, 0, 0)
    2d image: x(u), y(v). center(w/2, h/2), flip y-axis.

    :param vertices: [nver, 3]
    :param h: height of the rendering
    :param w : width of the rendering
    :return projected_vertices: [nver, 3]
    """
    if is_perspective:
        image_x = vertices[:, 0] * w / 2
        image_y = vertices[:, 1] * h / 2
    else:
        image_x = vertices[:, 0]
        image_y = vertices[:, 1]

    image_y = h - image_y - 1
    return tf.stack([image_x, image_y], 1)

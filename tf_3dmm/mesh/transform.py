from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf_constant_pi = tf.constant(np.pi)


def degree_to_rad(x):
    return x * tf_constant_pi / 180.


def angle_to_matrix(angles_rad, dtype=tf.float32):
    """
    get rotation matrix from three rotation angles(radian) as in 3DDFA.

    :param: angles_rad: [3,]. x, y, z angles
        x: pitch.
        y: yaw.
        z: roll.
    :return:
        R: 3x3. rotation matrix.
    """

    x, y, z = angles_rad[0], angles_rad[1], angles_rad[2]
    # rotation matrix on x axis
    rotate_x = tf.constant(
        [
            [1, 0, 0],
            [0, tf.math.cos(x), -tf.math.sin(x)],
            [0, tf.math.sin(x), tf.math.cos(x)]
        ],
        dtype=dtype
    )

    # rotation matrix on y axis
    rotate_y = tf.constant(
        [
            [tf.math.cos(y), 0, tf.math.sin(y)],
            [0, 1, 0],
            [-tf.math.sin(y), 0, tf.math.cos(y)]
        ],
        dtype=dtype
    )

    # rotation matrix on z axis
    rotate_z = tf.constant(
        [
            [tf.math.cos(z), -tf.math.sin(z), 0],
            [tf.math.sin(z), tf.math.cos(z), 0],
            [0, 0, 1]
        ],
        dtype=dtype
    )

    # rotation matrix
    rotate_xyz = tf.linalg.matmul(rotate_z, tf.linalg.matmul(rotate_y, rotate_x))

    return rotate_xyz


def rotate(vertices, angles_rad):
    """
    rotate vertices by angles specified in `angles`

    :param: vertices: [nver, 3].
    :param: angles_rad: angles in rad
        x: pitch. positive for looking down
        y: yaw. positive for looking left
        z: roll. positive for tilting head right
    :return:
        rotated vertices: [nver, 3]
    """

    rotate_matrix = angle_to_matrix(angles_rad)
    rotated_vertices = tf.linalg.matmul(vertices, tf.transpose(rotate_matrix))

    tf.debugging.assert_shapes(vertices.shape, rotated_vertices)

    return rotated_vertices


def affine_transform(vertices, scaling, angles_rad, t3d):
    """
    affine transformation in 3d

    in 3dmm, this is also the weak projection before rescaled to clip space

    s*R.dot(X) + t

    :param: vertices: [nver, 3].
        scaling: float
        angles_rad: [3, ], rotation degrees
            x: pitch. positive for looking down
            y: yaw. positive for looking left
            z: roll. positive for tilting head right
        t3d: [3, 1], 3d translation vector.
    :return:
        transformed vertices: [nver, 3]
    """

    transformed_vertices = scaling * rotate(vertices=vertices, angles_rad=angles_rad) + t3d

    return transformed_vertices

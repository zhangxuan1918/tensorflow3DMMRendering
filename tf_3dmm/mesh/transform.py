from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_graphics as tfg

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
    rotate_x = tf.Variable(
        [
            [1, 0, 0],
            [0, tf.math.cos(x), tf.math.sin(x)],
            [0, -tf.math.sin(x), tf.math.cos(x)]
        ],
        dtype=dtype
    )

    # rotation matrix on y axis
    rotate_y = tf.Variable(
        [
            [tf.math.cos(y), 0, -tf.math.sin(y)],
            [0, 1, 0],
            [tf.math.sin(y), 0, tf.math.cos(y)]
        ],
        dtype=dtype
    )

    # rotation matrix on z axis
    rotate_z = tf.Variable(
        [
            [tf.math.cos(z), tf.math.sin(z), 0],
            [-tf.math.sin(z), tf.math.cos(z), 0],
            [0, 0, 1]
        ],
        dtype=dtype
    )

    # rotation matrix
    # rotate_xyz = tf.linalg.matmul(rotate_z, tf.linalg.matmul(rotate_y, rotate_x))
    rotate_xyz = tf.linalg.matmul(tf.linalg.matmul(rotate_x, rotate_y), rotate_z)
    return rotate_xyz


def angle_to_matrix_batch(angles_rad, dtype=tf.float32):
    """
    get rotation matrix from three rotation angles(radian) as in 3DDFA.

    :param: angles_rad: [batch, 1, 3]. x, y, z angles
        x: pitch.
        y: yaw.
        z: roll.
    :return: R: [batch, 3, 3]. rotation matrix.
    """

    # x, y, z, (batch, 1)
    x, y, z = angles_rad[:, :, 0], angles_rad[:, :, 1], angles_rad[:, :, 2]

    # rotation matrix on x axis
    rotate_x = tftf.stack([tf.math.cos(x), -tf.math.sin(x)], axis=1), tf.stack([tf.math.sin(x), tf.math.cos(x)], axis=1)
    rotate_x = tf.Variable(
        [
            [1, 0, 0],
            [0, tf.math.cos(x), tf.math.sin(x)],
            [0, -tf.math.sin(x), tf.math.cos(x)]
        ],
        dtype=dtype
    )

    # rotation matrix on y axis
    rotate_y = tf.Variable(
        [
            [tf.math.cos(y), 0, -tf.math.sin(y)],
            [0, 1, 0],
            [tf.math.sin(y), 0, tf.math.cos(y)]
        ],
        dtype=dtype
    )

    # rotation matrix on z axis
    rotate_z = tf.Variable(
        [
            [tf.math.cos(z), tf.math.sin(z), 0],
            [-tf.math.sin(z), tf.math.cos(z), 0],
            [0, 0, 1]
        ],
        dtype=dtype
    )

    # rotation matrix
    # rotate_xyz = tf.linalg.matmul(rotate_z, tf.linalg.matmul(rotate_y, rotate_x))
    rotate_xyz = tf.linalg.matmul(tf.linalg.matmul(rotate_x, rotate_y), rotate_z)
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

    tf.debugging.assert_shapes([(rotated_vertices, vertices.shape)])

    return rotated_vertices


def rotate_batch(vertices, angles_rad):
    rotate_matrix = tfg.geometry.transformation.rotation_matrix_3d.from_euler(
        angles=angles_rad
    )
    rotated_vertices = tf.einsum('ijk,iks->ijs', vertices, rotate_matrix)
    tf.debugging.assert_shapes([(rotated_vertices, vertices.shape)])
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


def affine_transform_batch(vertices, scaling, angles_rad, t3d):
    """
        affine transformation in 3d

        in 3dmm, this is also the weak projection before rescaled to clip space

        s*R.dot(X) + t

        :param: vertices: [batch, n_ver, 3].
        :param: scaling: [batch, 1]
        :param: angles_rad: [batch, 3], rotation
            angles_rad[:, 0]: pitch. positive for looking down
            angles_rad[:, 1]: yaw. positive for looking left
            angles_rad[:, 2]: roll. positive for tilting head right
        :param: t3d: [batch, 3], 3d translation vector.
        :return:
            transformed vertices: [batch, n_ver, 3]
    """
    transformed_vertices = scaling * rotate_batch(vertices=vertices, angles_rad=angles_rad) + t3d

    return transformed_vertices



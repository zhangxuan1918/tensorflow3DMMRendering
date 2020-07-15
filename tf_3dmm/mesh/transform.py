from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_graphics as tfg


def rotate(vertices, angles_rad):
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
    transformed_vertices = tf.expand_dims(scaling, axis=2) * rotate(vertices=vertices,
                                                                    angles_rad=angles_rad) + t3d

    return transformed_vertices
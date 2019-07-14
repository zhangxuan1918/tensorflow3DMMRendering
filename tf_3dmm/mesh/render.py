# This demonstrates using Dirt for textured rendering with a UV map

import dirt
import imageio
import numpy as np
import scipy.io as sio
import tensorflow as tf

from face3d.morphable_model import MorphabelModel
from tf_3dmm.mesh.transform import affine_transform
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel
from tf_3dmm.tf_util import is_tf_expression

frame_width, frame_height = 450, 450


def normalize(vector):
    return tf.convert_to_tensor(vector) / tf.norm(vector)


def uvs_to_pixel_indices(uvs, texture_shape, mode='repeat'):
    # Note that this assumes u = 0, v = 0 is at the top-left of the image -- different to OpenGL!

    uvs = uvs[..., ::-1]  # change x, y coordinates to y, x indices
    texture_shape = tf.cast(texture_shape, tf.float32)
    if mode == 'repeat':
        return uvs % 1. * texture_shape
    elif mode == 'clamp':
        return tf.clip_by_value(uvs, 0., 1.) * texture_shape
    else:
        raise NotImplementedError


def sample_texture(texture, indices, mode='bilinear'):
    if mode == 'nearest':

        return tf.gather_nd(texture, tf.cast(indices, tf.int32))

    elif mode == 'bilinear':

        floor_indices = tf.floor(indices)
        frac_indices = indices - floor_indices
        floor_indices = tf.cast(floor_indices, tf.int32)

        neighbours = tf.gather_nd(
            texture,
            tf.stack([
                floor_indices,
                floor_indices + [0, 1],
                floor_indices + [1, 0],
                floor_indices + [1, 1]
            ]),
        )
        top_left, top_right, bottom_left, bottom_right = tf.unstack(neighbours)

        return \
            top_left * (1. - frac_indices[..., 1:]) * (1. - frac_indices[..., :1]) + \
            top_right * frac_indices[..., 1:] * (1. - frac_indices[..., :1]) + \
            bottom_left * (1. - frac_indices[..., 1:]) * frac_indices[..., :1] + \
            bottom_right * frac_indices[..., 1:] * frac_indices[..., :1]

    else:

        raise NotImplementedError


def process_uv(uv_coords, uv_height, uv_width):

    uv_x = uv_coords[:, 0] * (uv_width - 1)
    uv_y = uv_coords[:, 1] * (uv_height - 1)
    uv_y = uv_height - uv_y - 1

    uv_cords = tf.stack(
        [uv_x, uv_y],
        axis=0,
        name='stack'
    )
    return uv_cords


def render(
        pose_param,
        shape_param,
        exp_param,
        tex_param,
        color_param,
        illum_param,
        frame_width,
        frame_height,
        tf_bfm: TfMorphableModel
):
    assert is_tf_expression(pose_param)
    assert is_tf_expression(shape_param)
    assert is_tf_expression(exp_param)
    assert is_tf_expression(tex_param)
    assert is_tf_expression(color_param)
    assert is_tf_expression(illum_param)

    vertices = tf_bfm.get_vertices(shape_param=shape_param, exp_param=exp_param)
    vertex_norm = None # TODO

    colors = tf_bfm.get_vertex_colors(
        tex_param=tex_param,
        color_param=color_param,
        illum_param=illum_param,
        vertex_norm=vertex_norm
    )

    transformed_vertices = affine_transform(
        vertices=vertices,
        scaling=pose_param[0, 6],
        angles_rad=pose_param[0, 0:3],
        t3d=pose_param[0, 3:6]
    )
    transformed_vertices[:, 0] = transformed_vertices[:, 0] * 2 / frame_width - 1
    transformed_vertices[:, 1] = transformed_vertices[:, 1] * 2 / frame_height - 1
    transformed_vertices[:, 2] = -transformed_vertices[:, 2] / tf.math.reduce_max(tf.math.abs(transformed_vertices[:, 2]))

    # Convert vertices to homogeneous coordinates
    transformed_vertices = tf.concat([
        transformed_vertices,
        tf.ones_like(transformed_vertices[:, -1:])
    ], axis=1)

    # Render the G-buffer
    image = dirt.rasterise(
        vertices=transformed_vertices,
        faces=tf_bfm.triangles,
        vertex_colors=colors,
        background=tf.zeros([frame_height, frame_width, 3]),
        width=frame_width, height=frame_height, channels=3
    )

    return image * 255

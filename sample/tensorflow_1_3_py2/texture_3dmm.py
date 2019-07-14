# This demonstrates using Dirt for textured rendering with a UV map

import os
import imageio
import numpy as np
import tensorflow as tf
import scipy.io as sio
import dirt
import dirt.matrices as matrices
import dirt.lighting as lighting
import face3d
from face3d.morphable_model import MorphabelModel

frame_width, frame_height = 450, 450


def unit(vector):
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


def process_uv(uv_coords, uv_h=256, uv_w=256):
    uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
    uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
    uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))  # add z
    return np.asarray(uv_coords, np.float32)


def main():
    bfm = MorphabelModel('../../examples/Data/BFM/Out/BFM.mat')
    # --load mesh data
    pic_name = 'IBUG_image_008_1_0'
    # pic_name = 'IBUG_image_014_01_2'
    mat_filename = '../../examples/Data/{0}.mat'.format(pic_name)
    mat_data = sio.loadmat(mat_filename)
    sp = mat_data['Shape_Para']
    ep = mat_data['Exp_Para']
    vertices = bfm.generate_vertices(sp, ep)

    tp = mat_data['Tex_Para']
    tex = bfm.generate_tex_xuan(tex_para=tp)
    cp = mat_data['Color_Para']
    ip = mat_data['Illum_Para']

    import face3d.mesh_numpy as mesh
    norm = mesh.render.generate_vertex_norm(vertices=vertices, triangles=bfm.triangles,
                                            nver=bfm.nver, ntri=bfm.ntri)
    colors = bfm.generate_tex_color_xuan(tex, cp, ip, norm)
    colors = np.clip(colors / 255., 0., 1.)
    colors = np.asarray(colors, dtype=np.float32)

    triangles = bfm.triangles

    # colors = colors / np.max(colors)

    pp = mat_data['Pose_Para']
    s = pp[0, 6]
    # angles = [np.rad2deg(pp[0, 0]), np.rad2deg(pp[0, 1]), np.rad2deg(pp[0, 2])]
    angles = pp[0, 0:3]
    # angles = [0, 0, 0]
    t = pp[0, 3:6]

    transformed_vertices = bfm.transform_3ddfa(vertices, s, angles, t)
    transformed_vertices[:, 0] = transformed_vertices[:, 0] * 2 / frame_width - 1
    transformed_vertices[:, 1] = transformed_vertices[:, 1] * 2 / frame_height - 1
    transformed_vertices[:, 2] = -transformed_vertices[:, 2] / np.max(np.abs(transformed_vertices[:, 2]))
    transformed_vertices = np.asarray(transformed_vertices, dtype=np.float32)

    # Convert vertices to homogeneous coordinates
    transformed_vertices = tf.concat([
        transformed_vertices,
        tf.ones_like(transformed_vertices[:, -1:])
    ], axis=1)

    # Render the G-buffer channels (mask, UVs, and normals at each pixel) needed for deferred shading
    image = dirt.rasterise(
        vertices=transformed_vertices,  # (24, 4)
        faces=triangles.tolist(),  # [[]]
        vertex_colors=colors,
        background=tf.zeros([frame_height, frame_width, 3]),
        width=frame_width, height=frame_height, channels=3
    )

    session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    with session.as_default():
        image_eval = image.eval()
        imageio.imsave('textured_3dmm.jpg', (image_eval * 255).astype(np.uint8))


if __name__ == '__main__':
    main()

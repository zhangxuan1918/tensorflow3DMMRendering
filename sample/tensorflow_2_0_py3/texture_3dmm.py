# This demonstrates using Dirt for textured rendering with a UV map

import dirt
import imageio
import numpy as np
import scipy.io as sio
import tensorflow as tf
from dirt import lighting

from tf_3dmm.mesh.transform import affine_transform
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel

frame_width, frame_height = 450, 450


def main():
    tf_bfm = TfMorphableModel('../../examples/Data/BFM/Out/BFM.mat')
    # --load mesh data
    pic_name = 'IBUG_image_008_1_0'
    # pic_name = 'IBUG_image_014_01_2'
    mat_filename = '../../examples/Data/{0}.mat'.format(pic_name)
    mat_data = sio.loadmat(mat_filename)
    sp = tf.constant(mat_data['Shape_Para'], dtype=tf.float32)
    ep = tf.constant(mat_data['Exp_Para'], dtype=tf.float32)

    vertices = tf_bfm.get_vertices(sp, ep)
    triangles = tf_bfm.triangles

    tp = tf.constant(mat_data['Tex_Para'], dtype=tf.float32)
    cp = tf.constant(mat_data['Color_Para'], dtype=tf.float32)
    ip = tf.constant(mat_data['Illum_Para'], dtype=tf.float32)

    # import face3d.mesh_numpy as mesh
    # norm = mesh.render.generate_vertex_norm(vertices=vertices, triangles=tf_bfm.triangles,
    #                                         nver=tf_bfm.nver, ntri=tf_bfm.ntri)
    norm = lighting.vertex_normals(vertices, triangles)
    colors = tf_bfm.get_vertex_colors(tp, cp, ip, -norm)
    colors = tf.clip_by_value(colors / 255., 0., 1.)

    pp = tf.constant(mat_data['Pose_Para'], dtype=tf.float32)
    s = pp[0, 6]
    angles = pp[0, 0:3]
    t = pp[0, 3:6]

    transformed_vertices = affine_transform(vertices, s, angles, t)
    transformed_vertices_x = transformed_vertices[:, 0] * 2 / frame_width - 1
    transformed_vertices_y = transformed_vertices[:, 1] * 2 / frame_height - 1
    transformed_vertices_z = -transformed_vertices[:, 2] / tf.reduce_max(tf.abs(transformed_vertices[:, 2]))

    # Convert vertices to homogeneous coordinates
    transformed_vertices = tf.concat([
        tf.expand_dims(transformed_vertices_x, axis=1),
        tf.expand_dims(transformed_vertices_y, axis=1),
        tf.expand_dims(transformed_vertices_z, axis=1),
        tf.ones_like(transformed_vertices[:, -1:])
    ], axis=1)

    # Render the G-buffer channels (mask, UVs, and normals at each pixel) needed for deferred shading
    image = dirt.rasterise(
        vertices=transformed_vertices,  # (24, 4)
        faces=triangles,  # [[]]
        vertex_colors=colors,
        background=tf.zeros([frame_height, frame_width, 3]),
        width=frame_width, height=frame_height, channels=3
    )

    image_eval = image.numpy()
    imageio.imsave('./textured_3dmm.jpg', (image_eval * 255).astype(np.uint8))


if __name__ == '__main__':
    main()

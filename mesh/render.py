from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from .cython import mesh_core_cython


def rasterize_triangles(vertices, triangles, h, w):
    """
    raterize triangles
    Each triangle has 3 vertices & Each vertex has 3 coordinates x, y, z.
    h, w is the size of rendering

    :param: vertices:  [n_vertex,   3]
    :param: triangles: [n_triangle, 3]
    :param: h: height
    :param: w: width
    :return: depth_buffer: [h, w] saves the depth, here, the bigger the z, the fronter the point.
        triangle_buffer: [h, w] saves the tri id(-1 for no triangle).
        barycentric_weight: [h, w, 3] saves corresponding barycentric weight.
    """
    # initial
    depth_buffer = tf.zeros([h, w], dtype=tf.float32) - 999999.  # set the initial z to the furthest position
    triangle_buffer = tf.zeros([h, w], dtype=tf.int32) - 1 # if tri id = -1, the pixel has no triangle correspondance
    barycentric_weight = tf.zeros([h, w, 3], dtype=tf.float32)

    rasterize_triangles_core(
        tf.identity(vertices),
        tf.identity(triangles),
        depth_buffer,
        triangle_buffer,
        barycentric_weight,
        vertices.shape[0],
        triangles.shape[0],
        h,
        w
    )


def render_colors(vertices, triangles, colors, h, w, channels=3, background_img=None):
    """
    render mesh with colors
    :param: vertices:       [nver,       3]
    :param: triangles:      [n_triangle, 3]
    :param: colors:         [n_vertex,   3]
    :param: h:              height
    :param: w:              width
    :param: channels:       channel
    :param: background_img: background image
    :return: image:         [h, w, c]. rendered image
    """

    # initial 
    if background_img is None:
        image = tf.zeros((h, w, channels), dtype=np.float32)
    else:
        tf.debugging.assert_shapes((h, w, channels), background_img)
        image = background_img

    depth_buffer = tf.zeros([h, w], dtype=tf.float32) - 999999.

    mesh_core_cython.render_colors_core(
        tf.identity(image),
        tf.identity(vertices),
        tf.identity(triangles),
        colors,
        depth_buffer,
        vertices.shape[0],
        triangles.shape[0],
        h,
        w,
        channels)
    return image


def render_texture(vertices, triangles, texture, tex_coords, tex_triangles, h, w, channels=3, mapping_type='nearest',
                   background_img=None):
    """
    render mesh with texture map
    :param: vertices:       [n_vertex,     3]
    :param: triangles:      [n_triangle,   3]
    :param: texture:        [tex_h, tex_w, 3]
    :param: tex_coords:     [n_tex_coords, 3]
    :param: tex_triangles:  [ntri,         3]
    :param: h:              height of rendering
    :param: w:              width of rendering
    :param: channels:       channel
    :param: mapping_type:   "bilinear" or "nearest"
    """
    # initial 
    if background_img is None:
        image = tf.zeros((h, w, channels), dtype=tf.float32)
    else:
        tf.debugging.assert_shapes((h, w, channels), background_img)
        image = background_img

    depth_buffer = tf.zeros([h, w], dtype=np.float32) - 999999.

    tex_h, tex_w, tex_c = texture.shape
    if mapping_type == 'nearest':
        mt = 0
    elif mapping_type == 'bilinear':
        mt = 1
    else:
        mt = 0

    render_texture_core(
        tf.identity(image),
        tf.identity(vertices),
        tf.identity(triangles),
        tf.identity(texture),
        tf.identity(tex_coords),
        tf.identity(tex_triangles),
        depth_buffer,
        vertices.shape[0],
        tex_coords.shape[0],
        triangles.shape[0],
        h,
        w,
        channels,
        tex_h,
        tex_w,
        tex_c,
        mt)
    return image


def generate_vertex_norm(vertices, triangles, n_vertices, n_triangles):
    """
    generate vertex norm for each vertex using norms of triangles
    :param vertices: (n_vertex, 3)
    :param triangles: (n_triangle, 3)
    :param n_vertices: number of vertices
    :param n_triangles: number of triangles
    :return: norm: (n_vertex, 3)
    """
    # pt1 shape of (ntri, 3)
    pt1 = vertices[triangles[:, 0], :]
    pt2 = vertices[triangles[:, 1], :]
    pt3 = vertices[triangles[:, 2], :]

    # norm of triangle of shape (ntri, 3)
    norm_tri = tf.linalg.cross(pt1 - pt2, pt1 - pt3)

    # norm of vertices
    N = tf.zeros((n_vertices, 3), dtype=tf.float32)
    N = tnorm_to_vnorm(vertex_norm=N, n_vertex=n_vertices, triangle_norm=norm_tri, triangles=triangles,
                       n_triangle=n_triangles)
    # mag of shape (nver, 1)
    mag = tf.reduce_sum(tf.square(N), 1, keepdims=True)
    # deal with zero vector
    index = tf.equal(mag, 0)
    mag = tf.where(index, tf.ones(mag.shape, dtype=tf.float32) * 3, mag)

    # originally, if we find norm with magnitude is 0, we assign the norm to be 1 and
    # set the first axis to be 1
    # mag[index[0]] = 1
    # N[index[0], 1] = 1

    # TODO check if it works
    # it's not easy to do it in tensorflow, here we just assign the norm vector
    # with magnitude 1 and each axis has the same value
    N = tf.where(tf.tile(index, (1, 3)), tf.ones(N.shape, dtype=tf.float32), N)
    N = tf.divide(N, tf.math.sqrt(mag))
    return -N


def tnorm_to_vnorm(vertex_norm, n_vertex, triangle_norm, triangles, n_triangle):
    """
    triangle norm to vertex norm
    vertex norm is the average of the triangle norms to which it's adjacent
    :param vertex_norm:
    :param n_vertex:
    :param triangle_norm:
    :param triangles:
    :param n_triangle:
    :return:
    """
    mesh_core_cython.tnorm_to_vnorm(vertex_norm, n_vertex, triangle_norm, triangles, n_triangle)

    return vertex_norm

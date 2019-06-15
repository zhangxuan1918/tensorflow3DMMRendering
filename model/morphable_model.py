from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorboard.plugins import mesh

from model.morphable_model_util import load_BFM
from model.tf_util import get_shape, is_tf_expression


class TfMorphableModel(object):
    """
    3DMM Morphable Model implemented using tensorflow
    nver            :                           number of vertices
    ntri            :                           number of triangles
    shapeMU         : [3*nver, 1]               mean shape
    shapePC         : [3*nver, n_shape_para]    principle component for shape
    shapeEV         : [n_shape_para, 1]         standard deviation for shape principle component
    expMU           : [3*nver, 1]               mean expression
    expPC           : [3*nver, n_exp_para]      principle component for expression
    expEV           : [n_exp_para, 1]           standard deviation for expression principle component
    texMU           : [3*nver, 1]               mean tex
    texPC           : [3*nver, n_tex_para]      principle component for tex
    texEV           : [n_tex_para, 1]           standard deviation for tex principle component
    tri             : [ntri, 3]                 triangles of mesh wo month
    tri_mouth       : [114, 3]                  triangles of mouth mesh
    kpt_ind         : [68,]                     indices for landmarks
    """

    def __init__(self, model_path, model_type='BFM'):
        if model_type == 'BFM':
            self.model = load_BFM(model_path)
        else:
            print('only BFM09 is supported')
            raise Exception('model_type={0} is not supported; only BFM09 is supported'.format(
                self.model
            ))
        # 3dmm model
        self.shape_pc = tf.constant(self.model['shapePC'], dtype=tf.float32)
        self.shape_mu = tf.constant(self.model['shapeMU'], dtype=tf.float32)
        self.shape_ev = tf.constant(self.model['shapeEV'], dtype=tf.float32)

        self.tex_pc = tf.constant(self.model['texPC'], dtype=tf.float32)
        self.tex_mu = tf.constant(self.model['texMU'], dtype=tf.float32)
        self.tex_ev = tf.constant(self.model['texEV'], dtype=tf.float32)

        self.exp_pc = tf.constant(self.model['expPC'], dtype=tf.float32)
        self.exp_mu = tf.constant(self.model['expMU'], dtype=tf.float32)
        self.exp_ev = tf.constant(self.model['expEV'], dtype=tf.float32)

        # fixed attributes
        self.n_vertices = tf.divide(self.shape_pc.shape[0], 3)
        self.n_triangles = self.triangles.shape[0]
        self.n_shape_para = self.shape_pc.shape[1]
        self.n_exp_para = self.exp_pc.shape[1]
        self.n_tex_para = self.tex_mu.shape[1]
        self.kpt_ind = tf.constant(self.model['kpt_ind'], dtype=tf.int32)

        self.triangles = tf.constant(self.model['tri'], dtype=tf.float32)
        self.triangles_mouth = tf.constant(self.model['tri_moth'], dtype=tf.float32)
        self.full_triangles = tf.stack([self.triangles, self.triangles_mouth])

    def get_landmark_indices(self):
        return self.kpt_ind

    def generate_vertices(self, shape_param, exp_param):
        """
        generate vertices from shape_para and exp_para

        :param: shape_para: (n_shape_para, 1)
        :param: exp_para:   (n_exp_para, 1)
        :return: vertices:  (n_vertices, 3)
        """

        assert is_tf_expression(shape_param) and is_tf_expression(exp_param)

        vertices = self.shape_mu + tf.linalg.matmul(self.shape_pc, shape_param) + tf.linalg.matmul(self.exp_pc, exp_param)
        vertices = tf.transpose(tf.reshape(vertices, (3, self.n_vertices)))

        tf.debugging.assert_shapes((self.n_vertices, 3), vertices)
        return vertices

    def generate_tex(self, tex_param):
        """
        generate texture using tex_Para
        :param tex_param: (n_tex_para, 1)
        :return: tex: (n_vertices, 3)
        """

        assert is_tf_expression(tex_param)

        tex = self.tex_mu + tf.linalg.matmul(self.tex_pc, tex_param)
        tex = tf.transpose(tf.reshape(tex, (3, self.n_vertices)))

        tf.debugging.assert_shapes((self.n_vertices, 3), tex)
        return tex

    def _generate_color(self, color_param):
        """
        generate color from color_para

        :param color_param: (1, 7)
        :returns:
             o: (n_vertices, 3)
             M: constant matrix (3, 3)
             g: diagonal matrix (3, 3)
             c: float
        """

        assert is_tf_expression(color_param)

        gain_r = color_param[0, 0]
        gain_g = color_param[0, 1]
        gain_b = color_param[0, 2]

        offset_r = color_param[0, 3]
        offset_g = color_param[0, 4]
        offset_b = color_param[0, 5]

        c = color_param[0, 6]

        M = tf.constant(
            [[0.3, 0.59, 0.11],
             [0.3, 0.59, 0.11],
             [0.3, 0.59, 0.11]],
            shape=(3, 3)
        )

        g = tf.linalg.tensor_diag([gain_r, gain_g, gain_b])
        o = tf.constant([offset_r, offset_g, offset_b], shape=(1, 3))
        # o matrix of shape(n_vertices, 3)
        o = tf.tile(o, [self.n_vertices, 1])

        tf.debugging.assert_shapes((self.n_vertices, 1), o)

        return o, M, g, c

    def _generate_illuminate(self, illum_param):
        """
        genreate illuminate params
        :param illum_param:
        :return:
        """
        assert is_tf_expression(illum_param)

        amb_r = illum_param[0, 0]
        amb_g = illum_param[0, 1]
        amb_b = illum_param[0, 2]

        dir_r = illum_param[0, 3]
        dir_g = illum_param[0, 4]
        dir_b = illum_param[0, 5]

        thetal = illum_param[0, 6]
        phil = illum_param[0, 7]
        ks = illum_param[0, 8]
        v = illum_param[0, 9]

        amb = tf.linalg.diag([amb_r, amb_g, amb_b])
        dirt = tf.linalg.diag([dir_r, dir_g, dir_b])

        l = tf.constant([np.cos(thetal) * np.sin(phil), np.sin(thetal), np.cos(thetal) * np.cos(phil)], dtype=tf.float32)
        h = l + tf.constant([0, 0, 1], dtype=tf.float32)
        h = h / tf.sqrt(tf.reduce_sum(tf.square(h)))

        return tf.reshape(h, (-1, 1)), ks, v, amb, dirt, tf.reshape(l, (-1, 1))

    def generate_tex_color(self, tex, color_param, illum_param, vertex_norm):
        """
        generate texture and color for rendering
        :param tex: texture generated by method `generate_tex`, (n_vertices, 3)
        :param color_param: (1, 7)
        :param illum_param: (1, 10)
        :param vertex_norm: vertex norm (n_vertex, 3)
        :return: texture color (n_vertex, 3)
        """

        assert is_tf_expression(tex)
        assert is_tf_expression(color_param)
        assert is_tf_expression(illum_param)
        assert is_tf_expression(vertex_norm)

        o, M, g, c = self._generate_color(color_param=color_param)
        h, ks, v, amb, dirt, l = self._generate_illuminate(illum_param=illum_param)
        # n_l of shape (n_ver, 1)
        n_l = tf.clip_by_value(tf.linalg.matmul(vertex_norm, l), 0)
        # n_h of shape (n_ver, 1)
        n_h = tf.clip_by_value(tf.linalg.matmul(vertex_norm, h), 0)
        # n_l of shape (n_ver, 3)
        n_l = tf.tile(n_l, [1, 3])
        # n_h of shape (n_ver, 3)
        n_h = tf.tile(n_h, [1, 3])

        # L of shape (n_ver, 3)
        L = tf.linalg.matmul(tex, amb) + tf.linalg.matmul(tf.math.multiply(n_l, tex), dirt) + \
            ks * tf.math.pow(n_h, v)

        # CT of shape (3, 3)
        CT = tf.math.multiply(g, c * tf.eye(3) + (1 - c) * M)
        tex_color = tf.linalg.matmul(L, CT) + o

        tf.debugging.assert_shapes((self.n_vertices, 3), tex_color)
        return tex_color

    @staticmethod
    def rotate(vertices, angles):
        """
        rotate face
        :param: vertices: (n_vertex, 3)
        :param: angles: (3, ) x, y, z rotation angle(degree)
                x: pitch. positive for looking down
                y: yaw. positive for looking left
                z: roll. positive for tilting head right
        :return: vertices: rotated vertices
        """

        assert is_tf_expression(vertices) and is_tf_expression(angles)
        return mesh.transform.rotate(vertices, angles)

    @staticmethod
    def transform(vertices, scale, angles, t3d):
        """
        transform vertices, e.g. rescaling, rotating and translating
        :param vertices:
        :param scale:
        :param angles:
        :param t3d:
        :param vertices: (n_vertex, 3)
        :param scale: scale, float
        :param angles: (3,) x, y, z rotation angle(degree)
                x: pitch. positive for looking down
                y: yaw. positive for looking left
                z: roll. positive for tilting head right
        :param t3d: translation, (3, )
        :return: transformed vertices: (n_vertex, 3)
        """
        assert is_tf_expression(vertices)
        assert is_tf_expression(scale)
        assert is_tf_expression(angles)
        assert is_tf_expression(t3d)

        R = mesh.transform.angle2matrix(angles)
        return mesh.transform.similarity_transform(vertices, scale, R, t3d)

    @staticmethod
    def transform_3ddfa(vertices, scale, angles, t3d):
        """
        transform vertices, e.g. rescaling, rotating and translating
        only used for processing 300W_LP data

        :param vertices: (n_vertex, 3)
        :param scale: scale, float
        :param angles: (3,) x, y, z rotation angle(degree)
                x: pitch. positive for looking down
                y: yaw. positive for looking left
                z: roll. positive for tilting head right
        :param t3d: translation, (3, )
        :return:
        """
        assert is_tf_expression(vertices)
        assert is_tf_expression(scale)
        assert is_tf_expression(angles)
        assert is_tf_expression(t3d)

        R = mesh.transform.angle2matrix_3ddfa(angles)
        return mesh.transform.similarity_transform(vertices, scale, R, t3d)

    def render_3dmm(self, shape_param, exp_param, tex_param, color_param, illum_param,
                    pose_param, w=224, h=224):
        """
        render 3dmm to image
        :param shape_param: (n_shape_para, 1)
        :param exp_param:   (n_exp_para, 1)
        :param tex_param:   (n_tex_para, 1)
        :param color_param: (1, n_color_param)
        :param illum_param: (1, n_illum_param)
        :param pose_param:  (1, n_pose_param)
        :param w:
        :param h:
        :return:
        """
        vertices = self.generate_vertices(shape_param=shape_param, exp_param=exp_param)
        texture = self.generate_tex(tex_param=tex_param)

        vertex_norm = mesh.render.generate_vertex_norm(
            vertices=vertices,
            triangles=self.triangles,
            n_vertices=self.n_vertices,
            n_triangles=self.n_triangles)

        tex_color = self.generate_tex_color(
            tex=texture,
            color_param=color_param,
            illum_param=illum_param,
            vertex_norm=vertex_norm)

        scale = pose_param[0, 6]
        angles = pose_param[0, 0:3]
        translate = pose_param[0, 3:6]

        transformed_vertices = self.transform_3ddfa(
            vertices=vertices,
            scale=scale,
            angles=angles,
            t3d=translate)

        image_vertices = mesh.transform.to_image(
            vertices=transformed_vertices,
            h=h,
            w=w)
        image = mesh.render.render_colors(
            vertices=image_vertices,
            triangles=self.triangles,
            colors=tex_color,
            h=h,
            w=w)
        return image


if __name__ == '__main__':
    bfm = MorphableModel('G:\PycharmProjects\FaceFusion\project_code\data\\3dmm\BFM\BFM.mat')

    pic_name = 'IBUG_image_008_1_0'
    mat_filename = 'G:\PycharmProjects\FaceFusion\project_code\data\\3dmm\\300W_LP_samples/{0}.mat'.format(pic_name)
    import scipy.io as sio

    mat_data = sio.loadmat(mat_filename)

    image_filename = 'G:\PycharmProjects\FaceFusion\project_code\data\\3dmm\\300W_LP_samples/{0}.jpg'.format(pic_name)
    with open(image_filename, 'rb') as file:
        img = Image.open(file)
        img_np = np.asarray(img, dtype='int32')
        h, w, _ = img_np.shape
    shape_param = mat_data['Shape_Para']
    exp_param = mat_data['Exp_Para']
    tex_param = mat_data['Tex_Para']
    color_param = mat_data['Color_Para']
    illum_param = mat_data['Illum_Para']
    pose_param = mat_data['Pose_Para']

    image = bfm.render_3dmm(
        shape_param=shape_param,
        exp_param=exp_param,
        tex_param=tex_param,
        color_param=color_param,
        illum_param=illum_param,
        pose_param=pose_param,
        h=h,
        w=w
    )

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img_np)

    image = np.asarray(np.round(image), dtype=np.int).clip(0, 255)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(image)

    image_mask = image > 0
    img_np_mx = np.ma.masked_array(img_np, mask=image_mask, fill_value=0)
    image_overlay = img_np_mx + image
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(image_overlay)

    fig.show()

    xxx = 1

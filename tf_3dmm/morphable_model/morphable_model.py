from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_3dmm.morphable_model.morphable_model_util import load_BFM
from tf_3dmm.tf_util import is_tf_expression


class TfMorphableModel(object):
    """
    3DMM Morphable Model implemented using tensor in tensorflow
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
            model = load_BFM(model_path)
        else:
            print('only BFM09 is supported')
            raise Exception('model_type={0} is not supported; only BFM09 is supported'.format(
                model_type
            ))
        # 3dmm model
        self.shape_pc = tf.constant(model['shapePC'], dtype=tf.float32)
        self.shape_mu = tf.constant(model['shapeMU'], dtype=tf.float32)
        self.shape_ev = tf.constant(model['shapeEV'], dtype=tf.float32)

        self.tex_pc = tf.constant(model['texPC'], dtype=tf.float32)
        self.tex_mu = tf.constant(model['texMU'], dtype=tf.float32)
        self.tex_ev = tf.constant(model['texEV'], dtype=tf.float32)

        self.exp_pc = tf.constant(model['expPC'], dtype=tf.float32)
        self.exp_mu = tf.constant(model['expMU'], dtype=tf.float32)
        self.exp_ev = tf.constant(model['expEV'], dtype=tf.float32)

        self.triangles = tf.constant(model['tri'], dtype=tf.int32)
        self.triangles_mouth = tf.constant(model['tri_mouth'], dtype=tf.int32)
        self.full_triangles = tf.concat([self.triangles, self.triangles_mouth], axis=0)

        # fixed attributes
        self.n_vertices = self.shape_pc.shape[0] // 3
        self.n_triangles = self.triangles.shape[0]
        self.n_shape_para = self.shape_pc.shape[1]
        self.n_exp_para = self.exp_pc.shape[1]
        self.n_tex_para = self.tex_mu.shape[1]
        self.kpt_ind = tf.constant(model['kpt_ind'], dtype=tf.int32)

    def get_landmark_indices(self):
        return self.kpt_ind

    def get_vertices(self, shape_param, exp_param):
        """
        generate vertices from shape_para and exp_para
        :param: shape_para: [n_shape_para, 1]
        :param: exp_para:   [n_exp_para, 1]
        :return: vertices:  [n_vertices, 3]
        """

        assert is_tf_expression(shape_param) and is_tf_expression(exp_param)

        vertices = self.shape_mu + tf.linalg.matmul(self.shape_pc, shape_param) + tf.linalg.matmul(self.exp_pc,
                                                                                                   exp_param)
        vertices = tf.reshape(vertices, (self.n_vertices, 3))

        tf.debugging.assert_shapes({vertices: (self.n_vertices, 3)})
        return vertices

    def _get_texture(self, tex_param):
        """
        generate texture using tex_Para
        :param tex_param: [199, 1]
        :return: tex: [n_vertices, 3]
        """

        assert is_tf_expression(tex_param)

        tex = self.tex_mu + tf.linalg.matmul(self.tex_pc, tex_param)
        tex = tf.reshape(tex, (self.n_vertices, 3))

        tf.debugging.assert_shapes({tex: (self.n_vertices, 3)})
        return tex

    def _get_color(self, color_param):
        """
        generate color from color_para
        :param color_param: [1, 7]
        :returns:
             o: [n_vertices, 3]
             M: constant matrix [3, 3]
             g: diagonal matrix [3, 3]
             c: float
        """

        assert is_tf_expression(color_param)

        c = color_param[0, 6]

        M = tf.constant(
            [[0.3, 0.59, 0.11],
             [0.3, 0.59, 0.11],
             [0.3, 0.59, 0.11]],
            shape=(3, 3)
        )

        g = tf.linalg.tensor_diag(color_param[0, 0:3])
        o = tf.reshape(color_param[0, 3:6], (1, 3))
        # o matrix of shape(n_vertices, 3)
        o = tf.tile(o, [self.n_vertices, 1])

        tf.debugging.assert_shapes({o: (self.n_vertices, 3)})

        return o, M, g, c

    def _get_illum(self, illum_param):
        """
        genreate illuminate params
        :param illum_param:
        :return:
        """
        assert is_tf_expression(illum_param)

        thetal = illum_param[0, 6]
        phil = illum_param[0, 7]
        ks = illum_param[0, 8]
        v = illum_param[0, 9]

        amb = tf.linalg.diag(illum_param[0, 0:3])
        dirt = tf.linalg.diag(illum_param[0, 3:6])

        l = tf.Variable(
            [tf.math.cos(thetal) * tf.math.sin(phil), tf.math.sin(thetal), tf.math.cos(thetal) * tf.math.cos(phil)],
            dtype=tf.float32)
        h = l + tf.constant([0, 0, 1], dtype=tf.float32)
        h = h / tf.sqrt(tf.reduce_sum(tf.square(h)))

        return tf.reshape(h, (-1, 1)), ks, v, amb, dirt, tf.reshape(l, (-1, 1))

    def get_vertex_colors(self, tex_param, color_param, illum_param, vertex_norm):
        """
        generate texture and color for rendering
        :param tex_param: [199, 1]
        :param color_param: [1, 7]
        :param illum_param: [1, 10]
        :param vertex_norm: vertex norm [n_vertex, 3]
        :return: texture color [n_vertex, 3]
        """

        assert is_tf_expression(tex_param)
        assert is_tf_expression(color_param)
        assert is_tf_expression(illum_param)
        assert is_tf_expression(vertex_norm)

        tex = self._get_texture(tex_param=tex_param)

        o, M, g, c = self._get_color(color_param=color_param)
        h, ks, v, amb, dirt, l = self._get_illum(illum_param=illum_param)
        # n_l of shape (n_ver, 1)
        n_l = tf.maximum(tf.linalg.matmul(vertex_norm, l), 0)
        # n_h of shape (n_ver, 1)
        n_h = tf.maximum(tf.linalg.matmul(vertex_norm, h), 0)
        # n_l of shape (n_ver, 3)
        n_l = tf.tile(n_l, [1, 3])
        # n_h of shape (n_ver, 3)
        n_h = tf.tile(n_h, [1, 3])

        # L of shape (n_ver, 3)
        L = tf.linalg.matmul(tex, amb) + tf.linalg.matmul(tf.math.multiply(n_l, tex), dirt) + \
            ks * tf.math.pow(n_h, v)

        # CT of shape (3, 3)
        CT = tf.math.multiply(g, c * tf.eye(3) + (1 - c) * M)
        vertex_colors = tf.linalg.matmul(L, CT) + o

        tf.debugging.assert_shapes({vertex_colors: (self.n_vertices, 3)})
        return vertex_colors


if __name__ == '__main__':
    pic_name = 'IBUG_image_008_1_0'
    mat_filename = '../../examples/Data/{0}.mat'.format(pic_name)
    import scipy.io as sio

    mat_data = sio.loadmat(mat_filename)

    shape_param = tf.constant(mat_data['Shape_Para'], dtype=tf.float32)
    exp_param = tf.constant(mat_data['Exp_Para'], dtype=tf.float32)
    tex_param = tf.constant(mat_data['Tex_Para'], dtype=tf.float32)
    color_param = tf.constant(mat_data['Color_Para'], dtype=tf.float32)
    illum_param = tf.constant(mat_data['Illum_Para'], dtype=tf.float32)
    pose_param = tf.constant(mat_data['Pose_Para'], dtype=tf.float32)

    tf_bfm = TfMorphableModel('../../examples/Data/BFM/Out/BFM.mat')

    vertices = tf_bfm.get_vertices(
        shape_param=shape_param,
        exp_param=exp_param
    )

    from dirt import lighting

    vertex_norm = lighting.vertex_normals(vertices, tf_bfm.triangles)
    texture = tf_bfm.get_vertex_colors(tex_param, color_param, illum_param, -vertex_norm)

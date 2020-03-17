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

    def __init__(self, model_path, n_tex_para=40, model_type='BFM'):
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
        # self.shape_ev = tf.constant(model['shapeEV'], dtype=tf.float32)

        # Tex param: only take 40 params
        self.tex_pc = tf.constant(model['texPC'][:, :n_tex_para], dtype=tf.float32)
        self.tex_mu = tf.constant(model['texMU'], dtype=tf.float32)
        # self.tex_ev = tf.constant(model['texEV'][:n_tex_para, :], dtype=tf.float32)

        self.exp_pc = tf.constant(model['expPC'], dtype=tf.float32)
        self.exp_mu = tf.constant(model['expMU'], dtype=tf.float32)
        # self.exp_ev = tf.constant(model['expEV'], dtype=tf.float32)

        self.triangles = tf.constant(model['tri'], dtype=tf.int32)
        # self.triangles_mouth = tf.constant(model['tri_mouth'], dtype=tf.int32)
        # self.full_triangles = tf.concat([self.triangles, self.triangles_mouth], axis=0)

        # fixed attributes
        self.n_vertices = self.shape_pc.shape[0] // 3
        self.n_triangles = self.triangles.shape[0]
        self.n_shape_para = self.shape_pc.shape[1]  # 199
        self.n_exp_para = self.exp_pc.shape[1]  # 29
        self.n_tex_para = n_tex_para  # 40
        self.n_color_para = 7
        self.n_illum_para = 10
        self.n_pose_para = 7
        self.kpt_ind = tf.constant(model['kpt_ind'], dtype=tf.int32)

    def get_num_pose_param(self):
        return self.n_pose_para

    def get_landmark_indices(self):
        return self.kpt_ind

    def get_vertices(self, shape_param, exp_param, batch_size):
        """
        generate vertices from shape_para and exp_para
        :param: shape_para: [batch, n_shape_para, 1] or [batch, n_shape_para]
        :param: exp_para:   [batch, n_exp_para, 1] or [batch, n_exp_para]
        :param: batch_size:
        :return: vertices:  [batch, n_vertices, 3]
        """

        assert is_tf_expression(shape_param) and is_tf_expression(exp_param)

        sp_shape = tf.shape(shape_param)
        if sp_shape.shape[0] == 2:
            tf.debugging.assert_shapes(
                [(shape_param, (batch_size, self.n_shape_para))],
                message='shape_param shape wrong, dim != ({batch}, {dim})'.format(
                    batch=batch_size, dim=self.n_shape_para))
            shape_param = tf.expand_dims(shape_param, 2)
        elif sp_shape.shape[0] == 3:
            tf.debugging.assert_shapes(
                [(shape_param, (batch_size, self.n_shape_para, 1))],
                message='pose_param shape wrong, dim != ({batch}, {dim}, 1)'.format(
                    batch=batch_size, dim=self.n_shape_para))
        else:
            raise ValueError(
                'pose_param shape wrong, dim != ({batch}, {dim}, 1) or ({batch}, {dim})'.format(
                    batch=batch_size, dim=self.n_shape_para))

        ep_shape = tf.shape(exp_param)
        if ep_shape.shape[0] == 2:
            tf.debugging.assert_shapes(
                [(exp_param, (batch_size, self.n_exp_para))],
                message='exp_param shape wrong, dim != ({batch}, {dim})'.format(
                    batch=batch_size, dim=self.n_exp_para))
            exp_param = tf.expand_dims(exp_param, 2)
        elif ep_shape.shape[0] == 3:
            tf.debugging.assert_shapes(
                [(exp_param, (batch_size, self.n_exp_para, 1))],
                message='exp_param shape wrong, dim != ({batch}, {dim}, 1)'.format(
                    batch=batch_size, dim=self.n_exp_para))
        else:
            raise ValueError('exp_param shape wrong, dim != ({batch}, {dim}, 1) or ({batch}, {dim})'.format(
                batch=batch_size, dim=self.n_exp_para))

        vertices = tf.expand_dims(self.shape_mu, 0) + tf.einsum('ij,kjs->kis', self.shape_pc, shape_param) + tf.einsum('ij,kjs->kis', self.exp_pc, exp_param)

        vertices = tf.reshape(vertices, (batch_size, self.n_vertices, 3))
        return vertices

    def _get_texture(self, tex_param, batch_size):
        """
        generate texture using tex_Para
        :param tex_param: [batch, 40, 1] or [batch, 40]
        :return: tex: [n_vertices, 3]
        """
        assert is_tf_expression(tex_param)

        tp_shape = tf.shape(tex_param)
        if tp_shape.shape[0] == 2:
            tf.debugging.assert_shapes(
                [(tex_param, (batch_size, self.n_tex_para))],
                message='tex_param shape wrong, dim != ({batch}, {dim})'.format(
                    batch=batch_size, dim=self.n_tex_para))
            tex_param = tf.expand_dims(tex_param, 2)
        elif tp_shape.shape[0] == 3:
            tf.debugging.assert_shapes(
                [(tex_param, (batch_size, self.n_tex_para, 1))],
                message='tex_param shape wrong, dim != ({batch}, {dim}, 1)'.format(
                    batch=batch_size, dim=self.n_tex_para))
        else:
            raise ValueError('tex_param shape wrong, dim != ({batch}, {dim}, 1) or ({batch}, {dim})'.format(
                batch=batch_size, dim=self.n_tex_para))

        tex = tf.expand_dims(self.tex_mu, 0) + tf.einsum('ij,kjs->kis', self.tex_pc, tex_param)
        tex = tf.reshape(tex, (batch_size, self.n_vertices, 3))
        return tex

    def _get_color(self, color_param, batch_size):
        """
        # Color_Para: add last value as it's always 1

        generate color from color_para
        :param color_param: [batch, 1, 6] or [batch, 6] or [batch, 1, 7] or [batch, 7]
        the data we use to train the model has constant at position 6, e.g. color_param[:, 0, 6] == 1
        thus, we remove it from our training data
        :returns:
             o: [batch, n_vertices, 3]
             M: constant matrix [3, 3]
             g: diagonal matrix [batch, 3, 3]
             c: [batch, 1]
        """

        assert is_tf_expression(color_param)

        cp_shape = tf.shape(color_param)
        if cp_shape.shape[0] == 2:
            tf.debugging.assert_shapes(
                [(color_param, (batch_size, self.n_color_para))],
                message='color_param shape wrong, dim != ({batch}, {dim})'.format(
                    batch=batch_size, dim=self.n_color_para))
            color_param = tf.expand_dims(color_param, 1)
        elif cp_shape.shape[0] == 3:
            tf.debugging.assert_shapes(
                [(color_param, (batch_size, 1, self.n_color_para))],
                message='color_param shape wrong, dim != ({batch}, 1, {dim})'.format(
                    batch=batch_size, dim=self.n_color_para))
        else:
            raise ValueError('color_param shape wrong, dim != ({batch}, 1, {dim}) or ({batch}, {dim})'.format(
                batch=batch_size, dim=self.n_color_para))
        # c shape: (batch, 1)
        c = color_param[:, :, 6]
        M = tf.constant(
            [[0.3, 0.59, 0.11],
             [0.3, 0.59, 0.11],
             [0.3, 0.59, 0.11]],
            shape=(3, 3)
        )
        g = tf.linalg.diag(color_param[:, 0, 0:3])
        o = tf.reshape(color_param[:, 0, 3:6], (batch_size, 1, 3))
        # o matrix of shape(batch, n_vertices, 3)
        o = tf.tile(o, [1, self.n_vertices, 1])

        tf.debugging.assert_shapes([(o, (batch_size, self.n_vertices, 3))])

        return o, M, g, c

    def _get_illum(self, illum_param, batch_size):
        """
        genreate illuminate params
        :param illum_param: [batch, 1, 10] or  [batch, 10]
        :return:

        h: [batch, 3, 1]
        ks: [batch, 1]
        v: [batch, 1]
        amb: [batch, 3, 3]
        d: [batch, 3, 3]
        ks: [batch, 1]
        l: [batch, 3, 1]
        """
        assert is_tf_expression(illum_param)

        ip_shape = tf.shape(illum_param)
        if ip_shape.shape[0] == 2:
            tf.debugging.assert_shapes(
                [(illum_param, (batch_size, self.n_illum_para))],
                message='illum_param shape wrong, dim != ({batch}, {dim})'.format(
                    batch=batch_size, dim=self.n_illum_para))
            illum_param = tf.expand_dims(illum_param, 1)
        elif ip_shape.shape[0] == 3:
            tf.debugging.assert_shapes(
                [(illum_param, (batch_size, 1, self.n_illum_para))],
                message='illum_param shape wrong, dim != ({batch}, 1, {dim})'.format(
                    batch=batch_size, dim=self.n_illum_para))
        else:
            raise ValueError('illum_param shape wrong, dim != ({batch}, 1, {dim}) or ({batch}, {dim})'.format(
                    batch=batch_size, dim=self.n_illum_para))

        thetal = illum_param[:, :, 6]
        phil = illum_param[:, :, 7]
        ks = illum_param[:, :, 8]
        v = illum_param[:, :, 9]

        amb = tf.linalg.diag(illum_param[:, 0, 0:3])
        d = tf.linalg.diag(illum_param[:, 0, 3:6])

        # l, (batch, 3)
        l = tf.concat(
            [tf.math.cos(thetal) * tf.math.sin(phil), tf.math.sin(thetal), tf.math.cos(thetal) * tf.math.cos(phil)],
            axis=1)
        h = l + tf.expand_dims(tf.constant([0, 0, 1], dtype=tf.float32), axis=0)
        h = h / tf.sqrt(tf.reduce_sum(tf.square(h), axis=1, keepdims=True))

        return tf.reshape(h, (batch_size, -1, 1)), ks, v, amb, d, tf.reshape(l, (batch_size, -1, 1))

    def get_vertex_colors(self, tex_param, color_param, illum_param, vertex_norm, batch_size):
        """
        generate texture and color for rendering
        :param tex_param: [batch, 199, 1] or [batch, 199]
        :param color_param: [batch, 1, 7] or [batch, 7]
        :param illum_param: [batch, 1, 10] or [batch, 10]
        :param vertex_norm: vertex norm [batch, n_vertex, 3]
        :param batch_size
        :return: texture color [batch, n_vertex, 3]
        """

        assert is_tf_expression(tex_param)
        assert is_tf_expression(color_param)
        assert is_tf_expression(illum_param)
        assert is_tf_expression(vertex_norm)

        tex = self._get_texture(tex_param=tex_param, batch_size=batch_size)

        # o: [batch, n_vertices, 3]
        # M: constant, matrix[3, 3]
        # g: diagonal, matrix[batch, 3, 3]
        # c: 1

        o, M, g, c = self._get_color(color_param=color_param, batch_size=batch_size)

        # h: [batch, 3, 1]
        # ks: [batch, 1]
        # v: 20.0
        # amb: [batch, 3, 3]
        # d: [batch, 3, 3]
        # ks: [batch, 1]
        # l: [batch, 3, 1]

        h, ks, v, amb, d, l = self._get_illum(illum_param=illum_param, batch_size=batch_size)
        # n_l of shape (batch, n_ver, 1)
        n_l = tf.einsum('ijk,iks->ijs', vertex_norm, l)
        # n_h of shape (batch, n_ver, 1)
        n_h = tf.einsum('ijk,iks->ijs', vertex_norm, h)
        # n_l of shape (batch, n_ver, 3)
        n_l = tf.tile(n_l, [1, 1, 3])
        # n_h of shape (batch, n_ver, 3)
        n_h = tf.tile(n_h, [1, 1, 3])

        # L of shape (batch, n_ver, 3)
        L = tf.einsum('ijk,iks->ijs', tex, amb) + tf.einsum('ijk,iks->ijs', tf.math.multiply(n_l, tex), d) + tf.expand_dims(ks, axis=2) * tf.math.pow(n_h, tf.expand_dims(v, axis=1))  # <-(batch, 1, 1) * (batch, n_ver, 3)

        # c, (batch, 1)
        # tf.tile(c, (1, 3)), (batch, 3)
        # c_expanded, (batch, 3, 3)
        c_expanded = tf.linalg.diag(tf.tile(c, (1, 3)))
        nc_expanded = tf.linalg.diag(tf.tile(1 - c, (1, 3)))
        # CT of shape (batch, 3, 3)
        CT = tf.math.multiply(g, c_expanded + tf.einsum('ijk,ks->ijs', nc_expanded, M))

        # vertex_colors: (batch, n_ver, 3)
        vertex_colors = tf.einsum('ijk,iks->ijs', L, CT) + o

        tf.debugging.assert_shapes(
            [(vertex_colors, (batch_size, self.n_vertices, 3))],
            message='vertex_colors shape wrong, dim != ({batch}, {n_vert}, 3)'.format(
                batch=batch_size, n_vert=self.n_vertices))

        return vertex_colors


if __name__ == '__main__':
    pic_name = 'IBUG_image_008_1_0'
    mat_filename = '../../examples/Data/{0}.mat'.format(pic_name)
    import scipy.io as sio

    n_tex_para = 40

    mat_data = sio.loadmat(mat_filename)

    shape_param = tf.constant(mat_data['Shape_Para'], dtype=tf.float32)
    shape_param = tf.expand_dims(shape_param, 0)
    exp_param = tf.constant(mat_data['Exp_Para'], dtype=tf.float32)
    exp_param = tf.expand_dims(exp_param, 0)
    tex_param = tf.constant(mat_data['Tex_Para'][:n_tex_para, :], dtype=tf.float32)
    tex_param = tf.expand_dims(tex_param, 0)
    color_param = tf.constant(mat_data['Color_Para'], dtype=tf.float32)
    color_param = tf.expand_dims(color_param, 0)
    illum_param = tf.constant(mat_data['Illum_Para'], dtype=tf.float32)
    illum_param = tf.expand_dims(illum_param, 0)
    pose_param = tf.constant(mat_data['Pose_Para'], dtype=tf.float32)
    pose_param = tf.expand_dims(pose_param, 0)

    tf_bfm = TfMorphableModel(model_path='../../examples/Data/BFM/Out/BFM.mat', n_tex_para=n_tex_para)

    vertices = tf_bfm.get_vertices(
        shape_param=shape_param,
        exp_param=exp_param,
        batch_size=1
    )

    from dirt import lighting

    vertex_norm = lighting.vertex_normals(vertices, tf_bfm.triangles)
    texture = tf_bfm.get_vertex_colors(tex_param, color_param, illum_param, -vertex_norm, 1)

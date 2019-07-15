"""
modified from face3d [https://github.com/YadiraF/face3d]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from . import fit
from . import load


class MorphabelModel(object):
    """docstring for  MorphabelModel
    model: nver: number of vertices. ntri: number of triangles. *: must have. ~: can generate ones array for place holder.
            'shapeMU': [3*nver, 1].            * mean shape
            'shapePC': [3*nver, n_shape_para]. * principle component for shape
            'shapeEV': [n_shape_para, 1]. ~    * standard deviation for shape principle component
            'expMU': [3*nver, 1]. ~            * mean expression
            'expPC': [3*nver, n_exp_para]. ~   * principle component for expression
            'expEV': [n_exp_para, 1]. ~        * standard deviation for expression principle component
            'texMU': [3*nver, 1]. ~            * mean tex
            'texPC': [3*nver, n_tex_para]. ~   * principle component for tex
            'texEV': [n_tex_para, 1]. ~        * standard deviation for tex principle component
            'tri': [ntri, 3] (start from 1, should sub 1 in python and c++). *
            'tri_mouth': [114, 3] (start from 1, as a supplement to mouth triangles). ~
            'kpt_ind': [68,] (start from 1). ~
    """

    def __init__(self, model_path, model_type='BFM'):
        super(MorphabelModel, self).__init__()
        if model_type == 'BFM':
            self.model = load.load_BFM(model_path)
        else:
            print('sorry, not support other 3DMM model now')
            exit()

        # fixed attributes
        self.nver = int(self.model['shapePC'].shape[0] / 3)
        self.ntri = self.model['tri'].shape[0]
        self.n_shape_para = self.model['shapePC'].shape[1]
        self.n_exp_para = self.model['expPC'].shape[1]
        self.n_tex_para = self.model['texMU'].shape[1]

        self.kpt_ind = self.model['kpt_ind']
        self.triangles = self.model['tri']
        self.full_triangles = np.vstack((self.model['tri'], self.model['tri_mouth']))

    # ------------------------------------- shape: represented with mesh(vertices & triangles(fixed))
    def get_shape_para(self, type='random'):
        if type == 'zero':
            sp = np.random.zeros((self.n_shape_para, 1))
        elif type == 'random':
            sp = np.random.rand(self.n_shape_para, 1) * 1e04
        return sp

    def get_exp_para(self, type='random'):
        if type == 'zero':
            ep = np.zeros((self.n_exp_para, 1))
        elif type == 'random':
            ep = -1.5 + 3 * np.random.random([self.n_exp_para, 1])
            ep[6:, 0] = 0

        return ep

    def generate_vertices(self, shape_para, exp_para):
        '''
        Args:
            shape_para: (n_shape_para, 1)
            exp_para: (n_exp_para, 1) 
        Returns:
            vertices: (nver, 3)
        '''
        vertices = self.model['shapeMU'] + self.model['shapePC'].dot(shape_para) + self.model['expPC'].dot(exp_para)
        vertices = np.reshape(vertices, [int(3), int(len(vertices) / 3)], 'F').T

        return vertices

    # -------------------------------------- texture: here represented with rgb value(colors) in vertices.
    def get_tex_para(self, type='random'):
        if type == 'zero':
            tp = np.zeros((self.n_tex_para, 1))
        elif type == 'random':
            tp = np.random.rand(self.n_tex_para, 1)
        return tp

    def generate_colors(self, tex_para):
        '''
        Args:
            tex_para: (n_tex_para, 1)
        Returns:
            colors: (nver, 3)
        '''
        colors = self.model['texMU'] + self.model['texPC'].dot(tex_para * self.model['texEV'])
        colors = np.reshape(colors, [int(3), int(len(colors) / 3)], 'F').T / 255.

        return colors

    def generate_tex_xuan(self, tex_para):
        '''
        generate colors using Tex_Para
        :param tex_para: (n_tex_para, 1)
        :return: tex: (nver, 3)
        '''
        tex = self.model['texMU'] + self.model['texPC'].dot(tex_para)
        tex = np.reshape(tex, [int(3), int(len(tex) / 3)], 'F').T
        return tex

    def _generate_color_xuan(self, color_param):
        gain_r = color_param[0, 0]
        gain_g = color_param[0, 1]
        gain_b = color_param[0, 2]

        offset_r = color_param[0, 3]
        offset_g = color_param[0, 4]
        offset_b = color_param[0, 5]

        c = color_param[0, 6]

        M = np.array(
            [[0.3, 0.59, 0.11],
             [0.3, 0.59, 0.11],
             [0.3, 0.59, 0.11]]
        )

        g = np.diag([gain_r, gain_g, gain_b])
        o = np.array([offset_r, offset_g, offset_b]).reshape((-1, 1))
        # o matrix of shape(nver, 3)
        o = np.repeat(o, self.nver, axis=1).T
        return o, M, g, c

    def _generate_illuminate_xuan(self, illum_param):
        '''
        genreate illuminate params
        :param illum_param:
        :return:
        '''
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

        amb = np.diag([amb_r, amb_g, amb_b])
        dirt = np.diag([dir_r, dir_g, dir_b])

        l = np.asarray([np.cos(thetal) * np.sin(phil), np.sin(thetal), np.cos(thetal) * np.cos(phil)])
        h = l + np.asarray([0, 0, 1])
        h = h / np.sqrt(np.sum(np.square(h)))
        h = h / np.sqrt(np.sum(np.square(h)))

        return h.reshape((-1, 1)), ks, v, amb, dirt, l.reshape((-1, 1))

    def generate_tex_color_xuan(self, tex, color_param, illum_param, norm):
        '''
        generate tex color
        :param color_param:
        :param illum_param:
        :param norm: (nver, 3)
        :param n_vertex:
        :return:
        '''
        o, M, g, c = self._generate_color_xuan(color_param=color_param)
        h, ks, v, amb, dirt, l = self._generate_illuminate_xuan(illum_param=illum_param)
        # n_l of shape (n_ver, 1)
        n_l = norm.dot(l).clip(0)
        # n_h of shape (n_ver, 1)
        n_h = norm.dot(h).clip(0)
        # n_l of shape (n_ver, 3)
        n_l = np.repeat(n_l, 3, 1)
        # n_h of shape (n_ver, 3)
        n_h = np.repeat(n_h, 3, 1)

        # L of shape (n_ver, 3)
        L = tex.dot(amb) + np.multiply(n_l, tex).dot(dirt) + ks * np.power(n_h, v).dot(dirt)
        # CT of shape (3, 3)
        CT = g * (c * np.eye(3) + (1 - c) * M)
        tex_color = L.dot(CT) + o
        return tex_color

    # ------------------------------------------- transformation
    # -------------  transform
    def rotate(self, vertices, angles):
        ''' rotate face
        Args:
            vertices: [nver, 3]
            angles: [3] x, y, z rotation angle(degree)
            x: pitch. positive for looking down 
            y: yaw. positive for looking left
            z: roll. positive for tilting head right
        Returns:
            vertices: rotated vertices
        '''
        return np_3dmm.mesh_numpy.transform.rotate(vertices, angles)

    def transform(self, vertices, s, angles, t3d):
        R = np_3dmm.mesh_numpy.transform.angle2matrix(angles)
        return np_3dmm.mesh_numpy.transform.similarity_transform(vertices, s, R, t3d)

    def transform_3ddfa(self, vertices, s, angles, t3d):  # only used for processing 300W_LP data
        R = np_3dmm.mesh_numpy.transform.angle2matrix_3ddfa(angles)
        return np_3dmm.mesh_numpy.transform.similarity_transform(vertices, s, R, t3d)

    # --------------------------------------------------- fitting
    def fit(self, x, X_ind, max_iter=4, isShow=False):
        ''' fit 3dmm & pose parameters
        Args:
            x: (n, 2) image points
            X_ind: (n,) corresponding Model vertex indices
            max_iter: iteration
            isShow: whether to reserve middle results for show
        Returns:
            fitted_sp: (n_sp, 1). shape parameters
            fitted_ep: (n_ep, 1). exp parameters
            s, angles, t
        '''
        if isShow:
            fitted_sp, fitted_ep, s, R, t = fit.fit_points_for_show(x, X_ind, self.model, n_sp=self.n_shape_para,
                                                                    n_ep=self.n_exp_para, max_iter=max_iter)
            angles = np.zeros((R.shape[0], 3))
            for i in range(R.shape[0]):
                angles[i] = np_3dmm.mesh_numpy.transform.matrix2angle(R[i])
        else:
            fitted_sp, fitted_ep, s, R, t = fit.fit_points(x, X_ind, self.model, n_sp=self.n_shape_para,
                                                           n_ep=self.n_exp_para, max_iter=max_iter)
            angles = np_3dmm.mesh_numpy.transform.matrix2angle(R)
        return fitted_sp, fitted_ep, s, angles, t

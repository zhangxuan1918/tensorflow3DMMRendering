import imageio
import numpy as np

from example_utils import load_params
from tf_3dmm.mesh.render import render_batch
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel


def example_render_batch(pic_names: list, tf_bfm: TfMorphableModel, save_to_folder: str, n_tex_para:int):
    batch_size = len(pic_names)

    # pose_param: [batch, 1, n_pose_param]
    # shape_param: [batch, n_shape_para, 1]
    # exp_param:   [batch, n_exp_para, 1]
    # tex_param: [batch, n_tex_para, 1]
    # color_param: [batch, 1, n_color_para]
    # illum_param: [batch, 1, n_illum_para]

    shape_param_batch, exp_param_batch, tex_param_batch, color_param_batch, illum_param_batch, pose_param_batch = \
        load_params(pic_names=pic_names, n_tex_para=n_tex_para)

    images = render_batch(
        pose_param=pose_param_batch,
        shape_param=shape_param_batch,
        exp_param=exp_param_batch,
        tex_param=tex_param_batch,
        color_param=color_param_batch,
        illum_param=illum_param_batch,
        frame_height=450,
        frame_width=450,
        tf_bfm=tf_bfm,
        batch_size=batch_size
    )

    for i, pic_name in enumerate(pic_names):
        imageio.imsave('{folder}/rendered_{pic}.jpg'.format(folder=save_to_folder, pic=pic_name), images[i, :, :, :].numpy().astype(np.uint8))


if __name__ == '__main__':
    n_tex_para = 40
    tf_bfm = TfMorphableModel(model_path='./examples/Data/BFM/Out/BFM.mat', n_tex_para=n_tex_para)
    save_rendered_to = './output/render_batch/'
    pic_names = ['image00002', 'IBUG_image_014_01_2', 'AFW_134212_1_0', 'IBUG_image_008_1_0']
    example_render_batch(pic_names=pic_names, tf_bfm=tf_bfm, save_to_folder=save_rendered_to, n_tex_para=n_tex_para)

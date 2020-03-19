from example_utils import load_images, load_params
from sandbox.landmarks.check_get_landmarks2 import save_landmarks
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel


if __name__ == '__main__':
    n_tex_para = 40
    bfm = TfMorphableModel(model_path='/opt/project/examples/Data/BFM/Out/BFM.mat', n_tex_para=n_tex_para)
    output_folder = '/opt/project/output/landmarks/landmark3'
    pic_names = ['image00002', 'IBUG_image_014_01_2', 'AFW_134212_1_0', 'IBUG_image_008_1_0']
    # pic_names = ['IBUG_image_008_1_0']
    batch_size = len(pic_names)
    images = load_images(pic_names, '/opt/project/examples/Data')
    resolution = 450
    shape_param, exp_param, _, _, _, pose_param = load_params(pic_names=pic_names, n_tex_para=n_tex_para,
                                                              data_folder='/opt/project/examples/Data/')
    landmarks = bfm.get_landmarks(
        shape_param=shape_param,
        exp_param=exp_param,
        pose_param=pose_param,
        batch_size=batch_size,
        resolution=resolution,
        is_2d=True,
        is_plot=True
    )
    save_landmarks(images=images, landmarks=landmarks, output_folder=output_folder, resolution=resolution)
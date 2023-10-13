#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/7/5 16:54
# @Author  : Linxuan Jiang
# @File    : 评价指标.py
# @IDE     : PyCharm
# @Email   : 1195860834@qq.com
# Copyright MIT
import pandas as pd
from Metric import ID, FID, PSNR_SSIM_MAE, POSE,ID_Retrieval
import glob
import os
import shutil

class MetricScore:
    def __init__(self, Test_dir, 
                 Save_dir, 
                 checkpoint, 
                 sample=None,
                 to_video=False,
                 netG=None,
                 id_method=0,
                 simswap=False,
                 F_method=False,
                 use_f_method = False,
                 id_ls = [0,1,2]):
        
        self.sample = sample
        self.Test_dir = Test_dir
        self.Save_dir = Save_dir
        self.checkpoint = checkpoint
        self.to_video = to_video
        self.netG = netG
        self.id_method = id_method
        self.simswap = simswap
        self.F_method = F_method
        self.id_ls = id_ls
        self.use_f_method = use_f_method


    def psnr_ssim_mae_single(self, GanInverse_entity, weight, crop_size, dir_name):
        GanInverse_entity.init_swap_model(weight, crop_size=crop_size)
        GanInverse_entity.run(dir_name)



    def summary_single(self, weight, crop_size):
        temp_dir = os.path.basename(weight).split('.')[0] + "_" + f"{crop_size}"
        csv_paths = glob.glob(os.path.join(self.Save_dir, temp_dir, "*.csv"))
        for csv_path in csv_paths:
            temp_pd = pd.read_csv(csv_path, header=0)
            print(temp_pd)

    def fid(self):
        """
        Calculate FID
        """
        Fid_entity = FID(self.Test_dir, self.Save_dir, sample=self.sample, to_video=self.to_video)
        for weight, crop_size in self.checkpoint:
            Fid_entity.init_swap_model(weight, crop_size=crop_size)
            Fid_entity.run()
        del Fid_entity

    def psnr_ssim_mae(self, batch_size_inverse=4, batch_size_metric=16):
        """
        Calculate PSNR, SSIM, MAE
        """

        GanInverse_entity = PSNR_SSIM_MAE(batch_size_inverse=batch_size_inverse, batch_size_metric=batch_size_metric)

        for weight, crop_size in self.checkpoint:
            temp_dir = os.path.basename(weight).split('.')[0] + "_" + f"{crop_size}"
            dir_name = os.path.join(self.Save_dir, temp_dir, 'RowFace')
            GanInverse_entity.init_swap_model(weight, crop_size=crop_size)
            GanInverse_entity.run(dir_name)

        del GanInverse_entity

    def id(self, batch_size=16, method=0):
        """
        Calculate ID     method:  [(0, 'arcface-raw'), (1,'arcface-glin360'), (2,'curricularface')]
        """
        method_dict = {
            0: 'arcface-resnet50-raw',
            1: 'arcface-_r101_glin360',
            2: 'curricularface'
        }
        print(method_dict)
        Id_entity = ID(batch_size=batch_size, cos_metric='nn')
        for weight, crop_size in self.checkpoint:
            temp_dir = os.path.basename(weight).split('.')[0] + "_" + f"{crop_size}"
            dir_name = os.path.join(self.Save_dir, temp_dir, 'RowFace')
            Id_entity.init_id_extract(method=method, raw_face_dir=dir_name)
            Id_entity.run()
        del Id_entity

    def pose(self, batch_size=16):
        """
        Calculate POSE
        """
        Pose_entity = POSE(batch_size=batch_size)
        for weight, crop_size in self.checkpoint:
            temp_dir = os.path.basename(weight).split('.')[0] + "_" + f"{crop_size}"
            dir_name = os.path.join(self.Save_dir, temp_dir, 'RowFace')
            Pose_entity.init_dirname(dir_name)
            Pose_entity.run()
        del Pose_entity

    def calculate_all_score(self):
        self.fid(to_video=False)
        self.id(batch_size=16, method=0)
        self.psnr_ssim_mae(batch_size_inverse=4, batch_size_metric=16)
        self.pose(batch_size=16)

    def summary(self):
        for weight, crop_size in self.checkpoint:
            print('*' * 50)
            print(os.path.basename(weight), crop_size)
            print('*' * 50)
            temp_dir = os.path.basename(weight).split('.')[0] + "_" + f"{crop_size}"
            csv_paths = glob.glob(os.path.join(self.Save_dir, temp_dir, "*.csv"))
            for csv_path in csv_paths:
                temp_pd = pd.read_csv(csv_path, header=0)
                print(temp_pd)

    def run(self):

        for weight, crop_size in self.checkpoint:
            temp_dir = os.path.basename(weight).split('.')[0] + "_" + f"{crop_size}"
            dir_name = os.path.join(self.Save_dir, temp_dir, 'RowFace')
            '''--------------------------------------------------------------------'''
            Fid_entity = FID(Test_dir=self.Test_dir,
                             checkpoint = weight,
                             Save_dir=self.Save_dir,
                             sample=self.sample,
                             to_video=self.to_video,
                             crop_size=crop_size,
                             id_method=self.id_method,
                             simswap = self.simswap)
            Fid_entity.init_swap_model(netG=self.netG,
                                       weight_path=weight
                                       )
            print(Fid_entity.checkpoint_path)
            Fid_entity.run()
            Fid_entity.init_face_fusion(crop_size=crop_size)
            del Fid_entity
            '''--------------------------------------------------------------------'''

            # GanInverse_entity = PSNR_SSIM_MAE(batch_size_inverse=3,
            #                                   batch_size_metric=16)
            # self.psnr_ssim_mae_single(GanInverse_entity, weight, crop_size, dir_name)
            # del GanInverse_entity

            '''--------------------------------------------------------------------'''

            method_dict = {
                0: 'arcface-resnet50-raw',
                1: 'arcface-_r101_glin360',
                2: 'curricularface'
            }

            print(method_dict)
            Id_entity = ID(batch_size=128, cos_metric='F',use_f_method=self.use_f_method)
            for i in self.id_ls:
                Id_entity.init_id_extract(method=i, raw_face_dir=dir_name)
                Id_entity.run()
            del Id_entity

            '''--------------------------------------------------------------------'''
            Pose_entity = POSE(batch_size=128)
            Pose_entity.init_dirname(dir_name)
            Pose_entity.run()
            del Pose_entity
            '''--------------------------------------------------------------------'''
            print('*' * 50)
            self.summary_single(weight,crop_size)

        print('*' * 50)
        print('*' * 50)
        print('*' * 50)
        self.summary()

def init_temp_dir(temp_result_dir):
    print(" 删除缓存文件夹并重新创建 ")
    if os.path.exists(temp_result_dir):  # 删除整个文件夹并重新创建
        shutil.rmtree(temp_result_dir)  # 删除目录及其下面所有内容。
        os.makedirs(temp_result_dir, exist_ok=True)  # 重建文件夹
    else:
        os.makedirs(temp_result_dir, exist_ok=True)

if __name__ == "__main__":
    """
    cos_metric 采用F 比采用nn要高，一个可能原因是训练时用的F
    """
    Test_dir = '/media/shuangmu/F/Face_256/c0_ffmpeg'

    from models.Simswap import Generator
    checkpoint = [['Weight/latest_net_G.pth', 224]]
    Save_dir = '/media/shuangmu/F/Face_256/Metric/simswap' #0.8867,0.9019,0.8858, 0.8989

    # from models.mobilenet.model_2_ngate import Generator
    # checkpoint = [['checkpoints/two_image_onelatent/latest_net_G.pth', 256]] #0.9266,0.9486,0.9402, 0.946
    # Save_dir = '/media/shuangmu/F/Face_256/Metric/test'

    # from models.mobilenet.model_2 import Generator
    # checkpoint = [['Weight/mobilenet2_G.pth', 256]] 
    # Save_dir = '/media/shuangmu/F/Face_256/Metric/test'

    # from models.mobilenet.model_2_ngate import Generator
    # checkpoint = [['checkpoints/exp_method1_id24_down4_0.5ssim/latest_net_G.pth', 256]] 
    # Save_dir = '/media/shuangmu/F/Face_256/Metric/test'
    # id_method=1
    # from models.fasternet.fast_4 import Generator


    # from models.mobilenet.model_2_ngate import Generator
    # checkpoint = [['BEST_MODEL/best_5/latest_net_G.pth', 256]] 
    # Save_dir = '/media/shuangmu/F/Face_256/Metric/best_5'

    # init_temp_dir(Save_dir)
    # metric_entity = MetricScore(Test_dir=Test_dir, 
    #                             Save_dir=Save_dir, 
    #                             checkpoint=checkpoint, 
    #                             sample=None, 
    #                             netG=Generator(),
    #                             id_method=id_method,
    #                             simswap=False,
    #                             F_method=False,
    #                             id_ls = [0,1,2])
    # metric_entity.run()


    temp_dir = os.path.basename(checkpoint[0][0]).split('.')[0] + "_" + f"{checkpoint[0][1]}"
    dir_name = os.path.join(Save_dir, temp_dir, 'RowFace')
    Id_entity = ID_Retrieval(batch_size=512, cos_metric='F',use_f_method=False)
    for i in [0,1,2,3]:
        Id_entity.init_id_extract(method=i, raw_face_dir=dir_name)
        Id_entity.run()

    """
    'checkpoints/exp_method1_id24_down4_0.5ssim/latest_net_G.pth', 256
    **************************************************
        MODEL       FID
    0    Gnet  8.518880
    1  Gnet-F  6.198755
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.644234                     0.119226
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028715                     0.028715
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.669281                      0.088625
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.022833                      0.022833
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.628347               0.082978
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019185               0.019185
        MODEL      POSE
    0    Gnet  1.304962
    1  Gnet-F  1.227651
    ----------------------------------------------------------
    exp_method1_256_id24_loss_unsim_ssim24_1011
        MODEL       FID
    0    Gnet  8.139439
    1  Gnet-F  5.911146
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.644618                     0.096370
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028715                     0.028715
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.666393                      0.067481
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.022833                      0.022833
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.625789               0.065955
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019185               0.019185
        MODEL      POSE
    0    Gnet  1.442214
    1  Gnet-F  1.355322

    'checkpoints/exp_loss4_ssim0.5/latest_net_G.pth', 256
            MODEL       FID
    0    Gnet  9.606197
    1  Gnet-F  6.397835
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.638205                     0.065722
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028715                     0.028715
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.661512                      0.030005
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.022833                      0.022833
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.618850               0.035444
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019185               0.019185
        MODEL      POSE
    0    Gnet  1.806848
    1  Gnet-F  1.572746
    'checkpoints/exp_loss3/latest_net_G.pth', 256
            MODEL       FID
    0    Gnet  8.805003
    1  Gnet-F  6.313281
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.624416                     0.117727
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028715                     0.028715
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.650392                      0.082212
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.022833                      0.022833
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.606065               0.077645
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019185               0.019185
        MODEL      POSE
    0    Gnet  1.528222
    1  Gnet-F  1.375123
    'checkpoints/exp_loss2/latest_net_G.pth', 256
    MODEL       FID
    0    Gnet  9.989612
    1  Gnet-F  6.904281
    MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.645222                     0.058823
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028715                     0.028715
    MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.662083                      0.032783
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.022833                      0.022833
    MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.621489               0.034364
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019185               0.019185
    MODEL      POSE
    0    Gnet  1.567531
    1  Gnet-F  1.662275

    'checkpoints/exp_loss1/latest_net_G.pth', 256
    MODEL       FID
    0    Gnet  8.766448
    1  Gnet-F  6.572459
    MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.626134                     0.094625
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028715                     0.028715
    MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.645255                      0.062571
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.022835                      0.022835
    MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.603601               0.061626
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019184               0.019184
    MODEL      POSE
    0    Gnet  1.579084
    1  Gnet-F  1.611229
    BEST_MODEL/best_5/latest_net_G.pth
    **************************************************
        MODEL       FID
    0    Gnet  5.375508
    1  Gnet-F  3.995161
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.593975                     0.279399
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028715                     0.028715
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.626420                      0.262140
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.022834                      0.022834
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.579731               0.236792
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019184               0.019184
        MODEL      POSE
    0    Gnet  0.977376
    -----models.mobilenet.model_2_ngate--------'checkpoints/exp_method1_256_id24/latest_net_G.pth'', 256 -- id_method=1
            MODEL       FID
    0    Gnet  7.698661
    1  Gnet-F  5.678393
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.605839                     0.244053
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028715                     0.028715
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.635595                      0.227558
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.022835                      0.022835
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.590199               0.204858
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019184               0.019184
        MODEL      POSE
    0    Gnet  1.048010
    -----models.mobilenet.model_2_ngate--------'checkpoints/exp_method1_256_id24_info/latest_net_G.pth'', 256 -- id_method=1
        MODEL       FID
    0    Gnet  7.087771
    1  Gnet-F  5.218985
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.639823                     0.160720
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028715                     0.028715
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.667118                      0.132474
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.022833                      0.022833
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.622379               0.121831
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019185               0.019185
        MODEL      POSE
    0    Gnet  1.067330
    1  Gnet-F  0.976813
    -----models.mobilenet.model_2_ngate--------'checkpoints/exp_method1_256_id24_info_l1/latest_net_G.pth'', 256 -- id_method=1
        MODEL       FID
    0    Gnet  8.065987
    1  Gnet-F  6.056970
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.644381                     0.082272
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028714                     0.028714
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.668889                      0.045838
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.022833                      0.022833
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.624764               0.048599
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019185               0.019185
        MODEL      POSE
    0    Gnet  1.106605
    1  Gnet-F  1.056823

    -----models.mobilenet.model_2_ngate--------'checkpoints/exp_method1_256_id24_unsim/latest_net_G.pth'', 256 -- id_method=1
            MODEL       FID
    0    Gnet  8.492877
    1  Gnet-F  6.102316
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.649624                     0.083922
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028715                     0.028715
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.671950                      0.051629
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.022835                      0.022835
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.631170               0.053429
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019184               0.019184
        MODEL      POSE
    0    Gnet  1.486395
    1  Gnet-F  1.451417
    -----models.mobilenet.model_2_ngate--------'checkpoints/exp_method1_256_id25_unsim/latest_net_G.pth'', 256 -- id_method=1
    MODEL       FID
    0    Gnet  9.390716
    1  Gnet-F  6.599170
    MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.644206                     0.070424
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028715                     0.028715
    MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.660798                      0.043964
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.022833                      0.022833
    MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.620762               0.045153
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019183               0.019183
    MODEL      POSE
    0    Gnet  1.568503
    1  Gnet-F  1.574257
    -----models.mobilenet.model_2_ngate--------'checkpoints/exp_method1_256_id25_info/latest_net_G.pth'', 256 -- id_method=1
            MODEL       FID
    0    Gnet  8.120604
    1  Gnet-F  5.952565
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.614246                     0.200456
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028715                     0.028715
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.640153                      0.181608
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.022835                      0.022835
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.595706               0.161841
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019184               0.019184
        MODEL      POSE
    0    Gnet  1.324056
    1  Gnet-F  1.272185
    -----models.mobilenet.model_2_ngate--------'checkpoints/exp_method1_256_id25/latest_net_G.pth', 256 -- id_method=1
        MODEL       FID
    0    Gnet  8.155545
    1  Gnet-F  5.773780
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.603844                     0.229897
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028715                     0.028715
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.629463                      0.222150
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.022833                      0.022833
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.583996               0.196617
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019185               0.019185
        MODEL      POSE
    0    Gnet  1.297213
    1  Gnet-F  1.143704
    -----models.mobilenet.model_2_ngate--------'checkpoints/exp_method1_256_id30/latest_net_G.pth', 256 -- id_method=1
        MODEL       FID
    0    Gnet  9.187846
    1  Gnet-F  6.776178
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.614381                     0.177905
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028715                     0.028715
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.634669                      0.165472
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.022833                      0.022833
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.589770               0.145877
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019185               0.019185
        MODEL      POSE
    0    Gnet  1.545638
    1  Gnet-F  1.343247

    先在128上训练，然后在256上训练，
    -----models.mobilenet.model_2_ngate--------'checkpoints/exp_method1_256/latest_net_G.pth', 256 -- id_method=1,id35
        MODEL        FID
    0    Gnet  10.159967
    1  Gnet-F   7.103802
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.625798                     0.157699
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028715                     0.028715
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.648624                      0.146912
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.022835                      0.022835
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.603917               0.130314
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019184               0.019184
        MODEL      POSE
    0    Gnet  1.946520
    1  Gnet-F  1.424858

    -----models.mobilenet.model_2_ngate--------BEST_MODEL/best_5/latest_net_G.pth', 256 -- id_method=2
            MODEL       FID
    0    Gnet  5.585063
    1  Gnet-F  4.484363
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.551149                     0.119593
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028715                     0.028715
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.543423                      0.108544
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.022833                      0.022833
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.553492               0.087714
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019185               0.019185
        MODEL      POSE
    0    Gnet  3.260376
    1  Gnet-F  2.286715
    -----models.mobilenet.model_2_ngate--------checkpoints/exp_method1/latest_net_G.pth', 128 -- id_method=1
        MODEL        FID
    0    Gnet  15.388784
    1  Gnet-F   6.329062
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.598619                     0.193621
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028332                     0.028332
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.622460                      0.174933
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.023046                      0.023046
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.576401               0.159750
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019372               0.019372
        MODEL      POSE
    0    Gnet  1.339041
    1  Gnet-F  1.339061
    -----models.mobilenet.model_2_ngate--------checkpoints/exp_method5/latest_net_G.pth', 128 -- id_method=5
    # ID 模型中得到的cos 和实际类似， 有很强的潜力，鲁棒性强
        MODEL        FID
    0    Gnet  11.990592
    1  Gnet-F   5.355272
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.571485                     0.183824
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028332                     0.028332
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.572437                      0.183241
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.023046                      0.023046
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.551759               0.154673
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019372               0.019372
        MODEL      POSE
    0    Gnet  3.470295
    1  Gnet-F  1.735479
    -----models.mobilenet.model_2--------BEST/best_best/latest_net_G.pth -- id_method=0
            MODEL       FID
    0    Gnet  3.286227
    1  Gnet-F  2.482551
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.560967                     0.187361
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028715                     0.028715
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.423345                      0.334195
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.022833                      0.022833
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.424014               0.268192
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019184               0.019184
        MODEL      POSE
    0    Gnet  1.050658
    1  Gnet-F  0.932501
    -----models.mobilenet.model_2--------Weight/mobilenet2_G.pth
        MODEL       FID
    0    Gnet  2.889784
    1  Gnet-F  2.261879
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.528722                     0.222057
    1  Gnet-F                     0.000000                     0.000000
    2    Base                     0.028715                     0.028715
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.396037                      0.370395
    1  Gnet-F                      0.000000                      0.000000
    2    Base                      0.022833                      0.022833
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.393481               0.302994
    1  Gnet-F               0.000000               0.000000
    2    Base               0.019185               0.019185
        MODEL      POSE
    0    Gnet  0.837756
    1  Gnet-F  0.792316

    -----model_2_ngate--------checkpoints/two_image_onelatent/latest_net_G.pth
        MODEL       FID
    0    Gnet  4.639507
    1  Gnet-F  3.324602
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.537728                     0.160230
    1  Gnet-F                     0.510345                     0.199990
    2    Base                     0.028714                     0.028714
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.407460                      0.293097
    1  Gnet-F                      0.382684                      0.333173
    2    Base                      0.022833                      0.022833
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.405647               0.230011
    1  Gnet-F               0.380319               0.268019
    2    Base               0.019185               0.019185
        MODEL      POSE
    0    Gnet  1.334107
    1  Gnet-F  1.348326
    -----model_2_ngate--------checkpoints/one_image_onelatent/latest_net_G.pth
        MODEL       FID
    0    Gnet  4.941706
    1  Gnet-F  3.347489
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.535698                     0.149793
    1  Gnet-F                     0.508802                     0.191804
    2    Base                     0.028715                     0.028715
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.408979                      0.277030
    1  Gnet-F                      0.383935                      0.319602
    2    Base                      0.022833                      0.022833
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.406115               0.216746
    1  Gnet-F               0.380660               0.257535
    2    Base               0.019185               0.019185
        MODEL      POSE
    0    Gnet  1.494259
    1  Gnet-F  1.459788
    -------------------------------Simswap---------------------------------
        MODEL        FID
    0    Gnet  11.356283
    1  Gnet-F   6.604606
        MODEL  ID-arcface-resnet50-raw-pos  ID-arcface-resnet50-raw-neg
    0    Gnet                     0.585811                     0.105856
    1  Gnet-F                     0.546471                     0.167207
    2    Base                     0.028615                     0.028615
        MODEL  ID-arcface-_r101_glin360-pos  ID-arcface-_r101_glin360-neg
    0    Gnet                      0.483300                      0.173577
    1  Gnet-F                      0.444176                      0.242107
    2    Base                      0.022909                      0.022909
        MODEL  ID-curricularface-pos  ID-curricularface-neg
    0    Gnet               0.478514               0.142962
    1  Gnet-F               0.438620               0.203385
    2    Base               0.019192               0.019192
        MODEL       POSE
    0    Gnet  12.165376
    1  Gnet-F  13.498561








































    one_image_onelatent
    latest_net_G.pth 256
    **************************************************
        MODEL       FID
    0    Gnet  4.942100
    1  Gnet-F  3.347143
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.574897
    1  Gnet-F                 0.595898
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.638516
    1  Gnet-F                  0.659797
        MODEL  ID-curricularface
    0    Gnet           0.608375
    1  Gnet-F           0.628764
        MODEL      POSE
    0    Gnet  1.490359
    1  Gnet-F  1.457454


    checkpoints/test2/latest_net_G.pth
    latest_net_G.pth 256
    **************************************************
        MODEL        FID
    0    Gnet  16.348403
    1  Gnet-F  12.120818
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.536208
    1  Gnet-F                 0.553528
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.576030
    1  Gnet-F                  0.596112
        MODEL  ID-curricularface
    0    Gnet           0.556673
    1  Gnet-F           0.574959
        MODEL      POSE
    0    Gnet  2.084277
    1  Gnet-F  1.989987  


    test2_only_vgg
    latest_net_G.pth 256
    **************************************************
        MODEL        FID
    0    Gnet  11.620630
    1  Gnet-F   6.852443
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.549697
    1  Gnet-F                 0.574197
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.606332
    1  Gnet-F                  0.632421
        MODEL  ID-curricularface
    0    Gnet           0.579220
    1  Gnet-F           0.603437
        MODEL      POSE
    0    Gnet  1.638854
    1  Gnet-F  2.204605
    exp_6_256
    latest_net_G.pth 256
    **************************************************
        MODEL        FID
    0    Gnet  10.659465
    1  Gnet-F   5.813633
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.551439
    1  Gnet-F                 0.574626
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.604674
    1  Gnet-F                  0.629727
        MODEL  ID-curricularface
    0    Gnet           0.578626
    1  Gnet-F           0.601866
        MODEL      POSE
    0    Gnet  1.390874
    1  Gnet-F  1.331099
    exp_2
    latest_net_G.pth 128
    **************************************************
        MODEL        FID
    0    Gnet  26.329309
    1  Gnet-F  10.448769
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.537032
    1  Gnet-F                 0.553731
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.578044
    1  Gnet-F                  0.596872
        MODEL  ID-curricularface
    0    Gnet           0.556627
    1  Gnet-F           0.572827
        MODEL      POSE
    0    Gnet  4.061800
    1  Gnet-F  4.520305
        exp_3  128
        latest_net_G.pth 128
    **************************************************
        MODEL        FID
    0    Gnet  22.209988
    1  Gnet-F   9.824739
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.538404
    1  Gnet-F                 0.553513
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.583418
    1  Gnet-F                  0.600350
        MODEL  ID-curricularface
    0    Gnet           0.560076
    1  Gnet-F           0.574628
        MODEL      POSE
    0    Gnet  3.704512
    1  Gnet-F  3.087899

    exp_1   256
    latest_net_G.pth 128
    **************************************************
        MODEL        FID
    0    Gnet  17.992832
    1  Gnet-F   8.288809
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.546063
    1  Gnet-F                 0.559708
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.595718
    1  Gnet-F                  0.610347
        MODEL  ID-curricularface
    0    Gnet           0.570037
    1  Gnet-F           0.583020
        MODEL      POSE
    0    Gnet  3.997867
    1  Gnet-F  4.129488



    vgg smmoth_l1 and l1 BEST/best_5/latest_net_G.pth
    **************************************************
        MODEL        FID
    0    Gnet  10.944681
    1  Gnet-F   7.930268
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.560868
    1  Gnet-F                 0.578358
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.623386
    1  Gnet-F                  0.641245
        MODEL  ID-curricularface
    0    Gnet           0.592848
    1  Gnet-F           0.609383
        MODEL      POSE
    0    Gnet  1.771925
    1  Gnet-F  1.710781
    simswap latest_net_G.pth 224
    **************************************************
        MODEL        FID
    0    Gnet  11.356712
    1  Gnet-F   6.604936
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.552927
    1  Gnet-F                 0.583603
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.586785
    1  Gnet-F                  0.621050
        MODEL  ID-curricularface
    0    Gnet           0.571480
    1  Gnet-F           0.601692
        MODEL       POSE
    0    Gnet  12.164730
    1  Gnet-F  13.535238

    BEST/best_best/latest_net_G.pth
    *************************************************
        MODEL       FID
    0    Gnet  3.286646
    1  Gnet-F  2.482298
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.593678
    1  Gnet-F                 0.614796
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.667098
    1  Gnet-F                  0.687178
        MODEL  ID-curricularface
    0    Gnet           0.634094
    1  Gnet-F           0.653804
        MODEL      POSE
    0    Gnet  1.051131
    1  Gnet-F  0.928451


    
    /home/shuangmu/Project/paper_model/BEST/best_gmain5_id1_unsim5/latest_net_G.pth
    **************************************************
        MODEL       FID
    0    Gnet  7.946644
    1  Gnet-F  5.715031
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.608282
    1  Gnet-F                 0.625203
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.586887
    1  Gnet-F                  0.615123
        MODEL  ID-curricularface
    0    Gnet           0.583131
    1  Gnet-F           0.607820
        MODEL      POSE
    0    Gnet  2.177493
    1  Gnet-F  2.057135
    
    best_gmain5_id1/latest_net_G.pthjl
    **************************************************
        MODEL       FID
    0    Gnet  7.148264
    1  Gnet-F  5.145038
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.615839
    1  Gnet-F                 0.632473
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.600154
    1  Gnet-F                  0.626229
        MODEL  ID-curricularface
    0    Gnet           0.593606
    1  Gnet-F           0.616997
        MODEL      POSE
    0    Gnet  1.537446
    1  Gnet-F  1.636296
    BEST/best_gmain5_id2/latest_net_G.pth
    **************************************************
        MODEL       FID
    0    Gnet  7.416072
    1  Gnet-F  6.071415
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.599939
    1  Gnet-F                 0.620524
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.588114
    1  Gnet-F                  0.617608
        MODEL  ID-curricularface
    0    Gnet           0.575613
    1  Gnet-F           0.603110
        MODEL      POSE
    0    Gnet  2.574566
    1  Gnet-F  1.130942

    BEST/best_gmain10/latest_net_G.pth
    **************************************************
        MODEL       FID
    0    Gnet  5.355265
    1  Gnet-F  3.940472
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.573499
    1  Gnet-F                 0.591325
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.632628
    1  Gnet-F                  0.651575
        MODEL  ID-curricularface
    0    Gnet           0.604238
    1  Gnet-F           0.622054
        MODEL      POSE
    0    Gnet  1.624276
    1  Gnet-F  1.461710

    Gmain  BEST/best_gmain5/latest_net_G.pth
        **************************************************
        MODEL       FID
    0    Gnet  4.806973
    1  Gnet-F  3.910180
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.596465
    1  Gnet-F                 0.617517
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.661881
    1  Gnet-F                  0.683676
        MODEL  ID-curricularface
    0    Gnet           0.630448
    1  Gnet-F           0.651580
        MODEL      POSE
    0    Gnet  1.136642
    1  Gnet-F  1.106697


    BEST/best_vgg8/latest_net_G.pth
    **************************************************
        MODEL       FID
    0    Gnet  7.605381
    1  Gnet-F  5.339598
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.566256
    1  Gnet-F                 0.586868
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.623592
    1  Gnet-F                  0.645450
        MODEL  ID-curricularface
    0    Gnet           0.595283
    1  Gnet-F           0.615730
        MODEL      POSE
    0    Gnet  2.646768
    1  Gnet-F  1.912466
    
    BEST/best_s4_2/latest_net_G.pth
    **************************************************
        MODEL       FID
    0    Gnet  5.912260
    1  Gnet-F  4.668538
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.587732
    1  Gnet-F                 0.607838
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.652520
    1  Gnet-F                  0.673335
        MODEL  ID-curricularface
    0    Gnet           0.621006
    1  Gnet-F           0.641066
        MODEL      POSE
    0    Gnet  1.297721
    1  Gnet-F  1.024062
        BEST/best_base/latest_net_G.pth
    **************************************************
        MODEL       FID
    0    Gnet  4.694711
    1  Gnet-F  3.945529
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.581665
    1  Gnet-F                 0.600370
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.643173
    1  Gnet-F                  0.662381
        MODEL  ID-curricularface
    0    Gnet           0.612870
    1  Gnet-F           0.631379
        MODEL      POSE
    0    Gnet  1.482001
    1  Gnet-F  1.023544


    BEST/best_ngate/latest_net_G.pth
    **************************************************
        MODEL       FID
    0    Gnet  4.312363
    1  Gnet-F  3.238121
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.590892
    1  Gnet-F                 0.610121
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.655328
    1  Gnet-F                  0.675383
        MODEL  ID-curricularface
    0    Gnet           0.623607
    1  Gnet-F           0.642949
        MODEL      POSE
    0    Gnet  1.350450
    1  Gnet-F  1.170334

    '/home/shuangmu/Project/Baseline_v6/exp/mobilenet2_bilinear/latest_net_G.pth'
    **************************************************
        MODEL       FID
    0    Gnet  2.449056
    1  Gnet-F  1.974430
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.606074
    1  Gnet-F                 0.625227
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.677487
    1  Gnet-F                  0.695815
        MODEL  ID-curricularface 
    0    Gnet           0.644037
    1  Gnet-F           0.662240
        MODEL      POSE
    0    Gnet  0.726894
    1  Gnet-F  0.656434

    /home/shuangmu/Project/paper_model/Weight/mobilenet2_G.pth
    **************************************************
        MODEL       FID
    0    Gnet  2.666826
    1  Gnet-F  2.089470
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.610503
    1  Gnet-F                 0.629885
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.684859
    1  Gnet-F                  0.703291
        MODEL  ID-curricularface
    0    Gnet           0.650878
    1  Gnet-F           0.669149
        MODEL      POSE
    0    Gnet  0.747317
    1  Gnet-F  0.669419

    BEST_MODEL/mobilenet_2_nounsim/latest_net_G.pth
    **************************************************
        MODEL       FID
    0    Gnet  5.032731
    1  Gnet-F  3.369673
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.574729
    1  Gnet-F                 0.596016
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.642640
    1  Gnet-F                  0.664148
        MODEL  ID-curricularface
    0    Gnet           0.611268
    1  Gnet-F           0.632163
        MODEL      POSE
    0    Gnet  1.121573
    1  Gnet-F  0.942537
    

    BEST_MODEL/mobilenet_2_unsim_threshold0/latest_net_G.pth
            MODEL       FID
    0    Gnet  5.071332
    1  Gnet-F  3.559055
        MODEL  ID-arcface-resnet50-raw
    0    Gnet                 0.548016
    1  Gnet-F                 0.570742
        MODEL  ID-arcface-_r101_glin360
    0    Gnet                  0.611484
    1  Gnet-F                  0.635434
        MODEL  ID-curricularface
    0    Gnet           0.583058
    1  Gnet-F           0.605970
        MODEL      POSE
    0    Gnet  1.400141
    1  Gnet-F  1.259123
    """







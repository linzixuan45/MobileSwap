#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/7/5 16:54
# @Author  : Linxuan Jiang
# @File    : 评价指标.py
# @IDE     : PyCharm
# @Email   : 1195860834@qq.com
# Copyright MIT
import pandas as pd
from Metric import ID, FID, PSNR_SSIM_MAE, POSE, ID_Retrieval
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
	             use_f_method=False,
	             id_ls=[0, 1, 2]):

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
			                 checkpoint=weight,
			                 Save_dir=self.Save_dir,
			                 sample=self.sample,
			                 to_video=self.to_video,
			                 crop_size=crop_size,
			                 id_method=self.id_method,
			                 simswap=self.simswap)
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
			Id_entity = ID(batch_size=128, cos_metric='F', use_f_method=self.use_f_method)
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
			self.summary_single(weight, crop_size)

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
	# from models.mobilenet.model_2 import Generator
	from models.Simswap import Generator

	swap_model = Generator()
	'''init model weight & crop_size'''
	# checkpoint = [['Weight/10_net_G.pth', 256]]
	checkpoint = [['Weight/latest_net_G.pth', 224]]

	# '''init data path & output path'''
	# Test_dir = 'F:\Face_dataset\FF++\youtube\FF_test_dataset\c0_raw'
	# Save_dir = 'F:\Face_dataset\FF++\youtube\Metric_FF_Score\c0_mobilenet_unsim_test'
	# metric_entity = MetricScore(Test_dir=Test_dir, Save_dir=Save_dir, checkpoint=checkpoint, sample=None, netG=swap_model)
	# metric_entity.run()

	# Fid_entity = FID(Test_dir, Save_dir, sample=10, to_video=True)
	# for weight, crop_size in checkpoint:
	#     Fid_entity.init_swap_model(weight, crop_size=crop_size)
	#     Fid_entity.run()
	# del Fid_entity

	# /media/shuangmu/F/Face_256

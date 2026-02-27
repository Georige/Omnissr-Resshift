import ssl
ssl._create_default_https_context = ssl._create_unverified_context

"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
import sys
sys.path.append('../')
sys.path.append('./')

from multi_viewer import MultiViewer
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import math
import copy
import torch.nn.functional as F
import torchvision.utils as tvu
import cv2
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization

from overlapping_tile import partion_overlapping_window, reverse_overlapping_window

def space_timesteps(num_timesteps, section_counts):# 1000步里面选出要走的步数
	"""
	Create a list of timesteps to use from an original diffusion process,
	given the number of timesteps we want to take from equally-sized portions
	of the original process.
	For example, if there's 300 timesteps and the section counts are [10,15,20]
	then the first 100 timesteps are strided to be 10 timesteps, the second 100
	are strided to be 15 timesteps, and the final 100 are strided to be 20.
	If the stride is a string starting with "ddim", then the fixed striding
	from the DDIM paper is used, and only one section is allowed.
	:param num_timesteps: the number of diffusion steps in the original
						  process to divide up. 总步数
	:param section_counts: either a list of numbers, or a string containing
						   comma-separated numbers, indicating the step count
						   per section. As a special case, use "ddimN" where N
						   is a number of steps to use the striding from the
						   DDIM paper. 想走多少步
	:return: a set of diffusion steps from the original process to use.
	"""
	if isinstance(section_counts, str): # 查看传进来的是不是字符串
		if section_counts.startswith("ddim"): # 是不是以ddim开头
			desired_count = int(section_counts[len("ddim"):])# ddim50把50抠出来
			for i in range(1, num_timesteps): # 暴力遍历从跨度1开始，能不能刚好走50步
				if len(range(0, num_timesteps, i)) == desired_count:
					return set(range(0, num_timesteps, i))
			raise ValueError(
				f"cannot create exactly {num_timesteps} steps with an integer stride"
			) # 无法整除，就报错
		section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
	size_per = num_timesteps // len(section_counts)
	extra = num_timesteps % len(section_counts) # 除不尽的余数放在extra里面，放到前几段
	start_idx = 0 # 当前步数记录器
	all_steps = [] # 空篮子，装我们选上的步数
	for i, section_count in enumerate(section_counts): # 开始遍历每一段
		size = size_per + (1 if i < extra else 0) # 确定这一段的真是长度
		if size < section_count:
			raise ValueError( # 安全检查
				f"cannot divide section of {size} steps into {section_count}"
			)
		if section_count <= 1:
			frac_stride = 1
		else:
			frac_stride = (size - 1) / (section_count - 1) # 种树问题的间隔
		cur_idx = 0.0 # 这一段的小游标
		taken_steps = [] # 小篮子，装这一段选出来的步数
		for _ in range(section_count):
			taken_steps.append(start_idx + round(cur_idx)) 
			cur_idx += frac_stride # 游标向前挪动一个完美间距
		all_steps += taken_steps # 把这一段选好的点，倒进大篮子里
		start_idx += size # 把起点挪到下一段开头
	return set(all_steps)

def chunk(it, size): # 将输出切成一个个包（元组），每个包有size个数据，一直做到元组为空
	it = iter(it)
	return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
	print(f"Loading model from {ckpt}") # 读取模型参数文件
	pl_sd = torch.load(ckpt, map_location="cpu") # 将模型加载到cpu
	if "global_step" in pl_sd:
		print(f"Global Step: {pl_sd['global_step']}")
	sd = pl_sd["state_dict"] # key：是层的名字， value：具体的数字矩阵
	model = instantiate_from_config(config.model) # 创建模型
	m, u = model.load_state_dict(sd, strict=False) # 将sd权重字典加载到模型中，非严格模式
	if len(m) > 0 and verbose:
		print("missing keys:")
		print(m)
	if len(u) > 0 and verbose:
		print("unexpected keys:")
		print(u)

	model.cuda() # 模型搬到GPU
	model.eval() # 模型在推理而不是训练
	return model

def load_img(path): # 加载图片输出[-1,1]
	image = Image.open(path).convert("RGB") # 加载图片
	w, h = image.size  # pil_img.size = [w, h]
	print(f"loaded input image of size (width:{w}, height:{h}) from {path}")
	w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 32 
	image = image.resize((w, h), resample=PIL.Image.LANCZOS) # 高质量缩放算法，防止图片出现锯齿或模糊
	image = np.array(image).astype(np.float32) / 255.0 # [0,255]映射到[0,1]
	image = image[None].transpose(0, 3, 1, 2) # 维度重排 1,512,512,3变到 1,3,512,512
	image = torch.from_numpy(image) # 把numpy变成torch
	return 2.*image - 1. # [0,1]变到[-1,1]


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--init-img",
		type=str,
		nargs="?",
		help="path to the input image",
		default="inputs/user_upload"
	)
	parser.add_argument(
		"--outdir",
		type=str,
		nargs="?",
		help="dir to write results to",
		default="outputs/user_upload"
	)
	parser.add_argument(
		"--ddpm_steps",
		type=int,
		default=1000,
		help="number of ddpm sampling steps",
	)
	parser.add_argument(
		"--C",
		type=int,
		default=4,
		help="latent channels",
	)
	parser.add_argument(
		"--f",
		type=int,
		default=8,
		help="downsampling factor, most often 8 or 16", # 512图片变成64维度
	)
	parser.add_argument( # 同时并行出多少张结果（让用户可以选择）
		"--n_samples",
		type=int,
		default=2,
		help="how many samples to produce for each given prompt. A.k.a batch size",
	)
	parser.add_argument( # 如何搭建这个神经网络的架构
		"--config", 
		type=str,
		default="configs/stableSRNew/v2-finetune_text_T_512.yaml",
		help="path to config which constructs model",
	)
	parser.add_argument( # 这个神经网络架构里面的系数
		"--ckpt",
		type=str,
		default="./stablesr_000117.ckpt",
		help="path to checkpoint of model",
	)
	parser.add_argument( # 映射到潜空间的模型
		"--vqgan_ckpt",
		type=str,
		default="./vqgan_cfw_00011.ckpt",
		help="path to checkpoint of VQGAN model",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="the seed (for reproducible sampling)",
	)
	parser.add_argument(
		"--precision",
		type=str,
		help="evaluate at this precision",
		choices=["full", "autocast"],
		default="autocast"
	)
	parser.add_argument( # 模型每次看多大一块区域
		"--input_size",
		type=int,
		default=512,
		help="input size",
	)
	parser.add_argument( # 0 完全AI， 1.0 忠于原图
		"--dec_w",
		type=float,
		default=0.5,
		help="weight for combining VQGAN and Diffusion",
	)
	parser.add_argument(
		"--tile_overlap", # 窗口重复的步长
		type=int,
		default=32,
		help="tile overlap size",
	)
	parser.add_argument( # 超分辨率的倍数
		"--upscale",
		type=float,
		default=4.0,
		help="upsample scale",
	)
	parser.add_argument( # 调色
		"--colorfix_type",
		type=str,
		default="nofix",
		help="Color fix type to adjust the color of HR result according to LR input: adain (used in paper); wavelet; nofix",
	)

	parser.add_argument(
		"--dist_k_fold", # 数据要分成多少分
		type=str,
		default=None,
		help="Whether use distributed sampling, and the k-fold index",
	)

	opt = parser.parse_args()
	seed_everything(opt.seed)

	print('>>>>>>>>>>color correction>>>>>>>>>>>')
	if opt.colorfix_type == 'adain':
		print('Use adain color correction')
	elif opt.colorfix_type == 'wavelet':
		print('Use wavelet color correction')
	else:
		print('No color correction')
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

	config = OmegaConf.load(f"{opt.config}")
	model = load_model_from_config(config, f"{opt.ckpt}")   # UNet Model
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model = model.to(device)

	model.configs = config

	vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml") # 读取编码器的配置文件
	vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt) # 注入编码器的系数
	vq_model = vq_model.to(device) 
	vq_model.decoder.fusion_w = opt.dec_w  # from args: dec_w 给编码器增加一个融合权重的系数

	with torch.no_grad():
		factor = int(opt.upscale)
		def bicubic_kernel(x, a=-0.5): # 定义双三次插值，是一个核函数
			if abs(x) <= 1:
				return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
			elif 1 < abs(x) and abs(x) < 2:
				return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
			else:
				return 0
		k = np.zeros((factor * 4)) # 创建一个长度16的数组
		for i in range(factor * 4):
			x = (1 / factor)*(i - np.floor(factor*4/2) +0.5)
			k[i] = bicubic_kernel(x) # 将上面的数字变成16个具体的数字
		k = k / np.sum(k) # 归一化
		kernel = torch.from_numpy(k).float().cuda() 
		from svd_replacement import SRConv # 导入一个自定义的卷积层
		H_funcs = SRConv(kernel / kernel.sum(), \
							3, 256, 'cuda', stride = factor) # 做下采样，大图变成小图，步长为放大倍数4
		model.H_funcs = H_funcs # 将高清图扔给这个函数，可以得到一个严格按照双三次插值算法缩小的低清图
		# 为了将生成处理的高清图片变回低清的，然后比较原图和这个低清图
		H_funcs_cpu = SRConv(kernel.cpu() / kernel.sum().cpu(), \
							3, 256, 'cpu', stride = factor) # 放到cpu里面使用

	# OmniSSR Hyperparameters #
	gamma_l = 0.5    # gamma_l for latent z interpolation, default: 0.5 平滑过渡潜空间向量
	gamma_e = 1.0   # gamma_e for gradient decomposition in erp, default: 1.0 
	model.gamma_l = gamma_l
	model.gamma_e = gamma_e

	# Best ERP<->TAN config #
	pre_upscale = 4  # preupsampling in OTII 上采样系数，在切片之前先将全景图放大四倍，然后再切片
	nrows = 4 # 从北极到南极要分层多少层
	fov = (75, 75) # 平面图覆盖球面上多少度的范围 75*75
	patch_size = (512, 512)  # half of the gt height 切出来平面图的像素尺寸
	
	multi_viewer = MultiViewer() # 切球面以及还原球面的工具

	model.pre_upscale = pre_upscale
	model.nrows = nrows
	model.fov = fov
	# model.patch_size = (int(patch_size[0]/opt.f), int(patch_size[1]/opt.f))

	model.hr_erp_size = int(1024), int(2048)  # gt size

	if 'nrows' not in opt.outdir: # 把参数都写进文件夹里面
		outpath = opt.outdir + f"_gamma-latent-{gamma_l}_gamma-erp-{gamma_e}_input-size-{opt.input_size}_pre-upscale-{pre_upscale}_nrows-{nrows}_fov-{fov[0]}-{fov[1]}_patchsize-{patch_size[0]}-{patch_size[1]}"
	else:
		outpath = opt.outdir
	os.makedirs(outpath, exist_ok=True)

	batch_size = opt.n_samples  # default: 2

	img_list_ori = list(filter(lambda f: ".png" in f, os.listdir(opt.init_img))) # 读取 png数据
	img_list_ori = sorted(img_list_ori) # 数据排序
	if opt.dist_k_fold is not None: # 数据分发到GPU里面
		dist_k, dist_n = map(int, opt.dist_k_fold.split('/'))
		img_list_ori = list(chunk(img_list_ori, math.ceil(len(img_list_ori) / dist_n)))[dist_k-1]
		img_list_ori = list(img_list_ori)
		print(f"Using distributed sampling, k={dist_k}, n={dist_n}")

	img_list = copy.deepcopy(img_list_ori)
	init_image_list = []
	lr_image_list = []
	for item in img_list_ori:
		if os.path.exists(os.path.join(outpath, item)):
			img_list.remove(item)
			continue # 只能断点续传
			
		cur_image = load_img(os.path.join(opt.init_img, item))  # load image, transform to torch.tensor [1, C, H, W] range: [-1 ~ +1]
		lr_image_list.append(cur_image) # 储存原始小图
		# max size: 1800 x 1800 for V100
		if opt.upscale > 1.0: # 预处理放大
			cur_image = F.interpolate(
					cur_image,
					size=(int(cur_image.size(-2)*opt.upscale),
						int(cur_image.size(-1)*opt.upscale)),
					mode='bicubic', align_corners=True
					)
			# print(f"cur_image shape:{cur_image.shape}")
		init_image_list.append(cur_image) # 存放大后的模糊大图

	model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
						  linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3) # 制定加噪声计划，总共1000步，线性加噪声
	model.num_timesteps = 1000

	sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod) # 每一步原始图片的占比
	sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod) # 每一步噪声的占比

	use_timesteps = set(space_timesteps(1000, [opt.ddpm_steps])) # opt.ddpm_steps想走多少步， use_timesteps选中的步数
	last_alpha_cumprod = 1.0
	new_betas = []
	timestep_map = []
	for i, alpha_cumprod in enumerate(model.alphas_cumprod): # 重新计算跳步之间的噪声大小
		if i in use_timesteps:
			new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
			last_alpha_cumprod = alpha_cumprod
			timestep_map.append(i)
	new_betas = [beta.data.cpu().numpy() for beta in new_betas]
	model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas)) # 覆盖旧的时间表，模型会认为走50步就够了
	model.num_timesteps = 1000 # 保持逻辑上的总步数标记
	model.ori_timesteps = list(use_timesteps) # 记录50个新步数在原始1000步里面的真实位置
	model.ori_timesteps.sort()
	model = model.to(device)

	precision_scope = autocast if opt.precision == "autocast" else nullcontext  # default: autocast
		

	with torch.no_grad():
		with precision_scope("cuda"):
			with model.ema_scope(): # 使用平滑的EMA权重加载到模型上，换上后备箱里面的更好的权重
				all_samples = list()
				for n in trange(len(init_image_list), desc="Sampling"):
					tic = time.time()
					init_image = init_image_list[n]
					init_image = init_image.clamp(-1.0, 1.0)

					basename = os.path.splitext(os.path.basename(img_list[n]))[0]
					os.makedirs(f"{outpath}/sr_tan_examples/X{opt.upscale}/{basename}", exist_ok=True) # 储存AI画好的每一个高清切片
					os.makedirs(f"{outpath}/lr_tan_examples/X{opt.upscale}/{basename}", exist_ok=True) # 存原始的低清切片
					os.makedirs(f"{outpath}/erp_output", exist_ok=True)

					x_recon_path = f"{outpath}/x_recon/X{opt.upscale}/{basename}" # 存扩散模型每一步的去噪过程图
					os.makedirs(f"{outpath}/x_recon/X{opt.upscale}/{basename}", exist_ok=True)
					model.x_recon_path = x_recon_path
					model.x_recon_save_count = opt.ddpm_steps

					# 将球面图大卸八块
					pers, _, _, _ = multi_viewer.equi2pers(init_image.cpu(), fov=fov, nrows=nrows, patch_size=patch_size, pre_upscale=model.pre_upscale)
					pers = pers.cuda() # 切片堆
					# pers of shape: [N, C, p_h, p_w, patch_num=18]
					init_pers_inputs = []
					latent_pers_inputs = []

					tan_weights = model._gaussian_weights(tile_width=pers.shape[3], tile_height=pers.shape[2], nbatches=pers.shape[0]*pers.shape[-1]).float()
					tan_weights = tan_weights[:, :pers.shape[1], :, :] 
					tan_weights_pers = rearrange(tan_weights, '(B Np) C H W -> B C H W Np', Np=pers.shape[-1]) # 调整蒙板的维度和切片图完全对其
					model.img_tan_weights_pers = tan_weights_pers # 高斯权重，一张蒙版，中间为1，边缘过渡到0
					model.init_pers = pers # 存切片的高清画布
					model.lr_erp = lr_image_list[n].to(device) # 存切片的低清原图

					# Tangent Loop at Encoder#
					for idx in range(pers.shape[-1]):
						sub_init_image = pers[..., idx] # 取出第idx张切片
						# tvu.save_image(torch.clamp((sub_init_image + 1.0) / 2.0, min=0, max=1), f"{outpath}/lr_tan_examples/X{opt.upscale}/{basename}/lr_tan_{idx:02}.png")
						# print(f'>>>>>>>>>>>Tangent Loop [{idx:02}]>>>>>>>>>>>>')
						# print(sub_init_image.size())
						ori_size = None
						# 编码压缩
						init_template = sub_init_image
						init_pers_inputs.append(init_template)

						init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_template))  # move to latent space, LR image input through VQGAN Encoder, and output latent z
						latent_pers_inputs.append(init_latent) # 收集编码之后的潜变量

					init_pers_in = torch.stack(init_pers_inputs, dim=-1)  # [B, C, Hp, Wp, Np]  本身init_pers_inputs是一个张量列表，将张量列表打包成一个张量

					latent_pers_in = torch.stack(latent_pers_inputs, dim=-1)  # [B, C, Hp, Wp, Np]
					model.num_tan_patch = pers.shape[-1]
					latent_pers_in = rearrange(latent_pers_in, 'B C H W Np -> (B Np) C H W') # 把18个切片伪装成18个图片
					# 给模型的空指令，即无条件生成
					text_init = ['']*opt.n_samples  # n_samples (default): 2 
					semantic_c = model.cond_stage_model(text_init)  # [n_samples, 77, 1024]

					noise = torch.randn_like(latent_pers_in) # 制造纯噪声
					# If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
					t = repeat(torch.tensor([999]), '1 -> b', b=latent_pers_in.size(0))  # shape [B*Np,] 将时间点设为终点
					t = t.to(device).long()
					# 把原图和噪声混合，q_sample_respace是向前扩散过程，现在是纯噪声，但是可以看出来一点点图片的样子
					x_T = model.q_sample_respace(x_start=latent_pers_in, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)  # q_sample: 加噪。对 Latent LR 加噪，得到 x_T
						# x_T = noise

					# 逆向采样，最耗时间的过程，samples 是全新的，细节丰富的潜变量图
					samples, _ = model.sample_canvas_pano(cond=semantic_c, struct_cond=latent_pers_in, batch_size=latent_pers_in.size(0), timesteps=opt.ddpm_steps, time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True, tile_size=int(opt.input_size/8), tile_overlap=opt.tile_overlap, batch_size_sample=opt.n_samples)

					# 重新把张量变回18张切片
					samples_pers = rearrange(samples, "(B Np) C H W -> B C H W Np", Np=pers.shape[-1])

					final_pers_outputs = []
					for idx in range(pers.shape[-1]):
						init_template = init_pers_in[..., idx] # 原始低清切片的像素
						samples = samples_pers[..., idx]	# AI生成的高清切片潜变量
						_, enc_fea_lq = vq_model.encode(init_template) # 提取原始图的特征
						sub_x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq) # 解码高清切片潜变量，输入原始图的特征是为了，更好地保留原图结构，ResShift没有这个结构
						# sub_x_samples = model.decode_first_stage(samples_pers[..., idx]) # if no skip in encoder and decoder

						# if ori_size is not None:
						# 	sub_x_samples = sub_x_samples[:, :, :ori_size[-2], :ori_size[-1]]
						# 解决色偏问题
						if opt.colorfix_type == 'adain':
							sub_x_samples = adaptive_instance_normalization(sub_x_samples, init_template)  # 改进，针对tan patch做color fix
						elif opt.colorfix_type == 'wavelet':
							sub_x_samples = wavelet_reconstruction(sub_x_samples, init_template)
						# 将处理完，色彩准确的高清切片存入列表，准备最后的拼接
						final_pers_outputs.append(sub_x_samples)
						# tvu.save_image(torch.clamp((sub_x_samples + 1.0) / 2.0, min=0, max=1), f"{outpath}/sr_tan_examples/X{opt.upscale}/{basename}/sr_tan_{idx:02}.png")

					# Tangent Loop End 
					# 将列表转化为张量
					final_pers_out = torch.stack(final_pers_outputs, dim=-1)  # [B, C, H, W, Np]
					# 生成高斯分布蒙板，并且维度对其，这样每一个切片都有一个自己的专属蒙版
					tan_weights = model._gaussian_weights(tile_width=final_pers_out.shape[3], tile_height=final_pers_out.shape[2], nbatches=final_pers_out.shape[0]*final_pers_out.shape[-1]).float()
					tan_weights = tan_weights[:, :final_pers_out.shape[1], :, :]
					tan_weights_pers = rearrange(tan_weights, '(B Np) C H W -> B C H W Np', Np=final_pers_out.shape[-1])


					torch.cuda.empty_cache() # 清理显存垃圾
					# 带权重贴回到球体上
					weighted_erp_tensor = multi_viewer.pers2equi((final_pers_out * tan_weights_pers).cpu(), fov=fov, nrows=nrows, erp_size=(1024, 2048), pre_upscale=model.pre_upscale)
					# 制作权重地图，热力图球面，每个位置承载了多少比例的贡献
					erp_weights = multi_viewer.pers2equi(tan_weights_pers.cpu(), fov=fov, nrows=nrows, erp_size=(1024, 2048), pre_upscale=model.pre_upscale)
					# 实现平滑过渡
					erp_tensor = weighted_erp_tensor / erp_weights
					x_samples = erp_tensor

					# 像素归一化 【-1，1】到 [0,1]
					x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

					toc = time.time()
					# print(f"Runtime per ERP: {toc - tic:.2f}s")
					# 转换到255然后保存	
					for i in range(init_image.size(0)):
						x_sample = 255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
						Image.fromarray(x_sample.astype(np.uint8)).save(
							os.path.join(f"{outpath}/erp_output", basename+'.png'))
						# init_image = torch.clamp((init_image + 1.0) / 2.0, min=0.0, max=1.0)
						# init_image = 255. * rearrange(init_image[i].cpu().numpy(), 'c h w -> h w c')
						# Image.fromarray(init_image.astype(np.uint8)).save(
						# 	os.path.join(outpath, basename+'_lq.png'))


	print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
		f" \nEnjoy.")
	
	return 0


if __name__ == "__main__":
	main()

'''切片保存'''
import os
from torchvision.utils import save_image
from torchvision.io import read_image

'''ResShift'''
import os, sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from sampler import ResShiftSampler

from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url

'''Omnissr'''
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from einops import rearrange, repeat
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


'''resshift'''
_STEP = {
    'v1': 15,
    'v2': 15,
    'v3': 4,
    'bicsr': 4,
    'inpaint_imagenet': 4,
    'inpaint_face': 4,
    'faceir': 4,
    'deblur': 4,
    }
_LINK = {
    'vqgan': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth',
    'vqgan_face256': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/celeba256_vq_f4_dim3_face.pth',
    'vqgan_face512': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/ffhq512_vq_f8_dim8_face.pth',
    'v1': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s15_v1.pth',
    'v2': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s15_v2.pth',
    'v3': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s4_v3.pth',
    'bicsr': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_bicsrx4_s4.pth',
    'inpaint_imagenet': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_inpainting_imagenet_s4.pth',
    'inpaint_face': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_inpainting_face_s4.pth',
    'faceir': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_faceir_s4.pth',
    'deblur': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_deblur_s4.pth',
     }



def get_parser_res(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output path.")
    parser.add_argument("--mask_path", type=str, default="", help="Mask path for inpainting.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--bs", type=int, default=1, help="Batch size.")
    parser.add_argument(
            "-v",
            "--version",
            type=str,
            default="v1",
            choices=["v1", "v2", "v3"],
            help="Checkpoint version.",
            )
    parser.add_argument(
            "--chop_size",
            type=int,
            default=512,
            choices=[512, 256, 64],
            help="Chopping forward.",
            )
    parser.add_argument(
            "--chop_stride",
            type=int,
            default=-1,
            help="Chopping stride.",
            )
    parser.add_argument(
            "--task",
            type=str,
            default="realsr",
            choices=['realsr', 'bicsr', 'inpaint_imagenet', 'inpaint_face', 'faceir', 'deblur'],
            help="Chopping forward.",
            )
    parser.add_argument(
        "--colorfix_type",
        type = str,
        default="adain",
        choices=['adain','wavelet']
    )
    args = parser.parse_args()

    return args

def get_configs_res(args):
    ckpt_dir = Path('./weights')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()

    if args.task == 'realsr':
        if args.version in ['v1', 'v2']:
            configs = OmegaConf.load('./configs/realsr_swinunet_realesrgan256.yaml')
        elif args.version == 'v3':
            configs = OmegaConf.load('./configs/realsr_swinunet_realesrgan256_journal.yaml')
        else:
            raise ValueError(f"Unexpected version type: {args.version}")
        assert args.scale == 4, 'We only support the 4x super-resolution now!'
        ckpt_url = _LINK[args.version]
        ckpt_path = ckpt_dir / f'resshift_{args.task}x{args.scale}_s{_STEP[args.version]}_{args.version}.pth'
        vqgan_url = _LINK['vqgan']
        vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    elif args.task == 'bicsr':
        configs = OmegaConf.load('./configs/bicx4_swinunet_lpips.yaml')
        assert args.scale == 4, 'We only support the 4x super-resolution now!'
        ckpt_url = _LINK[args.task]
        ckpt_path = ckpt_dir / f'resshift_{args.task}x{args.scale}_s{_STEP[args.task]}.pth'
        vqgan_url = _LINK['vqgan']
        vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    elif args.task == 'inpaint_imagenet':
        configs = OmegaConf.load('./configs/inpaint_lama256_imagenet.yaml')
        assert args.scale == 1, 'Please set scale equals 1 for image inpainting!'
        ckpt_url = _LINK[args.task]
        ckpt_path = ckpt_dir / f'resshift_{args.task}_s{_STEP[args.task]}.pth'
        vqgan_url = _LINK['vqgan']
        vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    elif args.task == 'inpaint_face':
        configs = OmegaConf.load('./configs/inpaint_lama256_face.yaml')
        assert args.scale == 1, 'Please set scale equals 1 for image inpainting!'
        ckpt_url = _LINK[args.task]
        ckpt_path = ckpt_dir / f'resshift_{args.task}_s{_STEP[args.task]}.pth'
        vqgan_url = _LINK['vqgan_face256']
        vqgan_path = ckpt_dir / f'celeba256_vq_f4_dim3_face.pth'
    elif args.task == 'faceir':
        configs = OmegaConf.load('./configs/faceir_gfpgan512_lpips.yaml')
        assert args.scale == 1, 'Please set scale equals 1 for face restoration!'
        ckpt_url = _LINK[args.task]
        ckpt_path = ckpt_dir / f'resshift_{args.task}_s{_STEP[args.task]}.pth'
        vqgan_url = _LINK['vqgan_face512']
        vqgan_path = ckpt_dir / f'ffhq512_vq_f8_dim8_face.pth'
    elif args.task == 'deblur':
        configs = OmegaConf.load('./configs/deblur_gopro256.yaml')
        assert args.scale == 1, 'Please set scale equals 1 for deblurring!'
        ckpt_url = _LINK[args.task]
        ckpt_path = ckpt_dir / f'resshift_{args.task}_s{_STEP[args.task]}.pth'
        vqgan_url = _LINK['vqgan']
        vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    else:
        raise TypeError(f"Unexpected task type: {args.task}!")

    # prepare the checkpoint
    if not ckpt_path.exists():
         load_file_from_url(
            url=ckpt_url,
            model_dir=ckpt_dir,
            progress=True,
            file_name=ckpt_path.name,
            )
    if not vqgan_path.exists():
         load_file_from_url(
            url=vqgan_url,
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
            )
    # diffusion模型选择，编码器选择，scale参数注入
    configs.model.ckpt_path = str(ckpt_path) # set the checkpoint path for ResShift
    configs.diffusion.params.sf = args.scale # set the scale factor
    configs.autoencoder.ckpt_path = str(vqgan_path) # set the checkpoint path for VQGAN

    # save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)

    if args.chop_stride < 0:
        if args.chop_size == 512:
            chop_stride = (512 - 64) * (4 // args.scale)
        elif args.chop_size == 256:
            chop_stride = (256 - 32) * (4 // args.scale)
        elif args.chop_size == 64:
            chop_stride = (64 - 16) * (4 // args.scale)
        else:
            raise ValueError("Chop size must be in [512, 256]")
    else:
        chop_stride = args.chop_stride * (4 // args.scale)
    args.chop_size *= (4 // args.scale)
    print(f"Chopping size/stride: {args.chop_size}/{chop_stride}")

    return configs, chop_stride

'''omnissr'''

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

def chunk(it, size): # 将输出切成一个个包（元组），每个包有size个数据，一直做到元组为空
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


'''空壳模型'''

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 伪装参数：根据你的原代码，这里通常是 False 或者 1
        self.pre_upscale = False 
        
        # 你可以随时给它动态塞入属性，防止报错
        self.x_recon_path = ""
        self.x_recon_save_count = 0

    # 2. 伪装核心方法：生成高斯融合权重 (极其重要，否则拼接会有缝隙)
    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)  # outer product of two vectors
        return torch.tile(torch.tensor(weights, device=self.betas.device), (nbatches, self.configs.model.params.channels, 1, 1))

def main():
    # 加载空壳模型
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 实例化空壳模型，并把它推到 GPU 上 (虽然它没有任何权重)
    model = DummyModel().to(device)
    
    # 缩小函数
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

    # 模型系数
    model.gamma_l = 0.5
    model.gamma_e = 1.0
    print("✅ 空壳模型 (DummyModel) 加载完毕，绕过 UNet 权重！")
    
    # 加载ResShift
    
    args_res = get_parser_res()

    configs_res, chop_stride_res = get_configs_res(args_res)
    ## 创建ResShift对象
    resshift_sampler = ResShiftSampler(
            configs_res,
            sf=args_res.scale,
            chop_size=args_res.chop_size, # 切块大小
            chop_stride=chop_stride_res, # 窗口滑动距离
            chop_bs=1, # 切块批处理次数，如果是2，GPU一下子处理2块，决定并行计算的吞吐量
            use_amp=True, # 需不需要使用混合精度
            seed=args_res.seed,
            padding_offset=configs_res.model.params.get('lq_size', 64), # 检测是否尺寸是下采样的倍数，如果不是，那就补上黑边
            )
    resshift_sampler.H_funcs = model.H_funcs
    ## setting mask path for inpainting
    if args_res.task.startswith('inpaint'):
        assert args_res.mask_path, 'Please input the mask path for inpainting!'
        mask_path = args_res.mask_path
    else:
        mask_path = None
        
    # 数据预处理
    ## Best ERP<->TAN config #
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
    
    model = model.to(device)
    
    with torch.no_grad():
        ## 数据读取
        img_list_ori = list(filter(lambda f: ".png" in f, os.listdir(args_res.in_path))) # 读取 png数据
        img_list_ori = sorted(img_list_ori) # 数据排序
        img_list = copy.deepcopy(img_list_ori)
        init_image_list = []
        lr_image_list = []
        ## 图片放大
        for item in img_list_ori:
            if os.path.exists(os.path.join(args_res.out_path, item)):
                img_list.remove(item)
                continue # 支持断点续传
                
            cur_image = load_img(os.path.join(args_res.in_path, item))  # load image, transform to torch.tensor [1, C, H, W] range: [-1 ~ +1]
            # 图片会变到[-1,1]
            lr_image_list.append(cur_image) # 储存原始小图
            # max size: 1800 x 1800 for V100
            if args_res.scale > 1.0: # 预处理放大
                cur_image = F.interpolate(
                        cur_image,
                        size=(int(cur_image.size(-2)*args_res.scale),
                            int(cur_image.size(-1)*args_res.scale)),
                        mode='bicubic', align_corners=True
                        )
                # print(f"cur_image shape:{cur_image.shape}")
            init_image_list.append(cur_image) # 存放大后的模糊大图

        ## 取第一张图片出来
        init_image = init_image_list[0]
        
        basename = os.path.splitext(os.path.basename(img_list_ori[0]))[0]
        
        
        # 切片
        ## 执行
        pers, _, _, _ = multi_viewer.equi2pers(
            init_image.cpu(), 
            fov=fov, 
            nrows=nrows, 
            patch_size=patch_size, 
            pre_upscale=pre_upscale)
        ## 将最后一个维度调整到前面来
        pers_in = rearrange(pers, 'B C H W Np -> (B Np) C H W')# 变成 18 C H W
        ## 保存到切片文件夹
        # 1. 准备文件夹
        pers_dir = "pers"
        os.makedirs(pers_dir, exist_ok=True)  # 如果文件夹不存在则自动创建
        pers_out = "pers_out"
        os.makedirs(pers_out, exist_ok = True)
        # 2. 遍历并保存这 18 张图
        num_images = pers_in.shape[0]
        for i in range(num_images):
            # 取出单张图片的 Tensor，形状为 [C, H, W]
            single_img = pers_in[i]
            
            # 定义文件名，使用 :02d 让数字对齐，比如 patch_00.png, patch_01.png
            file_path = os.path.join(pers_dir, f"patch_{i:02d}.png")
            
            # 3. 核心保存动作！
            # 【高能预警】：请根据你的模型数据范围选择下面的一行代码
            
            # 情况 A：如果你的数据是 Diffusion 模型常见的 [-1, 1] 范围
            save_image(single_img, file_path, normalize=True, value_range=(-1, 1))
            
            # 情况 B：如果你的数据已经是 [0, 1] 的规范化浮点数
            # save_image(single_img, file_path)
            
        print(f"✅ 成功！{num_images} 张切片已保存至 {pers_dir} 文件夹。")
        
        tan_weights = model._gaussian_weights(tile_width=pers.shape[3], tile_height=pers.shape[2], nbatches=pers.shape[0]*pers.shape[-1]).float()
        tan_weights = tan_weights[:, :pers.shape[1], :, :] 
        tan_weights_pers = rearrange(tan_weights, '(B Np) C H W -> B C H W Np', Np=pers.shape[-1]) # 调整蒙板的维度和切片图完全对其
        resshift_sampler.img_tan_weights_pers = tan_weights_pers # 高斯权重，一张蒙版，中间为1，边缘过渡到0
        resshift_sampler.init_pers = pers # 存切片的高清画布
        resshift_sampler.lr_erp = lr_image_list[0].to(device) # 存切片的低清原图
        
        # 处理十八张切片然后储存到pers_out中，现在已经是255的图片了
        resshift_sampler.inference(
                pers_dir,
                pers_out,
                mask_path=mask_path,
                bs=args_res.bs,
                noise_repeat=False
                ) 
        
        ## 
        # 1. 基础配置
        input_dir = pers_out # 你之前存图片的文件夹
        num_patches = 18                # 切片总数
        patch_tensors = []

        # 2. 严格按顺序读取图片
        for i in range(num_patches):
            # 强制按编号读取，保证顺序绝对正确 (patch_00.png, patch_01.png...)
            file_path = os.path.join(input_dir, f"patch_{i:02d}.png") 
            
            # read_image 直接返回形状为 [C, H, W] 的张量，数值范围是 0-255 (uint8)
            img_tensor = read_image(file_path) 
            patch_tensors.append(img_tensor)

        # 3. 堆叠成一个大的 Batch 张量
        # 此时形状为 [18, 3, H, W] 也就是 [(B Np), C, H, W]
        stacked_tensor = torch.stack(patch_tensors, dim=0)

        # 4. 关键：像素归一化 (Value Range Normalization)
        # 将 0~255 的整数转换为浮点数
        # stacked_tensor = stacked_tensor.float() / 255.0  # 现在是 [0.0, 1.0] 范围

        # 【重要判断】：大部分扩散模型（如 SD/VQGAN）底层的输入是 [-1, 1]
        # 如果你的模型需要 [-1, 1]，请取消下面这行的注释：
        stacked_tensor = stacked_tensor * 2.0 - 1.0 

        # 5. 可选：变回全景图专用的 5D 维度 [B, C, H, W, Np]
        # 假设我们现在的 Batch Size 是 1
        pers_out = rearrange(stacked_tensor, '(B Np) C H W -> B C H W Np', B=1, Np=num_patches)

        print(f"✅张量切片读取完毕！最终张量形状为: {pers_out.shape}, 数据范围: [{pers_out.min():.2f}, {pers_out.max():.2f}]")
        
        # 解决色偏问题
        from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization
        final_pers_out = []
        for idx in range(pers_out[-1]):
            if args_res.colorfix_type == 'adain': #
                sub_pers = adaptive_instance_normalization(pers_out[...,idx],pers[...,idx]) 
            elif args_res.colorfix_type == 'wavelet':
                sub_pers = wavelet_reconstruction(sub_pers[...,idx],pers[...,idx])
            final_pers_out.append(sub_pers)
            
        ## 列表转化为张量
        final_pers_out = torch.stack(final_pers_out, dim = -1) # [B,C,H,W,Np]
        
        ## 生成高斯分布蒙版
        tan_weights = model._gaussian_weights(tile_width = final_pers_out.shape[3], tile_height=final_pers_out.shape[2], nbatches=final_pers_out.shape[0]*final_pers_out.shape[-1]).float()
        tan_weights = tan_weights[:,:final_pers_out.shape[1],:,:]
        tan_weights_pers = rearrange(tan_weights, '(B Np) C H W -> B C H W Np',Np = final_pers_out.shape[-1])
        
        ## 清理显存垃圾
        torch.cuda.empty_cache()
        
        ## 带权重贴合回球体上
        weighted_erp_tensor = multi_viewer.pers2equi((final_pers_out * tan_weights_pers).cpu(), fov=fov, nrows=nrows, erp_size=(1024, 2048), pre_upscale=model.pre_upscale)
        ## 制作权重地图，热力图球面
        erp_weights = multi_viewer.pers2equi(tan_weights_pers.cpu(), fov=fov, nrows=nrows, erp_size=(1024, 2048), pre_upscale=model.pre_upscale)
        erp_tensor = weighted_erp_tensor / erp_weights
        
        # 像素归一化从 [-1,1]到[0,1]
        x_sample =  erp_tensor
        x_sample = torch.clamp((x_sample + 1.0)/2.0 , min = 0.0, max = 1.0)
        
        for i in range(init_image.size[0]):
            x_sample = 255. * rearrange(x_sample[i].cpu().numpy(),
                                        Image.fromarray(x_sample.astype(np.uint8)).save(
                                            os.path.join(f"{args_res.out_path}/erp_output",basename + '.png') 
                                        ))
            
            
    print(f"成品完成:{args_res.out_path}") 
    
if __name__ == '__main__':
    main()

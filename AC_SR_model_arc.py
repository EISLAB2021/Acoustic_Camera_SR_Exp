from Sound_wave_attribute_simulation import random_add_speckle_noise_on_AC_IMG

import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


@MODEL_REGISTRY.register()
class Acoustic_Camera_SR_Model(SRGANModel):
    """migrate Real-ESRGAN model as the basic Acoustic camera (AC)-SR model.
    Core block:
    1. To randomly synthesize LR acoustic camera images from input HR data.
    2. To optimize the networks with AC-SR model training.
    """

    def __init__(self, opt):
        super(Acoustic_Camera_SR_Model, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # Simulate JPEG compression artifacts fo acoustic files
        self.usm_sharpener = USMSharp().cuda()  # Apply USM sharpening processing
        self.queue_size = opt.get('queue_size', 180)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """
        Training pair pool for increasing the diversity in a batch.
        """
        # Initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0

        if self.queue_ptr == self.queue_size:  # The pool is full
            # Do dequeue and enqueue
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # Only enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr += b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept acoustic camera input data, and then add two-order degradations to obtain LR acoustic images."""
        if self.is_train and self.opt.get('high_order_degradation', True):
            # Training data synthesis of acoustic camera images
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # Blur on acoustic camera images
            out = filter2D(self.gt_usm, self.kernel1)
            # Random resize of raw data
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])  # set by user

            out = F.interpolate(out, scale_factor=scale, mode=mode)
            ##############################################################################
            # Add noise on acoustic camera images--Simulating sound wave attribute
            gray_noise_prob = self.opt['gray_noise_prob']
            tmp_random = np.random.uniform()
            if tmp_random < self.opt['gaussian_noise_prob_1']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            elif tmp_random < self.opt['gaussian_noise_prob_2']:
                out = random_add_poisson_noise_pt(
                    out, scale_range=self.opt['poisson_scale_range'], gray_prob=gray_noise_prob, clip=True,
                    rounds=False)
            else:
                out = random_add_speckle_noise_on_AC_IMG(
                    out, shape_parameter=self.opt['shape_parameter'], scale_parameter=self.opt['scale_parameter'],
                    clip=True, rounds=False)
            # JPEG compression on raw acoustic sensor files
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # Clamp to [0, 1] to avoid unpleasant artifacts from JPEGer
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # Blur processing
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # Random resize processing
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out,
                                size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)),
                                mode=mode)
            # Add noise on acoustic camera images
            gray_noise_prob = self.opt['gray_noise_prob2']
            tmp_random2 = np.random.uniform()
            if tmp_random2 < self.opt['gaussian_noise_prob2_1']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            elif tmp_random2 < self.opt['gaussian_noise_prob2_2']:
                out = random_add_poisson_noise_pt(
                    out, scale_range=self.opt['poisson_scale_range2'], gray_prob=gray_noise_prob, clip=True,
                    rounds=False)
            else:
                out = random_add_speckle_noise_on_AC_IMG(
                    out, shape_parameter=self.opt['shape_parameter2'], scale_parameter=self.opt['scale_parameter2'],
                    clip=True, rounds=False)

            # JPEG compression on raw acoustic sensor files,video-like to JPEG format

            if np.random.uniform() < 0.5:

                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression on raw acoustic file
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                lq = self.jpeger(out, quality=jpeg_p)
            else:

                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)

                mode = random.choice(['area', 'bilinear', 'bicubic'])
                lq = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                lq = filter2D(lq, self.sinc_kernel)

            # Clamp and round
            clamp_and_round = False
            if clamp_and_round:
                lq = torch.clamp((lq * 255.0).round(), 0, 255) / 255.

            self.lq = lq.contiguous()  # for the training of VGG feature extractor

            # Crop borders
            gt_size = self.opt['gt_size']
            self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt['scale'])

            # Training pair pool
            self._dequeue_and_enqueue()
        else:
            # Validation data synthesis
            self.lq = data['lq'].to(self.device)
            self.gt = data['gt'].to(self.device)

    def calculate_perceptual_loss(self):
        l_g_percep = 0
        l_g_style = 0
        if self.cri_fea:
            real_fea = self.net_fea(self.gt).detach()
            fake_fea = self.net_fea(self.lq)
            l_g_percep = self.cri_fea(fake_fea, real_fea)
        if self.cri_style:
            real_style = self.net_style(self.gt).detach()
            fake_style = self.net_style(self.lq)
            l_g_style = self.cri_style(fake_style, real_style)
        return l_g_percep, l_g_style

    def calculate_gradient_penalty(self, k=''):
        # refer to wgan design
        alpha = torch.rand(self.gt.size(0), 1, 1, 1).to(self.device)

        # compute interpolation
        interpolates = alpha * self.gt + ((1 - alpha) * self.lq)
        interpolates = Variable(interpolates, requires_grad=True).to(self.device)

        disc_interpolates = self.net_d(interpolates)
        gradients = torch_grad(outputs=disc_interpolates, inputs=interpolates,
                               grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * k   # k is set by user
        return gradient_penalty

    def optimize_parameters(self, current_iter):
        # When sampling per iteration, the optimizer step size is determined
        # by 'num_cycles' * 'max_iters' / 'batch_size',use batch_size
        # 'num_cycles' and 'max_iters' to get the exact number of iterations.
        optimizer_step = (current_iter // self.opt['num_cycles']) * self.opt['max_iters'] / self.opt['batch_size']
        current_cycle = current_iter // self.opt['num_cycles']

        # generator and discriminator
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()
        l_g_total = 0
        l_d_total = 0

        # discriminator
        real_out = self.net_d(self.gt)
        l_d_real = F.relu(1.0 - real_out).mean()
        fake_out = self.net_d(self.lq.detach())
        l_d_fake = F.relu(1.0 + fake_out).mean()
        l_d_total += l_d_real + l_d_fake
        l_d_total.backward()
        self.optimizer_D.step()

        # generator
        fake_out = self.net_d(self.lq)
        l_g_gan = -fake_out.mean()
        l_g_total += l_g_gan

        # Gradient penalty introduced
        l_gp = self.calculate_gradient_penalty()
        l_g_total += l_gp

        # Reconstruction loss of AC-SR model
        l_g_pix = self.cri_pix(self.lq, self.gt)
        l_g_total += l_g_pix

        # Perceptual loss of AC-SR model
        l_g_percep, l_g_style = self.calculate_perceptual_loss()
        l_g_total += l_g_percep + l_g_style

        l_g_total.backward()
        self.optimizer_G.step()

        # Log recording
        self.log_dict = OrderedDict()
        self.log_dict['l_g_gan'] = l_g_gan.item()
        self.log_dict['l_g_pix'] = l_g_pix.item()
        self.log_dict['l_g_percep'] = l_g_percep.item()
        self.log_dict['l_g_style'] = l_g_style.item()
        self.log_dict['l_gp'] = l_gp.item()
        self.log_dict['l_d_real'] = l_d_real.item()
        self.log_dict['l_d_fake'] = l_d_fake.item()
        self.log_dict['l_d_total'] = l_d_total.item()
        self.log_dict['l_g_total'] = l_g_total.item()

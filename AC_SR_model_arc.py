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
    2. To approximate the complex acoustic camera image degradation process.
    3. To optimize the networks with AC-SR model training.
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

        if self.queue_ptr == self.queue_size:
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
        "Degradation process is flexible, and new degradation sub-modules can be designed by oneself and embedded."
        "Can also perform degradation preprocessing on the raw input data and then feed it to the model"
        "External scripts can provide common acoustic camera image preprocessing and degradation operations"

        if self.is_train and self.opt.get('high_order_degradation', True):
            # Training data synthesis of acoustic camera images
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # -------- The first degradation process on acoustic camera images----------- #
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
            mode = random.choice(['area', 'bilinear', 'bicubic'])  # set by user, others like Nearest Neighbor.

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
            # Superimposed noise degradation can be added optionally

            # JPEG compression on raw acoustic sensor files
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, -0.5, 1.5)  # adopt partial artifacts
            out = F.relu(out)  # highlight nonlinear

            out = self.jpeger(out, quality=jpeg_p)

            # Acoustic shadow model degradation is coming.

            # -------Custom degenerate submodule embedding---------#

            # ---------------- The second degradation process on acoustic camera images-------------------- #
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
            mode = random.choice(['area', 'bilinear', 'bicubic'])  # set by user, like Nearest Neighbor.
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

            # Superimposed noise degradation can be added optionally
            # JPEG compression on raw acoustic sensor files,video-like to JPEG format

            if np.random.uniform() < 0.5:

                mode = random.choice(['area', 'bilinear', 'bicubic'])  # set by user, others like Nearest Neighbor.
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression on raw acoustic file
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, -0.5, 1.5)  # adopt partial artifacts
                out = F.relu(out)  # highlight nonlinear

                lq = self.jpeger(out, quality=jpeg_p)
            else:

                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, -0.5, 1.5)  # adopt partial artifacts
                out = F.relu(out)
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

    def _gradient_penalty(self, real_data, generated_data, k=None):
        """
        Compute gradient penalty, refer to (WGAN).

        Args:
        - real_data (torch.Tensor): Real data samples.
        - generated_data (torch.Tensor): Generated data samples.

        Returns:
        - torch.Tensor: Computed gradient penalty.
        """
        batch_size = real_data.size(0)

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=real_data.device)
        alpha = alpha.expand_as(real_data)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated = interpolated.to(real_data.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.net_d(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size(), device=real_data.device),
                               create_graph=True, retain_graph=True)[0]

        # Flatten gradients to easily compute norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Compute norm of gradients
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Compute gradient penalty
        gradient_penalty = k * ((gradients_norm - 1) ** 2).mean()  # k is weight set by user,denotes lambda

        return gradient_penalty

    def optimize_parameters(self, current_iter):
        # Use GT or GT with USM based on options
        l1_gt = self.gt_usm if not self.opt['l1_gt_usm'] else self.gt
        percep_gt = self.gt_usm if not self.opt['percep_gt_usm'] else self.gt
        gan_gt = self.gt_usm if not self.opt['gan_gt_usm'] else self.gt

        # Freeze net_d parameters
        for p in self.net_d.parameters():
            p.requires_grad = False

        # Zero gradients and forward pass through net_g
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()

        if current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters:
            # Pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            # Perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            # GAN loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            # Backward pass and optimizer step for net_g
            l_g_total.backward()
            self.optimizer_g.step()

        # Unfreeze net_d parameters
        for p in self.net_d.parameters():
            p.requires_grad = True

        # Zero gradients and forward pass through net_d
        self.optimizer_d.zero_grad()

        # Real images
        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())

        # Fake images
        fake_d_pred = self.net_d(self.output.detach().clone())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        # Gradient penalty (GP) calculation
        gradient_penalty = self._gradient_penalty(gan_gt, self.output)
        loss_dict['d_gp'] = gradient_penalty

        # Total discriminator loss and backward pass
        total_d_loss = l_d_fake - l_d_real + gradient_penalty
        total_d_loss.backward()
        self.optimizer_d.step()

        # Optionally update model_ema
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        # Store reduced loss dictionary
        self.log_dict = self.reduce_loss_dict(loss_dict)

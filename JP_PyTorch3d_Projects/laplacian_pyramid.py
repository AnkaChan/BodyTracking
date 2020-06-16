import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class LaplacianPyramidSeparableKernel(nn.Module):
    def __init__(self, device, image_size: int, channels: int, n_res:int, kernel_size: int, alpha=0.45):
        """
        Supports only kernel_size 3 and 5.
        Gaussian blurring 2D kernel T is a separable convolution: T = P * Q.transpose()

        For 3x3 kernel:
            only possible P = [1/4, 1/2, 1/4]

        For 5x5 kernel, 1DoF exists:
            P = [1/4-a/2, 1/4, a, 1/4, 1/4-a/2]
            a is usually 0.3-0.6
        """
        assert(kernel_size == 5 or kernel_size == 3)
        assert(0.3 <= alpha <= 0.6)

        super().__init__()
        self.device = device
        self.image_size = image_size
        self.channels = channels
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.n_res = n_res

        self.gaussian_conv2d = self._gen_gaussian_conv(channels, kernel_size, alpha)

    def build_laplacian_pyramid(self, images):
        """
        input:
          image: (N, W, H, C)

        output:
          list of laplacians: [b_k]
          last image
        """
        bs = []
        I0 = images
        for _ in range(self.n_res):
            R_I1 = self._blur_then_downsample(I0)
            E_R_I1 = self._upsample_then_blur(R_I1)
            b0 = I0 - E_R_I1
            bs.append(b0)
            I0 = R_I1

        return bs, I0

    def reconstruct_original_images(self, I1, bs):
        """
        reconstruct the original image from Laplacians
        input:
          list of laplacians: [b_k]
          last image
        output:
          the original image reconstructed
        """
        I0 = I1
        for b_k in reversed(bs):
            I0 = b_k + self._upsample_then_blur(I0)

        return I0



    def _upsample_then_blur(self, images):
        """
        upsample then blur
        input:
          image: (N, W, H, C)
        """
        images = images.transpose(1, 3)
        images = F.interpolate(images, scale_factor=2.0, mode='nearest')
        images = self.gaussian_conv2d(images)
        images = images.transpose(1, 3)
        return images

    def _blur_then_downsample(self, images):
        """
        blur then downsample
        input:
          image: (N, W, H, C)
        """
        images = images.transpose(1, 3)
        images = self.gaussian_conv2d(images)
        images = F.interpolate(images, scale_factor=0.5, mode='nearest')
        images = images.transpose(1, 3)
        return images

    def _gen_gaussian_conv(self, channels, kernel_size, a):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        if kernel_size == 3:
            P = torch.FloatTensor([1/4, 1/2, 1/4])
        elif kernel_size == 5:
            P = torch.FloatTensor([1/4-a/2, 1/4, a, 1/4, 1/4-a/2])
        else:
            P = None

        Q = P.reshape(-1, 1)
        gaussian_kernel = P * Q

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1).to(self.device)

        # generate guassian filter as nn.Conv2D
        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1,
                                    padding=int(kernel_size / 2), groups=channels, bias=False).to(self.device)
        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        return gaussian_filter




class LaplacianPyramidExact(nn.Module):
    def __init__(self, device, image_size: int, channels: int, n_res:int, kernel_size: int, sigma: float):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.n_res = n_res

        self.gaussian_conv2d = self._gen_gaussian_conv(channels, kernel_size, sigma)

    def build_laplacian_pyramid(self, images):
        """
        input:
          image: (N, W, H, C)

        output:
          list of laplacians: [b_k]
          last image
        """
        bs = []
        I0 = images
        for _ in range(self.n_res):
            R_I1 = self._blur_then_downsample(I0)
            E_R_I1 = self._upsample_then_blur(R_I1)
            b0 = I0 - E_R_I1
            bs.append(b0)
            I0 = R_I1

        return bs, I0

    def reconstruct_original_images(self, I1, bs):
        """
        reconstruct the original image from Laplacians
        input:
          list of laplacians: [b_k]
          last image
        output:
          the original image reconstructed
        """
        I0 = I1
        for b_k in reversed(bs):
            I0 = b_k + self._upsample_then_blur(I0)

        return I0

    def _upsample_then_blur(self, images):
        """
        upsample then blur
        input:
          image: (N, W, H, C)
        """
        images = images.transpose(1, 3)
        images = F.interpolate(images, scale_factor=2.0, mode='nearest')
        images = self.gaussian_conv2d(images)
        images = images.transpose(1, 3)
        return images

    def _blur_then_downsample(self, images):
        """
        blur then downsample
        input:
          image: (N, W, H, C)
        """
        images = images.transpose(1, 3)
        images = self.gaussian_conv2d(images)
        images = F.interpolate(images, scale_factor=0.5, mode='nearest')
        images = images.transpose(1, 3)
        return images

    def _gen_gaussian_conv(self, channels, kernel_size, sigma):
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1).to(self.device)

        # generate guassian filter as nn.Conv2D
        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1,
                                    padding=int(kernel_size / 2), groups=channels, bias=False).to(self.device)
        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        return gaussian_filter
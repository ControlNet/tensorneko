# TODO: Need refactor
from typing import Sequence

from torch.nn import Module

from ..layer.alae_layers import *


class StyleGanGenerator(Module):
    """
    The StyleGAN generator.
    """
    def __init__(self, latent_dim, resolutions: Sequence[int], channels: Sequence[int],
        starting_dims: int = 4,
    ):
        """
        style: A list of tuples (<out_res>, <out_channels>) that describes the Generator blocks of this module
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.to_rgb = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])

        # Parse the module description given in "style"
        for i in range(len(resolutions)):
            self.to_rgb.append(Lreq_Conv2d(channels[i], 3, 1, 0))
            if i == 0:
                self.conv_blocks.append(StyleGeneratorBlock(latent_dim, channels[0], channels[i + 1],
                                                            is_first_block=True, starting_dims=starting_dims))
            else:
                upscale = (resolutions[i] * 2 == resolutions[i + 1])
                self.conv_blocks.append(StyleGeneratorBlock(latent_dim, channels[i], channels[i + 1],
                                                            upscale=upscale, starting_dims=starting_dims))

    def __str__(self):
        name = "Style-Generator:\n"
        name += "\ttoRgbs\n"
        for i in range(len(self.conv_blocks)):
            name += f"\t {self.to_rgb[i]}\n"
        name += "\tStyleGeneratorBlocks\n"
        for i in range(len(self.conv_blocks)):
            name += f"\t {self.conv_blocks[i]}\n"
        return name

    def forward(self, w, final_resolution_idx, alpha):
        generated_img = None
        feature_maps = None
        for i, block in enumerate(self.conv_blocks):
            # Separate noise for each block
            noise = torch.randn((w.shape[0], 1, 1, 1), dtype=torch.float32).to(w.device)

            prev_feature_maps = feature_maps
            feature_maps = block(feature_maps, w, noise)

            if i == final_resolution_idx:
                generated_img = self.to_rgb[i](feature_maps)

                # If there is an already stabilized last previous resolution layer. alpha blend with it
                if i > 0 and alpha < 1:
                    generated_img_without_last_block = self.to_rgb[i - 1](prev_feature_maps)
                    if block.upscale:
                        generated_img_without_last_block = upscale_2d(generated_img_without_last_block)
                    generated_img = alpha * generated_img + (1 - alpha) * generated_img_without_last_block
                break

        return generated_img


class StyleGeneratorBlock(Module):
    def __init__(self, latent_dim, in_channels, out_channels, is_first_block=False, upscale=False, starting_dims=4):
        super().__init__()
        assert not (is_first_block and upscale), "You should not upscale if this is the first block in the generator"
        self.is_first_block = is_first_block
        self.upscale = upscale
        if is_first_block:
            self.const_input = nn.Parameter(torch.randn(1, out_channels, starting_dims, starting_dims))
        else:
            self.blur = LearnablePreScaleBlur(out_channels)
            self.conv1 = Lreq_Conv2d(in_channels, out_channels, 3, padding=1)

        self.style_affine_transform_1 = StyleAffineTransform(latent_dim, out_channels)
        self.style_affine_transform_2 = StyleAffineTransform(latent_dim, out_channels)
        self.noise_scaler_1 = NoiseScaler(out_channels)
        self.noise_scaler_2 = NoiseScaler(out_channels)
        self.adain = AdaIn(in_channels)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv2 = Lreq_Conv2d(out_channels, out_channels, 3, padding=1)

        self.name = f"StyleBlock({latent_dim}, {in_channels}, {out_channels}, is_first_block={is_first_block}, upscale={upscale})"

    def __str__(self):
        return self. name

    def forward(self, input, latent_w, noise):
        if self.is_first_block:
            assert(input is None)
            result = self.const_input.repeat(latent_w.shape[0], 1, 1, 1)
        else:
            if self.upscale:
                input = upscale_2d(input)
            result = self.conv1(input)
            result = self.blur(result)

        result += self.noise_scaler_1(noise)
        result = self.adain(result, self.style_affine_transform_1(latent_w))
        result = self.lrelu(result)

        result = self.conv2(result)
        result += self.noise_scaler_2(noise)
        result = self.adain(result, self.style_affine_transform_2(latent_w))
        result = self.lrelu(result)

        return result
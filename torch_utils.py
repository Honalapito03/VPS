import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn



def imshow(title, img):
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.show()


def warp(
    img: torch.Tensor,
    M: torch.Tensor,
    dsize: tuple[int, int],
    mode: str = "bicubic",
    padding_mode: str = "border",
    align_corners: bool = True,
    device="cuda"
):
    if img.dim() != 4:
        raise ValueError("img must have shape (B, C, H, W)")

    B, C, H, W = img.shape
    out_w, out_h = dsize

    if M.dim() == 2:
        M = M.unsqueeze(0).expand(B, -1, -1)

    theta = M.clone()

    # Convert from pixel-space to normalized coordinates ([-1, 1])
    theta[:, 0, 2] /= (W - 1) / 2
    theta[:, 1, 2] /= (H - 1) / 2

    # create sampling grid
    grid = F.affine_grid(theta, size=(B, C, out_h, out_w), align_corners=align_corners)

    # sample pixels
    warped = F.grid_sample(
        img,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners
    )

    return warped

def log_polar_transform_torch(image, output_shape=None, center=None, interpolation="bicubic", device="cuda"):
    """
    Differentiable log-polar transform using PyTorch grid_sample.

    Parameters
    ----------
    image : torch.Tensor
        Input tensor of shape (N, 1, H, W) or (1, H, W) [grayscale].
    output_shape : (int, int), optional
        (num_log_r, num_theta). Defaults to input size.
    center : (float, float), optional
        Transformation center (y, x). Default: image center.
    interpolation  str, optional
        'bilinear' or 'bicubic'. Default='bicubic'.

    Returns
    -------
    log_polar_img : torch.Tensor
        Log-polar transformed image, shape (N, 1, num_log_r, num_theta).
    log_r_grid : torch.Tensor
        Log radius values, shape (num_log_r, num_theta).
    theta_grid : torch.Tensor
        Angle values, shape (num_log_r, num_theta).
    """

    # Ensure batch + channel
    if image.ndim == 3:  # (1, H, W)
        image = image.unsqueeze(0)  # (N=1, 1, H, W)
    if image.ndim != 4:
        raise ValueError("Image must be (N,1,H,W) or (1,H,W)")

    N, C, H, W = image.shape

    if output_shape is None:
        output_shape = (H, W)
    num_log_r, num_theta = output_shape

    if center is None:
        center = (H / 2.0, W / 2.0)
    cy, cx = center

    # Maximum radius
    max_radius = torch.tensor(max(cy / 2, cx / 2))

    # Log-radius and angle grids
    log_r = torch.linspace(0, torch.log(max_radius), num_log_r, device=image.device, dtype=image.dtype)
    theta = torch.linspace(0, 2*torch.pi, num_theta, device=image.device, dtype=image.dtype, requires_grad=False)
    log_r_grid, theta_grid = torch.meshgrid(log_r, theta, indexing='ij')

    # Convert to Cartesian coordinates
    r = torch.exp(log_r_grid)
    x = cx + r * torch.cos(theta_grid)
    y = cy + r * torch.sin(theta_grid)

    # Normalize coords to [-1, 1] for grid_sample
    x_norm = (x / (W - 1)) * 2 - 1
    y_norm = (y / (H - 1)) * 2 - 1
    grid = torch.stack((x_norm, y_norm), dim=-1)  # (num_log_r, num_theta, 2)
    grid = grid.unsqueeze(0).expand(N, -1, -1, -1)  # (N, num_log_r, num_theta, 2)

    # Interpolate
    mode = "bicubic" if interpolation == "bicubic" else "bilinear"
    log_polar_img = F.grid_sample(image, grid, mode=mode, align_corners=True, padding_mode="border")

    return log_polar_img



class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Real and imaginary parts of weights
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)

        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        # Split real and imaginary parts
        x_real = x.real
        if (torch.is_complex(x)):
            x_imag = x.imag
        else:
            x_imag = torch.zeros_like(x_real)

        # Perform the 4 real convolutions
        real = self.real_conv(x_real) - self.imag_conv(x_imag)
        imag = self.real_conv(x_imag) + self.imag_conv(x_real)

        # Combine back into a complex tensor
        return torch.complex(real, imag)

    def calc_out_size(self, in_shape):
        out = torch.Tensor(in_shape)
        out[1] = self.out_channels
        out[2:] = torch.floor((out[2:] - self.kernel_size + 2 * self.padding) / self.stride) + 1
        return(out)

class ComplexIdentity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.i = nn.Identity()

    def forward(self, x):
        x_real = x.real
        if (torch.is_complex(x)):
            x_imag = x.imag
        else:
            x_imag = torch.zeros_like(x_real)
        return(torch.complex(self.i(x_real), self.i(x_imag)))

    def calc_out_size(self, in_shape):
        return(torch.Tensor(in_shape))


def soft_argmax_2d(heatmap: torch.Tensor, beta=100.0):
    """
    Differentiable soft-argmax for 2D maps.

    Parameters
    ----------
    heatmap : (N, 1, H, W) torch.Tensor
    beta : float
        Sharpness parameter. Higher = closer to argmax.

    Returns
    -------
    coords : (N, 2) tensor
        Differentiable (y, x) coordinates of maximum.
    """
    N, C, H, W = heatmap.shape

    heatmap = heatmap.sum(1)  # (N, 1, H, W)
                              # other function?
    # Flatten
    heatmap_flat = heatmap.reshape(N, -1)

    # Softmax over all pixels
    prob = torch.softmax(beta * heatmap_flat, dim=-1)  # (N, H*W)

    # Coordinate grid
    ys = torch.linspace(0, H-1, H, device=heatmap.device)
    xs = torch.linspace(0, W-1, W, device=heatmap.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1)  # (H*W, 2)


    # Weighted average → expected coordinate
    expected = torch.matmul(prob, coords)  # (N, 2)

    return expected


# Imports
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torch import nn, einsum
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR

from inspect import isfunction
from functools import partial
import math
import imageio

# Install einops
get_ipython().system('pip install -q -U einops')
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


def exists(x):
    """
    Returns True if the input object "x" is not None, otherwise False.
    """
    return x is not None

def default(value, d):
    """
    Returns "value" if it exists, otherwise returns the default "d" value.
    """
    if exists(value):
        return value
    return d() if isfunction(d) else d

def num_to_groups(num, divisor):
    """
    Splits "num" into groups of size "divisor" and handles any remainder.
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class Residual(nn.Module):
    """
    Implements a residual connection.
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    """
    Upsamples input feature map by a factor of 2.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = "nearest"),
        nn.Conv2d(dim, default(dim_out, dim), kernel_size = 3, padding = 1),
    )

def Downsample(dim, dim_out = None):
    """
    Downsamples input feature map by a factor of 2.
    """
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Generates sinusoidal positional embeddings for given time steps of the diffusion process.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        # Calculate half the embedding dimension.
        half_dim = self.dim // 2
        # Create scaling factor for positional embeddings
        scale = math.log(10000) / (half_dim - 1)
        # Create a tensor of positional encodings using the scaling factor.
        positions = torch.exp(torch.arange(half_dim, device = device) * -scale)
        # Reshape the time tensor to match the shape of the positional encodings.
        embeddings = time[:, None] * positions[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim = -1)

        return embeddings

class WeightStandardizedConv2d(nn.Conv2d):
    """
    Implements weight standardized convolutional layer.The weights are z-score normalized
    (zero mean, unit variance) before being used in the convolution operation.
    """
    def forward(self, x):
        # Small constant to avoide division by zero
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight

        # z-score normalization
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased = False))

        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class Block(nn.Module):
    """
    A neural network block consisting of a weight-standardized convolution, followed by
    group normalization, and a SiLU activation. Optionally, the output can be scaled and shifted
    before applying the activation function.
    """
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, kernel_size = 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """
    A ResNet-style block with two sub-blocks (Block), where the output is added to the input after
    passing through a 1x1 convolution. Supports timestep-dependent scaling and shifting via an optional
    time embedding.
    """
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(),
                          nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        # Adjust number of channels if needed
        self.res_conv = nn.Conv2d(dim, dim_out, kernel_size = 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim  = 1)
        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)

class Attention(nn.Module):
    """
    Self-attention module for 2D feature maps.Flattens the spatial dimensions
    (height and width) into a sequence and applies multi-head self-attention over the resulting
    sequence.
    """
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h = self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x = h, y = w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    """
    Implements linear attention, which reduces memory complexity by applying softmax over queries and keys
    in a more efficient manner and computing attention in linear time.
    """
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, kernel_size = 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)

        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h = self.heads), qkv
        )

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h = self.heads, x = h, y = w)
        return self.to_out(out)

class PreNorm(nn.Module):
    """
    Applies normalization before passing input to a given function (a block or layer).
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Unet(nn.Module):
    """
    U-Net architecture for image-to-image tasks, with attention and residual connections.
    """
    def __init__(
            self,
            dim,
            init_dim = None,
            out_dim = None,
            dim_mults = (1, 2, 4, 8),
            channels = 3,
            self_condition = False,
            resnet_block_groups = 4,
    ):
        super().__init__()

        # Set dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding = 0)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults )]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # Time step embeddings
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding = 1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)


    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    Computes a cosine-based beta schedule for a given number of timesteps.
    """

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    """
    Generates a linearly spaced beta schedule for a given number of timesteps.
    """
    beta_start = 0.0001
    beta_end = 0.2
    return torch.linspace(beta_start, beta_end, timesteps)

class CorruptionHelper:
    """
        Creates a corruption helper with precomputed values for betas, alphas, and cumulative products.
    """
    def __init__(self, noise_scheduler, num_steps, device):

        self.device = device
        self.num_steps = num_steps

        # Precompute betas and alphas
        self.betas = noise_scheduler(num_steps).to(device)
        self.alphas = 1. - self.betas

        # Precompute cumulative alphas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim = 0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(device)

    def corrupt(self, image_batch, noise_steps):
        """
        Corrupts an image batch by adding noise according to the precomputed schedule.
        """
        # Get the alphas and square roots for the specified noise steps
        alphas_cumprod_steps = self.alphas_cumprod[noise_steps].to(self.device)
        sqrt_alphas_cumprod_steps = self.sqrt_alphas_cumprod[noise_steps].to(self.device)
        sqrt_one_minus_alphas_cumprod_steps = self.sqrt_one_minus_alphas_cumprod[noise_steps].to(self.device)

        # Generate noise for each image in the batch
        noise = torch.randn(image_batch.size(0), *image_batch.shape[1:]).to(self.device)

        # Compute the noise component and the corrupted images
        noise_component = sqrt_one_minus_alphas_cumprod_steps.unsqueeze(1).unsqueeze(2).unsqueeze(3) * noise
        corrupted_images = (
            sqrt_alphas_cumprod_steps.unsqueeze(1).unsqueeze(2).unsqueeze(3) * image_batch +
            noise_component
        )

        return corrupted_images, noise_component


class EarlyStopping:
    """
    Implements early stopping to halt training when validation loss stops improving.
    """
    def __init__(self, patience = 5, min_delta = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0
        self.best_model = None

    def check(self, current_loss, model):
        # Check if loss has improved
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.epochs_without_improvement = 0
            self.best_model = model.state_dict() # Save best model
        else:
            self.epochs_without_improvement += 1

        # Stop if patience is exceeded
        if self.epochs_without_improvement >= self.patience:
            return True
        return False

    def restore_best_model(self, model):
        # Load the best model
        if self.best_model is not None:
            model.load_state_dict(self.best_model)

def train_step(model, train_dataloader, loss_fn, optimizer, corruption_helper, device):
    # Set model to train mode
    model.train()
    train_loss = 0.
    train_losses = []
    for (data, labels) in train_dataloader:
        # Send data to device
        data = data.to(device)

        # Get batch size for forward process timesteps
        size = data.size(0)

        # Pick timesteps in the forward (corruption) process
        forward_timesteps = torch.randint(0, corruption_helper.num_steps, (size,), device=device).long()

        # Corrupt data using the corruption helper
        corrupted_data, noise_component = corruption_helper.corrupt(data, forward_timesteps)

        # Forward pass through the denoiser
        noise_prediction = model(corrupted_data, forward_timesteps)

        # Calculate loss
        loss = loss_fn(noise_prediction, noise_component)
        train_losses.append(loss.item())

        # Zero the gradient
        optimizer.zero_grad()

        # Backprop
        loss.backward()

        # Update parameters
        optimizer.step()

        # Track loss
        train_loss += loss.item()

    # Average loss for the epoch
    avg_train_loss = train_loss / len(train_dataloader)
    return avg_train_loss, train_losses


def validation_step(model, val_dataloader, loss_fn, corruption_helper, device):
    # Set model to eval mode
    model.eval()
    val_loss = 0
    val_losses = []

    with torch.no_grad():
        for data, labels in val_dataloader:
            # Send data to device
            data = data.to(device)

            # Get batch size for forward process timesteps
            size = data.size(0)

            # Pick timesteps in the forward (corruption) process
            forward_timesteps = torch.randint(0, corruption_helper.num_steps, (size,), device=device).long()

            # Corrupt data using the corruption helper
            corrupted_data, noise_component = corruption_helper.corrupt(data, forward_timesteps)

            # Forward pass through the denoiser
            noise_prediction = model(corrupted_data, forward_timesteps)

            # Calculate loss
            loss = loss_fn(noise_prediction, noise_component)
            val_losses.append(loss.item())

            # Accumulate loss for logging
            val_loss += loss.item()

    # Average validation loss for the epoch
    avg_val_loss = val_loss / len(val_dataloader)
    return avg_val_loss, val_losses


def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, corruption_helper, epochs, scheduler = None, early_stopping = None, writer = None, device = None):
    global_step = 0  # Keep track of global steps for TensorBoard
    # Lists to store all the batch losses for train and validation
    all_train_losses = []
    all_val_losses = []

    for epoch in range(epochs):
        # Train step
        avg_train_loss, train_losses = train_step(model, train_dataloader, loss_fn,
                                                  optimizer, corruption_helper, device)

        # Add batch losses to all_train_losses for tracking
        all_train_losses.extend(train_losses)

        # Val step
        avg_val_loss, val_losses = validation_step(model, val_dataloader, loss_fn, corruption_helper, device)

        # Add batch losses to all_val_losses for tracking
        all_val_losses.extend(val_losses)

        # Log losses and learning rate to TensorBoard
        if writer:
            writer.add_scalar("Loss/train", avg_train_loss, global_step)
            writer.add_scalar("Loss/val", avg_val_loss, global_step)
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], global_step)
            global_step += 1  # Increment global step for TensorBoard

        if scheduler is not None:
            # Step the scheduler *after* each epoch
            scheduler.step()

        # Print epoch results
        print(f"Epoch: {epoch + 1} | Train loss: {avg_train_loss:.5f} | Validation loss: {avg_val_loss:.5f} | Learning Rate: {optimizer.param_groups[0]['lr']}\n")

        if early_stopping is not None:
            # Check early stopping
            if early_stopping.check(avg_val_loss, model):
                print(f"Early stopping triggered at epoch {epoch+1}. Restoring best model...")
                early_stopping.restore_best_model(model)
                break

    print("Training completed.")
    return all_train_losses, all_val_losses


### Sampling
@torch.no_grad()
def sample(model, batch_size = 64, device = None, channels = 1, height = 28, width = 28, timesteps = 1000, s = 0.008, print_progress = False, gamma_scaling = None):
    """
    Implements the backward diffusion process for DDPM (sampling).
    """
    # Create list to store Xt
    x_t_backward_stages = []
    # Set model to eval
    model.eval()

    # Create randomn noise
    x_t = torch.randn((batch_size, channels, height, width), device = device)
    x_t_backward_stages.append(x_t.clone())

    # Calculate the forward/backward process hyperparameters
    betas = cosine_beta_schedule(timesteps, s=s).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim = 0).to(device)

    # Shifting all the values one step forward and inserting a 1.0 at the beginning to handle t = 0
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.0).to(device)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).to(device)

    # Equation (7) in the DDPM paper
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    for t in reversed(range(0, timesteps)):
        if print_progress:
            print(f"Step {timesteps - t}/{timesteps} - Denoising...")

        timesteps_tensor = torch.tensor([t]).repeat(batch_size).to(device)

        predicted_noise = model(x_t, timesteps_tensor)

        beta_t = betas[t]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas_t = sqrt_recip_alphas[t]
        posterior_variance_t = posterior_variance[t]

        # Equation (11) in the DDPM paper
        mean = sqrt_recip_alphas_t * (x_t - beta_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        # Sampling algorithm line 3 to 4
        if t > 0:
            variance = posterior_variance_t
            stddev = torch.sqrt(variance)

            if gamma_scaling is not None:
                if (gamma_scaling == "exponential_decay"):
                    gamma = np.exp(-5 * (1 - t / timesteps))  # Adjust 5 to control decay speed
                if (gamma_scaling == "sigmoid_decay"):
                    gamma = 1 / (1 + np.exp(-10 * (t / timesteps - 0.5)))  # Sigmoid centered at t=500
            else:
                gamma = 1

            x_t = mean + gamma * stddev * torch.randn_like(x_t, device = device)
        else:
            x_t = mean

        x_t_backward_stages.append(x_t.clone())

    return x_t, x_t_backward_stages


def save_gif(x_t_backward_stages, save_path = "/content/drive/MyDrive/Colab Notebooks/Diffusion/diffusion_playground/back_dif_stages.gif"):
    """
    Creates a GIF from the stored backward diffusion samples using imageio.
    """

    images = []

    for x_t in x_t_backward_stages:
        # Apply reverse transform
        img = reverse_transform(x_t)
        # Create grid
        grid = make_grid(img)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()

        # Map to range [0, 255] and type np.uint8
        grid_np = (grid_np * 255).astype(np.uint8)
        images.append(grid_np)

    # Save as GIF using imageio
    imageio.mimsave(save_path, images,  format='GIF', fps = 30, total_duration = 5)
    print(f"Saved GIF at {save_path}")


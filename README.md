# Diffusion Playground
Welcome to my diffusion playground! 🚀 This repository is dedicated to my journey into the fascinating world of diffusion models. Starting with the simplest possible version of a denoising algorithm, I aim to progressively explore, understand, and implement increasingly complex versions of these algorithms.

This is more than just a learning project—it's a hands-on exploration of the mathematical foundations, practical applications, and creative possibilities of diffusion models. I'll document each step, sharing insights, challenges, and breakthroughs along the way.

If you're curious about diffusion models, their inner workings, or their applications in generative modeling, this repository will grow into a resource for learners and enthusiasts alike.

Let’s dive into diffusion!

The project is currently in active development.

## Project Structure
- **`toy_diffusion.ipynb`**: A very basic implementation of a diffusion model

- **`DDPM.ipynb`**: A diffusion model implementation based on the paper [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

- **`Secondary files:`**:
    - **`DDPM.py:`**: The code developed in DDPM.ipynb as a python module
    - **`experiment.py:`**: A standard implementation of an experiment and experiment manager
    - **`fmnist_experiments.ipynb:`**: Training experiments on the Fasion MNIST dataset
  

## Results

## Toy Diffusion

Features of the algorithm design used in toy_diffusion:
* The architecture of the model is a very basic version of a U-Net.
* Spatial up-sampling and down-sampling don't use learnable parameters.
* Given a noisy (corrupted) version of a clean image the model tries to predict the clean image directly.
* The amount of noise added to each input during training is random and does not follow a schedule.
* The sampling strategy used is a very basic one, making smaller steps in the backward process when the input is very noisy and larger steps when it is cleaner.

The forward (corruption) process for toy_diffusion:
<p align="center">
  <img src="data/toy_forward_process.png" width="1200"/>
</p>

The denoiser inputs and outputs:
<p align="center">
  <img src="data/toy_denoiser_input_output.png" width="1200"/>
</p>

Sampling with the denoiser with five steps:
<p align="center">
  <img src="data/toy_sampling_5_steps.png" width="1200"/>
</p>

References for toy_diffusion:
- &nbsp; The notebook follows the [Diffusion Models from Scratch](https://huggingface.co/learn/diffusion-course/en/unit1/3) notebook in the Hugging Face Diffusion course, but the U-Net used is implemented in a different way - using two layers of 3X3 convolutions instead of one layer of 5X5 and without making use of torch.nn.ModuleList to access layers by index(the goal is to make the architecture more transparent).

## DDPM Diffusion

Features of the algorithm design used in the DDPM notbook:
* The architecture of the model is a again a U-Net, but is has about ten times more parameters. It is also more complex. It uses ResNet blocks which have a residual connection within themselves adding their input to their output. Linear attention is used in the downsample and upsample paths and standard multi-head self-attention at the bottleneck of the U-Net.
* Spatial downsampling is done without any information loss by sending extra pixels to new feature maps and feeding the result to a convolutional layer to achieve the desired tensor shape.
* Skip connections between the corresponding levels (the ones with the same tensor shapes) of the downsample and upsample do not add input to output like in toy_diffusion, but rather concatenate the input feature maps to the output feature maps.
* The model is not trying to predict the clean image directly, but rather given a noisy image it tries to predict the noise.
* The amount of noise added to each input during training follows a strict predetermined schedule consisting of a fixed number of steps.
* Because the model parameters are shared (identical) for all steps in the forward/backward diffusion process, the model is provided additional information by being fed a tensor indicating the corresponding timesteps in the forward process for each input sample. The embeddings created from these timesteps shift and scale activations in the first sub-block of each ResNet block in the U-net.
* The sampling strategy is much more sophisticated and seeks to model the reverse probability density function for a step in the backward diffusion process. It used the model's prediction for the noise and the forward process parameters to calculate the mean of this distribution and then adds new noise to it according to a fixed variance parameter. This introduces probabilistic outcomes and makes the sampling algorithm non-deterministic. A major downside is that generation requires a lot of steps and therefore a lot of forward passes through the model.
* Adjusting the variance of the backward process has a large effect on the generated samples - too low variance leads to something close to mode collapse when only a small set of very similar samples are generated, too high variance can lead to samples which still contain a lot of noise.

Sinusoidal time step embeddings for DDPM visualized:
<p align="center">
  <img src="data/DDPM_sin_timestep_emb.png" width="600"/>
</p>

Noise schedules betas compared:
<p align="center">
  <img src="data/DDPM_ns_betas.png" width="600"/>
</p>

Noise schedules cumulative alpha products compared:
<p align="center">
  <img src="data/DDPM_ns_alphas_cumprod.png" width="600"/>
</p>

The forward diffusion process visualized for 300 steps:
<p align="center">
  <img src="data/DDPM_forward_300.png" width="1200"/>
</p>

Results of the backward diffusion process with 1000 steps -on the MNIST dataset (generated samples):
<p align="center">
  <img src="data/DDPM_gen_samples.png" width="600"/>
</p>

The backward diffusion process in action:
<p align="center">
  <img src="data/DDPM_back_dif_stages_loop.gif" width="600"/>
</p>

The same model trained on FashionMNIST:
<p align="center">
  <img src="data/DDPM_fas_mnist_gen_samples.png" width="600"/>
</p>

References for DDPM diffusion:
- &nbsp; The U-Net used is based on Phil Wang's [implementation](https://github.com/lucidrains/denoising-diffusion-pytorch) - which itself is based on [the original TensorFlow implementation](https://github.com/hojonathanho/diffusion).

### Paper and Blog References:

#### Papers:
- [x] [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) <br>
- [x] [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)
- [x] [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

#### Blogs and other resources:
- &nbsp; [Hugging Face Diffusion Models Course](https://huggingface.co/learn/diffusion-course/unit0/1)
- &nbsp; [A university course by the Israel Institute of Technology, taught by Michael Elad](https://www.youtube.com/playlist?list=PL0H3pMD88m8XPBlWoWGyal45MtnwKLSkQ)
- &nbsp; [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
- &nbsp; [Lilian Weng's blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- &nbsp; [Score based approach - Yang Song](https://yang-song.net/blog/2021/score/)









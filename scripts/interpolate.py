"""
Interpolation between weighted compositions:
Creates  moving videos by smoothly walking through the space of compositions for a given collection of prompts.
Composition operator implemented from https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/
Everything else adapted from Karpathy's 'stablediffusionwalk.py` script for making videos from latent space interpolations

example way to run this script:
$ python sd_conceptual_blend.py --prompts "a red car parked in a desert","hills behind the car","Aurora in the sky" --weights [0.5,0.5,1.0] 
--name composition_normalized_test8 --num_steps 50
to stitch together the images, e.g.:
$ ffmpeg -r 10 -f image2 -s 512x512 -i blend/frame%06d.jpg -vcodec libx264 -crf 10 -pix_fmt yuv420p blend.mp4


you have to have access to stablediffusion checkpoints from https://huggingface.co/CompVis
and install all the other dependencies (e.g. diffusers library)
"""

import fire
import os
import inspect
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from time import time
from PIL import Image
from einops import rearrange
import numpy as np
import torch
from torch import autocast
from torchvision.utils import make_grid

# -----------------------------------------------------------------------------


@torch.no_grad()
def diffuse(
        pipe,
        time_step,
        cond_embeddings, # text conditioning, should be (1, 77, 768)
        cond_latents,  # image conditioning, should be (1, 4, 64, 64),
        num_inference_steps,
        guidance_scale,
        eta,
        weights,
    ):
    torch_device = cond_latents.get_device()

    # classifier guidance: add the unconditional embedding
    max_length = cond_embeddings[0].shape[1] # 77
    uncond_input = pipe.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings[0]])

    # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
    if isinstance(pipe.scheduler, LMSDiscreteScheduler):
        cond_latents = cond_latents * pipe.scheduler.sigmas[0]
      
    # init the scheduler
    accepts_offset = "offset" in set(inspect.signature(pipe.scheduler.set_timesteps).parameters.keys())
    extra_set_kwargs = {}
    if accepts_offset:
        extra_set_kwargs["offset"] = 1
    pipe.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]
    accepts_eta = "eta" in set(inspect.signature(pipe.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # diffuse!
    for i, t in enumerate(pipe.scheduler.timesteps):

        # expand the latents for classifier free guidance
        latent_model_input = torch.cat([cond_latents] * 2)
        if isinstance(pipe.scheduler, LMSDiscreteScheduler):
            sigma = pipe.scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

        # predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        # cfg
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                
        old_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
        # This is where the compositional code goes.
        noise_pred_concept_lst = []
        for concept in cond_embeddings:
            # Get the noise_pred for each input prompt
            latent_model_input = cond_latents
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)
                
            noise_pred_concept = pipe.unet(latent_model_input, t, encoder_hidden_states=concept)["sample"]
            noise_pred_concept_lst.append(noise_pred_concept)
           
        # Compute the weighted noise pred
        noise_sum = torch.zeros(noise_pred_concept.size(),device=torch_device,dtype=torch.half)
            
        for cn in range(len(noise_pred_concept_lst)):
            noise_sum += weights[cn]*(noise_pred_concept_lst[cn]-noise_pred_uncond)
        
        new_noise_pred = noise_pred_uncond + guidance_scale * noise_sum
           
        # compute the previous noisy sample x_t -> x_t-1
        if isinstance(pipe.scheduler, LMSDiscreteScheduler):
            cond_latents = pipe.scheduler.step(new_noise_pred, i, cond_latents, **extra_step_kwargs)["prev_sample"]
        else:
            cond_latents = pipe.scheduler.step(noise_pred, t, cond_latents, **extra_step_kwargs)["prev_sample"]
       
    # scale and decode the image latents with vae
    cond_latents = 1 / 0.18215 * cond_latents  # Why this?
    image = pipe.vae.decode(cond_latents)  # What kind of vae is being used here? What does it's latent space look like?

    # generate output numpy image as uint8
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).astype(np.uint8)

    return image

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """ helper function to spherically interpolate two arrays v1 v2 """

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def run(
        # --------------------------------------
        # args you probably want to change
        prompts = ["A cloudy blue sky","A mountain in the horizon","Cherry Blossoms in front of the mountain"], # prompt to dream about
        gpu = 0, # id of the gpu to run on
        name = 'starry_night_bear', # name of this project, for the output directory
        rootdir = '/home/ubuntu/filesystem/blends/',
        num_steps = 200, # number of steps between each pair of sampled points
        max_frames = 10000, # number of frames to write and then exit the script
        num_inference_steps = 50, # more (e.g. 100, 200 etc) can create slightly better images.  Number of latent spaces to induce?
        guidance_scale = 7.5, # can depend on the prompt. usually somewhere between 3-10 is good... What is this?
        seed = 1456,  # Random seed. Uniquely determines the trajectory!
        # --------------------------------------
        # args you probably don't want to change
        quality = 90, # for jpeg compression of the output images
        eta = 0.0,
        width = 512,
        height = 512,
        weights_path = "/home/ubuntu/filesystem/stable-diffusion-v1-4",
        weights = [0.5,0.5],
        # --------------------------------------
    ):
    assert torch.cuda.is_available()
    assert height % 8 == 0 and width % 8 == 0
    torch.manual_seed(seed)
    torch_device = f"cuda:{gpu}"
    try:
        prompts = prompts.split(",")
    except:
        pass

    weights = [i/sum(weights) for i in weights]
    
    # init the output dir
    outdir = os.path.join(rootdir, name)
    os.makedirs(outdir, exist_ok=True)

    # init all of the models and move them to a given GPU
    lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    pipe = StableDiffusionPipeline.from_pretrained(weights_path, scheduler=lms, use_auth_token=True)

  
    pipe.unet.to(torch_device)
    pipe.vae.to(torch_device)
    pipe.text_encoder.to(torch_device)

    # get the conditional text embeddings based on the prompt
    text_inputs = []
    cond_embeddings = []
    for i in range(len(prompts)):
        text_inputs.append(pipe.tokenizer(prompts[i], padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt"))
        cond_embeddings.append(pipe.text_encoder(text_inputs[i].input_ids.to(torch_device))[0]) # shape [1, 77, 768]
        
        
    # Create the extremal points of the weights probability simplex
    extremal_points = []
    for i in range(len(weights)):
        centre = [1/len(weights) for i in weights]
        new_extremal = [0.0 for i in weights]
        new_extremal[i] = 1.0

        if i == 0:
            extremal_points.append(new_extremal)
            extremal_points.append(centre)

        else:
            extremal_points.append(new_extremal)
        
    
    # sample a source
    init1 = torch.randn((1, pipe.unet.in_channels, height // 8, width // 8), device=torch_device)

    # iterate the loop
    #max_frames = 1000
    frame_index = 0
    count = 1
    
    random_walk = False
    
    # Start with one extreme -> go through an equal composition -> then go to the other extreme
    weights = extremal_points[0]
    #while frame_index < max_frames:
    while count < len(extremal_points):

        # sample the destination

        extremal_point = extremal_points[count]
            
        for i, t in enumerate(np.linspace(0, 1, num_steps)):
            inter_weights = slerp(float(t),torch.from_numpy(np.array(weights)),torch.from_numpy(np.array(extremal_point)))
            print(inter_weights)
            print("dreaming... ", frame_index)
            with autocast("cuda"):
                image = diffuse(pipe,t, cond_embeddings,init1,num_inference_steps, guidance_scale, eta,inter_weights)
            im = Image.fromarray(image)
            new_str = ""
            for w in inter_weights.numpy():
                new_str += str(w)
            outpath = os.path.join(outdir,'frame%06d.jpg' % frame_index)
            im.save(outpath, quality=quality)
            frame_index += 1

        weights = inter_weights
        count += 1
            

if __name__ == '__main__':
    fire.Fire(run)
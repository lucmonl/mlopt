
import os

#import skimage.io
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import sys

def load_image(image_path):
    image_path = "plots/dog.png"
    if image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {image_path} is non valid.")
        sys.exit(1)
    #print(np.array(img))
    return img


def get_attention_map(graphs, model, device, patch_size, image_path=None, num_register=0):
    img = load_image(image_path)
    image_size = (280, 280)
    threshold = None
    transform = pth_transforms.Compose([
        pth_transforms.Resize(image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    raw_image = img
    img = transform(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    attentions = model.get_all_selfattention(img.to(device))#.detach().cpu() #[1,6,2601,3601]
    nh = attentions[0].shape[1] # number of head

    # we keep only the output patch attention
    for i in range(len(attentions)):
        attentions[i] = attentions[i][0, :, 0, 1+num_register:].reshape(nh, -1)
    #print(torch.sum(attentions, dim=-1), torch.max(attentions))

        if threshold is not None:
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

        attentions[i] = attentions[i].reshape(nh, w_featmap, h_featmap)
        attentions[i] = nn.functional.interpolate(attentions[i].unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    graphs.test_img.append(raw_image)
    #graphs.attention_map.append([1,2,3])
    graphs.attention_map = attentions

    output_norm = model.get_intermediate_layers(img.to(device), n=1, reshape=True, norm=True)[-1].detach().cpu()
    output_norm = torch.norm(output_norm, dim=1).tolist()
    graphs.output_norm.append(output_norm)

    # save attentions heatmaps
    """
    os.makedirs(args.output_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
    for j in range(nh):
        fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")

    if args.threshold is not None:
        image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))
        for j in range(nh):
            display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)
    """


def get_attention_map_path(graphs, model, device, patch_size, image_path=None, num_register=0):
    img = load_image(image_path)
    image_size = (600, 600)
    threshold = None
    transform = pth_transforms.Compose([
        pth_transforms.Resize(image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    raw_image = img
    img = transform(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    attentions = model.get_all_selfattention(img.to(device))#.detach().cpu() #[1,6,2601,3601]
    nh = attentions[0].shape[1] # number of head

    # we keep only the output patch attention
    prominent_idx = 0
    for i in range(len(attentions)-1, -1, -1):
        attentions[i] = attentions[i][0, :, prominent_idx, 1+num_register:].reshape(nh, -1)
        _, prom_ids = torch.max(attentions[i], dim=-1, out=None)
        print(prom_ids)
        prominent_idx, _ = torch.mode(prom_ids.cpu())
        print(prominent_idx)
        prominent_idx += 1+num_register

        if threshold is not None:
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

        attentions[i] = attentions[i].reshape(nh, w_featmap, h_featmap).cpu().numpy()
        #print(torch.amax(attentions[i], dim=[-2,-1]))
        #attentions[i] = nn.functional.interpolate(attentions[i].unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    graphs.test_img.append(raw_image)
    #graphs.attention_map.append([1,2,3])
    #for i in range(len(attentions)):
    #    print([np.max(attentions[i][j].reshape(-1)) for j in range(attentions[i].shape[0])])
    graphs.attention_map = attentions

    output_norm = model.get_intermediate_layers(img.to(device), n=1, reshape=True, norm=True)[-1].detach().cpu()
    output_norm = torch.norm(output_norm, dim=1).tolist()
    graphs.output_norm.append(output_norm)

def get_attention_prominent(attentions, layer, prominent_idx, num_register, k):
    if layer == -1:
        return {}
    #token_attn_map = torch.sum(attentions[layer][0, :, prominent_idx, 1+num_register:], dim=0) #w * h
    token_attn_map = attentions[layer][0, 5, prominent_idx, 1+num_register:] #w * h
    prom_vals, prom_ids = torch.topk(token_attn_map, k)
    print(layer, prominent_idx, prom_ids // 42, prom_ids % 42)
    path = {}
    for i in prom_ids:
        i = (i+1+num_register).item()
        path[i] = get_attention_prominent(attentions, layer-1, i, num_register, k)
        #= return_dict if return_dict is not None
    return path

def get_attention_map_path_topk(graphs, model, device, patch_size, image_path=None, num_register=0, k=1):
    img = load_image(image_path)
    image_size = (600, 600)
    threshold = None
    transform = pth_transforms.Compose([
        pth_transforms.Resize(image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    raw_image = img
    img = transform(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    attentions = model.get_all_selfattention(img.to(device))#.detach().cpu() #[1,6,2601,3601]
    nh = attentions[0].shape[1] # number of head

    # we keep only the output patch attention
    """
    for i in range(len(attentions)-1, -1, -1):
        for prominent_idx in attn_path:
            token_attn_path = attn_path[prominent_idx]
            token_attn_map = torch.sum(attentions[i][0, :, prominent_idx, 1+num_register:], dim=0) #w * h
            prom_vals, prom_ids = torch.topk(token_attn_map, k)
            print(prom_ids)
            for id in prom_ids:
                token_attn_path[id] = []
            prominent_idx += 1+num_register

            attentions[i] = attentions[i].reshape(nh, w_featmap, h_featmap).cpu().numpy()
        #print(torch.amax(attentions[i], dim=[-2,-1]))
        #attentions[i] = nn.functional.interpolate(attentions[i].unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    """
    print("in recursion")
    attn_path = get_attention_prominent(attentions, len(attentions)-1, 0, num_register, k)
    print("out recursion")
    graphs.test_img.append(raw_image)
    #graphs.attention_map.append([1,2,3])
    #for i in range(len(attentions)):
    #    print([np.max(attentions[i][j].reshape(-1)) for j in range(attentions[i].shape[0])])
    #graphs.attention_map = attentions
    graphs.attention_path = attn_path

    output_norm = model.get_intermediate_layers(img.to(device), n=1, reshape=True, norm=True)[-1].detach().cpu()
    output_norm = torch.norm(output_norm, dim=1).tolist()
    graphs.output_norm.append(output_norm)
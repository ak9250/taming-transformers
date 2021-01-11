import runway
import sys
sys.path.append(".")
from omegaconf import OmegaConf
import yaml
from taming.models.cond_transformer import Net2NetTransformer
import torch
from PIL import Image
import numpy as np
import time
import os

from omegaconf import OmegaConf
import yaml

from taming.models.cond_transformer import Net2NetTransformer

import torch

def show_image(s):
  s = s.detach().cpu().numpy().transpose(0,2,3,1)[0]
  s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
  s = Image.fromarray(s)
  return s

@runway.command('generate', inputs={"source": runway.image(default_output_format='PNG'),}, outputs={'image': runway.image})
def generate(model, inputs):
    config_path = "logs/2020-11-09T13-31-51_sflckr/configs/2020-11-09T13-31-51-project.yaml"
    config = OmegaConf.load(config_path)
    model = Net2NetTransformer(**config.model.params)
    ckpt_path = "logs/2020-11-09T13-31-51_sflckr/checkpoints/last.ckpt"
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.cuda().eval()
    torch.set_grad_enabled(False)
    os.makedirs('images', exist_ok=True)
    inputs['source'].save('images/temp.png')
    segmentation_path = os.path.join('images','temp.png')
    segmentation = Image.open(segmentation_path)
    segmentation = segmentation.convert("L")
    segmentation = np.array(segmentation)
    segmentation = segmentation.clip(0, 181)
    segmentation = np.eye(182)[segmentation]
    segmentation = torch.tensor(segmentation.transpose(2,0,1)[None]).to(dtype=torch.float32, device=model.device)

    c_code, c_indices = model.encode_to_c(segmentation)
    assert c_code.shape[2]*c_code.shape[3] == c_indices.shape[1]
    segmentation_rec = model.cond_stage_model.decode(c_code)

    codebook_size = config.model.params.first_stage_config.params.embed_dim
    z_indices_shape = c_indices.shape
    z_code_shape = c_code.shape
    z_indices = torch.randint(codebook_size, z_indices_shape, device=model.device)
    x_sample = model.decode_to_img(z_indices, z_code_shape)



    idx = z_indices
    idx = idx.reshape(z_code_shape[0],z_code_shape[2],z_code_shape[3])

    cidx = c_indices
    cidx = cidx.reshape(c_code.shape[0],c_code.shape[2],c_code.shape[3])

    temperature = 1.0
    top_k = 100
    update_every = 50

    start_t = time.time()
    for i in range(0, z_code_shape[2]-0):
      if i <= 8:
        local_i = i
      elif z_code_shape[2]-i < 8:
        local_i = 16-(z_code_shape[2]-i)
      else:
        local_i = 8
      for j in range(0,z_code_shape[3]-0):
        if j <= 8:
          local_j = j
        elif z_code_shape[3]-j < 8:
          local_j = 16-(z_code_shape[3]-j)
        else:
          local_j = 8

        i_start = i-local_i
        i_end = i_start+16
        j_start = j-local_j
        j_end = j_start+16
        
        patch = idx[:,i_start:i_end,j_start:j_end]
        patch = patch.reshape(patch.shape[0],-1)
        cpatch = cidx[:, i_start:i_end, j_start:j_end]
        cpatch = cpatch.reshape(cpatch.shape[0], -1)
        patch = torch.cat((cpatch, patch), dim=1)
        logits,_ = model.transformer(patch[:,:-1])
        logits = logits[:, -256:, :]
        logits = logits.reshape(z_code_shape[0],16,16,-1)
        logits = logits[:,local_i,local_j,:]

        logits = logits/temperature

        if top_k is not None:
          logits = model.top_k_logits(logits, top_k)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx[:,i,j] = torch.multinomial(probs, num_samples=1)

        step = i*z_code_shape[3]+j
        if step%update_every==0 or step==z_code_shape[2]*z_code_shape[3]-1:
          x_sample = model.decode_to_img(idx, z_code_shape)      
    return show_image(x_sample)
    
if __name__ == '__main__':
    runway.run(port=8889)

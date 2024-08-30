import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
from ldm.modules.attention import CrossAttention, SpatialTransformer, BasicTransformerBlock

class CrossAttentionHook(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.generated_image = True
        
        self.current_time_step = 0
        self.lst_idx = 0
        
        self.per_time_step = False
        self.time_division = 5
        self.total_timestep = 50
        self.per_attention_head = False
        
        if self.per_time_step:
            self.cross_attention_forward_hooks = defaultdict(lambda: defaultdict(int))
        else:
            self.cross_attention_forward_hooks = defaultdict(lambda:0)
        
        
    def clear(self):
        self.cross_attention_forward_hooks.clear()
        self.current_time_step = 0
        self.lst_idx = 0
    
    def make_images(self, input_image, cloth, attention_maps, save_dir, batch_idx):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            # input_image: [batch_size, 3, 512, 384]
            # attention_maps: [64, 48]

            # range: [0, 1]
            attention_maps = (attention_maps - attention_maps.min()) / (attention_maps.max() - attention_maps.min() + 1e-8)

            # [512, 384, 3], range: [-1, 1]
            if self.generated_image:
                input_image = input_image[0]
            else:
                input_image = cloth[0]
            input_image = np.uint8(((input_image.permute(1,2,0) + 1.0) * 127.5).numpy())
            
            # [512, 384, 1] 
            if self.generated_image:
                resized_attention_maps = F.interpolate(attention_maps.cpu().unsqueeze(0).unsqueeze(0), size=(512,512), mode='bicubic')
            else:
                resized_attention_maps = F.interpolate(attention_maps.cpu().unsqueeze(0).unsqueeze(0), size=(224,224), mode='bicubic')
            resized_attention_maps = ((resized_attention_maps - resized_attention_maps.min()) / (resized_attention_maps.max() - resized_attention_maps.min() + 1e-8)).squeeze(0).squeeze(0).numpy()
            
            #resized_attention_maps = 1.0 - resized_attention_maps
            
            resized_attention_maps[resized_attention_maps >= 0.6] = 1
                
            resized_attention_maps = cv2.applyColorMap(np.uint8(resized_attention_maps*255), cv2.COLORMAP_JET)
            
            heat_map = cv2.addWeighted(input_image, 0.7, resized_attention_maps, 0.5, 0)
            
            attention_maps = attention_maps.cpu().numpy()
            attention_maps = 1.0 - attention_maps 
            attention_maps[attention_maps <= 0.4] = 0
            
            plt.imshow(attention_maps, cmap='jet')
            attention_map_filename = f"idx-{batch_idx}_attention_map_generated_image-{self.generated_image}.png"
            attetion_map_save_pth = f"{save_dir}/{attention_map_filename}"
            plt.axis('off')
            plt.savefig(attetion_map_save_pth, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            plt.imshow(heat_map)
            heat_map_filename = f"idx-{batch_idx}_heatmap_generated_image-{self.generated_image}.png"
            heat_map_save_pth = f"{save_dir}/{heat_map_filename}"
            plt.axis('off')
            plt.savefig(heat_map_save_pth, bbox_inches='tight', pad_inches=0)
            plt.close()
    
    def make_attention_maps(self):
        # 하나로 합치기 
        with torch.cuda.amp.autocast(dtype=torch.float32):
            # [8, seq_len(1), height, width] * 15
            if self.per_time_step:
                attention_maps = {}
                for time_pack in range(self.time_division):
                    attention_maps[time_pack] = []
            else:
                attention_maps = []
            for key in self.cross_attention_forward_hooks.keys():
                if self.per_time_step:
                    for time_pack in range(self.time_division):
                        for t in range(time_pack * int(self.total_timestep//self.time_division), (time_pack + 1) * int(self.total_timestep//self.time_division)):
                            attention_maps[time_pack].append(F.interpolate(self.cross_attention_forward_hooks[key][t], size=(64, 64), mode='bicubic').clamp_(min=0))
                else:
                    attention_maps.append(F.interpolate(self.cross_attention_forward_hooks[key], size=(64, 64), mode='bicubic').clamp_(min=0))

            if self.per_time_step:
                if self.generated_image:
                    for time_pack in range(self.time_division):
                        attention_maps[time_pack] = torch.cat(attention_maps[time_pack], dim=0)
                        attention_maps[time_pack] = attention_maps[time_pack].mean(0).mean(0)
            elif self.per_attention_head:
                for i, maps in enumerate(attention_maps):
                    attention_maps[i] = maps.unsqueeze(1)
                attention_maps = torch.cat(attention_maps, dim = 1)
                attention_maps = list(torch.chunk(attention_maps, 8, dim=0))
                for i, maps in enumerate(attention_maps):
                    attention_maps[i] = maps.squeeze(0).mean(0).mean(0)
            else:
                if self.generated_image:
                    # [num, 64, 48]
                    attention_maps = torch.cat(attention_maps, dim=0)
                    attention_maps = attention_maps.mean(0).mean(0)
                else:
                    attention_maps = [attention_map.mean(dim=1) for attention_map in attention_maps]
                    attention_maps = torch.cat(attention_maps, dim=0)
                    attention_maps = attention_maps.mean(0)
            self.clear()
        
        return attention_maps
    
    def cross_attention_hook(self, module, input, output, name):
        # Get heat maps
        # print(f"Input size: {len(input)}")
        # print(f"Output size: {len(output)}")
        # print(f"Module Name: {name}")
        # x: [batch_size(1), 1, H, W]
        # y: [batch_size(1), 1, 768]
        x, y = input[0], input[1]
        
        # [num_heads(8), context_seq_len(1), height, width]
        self.cross_attention_forward_hooks, self.lst_idx, self.current_time_step = module.get_attention_score(x, y, self.cross_attention_forward_hooks, self.lst_idx,
                                                                            self.current_time_step, generated_image=self.generated_image, per_time_step=self.per_time_step)
            
    def take_module(self, model):
        for name, module in model.named_modules():
            if isinstance(module, SpatialTransformer) and not 'middle_block' in name: #and module.to_k.in_features == 768 and not 'middle_block' in name:
                module.register_forward_hook(lambda m, inp, out, n=name: self.cross_attention_hook(m, inp, out, n))
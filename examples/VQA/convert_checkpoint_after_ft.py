import torch

pl_ckpt = "./step=0.ckpt"
output_dir = './'
hf_ckpt = {}

loaded = torch.load(pl_ckpt,map_location='cpu')
if 'state_dict' in loaded:
    sd = loaded['state_dict']
else:
    sd = loaded
for k,v in sd.items():
    if k.startswith('model.'):
        new_key = k.replace('model.', '')
        hf_ckpt[new_key] = v
    elif k.startswith('module.model.'):
        new_key = k.replace('module.model.', '')
        hf_ckpt[new_key] = v
    else:
        print("unhandled keys:",k)

torch.save(hf_ckpt, output_dir + 'pytorch_model.bin')
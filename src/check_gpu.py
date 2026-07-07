import torch

print('Torch:', torch.__version__)
print('Torch CUDA version:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print('VRAM GB:', round(props.total_memory / (1024**3), 2))
else:
    print('GPU: CPU only')

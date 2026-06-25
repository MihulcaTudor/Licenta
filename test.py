import torch

print("CUDA Disponibil:", torch.cuda.is_available())
print("Nume GPU:", torch.cuda.get_device_name(0))

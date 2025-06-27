import torch
import torch_directml

print(torch.__version__)

try:
    _ = torch.scan
    print('torch.scan is available!')
except AttributeError:
    print('torch.scan is NOT available.')
    if torch_directml.is_available():
        print('DirectML is available!')
    else:
        print('DirectML is NOT available.')

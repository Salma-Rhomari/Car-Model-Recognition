import torch
checkpoint = torch.load('models/resnet50_best.pth', map_location='cpu')
print(type(checkpoint))
if isinstance(checkpoint, dict):
    print(checkpoint.keys())
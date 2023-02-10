import torch
import torchvision

BATCH_SIZE = 20
DEVICE = torch.device('cpu')
#DEVICE = torch.device('cuda')
#DEVICE = torch.device('mps')
EPOCHS = 1
K = 5
LEARNING_RATE = 0.001
MOMENTUM = 0.9
NUM_OF_WORKERS = 0
SEED = 47
TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(416),
    torchvision.transforms.CenterCrop(384),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
TOKEN = "dropbox token goes here"

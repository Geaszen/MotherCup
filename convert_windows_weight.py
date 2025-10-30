import torch

# state = torch.load('checkpoints/classifier/best.pt', map_location='cpu')
# torch.save(state['model'] if 'model' in state else state, 'checkpoints/classifier/classifier_best2.pt')

state = torch.load('checkpoints/classifier_svdd/best.pt', map_location='cpu')
torch.save(state['model'] if 'model' in state else state, 'checkpoints/classifier_svdd/classifier_svdd_best3.pt')

state = torch.load('checkpoints/detector/best.pt', map_location='cpu')
torch.save(state['model'] if 'model' in state else state, 'checkpoints/detector/detector_best3.pt')
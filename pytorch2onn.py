from torch.autograd import Variable
import torch.onnx
import torchvision
from EdgeCNN import *

dummy_input = Variable(torch.randn(1, 3, 44, 44))
model = EdgeCNN()

model.load_state_dict(torch.load("/Users/pauldamsa/Desktop/licenta/PyImageSearch/face-detection/raspberrypi/RAF_EdgeCNN/PrivateTest_model.t7"), map_location=torch.device('cpu'))

torch.onnx.export(model, dummy_input, "/Users/pauldamsa/Desktop/licenta/PyImageSearch/face-detection/raspberrypi/RAF_EdgeCNN/torch_model.onnx", verbose=True)

print("Export of torch_model.onnx complete!")

from __future__ import print_function
import os
import torch
import torchvision.models as models

net = models.load('./weights/bestofbest.pt')
net.eval()
print('Finished loading model!')
device = torch.device("cuda:0")
net = net.to(device)

output_onnx = 'test_batch2.onnx'
input_names = ["input_0"]
output_names = ["output_0"]

inputs = torch.randn(2, 3, 256, 256).to(device)

torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
                                   input_names=input_names, output_names=output_names, opset_version=11)
from Src.layers import Linear , ReLU, BatchNorm2d, Softmax, Dropout
from Src.conv import Conv2d, Pooling2d
from Src.containers import Sequential
def LeNet_arch():
  return Sequential(
      Conv2d(num_filters = 32, filter_size = (5,5), in_channels=3,padding = 1, bias=False),
      BatchNorm2d(32,momentum = 0.999),
      ReLU(),
      Pooling2d(kernel_size = 2 ,pool_type = 'MAX'),
      Conv2d(num_filters = 64, filter_size = (5,5), in_channels=32,padding = 1,bias=False), 
      BatchNorm2d(64,momentum = 0.999),ReLU(),
      Pooling2d(kernel_size = 2 ,pool_type = 'MAX'),
      Flatten(),
      Linear(in_dim= 6*6*64, out_dim =120),ReLU(),
      Linear(in_dim= 120 , out_dim = 84), ReLU(),
      Linear(in_dim= 84 , out_dim = 10)
  )

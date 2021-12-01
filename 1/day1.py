import torch

# Read in lines
in_file = open("input", 'r')
measurements = torch.Tensor([int(x) for x in in_file.readlines()])

# Part 1
increase = (measurements[:-1] - measurements[1:]) < 0
print(torch.sum(increase))

# Part 2

sums = torch.nn.AvgPool1d(3, stride=1)
pooled = sums(measurements.view(1, 1, -1)).squeeze()
increase = (pooled[:-1] - pooled[1:]) < 0
print(torch.sum(increase))

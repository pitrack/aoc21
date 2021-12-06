import torch
import re
torch.set_default_tensor_type('torch.FloatTensor')
torch.set_printoptions(sci_mode=False)
# cpu: 0.6 gpu: RuntimeError: bincount/addmm not implemented on cuda

# Read in
line = open("input", 'r').read().strip()
counts = torch.bincount(torch.LongTensor([int(x) for x in line.split(",")]))
counts = torch.cat((counts, torch.zeros(9 - counts.shape[0]).long()))

# Part 1
sim_matrix = torch.Tensor([
    [0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
])
final_counts = counts.matmul(torch.matrix_power(sim_matrix, 80).long())
print(torch.sum(final_counts))

# Part 2 - matrix_power gets inexact if we go straight to 256 (1601616884048 instead of 1601616884019)
sim_128 = torch.matrix_power(sim_matrix, 128).long()
final_counts = counts.matmul(sim_128).matmul(sim_128)
print(torch.sum(final_counts))

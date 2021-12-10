import torch
import torch.nn.functional as F
import time
torch.set_default_tensor_type('torch.FloatTensor')
torch.set_printoptions(sci_mode=False)
start = time.time()
# cpu: 0.65s [0.004s + 0.006s]
# gpu: 4s [3s + 0.006s]

# Read in - this could be done more elegantly too
in_file = open("input", 'r')
numbers = torch.stack([torch.Tensor([int(x) for x in line.strip()])
                      for line in in_file.readlines()])

# Part 1, use 0.999 because we need strict inequalities
kernels = torch.Tensor(
    [[[0, 0.999, 0], [0, 0, 0], [0, 0, 0]],
     [[0, 0, 0], [0.999, 0, 0], [0, 0, 0]],
     [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
     [[0, 0, 0], [0, 0, 0.999], [0, 0, 0]],
     [[0, 0, 0], [0, 0, 0], [0, 0.999, 0]]
 ])
filters = kernels.unsqueeze(1)

h = int(numbers.shape[0])
w = int(numbers.shape[1])
padded_numbers = F.pad(numbers, (1, 1, 1, 1), mode='constant', value=9).unsqueeze(0)
channels = F.conv2d(padded_numbers.unsqueeze(0),
                    filters,
                    stride=(1,1),
                    padding=0)
min_values = channels.min(dim=1).values
print((numbers + 1)[min_values[0] == numbers].sum())
part1 = time.time()

# Part 2 -- need some notion of backpointers?
walls = padded_numbers == 9
ids = torch.arange(h * w).view(1, 1, h, w).float()
idx_kernel = torch.Tensor(
    [[[0, 1, 0], [0, 0, 0], [0, 0, 0]],
     [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
     [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
     [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
     [[0, 0, 0], [0, 0, 0], [0, 1, 0]]
 ]).unsqueeze(1)

for i in range(h * w):
    # h * w is an upper bound for the path length/area.
    # In practice, we only loop 8 times on our dataset
    channels = F.conv2d(padded_numbers.unsqueeze(0),
                        idx_kernel,
                        stride=(1,1),
                        padding=0)
    indices = F.conv2d(F.pad(ids, (1, 1, 1, 1), mode='constant', value=h*w+1),
                       idx_kernel,
                       stride=(1,1),
                       padding=0)
    min_values, min_indices = channels.min(dim=1)
    ids = torch.gather(input=indices.squeeze(0),
                       index=min_indices,
                       dim=0).view(1, 1, h, w)
    new_padded_numbers = F.pad(min_values, (1, 1, 1, 1), mode='constant', value=9)
    new_padded_numbers = torch.maximum(new_padded_numbers, 9 * walls.long())
    if (new_padded_numbers == padded_numbers).all():
        break
    else:
        padded_numbers = new_padded_numbers

# Fix the ids
padded_ids = torch.maximum(ids.view(h,w),
                           walls[0,1:1+h, 1:1+w].long() * (h * w +1))

counts = padded_ids.long().view(-1).bincount()[:h*w]
sorted_counts = counts.sort(descending=True).values
print(torch.prod(sorted_counts[:3]))

# Profiling
part2 = time.time()
print(part1-start, part2 - part1)

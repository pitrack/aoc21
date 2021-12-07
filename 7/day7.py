import torch
torch.set_default_tensor_type('torch.FloatTensor')
torch.set_printoptions(sci_mode=False)
# cpu: 0.66s gpu: 3.7s

# Read in -- this part is being made more generic
def process(line):
    return [int(x) for x in line.split(",")]
in_file = open("input", 'r')
numbers = torch.Tensor([process(line) for line in in_file.readlines()])

# Part 1
ranges = torch.arange(torch.max(numbers).long())
distances = torch.abs(ranges.unsqueeze(1) - numbers)
print(torch.min(distances.sum(dim=1)))

# Part 2
# Somehow this part was nontrivial because pytorch doesn't know how
# to divide by the integer 2.
#
# (Pdb) x
# tensor(209644488)
# (Pdb) x/2
# tensor(104822240.)
#
# The solution is to either use float64 or use // (integer div)
# instead of / (cast both arguments to float, even if they were
# both LongTensors to begin with.
costs = (torch.square(distances) + distances).long() // 2
print(torch.min(costs.sum(dim=1)))

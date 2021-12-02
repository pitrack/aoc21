import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_printoptions(sci_mode=False)
# CPU: 0.55s, GPU: 2.1s

# Read in lines
word_map = {
    "forward": torch.Tensor([1, 0]),
    "down": torch.Tensor([0, 1]),
    "up": torch.Tensor([0, -1])
}

in_file = open("input", 'r')
movements = torch.stack([int(line.split()[1]) * word_map[line.split()[0]]
                          for line in in_file.readlines()])

# Part 1
final_loc = torch.sum(movements, dim=0)
print(torch.prod(final_loc))

# Part 2
h_dir, a_dir = torch.unbind(movements, dim=1)
new_movements = torch.stack((h_dir, h_dir * torch.cumsum(a_dir, dim=0)), dim=1)
print(torch.prod(torch.sum(new_movements, dim=0).int()))

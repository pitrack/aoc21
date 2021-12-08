import torch
import time
torch.set_default_tensor_type('torch.FloatTensor')
torch.set_printoptions(sci_mode=False)
# cpu: 0.062 + 0.068
# gpu: 2.8 + 3.8
start = time.time()

MAPPING = {
    "a": torch.Tensor([1, 0, 0, 0, 0, 0, 0]),
    "b": torch.Tensor([0, 1, 0, 0, 0, 0, 0]),
    "c": torch.Tensor([0, 0, 1, 0, 0, 0, 0]),
    "d": torch.Tensor([0, 0, 0, 1, 0, 0, 0]),
    "e": torch.Tensor([0, 0, 0, 0, 1, 0, 0]),
    "f": torch.Tensor([0, 0, 0, 0, 0, 1, 0]),
    "g": torch.Tensor([0, 0, 0, 0, 0, 0, 1]),
}

# Read in - this could be done more elegantly too
in_file = open("input", 'r')
lefts = []
rights = []
for line in in_file.readlines():
    parts = line.split("|")
    # Create 10 x 7
    left = torch.stack([torch.sum(torch.stack([MAPPING[char] for char in token]), dim=0) for token in parts[0].split()])
    # Create 4 x 7 -- It would be nice/easier if this were a 4 x 1 set of indices into `left`
    right = torch.stack([torch.sum(torch.stack([MAPPING[char] for char in token]), dim=0) for token in parts[1].split()])
    lefts.append(left)
    rights.append(right)
known = torch.stack(lefts)
target = torch.stack(rights)

# Part 1
num_segs = torch.sum(target, dim=2)
print(torch.sum(num_segs < 5) + torch.sum(num_segs > 6))
print(time.time() - start)
# Part 2
DISPLAY = torch.Tensor(
    [[1, 1, 1, 0, 1, 1, 1], # 0
     [0, 0, 1, 0, 0, 1, 0], # 1
     [1, 0, 1, 1, 1, 0 ,1], # 2
     [1, 0, 1, 1, 0, 1, 1], # 3
     [0, 1, 1, 1, 0, 1, 0], # 4
     [1, 1, 0, 1, 0, 1, 1], # 5
     [1, 1, 0, 1, 1, 1, 1], # 6
     [1, 0, 1, 0, 0, 1, 0], # 7
     [1, 1, 1, 1, 1, 1, 1], # 8
     [1, 1, 1, 1, 0, 1, 1], # 9
 ])
DISPLAY_V_inv = DISPLAY.svd().V.abs().inverse()

known_v = known.svd().V.abs()
all_perms = known_v.matmul(DISPLAY_V_inv).transpose(2, 1) # we could be done here! instad, more math:
fixed_wires = all_perms.matmul(target.transpose(2, 1)).transpose(2, 1)
diff = torch.abs(fixed_wires.unsqueeze(2) - DISPLAY.view(1, 1, 10, 7))
digits = torch.argmin(diff.sum(dim=-1), dim=-1)#.float().cuda()
base_10 = torch.Tensor([1000, 100, 10, 1])#.float().cuda()

print(torch.sum(digits * base_10))
print(time.time() - start)

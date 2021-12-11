import torch
import torch.nn.functional as F
import time
torch.set_default_tensor_type('torch.FloatTensor')
torch.set_printoptions(sci_mode=False)
start = time.time()
# cpu: 0.7s [0.5 + 0.0015]
# gpu: 4s [2.9 + 0.0012]

# Read in - this could be done more elegantly too
MAPPING = {
    "(": torch.Tensor([1, 0, 0, 0]),
    ")": torch.Tensor([-1, 0, 0, 0]),
    "[": torch.Tensor([0, 1, 0, 0]),
    "]": torch.Tensor([0, -1, 0, 0]),
    "{": torch.Tensor([0, 0, 1, 0]),
    "}": torch.Tensor([0, 0, -1, 0]),
    "<": torch.Tensor([0, 0, 0, 1]),
    ">": torch.Tensor([0, 0, 0, -1]),
}

in_file = open("input", 'r')
numbers = [torch.stack([MAPPING[x] for x in line.strip()])
           for line in in_file.readlines()]
padded_numbers = torch.cat((torch.zeros(len(numbers), 1, 4),
                            torch.nn.utils.rnn.pad_sequence(numbers, batch_first=True)),
                           dim=1)

def index_gather(tensor, indices):
    """Maybe this exists already?
    Takes tensor T: [m, n] and U: [m, n] indices and keeps only values in T that are
    the row-wise max of U.
    """
    max_index = torch.argmax(indices, dim=-1)
    zeros = torch.zeros(tensor.shape[0], tensor.shape[1])
    mask = zeros.index_put((torch.arange(tensor.shape[0]), max_index), torch.tensor(1.))
    return mask * tensor

# Part 1 - in an alternate universe, we'd train an RNN on paren matching
num_exps = padded_numbers.shape[0]
length = padded_numbers.shape[1]

cumsum = padded_numbers.cumsum(dim=1)
closes = padded_numbers.sum(dim=2) < 0
counters = cumsum * closes.long().unsqueeze(-1)

# Check if each stack state has been seen before
# [batch, len, 1, 4] - [batch, 1, len, 4] = [batch, len, len, 4] --> [batch, len, len]
seen_stack = torch.all((counters.unsqueeze(2) - cumsum.unsqueeze(1) == 0), dim=-1)
masked_stack = torch.tril(seen_stack, diagonal=-1)

# Check when the same num open parens has been seen before
seen_totals = counters.sum(dim=-1).unsqueeze(2) - cumsum.sum(dim=-1).unsqueeze(1) == 0
masked_seen_totals = torch.tril(seen_totals, diagonal=-1)

# Find the last time the same num open parens was seen
reshaped_mst = masked_seen_totals.view(-1, length) * torch.arange(length).repeat(num_exps * length, 1)
most_recent_seen_totals = index_gather(masked_seen_totals.view(-1, length),
                                       reshaped_mst).view(num_exps, length, length)

# and that should match the last time we saw this stack state
is_matched = seen_stack * most_recent_seen_totals
is_safe = torch.sum(is_matched, dim=2) # 1 if it matches
is_problem = (1 - is_safe) * closes

# Now we know which brackets don't close, find the first one
indices = torch.arange(padded_numbers.shape[1], 0, -1) * is_problem.long()

# And calculate total score
scores = torch.Tensor([3, 57, 1197, 25137])
all_scores = padded_numbers.matmul(-1 * scores)
syntax_score = index_gather(all_scores, indices)
print(torch.sum(syntax_score))
part1 = time.time()

# Part 2 - first filter with what's leftover
incomplete = torch.sum(syntax_score, dim=1) == 0
closes = closes[incomplete]
num_incomplete = closes.shape[0]

# is_matched[i,j,k] means for seq i, if j is closed then k is right before open
is_matched = is_matched[incomplete] * closes.unsqueeze(2)

# this means the open is the index to the right, and we can sum all of them and reshape a bit (the last open can't be consumed)
used_opens = torch.cat((torch.zeros(num_incomplete, length, 1), is_matched), dim=2)[:,:,:length].sum(dim=1)

# And add in all the closed, trim out the dummy at start...
unconsumed = (1 - (closes.long() + used_opens))[:,1:].long()

# Scoring - be careful not to introduce float64s, remove padding
scores = torch.Tensor([1, 2, 3, 4])
all_scores = padded_numbers[incomplete].matmul(scores)[:, 1:].long()
base_5_exp = unconsumed.cumsum(dim=1)
base_5 = torch.pow(5, (base_5_exp - 1))
autocomplete_scores = (all_scores * unconsumed * base_5).sum(dim=1)
print(autocomplete_scores.median())

# Profiling
part2 = time.time()
print(part1-start, part2 - part1)

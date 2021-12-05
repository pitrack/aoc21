import torch
import re
torch.set_default_tensor_type('torch.FloatTensor')
torch.set_printoptions(sci_mode=False)
# cpu: 0.68 gpu: RuntimeError

# Read in
in_file = open("input", 'r')
exp = "(.*),(.*) -> (.*),(.*)\W*"
numbers = torch.stack([
    torch.Tensor([int(x) for x in re.match(exp, line).groups()])
    for line in in_file.readlines()]
)
grid_size = torch.max(numbers + 1)

# Helper functions
def tensorized_arange(bounds, max_range):
    # Looks like this needs to be custom written
    step_sizes = torch.sign(torch.stack((bounds[:, 2] - bounds[:, 0], bounds[:, 3] - bounds[:, 1]),
                                        dim=1)).unsqueeze(1)
    arange_2d_tensor = torch.stack((torch.arange(max_range), torch.arange(max_range)), dim=1)
    arange_values = step_sizes * arange_2d_tensor.unsqueeze(0)
    all_range = bounds[:, :2].unsqueeze(1) + arange_values
    good_x = step_sizes[:,:,0] * all_range[:,:,0] <= (step_sizes[:,0,0] * bounds[:, 2]).unsqueeze(1)
    good_y = step_sizes[:,:,1] * all_range[:,:,1] <= (step_sizes[:,0,1] * bounds[:, 3]).unsqueeze(1)
    range_mask = torch.logical_and(good_x, good_y)
    return all_range, range_mask

def get_overlaps(numbers, grid_size, orthogonal=True):
    if orthogonal:
        numbers = numbers[torch.logical_or(numbers[:, 0] == numbers[:, 2], numbers[:, 1] == numbers[:, 3])]
    locations, range_mask = tensorized_arange(numbers, grid_size)
    flat_locations = torch.flatten(locations, end_dim=1)
    flat_range_mask = torch.flatten(range_mask, end_dim=1)
    all_locations = flat_locations[flat_range_mask]
    grid = torch.zeros((grid_size, grid_size))
    grid = grid.index_put(indices=all_locations.long().unbind(dim=1),
                          values=torch.Tensor([1]),
                          accumulate=True)
    return grid

# Part 1
part1_grid = get_overlaps(numbers, int(grid_size))
print(torch.sum(part1_grid > 1))

# Part 2
part2_grid = get_overlaps(numbers, int(grid_size), orthogonal=False)
print(torch.sum(part2_grid > 1))

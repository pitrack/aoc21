import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_printoptions(sci_mode=False)
# cpu: 0.62s gpu: 3.7s

# Read in
in_file = open("input", 'r')
numbers = None
boards = []
board = []
for line in in_file.readlines():
    if line == "\n":
        if board:
            boards.append(torch.stack(board))
        board = []
        continue
    if numbers is None:
        numbers = torch.Tensor([int(x) for x in line.split(",")])
        continue
    board.append(torch.Tensor([int(x) for x in line.split()]))
all_boards = torch.stack(boards)

# Part 1

_, indices = torch.sort(numbers)
emb_boards = torch.nn.functional.embedding(all_boards.long(), indices.view(-1, 1)).squeeze()
board_scores = torch.cat((torch.max(emb_boards, dim=2).values.squeeze(),
                          torch.max(emb_boards, dim=1).values.squeeze()),
                         dim=1)
first_win = torch.min(board_scores, dim=1).values
min_score, first_board_idx = torch.min(first_win, 0)
first_emb_board = emb_boards[first_board_idx]
first_board = all_boards[first_board_idx]
unused_sum = torch.sum((first_emb_board > min_score) * first_board)
print(unused_sum * numbers[min_score])


# Part 2: copy and switch "min" to "max"

first_win = torch.min(board_scores, dim=1).values
max_score, last_board_idx = torch.max(first_win, 0)
last_emb_board = emb_boards[last_board_idx]
last_board = all_boards[last_board_idx]
unused_sum = torch.sum((last_emb_board > max_score) * last_board)
print(unused_sum * numbers[max_score])

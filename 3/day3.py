import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_printoptions(sci_mode=False)
# cpu: 0.55 gpu: 2.1s

# Read in
in_file = open("input", 'r')
bits = torch.stack([torch.Tensor([int(x) for x in list(line.strip())])
                    for line in in_file.readlines()])
BASE = torch.pow(2, torch.arange(bits.shape[1] - 1, -1, -1))

# Part 1
def convert_to_answer(vals):
    # Vals should have shape [2, len(bits)]
    answer = torch.prod(torch.sum(vals * BASE, dim=1)).int().item()
    print(answer)

gamma, _ = torch.mode(bits, dim=0)
found_values = torch.stack([gamma, 1 - gamma])
convert_to_answer(found_values)

# Part 2
# torch.mode returns the lower one by default, so we're
# going to multiply bits by -1
# nevermind, torch.mode is still inconsistent?

oxygen = bits.clone()
co2 = bits.clone()
for i in range(12): # :( in theory I can unroll this since it is only 12
    oxygen_mode = torch.sum(oxygen[:, i], dim=0) >= oxygen.shape[0] / 2
    oxygen = oxygen[oxygen[:, i] == oxygen_mode.long()]
    if co2.shape[0] > 1: #:( conditional
        co2_mode = torch.sum(co2[:, i], dim=0) < co2.shape[0] / 2
        co2 = co2[co2[:, i] == co2_mode]
answer_values = torch.stack([oxygen, co2]).squeeze()
convert_to_answer(answer_values)


# This cursed approach didn't work so it's commented out for posterity. It worked
# on the example case (12 x 5) most of the time. In retrospect, I ran this code with
# several different hyperparameters  10+ times and printed a bunch of intermediate
# predictions and never once remember seeing an guess starting with the correct digit.

# This language model solution was a cool idea even though it's super inefficient
# and doesn't work.

# We'll define a small RNN. The theory is that we just want the most common
# and least common prefix, which is something an RNN can learn once fully trained.
# We're going to have issues tiebreaking 0 and 1s -- and as it turns out, actually optimizing
# the model correctly. And dealing with the case where {101, 111, 011} means that P(0) > P(1)
# because I comparing P(101), P(111), P(011). There's probably a way to fix this and decode the
# right thing, but the tiebreaking made this too hard anyway, with ties being fairly frequent.

# import time
# from collections import Counter

# INPUT_SIZE = 8
# HIDDEN_SIZE = 32
# NUM_EPOCHS = 50000
# rnn = torch.nn.RNN(input_size=INPUT_SIZE,
#                    hidden_size=HIDDEN_SIZE,
#                    num_layers=2,
#                    nonlinearity="relu",
#                    bias=True,
#                    batch_first=True,
# )
# output_layer = torch.nn.Linear(HIDDEN_SIZE, 1)
# embedding = torch.nn.Embedding(3, INPUT_SIZE) # 2 16-dim embs

# bits_with_bos = torch.cat((2 * torch.ones(bits.shape[0], 1), bits), dim=1).long()
# loss = torch.nn.BCELoss()
# optimizer = torch.optim.SGD([
#     {'params': embedding.parameters()},
#     {'params': rnn.parameters()},
#     {'params': output_layer.parameters()}
# ], lr=0.001)

# # We still need a for loop for training to update model parameters

# start_time = time.time()
# best_values = Counter()
# for epoch in range(NUM_EPOCHS):
#     optimizer.zero_grad()
#     model_input = embedding(bits_with_bos)
#     rnn_output = rnn(model_input)
#     logits = torch.sigmoid(output_layer(rnn_output[0][:,:-1,:])).squeeze()
#     loss = torch.nn.functional.binary_cross_entropy(logits, bits, reduction="none")
#     # per_example_loss = loss.matmul(BASE.float())
#     batch_loss = torch.mean(loss)
#     batch_loss.backward()
#     optimizer.step()
#     if epoch % 500 == 0:
#         # Failed attempts at tiebreaking: add a very small epsilon in favor of 1.
#         log_half = -1 * torch.log(torch.Tensor([0.5]))
#         # favorable = (loss - (bits * 0.002) > (log_half - 0.001)).float().matmul(BASE.float())
#         favorable = (loss >= log_half).float().matmul(BASE.float())
#         min_ppl, most_likely = torch.min(favorable, 0)
#         max_ppl, least_likely = torch.max(favorable, 0)
#         print(f"Epoch {epoch} {time.time() - start_time:.3f}s [loss: {batch_loss}] (min: {min_ppl}, max: {max_ppl})")
#         start_time = time.time()
#         found_values = torch.stack((bits[most_likely], bits[least_likely]))
#         found_loss = torch.stack((loss[most_likely], loss[least_likely]))
#         print(found_values)
#         print(found_loss)
#         answer = convert_to_answer(found_values)
#         best_values[answer] += 1

# print(best_values)

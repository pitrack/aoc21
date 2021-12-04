# Advent of Code 2021

This is my first year doing Advent of Code. My aim is to primarily use pytorch functions (and some preprocessing). Main rules:

1. Single pass over input file to store input data into a Tensor
2. ~~No use of for loops or list/set/dict comprehensions~~ For loop used on 3.2 although in theory that could be unrolled.

For loops are allowed now but I'll still try to keep all the data vectorized and use pytorch primitives/tensors.

# brain-hs

An attempt to write simple and legible MNIST classifier in haskell.

## About

It trains with on-line learning, processing all examples one-by-one for a five minutes on my pc.

Main inefficiency is caused by using matrices and vectors as a single linked lists in Haskell.

As for a C++ developer, it sounds funny, but it works anyway under 200 lines of code, not all of which is even used.

Current training gets about 40% success rate, which is better than random...

## Usage

1. Install stack(apt-get stack or something)
2. In the project directory, run:

```sh
stack run download   # download database
stack run train <x>  # train model on the first x images
stack run test <x>   # test model
stack run draw <x>   # draw record number x in the console
```

## Inspiration source

<https://crypto.stanford.edu/~blynn/haskell/brain.html>
(but it actually is brainfuck there, not haskell.)

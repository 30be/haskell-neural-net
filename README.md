# brain-hs

An attempt to write simple and legible MNIST classifier in haskell.

## About

It trains with on-line learning, processing all examples one-by-one by five minutes on my laptop.

Main inefficiency is caused by using matrices and vectors as a single linked lists in Haskell. As for a C++ developer, it sounds funny, but it works anyway.

Current one-epoch training gets about 40% success rate, which is better than random...

## Usage

1. Install stack(apt-get stack or something)
2. In the project directory, run:

```sh
stack run download          # download database
stack run train <x>         # train model on the first x images
stack run train further<x>  # train model on the first x images, taking existing model as a source
stack run test <x>          # test model
stack run test on <x>       # test model on test sample x
stack run draw <x>          # draw record number x in the console
```

## Inspiration source

<https://crypto.stanford.edu/~blynn/haskell/brain.html>
(but it actually is brainfuck there, not haskell.)

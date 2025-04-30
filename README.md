# brain-hs

An attempt to write simple and legible MNIST classifier in haskell.

## Usage

1. Install stack(apt-get stack or something)
2. In the project directory, run:

```sh
stack run download  # download database
stack run train <amount>   # train model on the first x images
stack run test     # test model
```

## Inspiration source

<https://crypto.stanford.edu/~blynn/haskell/brain.html>
(but it actually is brainfuck there, not haskell.)

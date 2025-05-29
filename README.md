# brain-hs

An attempt to write simple and legible MNIST classifier in haskell.

## About

It trains with on-line learning, processing all examples one-by-one by five minutes on my laptop.

Main inefficiency is caused by using matrices and vectors as a single linked lists in Haskell. As for a C++ developer, it sounds funny, but it works anyway.

Current one-epoch training gets 88.97% success rate, which can be improved with multiple epochs.
But i will fix the bug some time..

## Usage

1. Install stack(apt-get stack or something)
2. In the project directory, run:

```sh
stack run download          # download database
stack run train <x>         # train model on the first x images
stack run train further<x>  # train model on the first x images, taking existing model as a source(one-epoch, basically)
stack run test <x>          # test model
stack run test on <x>       # test model on test sample x
stack run draw <x>          # draw record number x in the console
```

## Notes

```
~/dev/haskell-neural-net % stack run test on 10




                      ..    ::OO::::..
                    OO@@@@@@@@@@@@@@@@@@..
                  oo@@@@@@OOOOOOOOOO@@@@@@..
                ..@@@@@@..          oo@@@@OO
                ..@@@@::              oo@@@@..
                ..@@oo                  @@@@oo
                OO@@::                  oo@@OO
                @@@@..                  ..@@OO
              ::@@OO                    ..@@OO
              @@@@..                    ..@@OO
              @@OO                      ..@@OO
              @@OO                      ..@@OO
            ::@@OO                      oo@@OO
            ::@@OO                    ..@@@@..
            ::@@OO                  ..oo@@OO
            ..@@OO                  oo@@@@..
              @@OO            ..ooOO@@@@..
              @@@@OOOOOOOOOOOO@@@@@@@@oo
              ..OO@@@@@@@@@@@@@@@@@@::
                  OO@@@@@@@@@@oo::





0(0.84): ########
1(0.00):
2(0.00):
3(0.00):
4(0.00):
5(0.00):
6(0.00):
7(0.00):
8(0.00):
9(0.00):

best guess: 0.0


```

## Inspiration source

<https://crypto.stanford.edu/~blynn/haskell/brain.html>
(but it actually is brainfuck there, not haskell.)

i havent yet read but <https://jpmoresmau.blogspot.com/2007/06/very-dumb-neural-network-in-haskell.html> may have something to do with it

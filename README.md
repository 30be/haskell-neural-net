# brain-hs

An attempt to write simple and legible MNIST classifier in haskell.

## About

It trains with on-line learning, processing all examples one-by-one by five minutes on my laptop.

Main inefficiency is caused by using matrices and vectors as a single linked lists in Haskell. As for a C++ developer, it sounds funny, but it works anyway.

Current one-epoch training gets about only 40% success rate, which is better than random...
But i will fix the bug some time..

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

## Notes

```


        ..@@@@@@@@@@oooooo  ....
        ::@@@@@@@@@@@@@@@@@@@@@@OO::..
          ::oo::::::OOOOOO@@@@@@@@@@@@@@oo
                          ..::::OO@@@@@@@@
                                    oo@@@@oo
                                      @@@@
                                    ::@@@@::
                                    @@@@@@
                                  ::@@@@oo
                                  @@@@@@
                                OO@@@@::
                              oo@@@@@@
                            ::@@@@@@..
                          ..@@@@@@..
                          @@@@@@oo
                        ::@@@@OO
                      ..@@@@@@..
                      @@@@@@OO
                    ..@@@@OO
                      @@OO


0(-1.46):
1(-4.44):
2(-1.47):
3(-1.42):
4(-2.64):
5(-1.49):
6(-1.62):
7(1.30): #############
8(-1.47):
9(-1.77):
```

Output looks like this. It has a bug now, which affects amount of images guessed right - many coefficients are negative

## Inspiration source

<https://crypto.stanford.edu/~blynn/haskell/brain.html>
(but it actually is brainfuck there, not haskell.)

i havent yet read but <https://jpmoresmau.blogspot.com/2007/06/very-dumb-neural-network-in-haskell.html> may have something to do with it

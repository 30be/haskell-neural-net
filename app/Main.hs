module Main (main) where

import Lib
import Codec.Compression.GZip (decompress)

trainingImages = "train-images-idx3-ubyte.gz"
trainingLabels = "train-labels-idx1-ubyte.gz"
testingImages = "t10k-images-idx3-ubyte.gz"
testingImages = "t10k-labels-idx3-ubyte.gz"
mnistLink = "https://storage.googleapis.com/cvdf-datasets/mnist/"

downloadMNIST :: IO ()
downloadMNIST = mapM_ download [trainingImages, trainingLabels, testingImages, testingLabels]
  where download fileName = simpleHTTP (mnistLink ++ fileName) >>= saveFile fileName

drawImageN :: Int -> IO ()
drawImageN n = renderImage . getImage n . decompress <$> readFile trainingImages >>= putStrLn

main :: IO ()
main = getArgs >>= \case
  ["download"] -> downloadMNIST
  --["train"] -> train
  ["draw", n] -> drawImageN $ read n
  _ -> putStrLn "Only download, draw n commands are supported"

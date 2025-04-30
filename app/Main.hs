{-# LANGUAGE LambdaCase #-}

module Main (main) where

import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as L
import Lib
import Network.HTTP.Conduit (simpleHttp)
import System.Directory.Internal.Prelude (getArgs)

trainingImages, trainingLabels, testingImages, testingLabels, mnistLink :: String
trainingImages = "train-images-idx3-ubyte.gz"
trainingLabels = "train-labels-idx1-ubyte.gz"
testingImages = "t10k-images-idx3-ubyte.gz"
testingLabels = "t10k-labels-idx1-ubyte.gz"
mnistLink = "https://storage.googleapis.com/cvdf-datasets/mnist/"

downloadMNIST :: IO ()
downloadMNIST = mapM_ download [trainingImages, trainingLabels, testingImages, testingLabels]
 where
  download fileName = putStrLn ("downloading " ++ fileName) >> simpleHttp (mnistLink ++ fileName) >>= L.writeFile fileName

drawImageN :: Int -> IO ()
drawImageN n = L.readFile trainingImages >>= putStrLn . renderImage . getImage n . decompress

printLabelN :: Int -> IO ()
printLabelN n = L.readFile trainingLabels >>= print . getLabel n . decompress

trainModel :: Int -> IO ()
trainModel amount = do
  images <- L.readFile trainingImages
  labels <- L.readFile trainingLabels
  putStrLn "Started training..."
  let model = train images labels amount
  putStrLn "Training complete."
  writeFile "model" $ show model
  putStrLn "model is saved to 'model' file"

testModel :: IO ()
testModel = do
  putStrLn "Loading model..."
  model <- read <$> readFile "model"
  trainImages <- L.readFile trainingImages
  trainLabels <- L.readFile trainingLabels
  putStrLn "Testing..."
  putStrLn $ test model trainImages trainLabels
main :: IO ()
main =
  getArgs >>= \case
    ["download"] -> downloadMNIST
    ["train", amount] -> trainModel (read amount)
    ["test"] -> testModel
    ["draw", n] -> drawImageN (read n) >> printLabelN (read n)
    _ -> putStrLn "Only [download, draw n, train, test] commands are supported"

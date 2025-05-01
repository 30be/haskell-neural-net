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

readGz :: FilePath -> IO L.ByteString
readGz = (decompress <$>) . L.readFile

drawImageN :: Int -> IO ()
drawImageN n = readGz trainingImages >>= putStrLn . renderImage . getImage n

printLabelN :: Int -> IO ()
printLabelN n = readGz trainingLabels >>= print . getLabel n

trainModel :: Int -> IO ()
trainModel amount = do
  images <- readGz trainingImages
  labels <- readGz trainingLabels
  putStrLn "Started training..."
  let model = train images labels amount
  writeFile "model" $ show model
  putStrLn "Training complete."
  putStrLn "model is saved to 'model' file"

testModel :: Int -> IO ()
testModel amount = do
  putStrLn "Loading model..."
  model <- read <$> readFile "model"
  testImages <- readGz testingImages
  testLabels <- readGz testingLabels
  putStrLn "Testing..."
  putStrLn $ test model testImages testLabels amount

main :: IO ()
main =
  getArgs >>= \case
    ["download"] -> downloadMNIST
    ["train", amount] -> trainModel (read amount)
    ["test", amount] -> testModel (read amount)
    ["draw", n] -> drawImageN (read n) >> printLabelN (read n)
    _ -> putStrLn "Only [download, draw n, train, test n] commands are supported"

{-# LANGUAGE LambdaCase #-}

module Main (main) where

import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as L
import Lib
import Network.HTTP.Conduit (simpleHttp)
import System.Directory.Internal.Prelude (getArgs)
import System.IO (readFile')

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

trainModel :: Int -> [Layer] -> IO ()
trainModel amount model = do
  images <- readGz trainingImages
  labels <- readGz trainingLabels
  putStrLn "Started training..."
  let trained = train images labels amount model
  writeFile "model" $ show trained
  putStrLn "Trained model. The model is saved to 'model' file"

testModel :: Int -> IO ()
testModel amount = do
  model <- read <$> readFile "model"
  testImages <- readGz testingImages
  testLabels <- readGz testingLabels
  putStrLn "Testing..."
  putStrLn $ test model testImages testLabels amount

testModelOn :: Int -> IO ()
testModelOn n = do
  testImage <- getImage n <$> readGz testingImages
  model <- read <$> readFile "model" :: IO [Layer]
  let result = foldl applyLayer testImage model
  putStrLn $ renderImage testImage
  putStrLn $ drawActivations result
  putStrLn $ "best guess: " ++ show (maximumIndex result :: Float)

main :: IO ()
main =
  getArgs >>= \case
    ["download"] -> downloadMNIST
    ["train", amount] -> trainModel (read amount) $ newBrain [imageSize, 30, 10]
    ["train", "further", amount] -> readFile' "model" >>= trainModel (read amount) . read
    ["test", amount] -> testModel (read amount)
    ["test", "on", n] -> testModelOn (read n)
    ["draw", n] -> drawImageN (read n) >> printLabelN (read n)
    _ -> putStrLn "Only [download, draw n, train n, train further n, test n, test on n] commands are supported"

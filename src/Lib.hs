module Lib where

import qualified Data.ByteString.Lazy as L

import Data.List.Split (chunksOf)
import System.Random

type Bias = Float
type Weight = Float
type Activation = Float
type Layer = ([Bias], [[Weight]])
type LayerSize = Int

eta :: Float
eta = 0.002
imageWidth, imageSize, imagesHeaderSize, labelsHeaderSize :: Int
imageWidth = 28
imageSize = imageWidth ^ (2 :: Integer)
imagesHeaderSize = 16 -- magic number + amount + width + height
labelsHeaderSize = 8

-- List for convenience
gauss :: Float -> [Float] -> Float
gauss scale [x1, x2] = scale * sqrt ((-2) * log x1) * cos (2 * pi * x2)
gauss _ _ = error "Gauss distribution has 2 args"

gaussSequence :: StdGen -> Float -> [Float]
gaussSequence gen scale = gauss scale <$> chunksOf 2 (randomRs (0, 1) gen)

newBrain :: [LayerSize] -> [Layer]
newBrain sizes = zipWith3 makeLayer sizes (tail sizes) [0 ..]
 where
  makeLayer m n i = (replicate n 1.0, replicate n $ take m $ gaussSequence (mkStdGen i) 0.01)

relu :: Float -> Activation
relu = max 0

relu' :: Float -> Float
relu' = max 0 . signum

renderImage :: [Activation] -> [Char]
renderImage = unlines . chunksOf imageWidth . map render
 where
  render color = " .:oO@X" !! floor (color * 6.0) -- X should never happen?

getImage :: Int -> L.ByteString -> [Activation]
getImage n images = getColor <$> indexes
 where
  getColor = (/ 256) . fromIntegral . L.index images . fromIntegral
  indexes = (n * imageSize + imagesHeaderSize +) <$> [0 .. imageSize - 1]

getLabel :: Int -> L.ByteString -> Int
getLabel n labels = fromIntegral $ L.index labels $ fromIntegral $ labelsHeaderSize + 8 + n

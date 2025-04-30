module Lib where

import qualified Data.ByteString.Lazy as L

import Data.Bifunctor (bimap)
import Data.Bits (Bits (xor))
import Data.Char (digitToInt)
import Data.Function (on)
import Data.List (transpose, unfoldr)
import Data.List.Split (chunksOf)
import Data.Word (Word8)
import Debug.Trace (trace)
import GHC.Char (chr)
import System.Random

type LayerSize = Int
type Bias = Float
type Weight = Float
type Activation = Float
type WeightedInput = Float
type ErrorDelta = Float
type Layer = ([Bias], [[Weight]])

learningRate :: Float
learningRate = 0.002
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

relu :: WeightedInput -> Activation
relu = max 0

relu' :: Float -> Float
relu' x = if x < 0 then 0 else 1

-- derivative of the cost function
-- TODO:
cost' :: Activation -> Activation -> Float
cost' actual desired = if desired == 1 && actual >= desired then 0 else actual - desired

(+.) :: (Num c) => [c] -> [c] -> [c]
(+.) = zipWith (+)

(*.) :: (Num c) => [[c]] -> [c] -> [c]
(*.) matrix vector = sum . zipWith (*) vector <$> matrix

dot :: (Num a) => [a] -> [a] -> a
dot = (sum .) . zipWith (*)

zLayer :: [Activation] -> Layer -> [WeightedInput]
zLayer inputActivations (biases, weightMatrix) = weightMatrix *. inputActivations +. biases

applyLayer :: [WeightedInput] -> Layer -> [WeightedInput]
applyLayer = zLayer . map relu

feed :: [Activation] -> [Layer] -> [Activation]
feed = foldl applyLayer

backpropagate :: [[ErrorDelta]] -> ([[Weight]], [WeightedInput]) -> [[ErrorDelta]]
backpropagate (e : rrors) (weightMatrix, weightedInput) = currentErrors : e : rrors
 where
  currentErrors = zipWith (*) (dot e <$> weightMatrix) (relu' <$> weightedInput)
backpropagate _ _ = error "backpropagate called with empty error vector"

deltas :: [Activation] -> [Activation] -> [Layer] -> ([[Activation]], [[ErrorDelta]])
deltas inputs desiredOutputs layers = (activations, foldl backpropagate [delta0] $ reverse weightsInputs)
 where
  weightedInputs = scanl applyLayer inputs layers
  activations = map relu <$> weightedInputs
  delta0 = zipWith3 deltaI (last weightedInputs) (last activations) desiredOutputs
  deltaI input activation desired = cost' activation desired * relu' input -- Derivative of cost function wrt weighted input
  weightsInputs = zip (transpose . snd <$> layers) (init weightedInputs)

descend :: [Float] -> [Float] -> [Float]
descend values gradients = zipWith (-) values $ (learningRate *) <$> gradients

learn :: [Layer] -> ([Activation], [Activation]) -> [Layer]
learn layers (input, output) = trace "x" $ zip biasVectors weightMatrices
 where
  (layerActivations, layerGradients) = deltas input output layers
  biasVectors = zipWith descend (fst <$> layers) layerGradients -- How does that work that i apply gradients to biases just like this
  weightMatrices = zipWith3 (zipWith . updateWeightMatrix) layerActivations (snd <$> layers) layerGradients
  updateWeightMatrix activations weights gradients = descend weights ((gradients *) <$> activations)

renderImage :: [Activation] -> [Char]
renderImage = unlines . chunksOf (imageWidth * 2) . concatMap (replicate 2 . render)
 where
  gradient = " .:oO@"
  render color = gradient !! floor (color * fromIntegral (length gradient)) -- ? should never happen

getImage :: Int -> L.ByteString -> [Activation]
getImage n images = getColor <$> indexes
 where
  getColor = (/ 256) . fromIntegral . L.index images . fromIntegral
  indexes = (n * imageSize + imagesHeaderSize +) <$> [0 .. imageSize - 1]

getLabel :: Int -> L.ByteString -> Int
getLabel n labels = toEnum $ fromEnum $ L.index labels $ fromIntegral $ labelsHeaderSize + n

-- | Splits a lazy ByteString into chunks of size n. - not tested
chunksOf' :: Int -> L.ByteString -> [L.ByteString]
chunksOf' n = unfoldr (\b -> if L.null b then Nothing else Just (L.splitAt (fromIntegral n) b))

getLabelActivation :: Word8 -> [Activation]
getLabelActivation = map (fromIntegral . fromEnum) . flip map [0 .. 9] . (==)

parseImages, parseLabels :: L.ByteString -> [[Activation]]
parseImages = map (map ((/ 256) . fromIntegral) . L.unpack) . chunksOf' imageSize . L.drop (fromIntegral imagesHeaderSize)
parseLabels = map getLabelActivation . L.unpack . L.drop (fromIntegral labelsHeaderSize)

train :: L.ByteString -> L.ByteString -> [Layer]
train images labels = foldl learn (newBrain [imageSize, 30, 10]) $ zip (parseImages images) (parseLabels labels)

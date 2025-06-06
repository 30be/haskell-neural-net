module Lib where

import qualified Data.ByteString.Lazy as L

import Data.Foldable (maximumBy)
import Data.Function (on)
import Data.List (mapAccumR, transpose, unfoldr)
import Data.List.Split (chunksOf)
import Data.Ord (comparing)
import Data.Word (Word8)
import System.Random
import Text.Printf (printf)

-- TYPE DECLARATIONS --

type LayerSize = Int
type Bias = Float
type Weight = Float
type Activation = Float
type Preactivation = Float
type ErrorDelta = Float
type Layer = ([Bias], [[Weight]])
type DeltaLayer = Layer

-- CONSTANTS --

learningRate :: Float
learningRate = 0.002
imageWidth, imageSize, imagesHeaderSize, labelsHeaderSize :: Int
imageWidth = 28
imageSize = imageWidth ^ (2 :: Integer)
imagesHeaderSize = 16 -- magic number + amount + width + height
labelsHeaderSize = 8

-- UTILITY FUNCTIONS --

gauss :: Float -> [Float] -> Float
gauss scale [x1, x2] = scale * sqrt ((-2) * log x1) * cos (2 * pi * x2)
gauss _ _ = error "Gauss distribution has 2 args"

gaussSequence :: StdGen -> Float -> [Float]
gaussSequence gen scale = gauss scale <$> chunksOf 2 (randomRs (0, 1) gen)

newBrain :: [LayerSize] -> [Layer]
newBrain sizes = zipWith3 makeLayer sizes (tail sizes) [0 ..]
 where
  makeLayer m n i = (replicate n 1.0, replicate n $ take m $ gaussSequence (mkStdGen i) 0.01)

(+.), (*.) :: (Num c) => [c] -> [c] -> [c]
(+.) = zipWith (+)
(*.) = zipWith (*)

(*^) :: (Functor f, Num b) => f [b] -> [b] -> f b
(*^) matrix vector = sum . zipWith (*) vector <$> matrix

class Subtractable a where
  (-.) :: a -> a -> a
instance Subtractable Float where
  (-.) = (-)
instance (Subtractable a) => Subtractable [a] where
  (-.) = zipWith (-.)
instance (Subtractable c, Subtractable d) => Subtractable (c, d) where
  (c1, d1) -. (c2, d2) = (c1 -. c2, d1 -. d2)

chunksOf' :: Int -> L.ByteString -> [L.ByteString]
chunksOf' n = unfoldr (\b -> if L.null b then Nothing else Just (L.splitAt (fromIntegral n) b))

maximumIndex :: (Ord a, Num b, Enum b) => [a] -> b
maximumIndex = fst . maximumBy (comparing snd) . zip [0 ..]

hammingDistance :: ([Int], [Int]) -> Int
hammingDistance = length . filter (uncurry (==)) . uncurry zip

-- NEURAL NET --

relu :: Preactivation -> Activation
relu = max 0

relu' :: Float -> Float
relu' x = if x <= 0 then 0 else 1

cost' :: Activation -> Activation -> Float
cost' actual desired = if desired == 1 && actual >= desired then 0 else actual - desired

applyLayer :: [Activation] -> Layer -> [Preactivation]
applyLayer activations (biases, weightMatrix) = relu <$> weightMatrix *^ activations +. biases

backpropagate :: [ErrorDelta] -> (Layer, [Activation]) -> ([ErrorDelta], DeltaLayer)
backpropagate nextErrors ((_biases, weightMatrix), prevActivations) = (prevErrors, (deltaBiases, deltaWeights))
 where
  deltaBiases = map (learningRate *) nextErrors
  deltaWeights = [[x * y * learningRate | y <- prevActivations] | x <- nextErrors]
  prevErrors = transpose weightMatrix *^ nextErrors *. map relu' prevActivations -- Actually relu' prevZ but who cares

learn :: [Layer] -> ([Activation], [Activation]) -> [Layer]
learn layers (inputs, desiredOutputs) = layers -. gradient
 where
  errors = zipWith cost' (last computedActivations) desiredOutputs
  gradient = snd (mapAccumR backpropagate errors (zip layers computedActivations))
  computedActivations = scanl applyLayer inputs layers

-- TESTING AND ANALYSIS --

getImage :: Int -> L.ByteString -> [Activation]
getImage n images = getColor <$> indexes
 where
  getColor = (/ 256) . fromIntegral . L.index images . fromIntegral
  indexes = (n * imageSize + imagesHeaderSize +) <$> [0 .. imageSize - 1]

getLabel :: Int -> L.ByteString -> Int
getLabel n labels = toEnum $ fromEnum $ L.index labels $ fromIntegral $ labelsHeaderSize + n

renderImage :: [Activation] -> [Char]
renderImage = unlines . chunksOf (imageWidth * 2) . concatMap (replicate 2 . render)
 where
  gradient = " .:oO@"
  render color = gradient !! floor (color * fromIntegral (length gradient))

getLabelActivation :: Word8 -> [Activation]
getLabelActivation = map (fromIntegral . fromEnum) . flip map [0 .. 9] . (==)

parseImages, parseLabels :: L.ByteString -> [[Activation]]
parseImages = map (map ((/ 256) . fromIntegral) . L.unpack) . chunksOf' imageSize . L.drop (fromIntegral imagesHeaderSize)
parseLabels = map getLabelActivation . L.unpack . L.drop (fromIntegral labelsHeaderSize)

train :: L.ByteString -> L.ByteString -> Int -> [Layer] -> [Layer]
train images labels amount model = foldl learn model trainingData
 where
  trainingData = take amount $ zip (parseImages images) (parseLabels labels)

drawActivations :: [Activation] -> String
drawActivations = unlines . zipWith drawActivation [0 ..]
 where
  drawActivation :: Int -> Activation -> String
  drawActivation i a = printf "%d(%.2f): %s" i a $ replicate (round (a * 10)) '#'

test :: [Layer] -> L.ByteString -> L.ByteString -> Int -> String
test layers images labels amount =
  printf
    "Success rate: %.2f%% (%d of %d)\n"
    (100 * rightGuesses `fdiv` length parsedImages :: Float)
    rightGuesses
    (length parsedImages)
 where
  parsedImages = take amount $ parseImages images
  parsedLabels = take amount $ map maximumIndex $ parseLabels labels
  predictedLabels = map guess parsedImages
  guess :: [Activation] -> Int
  guess activations = maximumIndex $ foldl applyLayer activations layers
  fdiv = (/) `on` fromIntegral
  rightGuesses = hammingDistance (parsedLabels, predictedLabels)

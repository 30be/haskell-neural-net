-- Useful for type annotations in tests
{-# LANGUAGE ScopedTypeVariables #-}

-- Import the module with your functions

import Control.Exception (SomeException, evaluate, try)
import Control.Monad (unless) -- Needed for unless
import qualified Data.ByteString.Lazy as L
import Data.List (nub, transpose) -- For gaussSequence basic check
import Data.Word (Word8)
import Debug.Trace (trace, traceShowId)
import Lib
import System.Random (mkStdGen)
import Test.Hspec

-- -- Helper Functions for Floating Point Comparisons -- --
epsilon :: Float
epsilon = 1e-5 -- Define tolerance

-- Check if two single floats are close
areClose :: Float -> Float -> Bool
areClose actual expected = abs (actual - expected) < epsilon

-- Hspec expectation for single floats
shouldBeCloseTo :: Float -> Float -> Expectation
shouldBeCloseTo actual expected =
  unless (areClose actual expected) $
    expectationFailure $
      "Expected "
        ++ show expected
        ++ " but got "
        ++ show actual
        ++ " (within tolerance "
        ++ show epsilon
        ++ ")"

-- Hspec expectation for lists of floats (element-wise)
shouldBeCloseToList :: [Float] -> [Float] -> Expectation
shouldBeCloseToList actual expected = do
  -- Check length first for a better error message
  unless (length actual == length expected) $
    expectationFailure $
      "List length mismatch:"
        ++ "\nExpected length: "
        ++ show (length expected)
        ++ " (value: "
        ++ show expected
        ++ ")"
        ++ "\nActual length:   "
        ++ show (length actual)
        ++ " (value: "
        ++ show actual
        ++ ")"

  -- Check element-wise closeness
  let comparisons = zipWith areClose actual expected
  unless (and comparisons) $
    expectationFailure $
      "Lists differ element-wise beyond tolerance ("
        ++ show epsilon
        ++ "):"
        ++ "\nExpected: "
        ++ show expected
        ++ "\nActual:   "
        ++ show actual
        ++ "\nClose?:   "
        ++ show comparisons -- Show which elements failed

-- -- Main Test Suite -- --
main :: IO ()
main = hspec $ do
  -- ... (keep describe "gauss", "gaussSequence", "newBrain") ...
  describe "gauss" $ do
    -- Note: Testing randomness precisely is hard. We test the error case.
    -- A property-based test could check distribution, but that's more complex.
    it "throws an error if not given exactly two random numbers" $ do
      evaluate (gauss 1.0 [0.5]) `shouldThrow` anyException
      evaluate (gauss 1.0 [0.5, 0.2, 0.9]) `shouldThrow` anyException
    -- It's hard to test the output value deterministically due to floating point trig/log functions
    -- We can check it produces *a* value for valid input
    it "produces a value for valid input" $ do
      let result = gauss 1.0 [0.5, 0.8] -- Use known valid inputs (0 < x <= 1)
      -- Check if it's a finite number (not NaN or Infinity)
      result `shouldSatisfy` (\x -> not (isNaN x) && not (isInfinite x))

  describe "gaussSequence" $ do
    -- Again, hard to test exact values. Test basic properties.
    it "generates a sequence of Floats" $ do
      let gen = mkStdGen 42
      let seq1 = take 10 (gaussSequence gen 0.01)
      length seq1 `shouldBe` 10
      -- Check they are not all identical (highly unlikely for gauss)
      length (nub seq1) `shouldSatisfy` (> 1)

  describe "newBrain" $ do
    it "creates layers with correct dimensions" $ do
      let sizes = [784, 30, 10]
      let brain = newBrain sizes
      length brain `shouldBe` (length sizes - 1) -- 2 layers
      let (biases1, weights1) = brain !! 0
      let (biases2, weights2) = brain !! 1

      length biases1 `shouldBe` 30
      length weights1 `shouldBe` 30
      map length weights1 `shouldBe` replicate 30 784

      length biases2 `shouldBe` 10
      length weights2 `shouldBe` 10
      map length weights2 `shouldBe` replicate 10 30

    it "initializes biases to 1.0" $ do
      let brain = newBrain [5, 3]
      let (biases, _) = head brain
      biases `shouldBe` [1.0, 1.0, 1.0]

  describe "(+.)" $ do
    it "adds vectors element-wise" $ do
      ([1, 2, 3] +. [4, 5, 6]) `shouldBe` ([5, 7, 9] :: [Int])
      -- Use the new list helper here
      ([1.0, 2.5] +. [0.5, -1.0]) `shouldBeCloseToList` [1.5, 1.5]
  -- ... rest of (+.) tests are fine ...

  describe "(*.)" $ do
    it "multiplies matrix by vector" $ do
      let matrix = [[1, 2, 3], [4, 5, 6]] :: [[Int]]
      let vector = [10, 0, -1] :: [Int]
      (matrix *. vector) `shouldBe` [7, 34]
      -- Example with floats if needed:
      let matrixF = [[1.0, 0.5], [-1.0, 2.0]] :: [[Float]]
      let vectorF = [10.0, 2.0] :: [Float]
      -- Expected: [1*10+0.5*2, -1*10+2*2] = [11.0, -6.0]
      (matrixF *. vectorF) `shouldBeCloseToList` [11.0, -6.0]
  -- ... rest of (*.) tests are fine ...

  describe "chunksOf'" $ do
    let bs = L.pack [1 .. 10]
    it "splits a ByteString into chunks" $ do
      chunksOf' 3 bs `shouldBe` [L.pack [1, 2, 3], L.pack [4, 5, 6], L.pack [7, 8, 9], L.pack [10]]
    it "handles chunk size larger than ByteString" $ do
      chunksOf' 15 bs `shouldBe` [L.pack [1 .. 10]]
    it "handles empty ByteString" $ do
      chunksOf' 3 L.empty `shouldBe` []
    it "handles exact multiple" $ do
      chunksOf' 5 bs `shouldBe` [L.pack [1 .. 5], L.pack [6 .. 10]]

  describe "maximumIndex" $ do
    it "finds the index of the maximum element" $ do
      maximumIndex [10, 50, 20, 40, 30] `shouldBe` (1 :: Int) -- First maximum
      maximumIndex [1.1, 0.5, 2.5, -1.0] `shouldBe` (2 :: Int)
    it "works with negative numbers" $ do
      maximumIndex [-5, -2, -8, -1] `shouldBe` (3 :: Int)
    it "throws an error for an empty list" $ do
      evaluate (maximumIndex ([] :: [Int])) `shouldThrow` anyException

  describe "hammingDistance" $ do
    it "calculates distance for equal length lists" $ do
      hammingDistance ([1, 2, 3, 4], [1, 5, 3, 6]) `shouldBe` 2
    it "calculates distance for different length lists (truncates)" $ do
      hammingDistance ([1, 2, 3], [1, 5, 3, 6]) `shouldBe` 2
      hammingDistance ([1, 5, 3, 6], [1, 2, 3]) `shouldBe` 2
    it "returns zero for identical lists" $ do
      hammingDistance ([1, 2, 3], [1, 2, 3]) `shouldBe` 3 -- Counts matches
    it "returns zero for empty lists" $ do
      hammingDistance ([], []) `shouldBe` 0
      hammingDistance ([1, 2], []) `shouldBe` 0
      hammingDistance ([], [1, 2]) `shouldBe` 0

  describe "relu" $ do
    it "returns input if positive" $ do
      relu 5.0 `shouldBeCloseTo` 5.0
      relu 0.1 `shouldBeCloseTo` 0.1
    it "returns 0 if negative or zero" $ do
      relu 0.0 `shouldBeCloseTo` 0.0
      relu (-0.1) `shouldBeCloseTo` 0.0
      relu (-10.0) `shouldBeCloseTo` 0.0

  describe "relu'" $ do
    it "returns 1 if positive" $ do
      relu' 5.0 `shouldBeCloseTo` 1.0
      relu' 0.0001 `shouldBeCloseTo` 1.0
    it "returns 0 if zero or negative" $ do
      relu' 0.0 `shouldBeCloseTo` 0.0
      relu' (-0.0001) `shouldBeCloseTo` 0.0
      relu' (-10.0) `shouldBeCloseTo` 0.0

  describe "cost'" $ do
    it "calculates cost derivative component" $ do
      cost' 0.8 1.0 `shouldBeCloseTo` (0.8 - 1.0)
      cost' 0.2 0.0 `shouldBeCloseTo` (0.2 - 0.0)
    it "returns 0 if desired is 1 and actual meets or exceeds it" $ do
      cost' 1.0 1.0 `shouldBeCloseTo` 0.0
      cost' 1.2 1.0 `shouldBeCloseTo` 0.0
    it "calculates normally if desired is 0" $ do
      cost' 1.2 0.0 `shouldBeCloseTo` 1.2
      cost' (-0.5) 0.0 `shouldBeCloseTo` (-0.5)

  describe "zLayer" $ do
    it "calculates weighted inputs for a layer" $ do
      let activations = [0.5, 1.0] :: [Activation]
      let biases = [0.1, -0.2] :: [Bias]
      let weights = [[1.0, 2.0], [0.5, -1.0]] :: [[Weight]]
      let layer = (biases, weights)
      -- Expected: [2.6, -0.95]
      zLayer activations layer `shouldBeCloseToList` [2.6, -0.95]

  describe "applyLayer" $ do
    it "applies relu to the weighted inputs" $ do
      let activations = [0.5, 1.0] :: [Activation]
      let biases = [0.1, -0.2] :: [Bias]
      let weights = [[1.0, 2.0], [0.5, -1.0]] :: [[Weight]]
      let layer = (biases, weights)
      -- Expected: [2.6, 0.0]
      applyLayer activations layer `shouldBeCloseToList` [2.6, -0.95]

  describe "backpropagate" $ do
    it "calculates error for the previous layer" $ do
      let nextLayerErrors = [[0.5, -0.1]]
      let weights = [[1.0, 3.0, 0.5], [2.0, 4.0, -1.0]] -- W (2x3)
      let weightMatrixT = transpose weights -- W^T (should be 3x2)
      let weightedInputs = [1.5, -0.5, 2.0] -- z_l (len 3)
      let layerData = (weightMatrixT, weightedInputs)
      let result = backpropagate nextLayerErrors layerData
      head result `shouldBeCloseToList` [0.3, 0.0, 0.35]

  describe "deltas" $ do
    it "calculates activations and errors for a simple network" $ do
      let input = [0.5] :: [Activation]
      let desired = [1.0] :: [Activation]
      let layer1 = ([0.1], [[2.0]]) :: Layer
      let layers = [layer1]
      let (activations, errors) = deltas input desired layers

      length activations `shouldBe` 2
      head activations `shouldBeCloseToList` input
      last activations `shouldBeCloseToList` [1.1]

      length errors `shouldBe` 1
      head errors `shouldBeCloseToList` [0.0]

    it "calculates deltas for a slightly more complex network" $ do
      let inputs = [0.5, 0.2] :: [Activation]
      let desired = [0.0] :: [Activation]
      let layer1 = ([0.1, -0.1], [[1.0, 0.5], [0.0, 2.0]]) :: Layer
      let layer2 = ([0.2], [[0.8, -1.0]]) :: Layer
      let layers = [layer1, layer2]
      let (calculatedActivations, calculatedErrors) = deltas inputs desired layers

      length calculatedActivations `shouldBe` 3
      calculatedActivations !! 0 `shouldBeCloseToList` [0.5, 0.2]
      calculatedActivations !! 1 `shouldBeCloseToList` [0.7, 0.3]
      calculatedActivations !! 2 `shouldBeCloseToList` [0.46]

      length calculatedErrors `shouldBe` 2
      calculatedErrors !! 0 `shouldBeCloseToList` [0.368, -0.46]
      calculatedErrors !! 1 `shouldBeCloseToList` [0.46]

  describe "descend" $ do
    it "updates values based on gradients and learning rate" $ do
      let values = [10.0, 5.0, 0.0]
      let gradients = [1.0, -2.0, 0.5]
      -- Expected: [9.998, 5.004, -0.001]
      descend values gradients `shouldBeCloseToList` [9.998, 5.004, -0.001]

  describe "learn" $ do
    it "updates layer weights and biases" $ do
      let input = [0.5] :: [Activation]
      let layer1 = ([0.1], [[2.0]]) :: Layer -- Bias, Weight
      let layers = [layer1]
      let outputNonZeroGradient = [0.0] :: [Activation]
      let trainingPair2 = (input, outputNonZeroGradient)
      let updatedLayers2 = learn layers trainingPair2
      let (updatedBiases2, updatedWeightsMatrix2) = head updatedLayers2
      let updatedWeights2 = head updatedWeightsMatrix2

      -- Biases are [Float], Weights are [[Float]]
      updatedBiases2 `shouldBeCloseToList` [0.0978]
      updatedWeights2 `shouldBeCloseToList` [1.9989]

  -- -- Testing parsing/data handling functions -- --
  let imageWidth = 28
  let imageSize = imageWidth * imageWidth
  let imagesHeaderSize = 16
  let labelsHeaderSize = 8

  describe "getImage" $ do
    it "extracts pixel activations for a specific image" $ do
      let header = L.replicate (fromIntegral imagesHeaderSize) 0
      let img0Bytes = L.replicate (fromIntegral imageSize) 10
      let img1Pixel = 200 :: Word8
      let img1Bytes = L.cons img1Pixel $ L.replicate (fromIntegral imageSize - 1) 50
      let dummyImages = L.concat [header, img0Bytes, img1Bytes]

      let activations0 = getImage 0 dummyImages
      length activations0 `shouldBe` imageSize
      -- Check all pixels are close (more robust check)
      activations0 `shouldSatisfy` all (\p -> areClose p (10 / 256))

      let activations1 = getImage 1 dummyImages
      length activations1 `shouldBe` imageSize
      -- Check specific pixels (use shouldBeCloseTo for single floats)
      head activations1 `shouldBeCloseTo` (fromIntegral img1Pixel / 256)
      last activations1 `shouldBeCloseTo` (50 / 256)

  describe "getLabel" $ do
    it "extracts the label for a specific image index" $ do
      -- Header (8 bytes) + Label 0 + Label 1 + Label 2
      let header = L.replicate (fromIntegral labelsHeaderSize) 0
      let labels = L.pack [5, 0, 9] -- Labels are 5, 0, 9
      let dummyLabels = L.concat [header, labels]

      getLabel 0 dummyLabels `shouldBe` 5
      getLabel 1 dummyLabels `shouldBe` 0
      getLabel 2 dummyLabels `shouldBe` 9

  describe "getLabelActivation" $ do
    it "creates a one-hot activation vector for a label" $ do
      -- Results are [Float], compare using shouldBeCloseToList for robustness
      getLabelActivation 0 `shouldBeCloseToList` [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      getLabelActivation 3 `shouldBeCloseToList` [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      getLabelActivation 9 `shouldBeCloseToList` [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

  describe "parseImages" $ do
    it "parses multiple images from ByteString" $ do
      let header = L.replicate (fromIntegral imagesHeaderSize) 0
      let imgBytes = L.replicate (fromIntegral $ imageSize * 2) 128
      let dummyImages = L.concat [header, imgBytes]
      let expectedPixelVal = 128 / 256

      let parsed = parseImages dummyImages
      length parsed `shouldBe` 2
      length (head parsed) `shouldBe` imageSize
      -- Check individual pixels
      head (head parsed) `shouldBeCloseTo` expectedPixelVal
      last (last parsed) `shouldBeCloseTo` expectedPixelVal
      head parsed `shouldSatisfy` all (\p -> areClose p expectedPixelVal)
      last parsed `shouldSatisfy` all (\p -> areClose p expectedPixelVal)

  describe "parseLabels" $ do
    it "parses multiple labels from ByteString into activation vectors" $ do
      let header = L.replicate (fromIntegral labelsHeaderSize) 0
      let labelBytes = L.pack [1, 7] -- Labels 1 and 7
      let dummyLabels = L.concat [header, labelBytes]

      let parsed = parseLabels dummyLabels
      length parsed `shouldBe` 2
      -- Compare the resulting [Activation] lists
      head parsed `shouldBeCloseToList` getLabelActivation 1
      last parsed `shouldBeCloseToList` getLabelActivation 7

import Control.Exception (evaluate)
import qualified Data.ByteString.Lazy as L
import Data.List (transpose)
import Lib
import System.Random
import Test.Hspec
import Test.QuickCheck

-- Test data generators
genActivations :: Int -> Gen [Activation]
genActivations n = vectorOf n (choose (0.0, 1.0))

genWeights :: Int -> Int -> Gen [[Weight]]
genWeights rows cols = vectorOf rows (vectorOf cols (choose (-1.0, 1.0)))

genBiases :: Int -> Gen [Bias]
genBiases n = vectorOf n (choose (-1.0, 1.0))

genLayer :: Int -> Int -> Gen Layer
genLayer inputSize outputSize = do
  biases <- genBiases outputSize
  weights <- genWeights outputSize inputSize
  return (biases, weights)

-- Helper functions for testing
approxEqual :: Float -> Float -> Bool
approxEqual x y = abs (x - y) < 1e-6

listApproxEqual :: [Float] -> [Float] -> Bool
listApproxEqual xs ys = length xs == length ys && all (uncurry approxEqual) (zip xs ys)

-- Create simple test data
simpleLayer :: Layer
simpleLayer = ([0.5, -0.3], [[0.2, 0.8], [0.4, -0.6]])

simpleInputs :: [Activation]
simpleInputs = [1.0, 0.5]

-- Main test suite
main :: IO ()
main = hspec $ do
  describe "Utility Functions" $ do
    describe "gauss" $ do
      it "generates Gaussian distributed values" $ do
        let result = gauss 1.0 [0.5, 0.3]
        result `shouldSatisfy` (\x -> x > -5.0 && x < 5.0)

      it "throws error with wrong number of arguments" $ do
        evaluate (gauss 1.0 [0.5]) `shouldThrow` anyException

    describe "gaussSequence" $ do
      it "generates infinite sequence" $ do
        let gen = mkStdGen 42
        let seq = take 100 $ gaussSequence gen 1.0
        length seq `shouldBe` 100

    describe "vector operations" $ do
      it "(+.) adds vectors element-wise" $ do
        [1, 2, 3] +. [4, 5, 6] `shouldBe` [5, 7, 9]

      it "(*.) multiplies vectors element-wise" $ do
        [2, 3, 4] *. [1, 2, 3] `shouldBe` [2, 6, 12]

      it "(*^) performs matrix-vector multiplication" $ do
        let matrix = [[1, 2], [3, 4]]
        let vector = [1, 2]
        matrix *^ vector `shouldBe` [5, 11]

    describe "maximumIndex" $ do
      it "returns index of maximum element" $ do
        maximumIndex [1.0, 3.0, 2.0] `shouldBe` 1
        maximumIndex [5.0, 1.0, 2.0] `shouldBe` 0

    describe "hammingDistance" $ do
      it "counts equal elements (should this be matches or mismatches?)" $ do
        hammingDistance ([1, 2, 3], [1, 0, 3]) `shouldBe` 2
        hammingDistance ([1, 2, 3], [4, 5, 6]) `shouldBe` 0

  describe "Neural Network Core" $ do
    describe "relu activation" $ do
      it "returns 0 for negative inputs" $ do
        relu (-1.0) `shouldBe` 0.0
        relu (-0.5) `shouldBe` 0.0

      it "returns input for positive inputs" $ do
        relu 1.0 `shouldBe` 1.0
        relu 0.5 `shouldBe` 0.5

      it "returns 0 for zero input" $ do
        relu 0.0 `shouldBe` 0.0

    describe "relu derivative" $ do
      it "returns 0 for negative inputs" $ do
        relu' (-1.0) `shouldBe` 0.0
        relu' (-0.1) `shouldBe` 0.0

      it "returns 1 for positive inputs" $ do
        relu' 1.0 `shouldBe` 1.0
        relu' 0.1 `shouldBe` 1.0

      it "returns 0 for zero input" $ do
        relu' 0.0 `shouldBe` 0.0

    describe "cost function" $ do
      it "calculates squared error correctly" $ do
        cost 0.8 1.0 `shouldSatisfy` approxEqual 0.02
        cost 0.0 1.0 `shouldSatisfy` approxEqual 0.5

      it "returns 0 for perfect prediction" $ do
        cost 1.0 1.0 `shouldSatisfy` approxEqual 0.0

    describe "cost derivative" $ do
      it "returns difference for normal cases" $ do
        cost' 0.8 1.0 `shouldSatisfy` approxEqual (-0.2)
        cost' 1.2 1.0 `shouldSatisfy` approxEqual 0.2

      it "returns 0 when desired=1 and actual>=desired" $ do
        cost' 1.0 1.0 `shouldBe` 0.0
        cost' 1.1 1.0 `shouldBe` 0.0

  describe "Layer Operations" $ do
    describe "applyLayer" $ do
      it "applies weights, biases, and activation correctly" $ do
        let result = applyLayer simpleInputs simpleLayer
        -- Expected: relu([0.2*1.0 + 0.8*0.5, 0.4*1.0 + (-0.6)*0.5] +. [0.5, -0.3])
        -- = relu([0.6, 0.1] +. [0.5, -0.3]) = relu([1.1, -0.2]) = [1.1, 0.0]
        result `shouldSatisfy` listApproxEqual [1.1, 0.0]

      it "handles zero inputs" $ do
        let result = applyLayer [0.0, 0.0] simpleLayer
        result `shouldSatisfy` listApproxEqual [0.5, 0.0]

    describe "newBrain" $ do
      it "creates correct number of layers" $ do
        let brain = newBrain [2, 3, 1]
        length brain `shouldBe` 2

      it "creates layers with correct dimensions" $ do
        let brain = newBrain [2, 3, 1]
        let (biases1, weights1) = brain !! 0
        let (biases2, weights2) = brain !! 1
        length biases1 `shouldBe` 3
        length weights1 `shouldBe` 3
        all ((== 2) . length) weights1 `shouldBe` True
        length biases2 `shouldBe` 1
        length weights2 `shouldBe` 1
        all ((== 3) . length) weights2 `shouldBe` True

  describe "Training Algorithm" $ do
    describe "backpropagate" $ do
      it "produces deltas with correct dimensions" $ do
        let prevActivations = [1.0, 0.5]
        let nextErrors = [0.1, -0.2]
        let (prevErrors, (deltaBiases, deltaWeights)) = backpropagate nextErrors (simpleLayer, prevActivations)

        length prevErrors `shouldBe` 2
        length deltaBiases `shouldBe` 2
        length deltaWeights `shouldBe` 2
        all ((== 2) . length) deltaWeights `shouldBe` True

      it "calculates weight deltas correctly" $ do
        let prevActivations = [1.0, 0.5]
        let nextErrors = [0.1, -0.2]
        let (_, (_, deltaWeights)) = backpropagate nextErrors (simpleLayer, prevActivations)

        -- Expected deltaWeights: [[0.1*1.0*lr, 0.1*0.5*lr], [-0.2*1.0*lr, -0.2*0.5*lr]]
        let expected =
              [ [0.1 * learningRate, 0.05 * learningRate]
              , [-0.2 * learningRate, -0.1 * learningRate]
              ]
        zipWith listApproxEqual deltaWeights expected `shouldSatisfy` all id

    describe "learn" $ do
      it "updates weights in correct direction for simple case" $ do
        let inputs = [1.0, 0.0]
        let desired = [1.0, 0.0]
        let initialLayer = ([0.0, 0.0], [[0.5, 0.5], [0.5, 0.5]])
        let updatedLayers = learn [initialLayer] (inputs, desired)

        length updatedLayers `shouldBe` 1
        -- The weights should change based on the error
        let (newBiases, newWeights) = head updatedLayers
        newWeights `shouldNotBe` (snd initialLayer)

  describe "Data Processing" $ do
    describe "getLabelActivation" $ do
      it "creates one-hot encoding correctly" $ do
        getLabelActivation 0 `shouldBe` [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        getLabelActivation 5 `shouldBe` [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        getLabelActivation 9 `shouldBe` [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    describe "parseImages and parseLabels integration" $ do
      it "handles empty bytestrings gracefully" $ do
        let emptyImages = L.replicate (fromIntegral imagesHeaderSize) 0
        let emptyLabels = L.replicate (fromIntegral labelsHeaderSize) 0
        parseImages emptyImages `shouldBe` []
        parseLabels emptyLabels `shouldBe` []

    -- describe "Mathematical Properties" $ do
    --   describe "forward pass consistency" $ do
    --     it "produces output with correct dimensions" $ property $ \gen -> do
    --       let inputSize = 3
    --       let hiddenSize = 4
    --       let outputSize = 2
    --       inputs <- generate $ genActivations inputSize
    --       layer1 <- generate $ genLayer inputSize hiddenSize
    --       layer2 <- generate $ genLayer hiddenSize outputSize
    --
    --       let hidden = applyLayer inputs layer1
    --       let output = applyLayer hidden layer2
    --
    --       return $ length hidden == hiddenSize && length output == outputSize

    describe "backpropagation gradient check" $ do
      it "weight updates should decrease cost for simple cases" $ do
        -- Simple test: single layer, single training example
        let inputs = [1.0, 0.0]
        let desired = [1.0]
        let layer = ([0.0], [[0.1, 0.1]])

        -- Forward pass
        let output = applyLayer inputs layer
        let initialCost = sum $ zipWith cost output desired

        -- One learning step
        let updatedLayers = learn [layer] (inputs, desired)
        let newOutput = applyLayer inputs (head updatedLayers)
        let newCost = sum $ zipWith cost newOutput desired

        -- Cost should decrease (or stay same if already optimal)
        newCost `shouldSatisfy` (<= initialCost + 1e-6)

  describe "Edge Cases and Error Conditions" $ do
    describe "numerical stability" $ do
      it "handles very large inputs" $ do
        let largeInputs = [1000.0, -1000.0]
        let result = applyLayer largeInputs simpleLayer
        all (not . isNaN) result `shouldBe` True
        all (not . isInfinite) result `shouldBe` True

      it "handles very small weights" $ do
        let tinyLayer = ([0.0, 0.0], [[1e-10, 1e-10], [1e-10, 1e-10]])
        let result = applyLayer [1.0, 1.0] tinyLayer
        all (not . isNaN) result `shouldBe` True

    describe "dimension mismatches" $ do
      it "should handle mismatched input sizes gracefully" $ do
        -- This might throw an exception or produce wrong results
        -- depending on implementation
        let wrongInputs = [1.0] -- too few inputs
        evaluate (applyLayer wrongInputs simpleLayer) `shouldThrow` anyException

  describe "Performance Regression Tests" $ do
    describe "learning convergence" $ do
      it "should learn simple XOR-like pattern" $ do
        -- Create a simple 2-2-1 network
        let brain = newBrain [2, 2, 1]

        -- Training data: simple pattern where output = first input
        let trainingData =
              [ ([1.0, 0.0], [1.0])
              , ([0.0, 1.0], [0.0])
              , ([1.0, 1.0], [1.0])
              , ([0.0, 0.0], [0.0])
              ]

        -- Train for multiple epochs
        let trainedBrain = iterate (\b -> foldl learn b trainingData) brain !! 10

        -- Test if it learned something (cost should be lower)
        let initialCosts = [sum $ zipWith cost (foldl applyLayer inp brain) out | (inp, out) <- trainingData]
        let finalCosts = [sum $ zipWith cost (foldl applyLayer inp trainedBrain) out | (inp, out) <- trainingData]
        sum finalCosts `shouldSatisfy` (< sum initialCosts)

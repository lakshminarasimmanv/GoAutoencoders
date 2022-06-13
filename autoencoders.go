// Autoencoders using Go.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

const (
	// Number of neurons in the input layer.
	inputNeurons = 2
	// Number of neurons in the hidden layer.
	hiddenNeurons = 3
	// Number of neurons in the output layer.
	outputNeurons = inputNeurons
	// Number of training iterations.
	iterations = 10000
	// Learning rate.
	learningRate = 0.3
	// Momentum.
	momentum = 0.6
)

// Sigmoid function.
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivative of the sigmoid function.
func dsigmoid(y float64) float64 {
	return y * (1.0 - y)
}

// Neural network.
type NeuralNetwork struct {
	inputLayer  [inputNeurons]float64
	hiddenLayer [hiddenNeurons]float64
	outputLayer [outputNeurons]float64
	weights1    [inputNeurons][hiddenNeurons]float64
	weights2    [hiddenNeurons][outputNeurons]float64
	bias1       [hiddenNeurons]float64
	bias2       [outputNeurons]float64
	change1     [inputNeurons][hiddenNeurons]float64
	change2     [hiddenNeurons][outputNeurons]float64
}

// Initialize the neural network.
func (nn *NeuralNetwork) Init() {
	rand.Seed(time.Now().UTC().UnixNano())
	for i := 0; i < inputNeurons; i++ {
		for j := 0; j < hiddenNeurons; j++ {
			nn.weights1[i][j] = rand.Float64()
			nn.change1[i][j] = 0.0
		}
	}
	for i := 0; i < hiddenNeurons; i++ {
		for j := 0; j < outputNeurons; j++ {
			nn.weights2[i][j] = rand.Float64()
			nn.change2[i][j] = 0.0
		}
	}
	for i := 0; i < hiddenNeurons; i++ {
		nn.bias1[i] = rand.Float64()
	}
	for i := 0; i < outputNeurons; i++ {
		nn.bias2[i] = rand.Float64()
	}
}

// Feed forward.
func (nn *NeuralNetwork) FeedForward(inputs [inputNeurons]float64) [outputNeurons]float64 {
	for i := 0; i < inputNeurons; i++ {
		nn.inputLayer[i] = inputs[i]
	}
	for i := 0; i < hiddenNeurons; i++ {
		var sum float64
		for j := 0; j < inputNeurons; j++ {
			sum += nn.inputLayer[j] * nn.weights1[j][i]
		}
		sum += nn.bias1[i]
		nn.hiddenLayer[i] = sigmoid(sum)
	}
	for i := 0; i < outputNeurons; i++ {
		var sum float64
		for j := 0; j < hiddenNeurons; j++ {
			sum += nn.hiddenLayer[j] * nn.weights2[j][i]
		}
		sum += nn.bias2[i]
		nn.outputLayer[i] = sigmoid(sum)
	}
	return nn.outputLayer
}

// Back propagation.
func (nn *NeuralNetwork) BackPropagation(targets [outputNeurons]float64) {
	// Calculate the output layer error.
	var outputError [outputNeurons]float64
	for i := 0; i < outputNeurons; i++ {
		outputError[i] = targets[i] - nn.outputLayer[i]
	}
	// Calculate the hidden layer error.
	var hiddenError [hiddenNeurons]float64
	for i := 0; i < hiddenNeurons; i++ {
		var sum float64
		for j := 0; j < outputNeurons; j++ {
			sum += outputError[j] * nn.weights2[i][j]
		}
		hiddenError[i] = sum
	}
	// Update the weights of the output layer.
	for i := 0; i < hiddenNeurons; i++ {
		for j := 0; j < outputNeurons; j++ {
			change := outputError[j] * nn.hiddenLayer[i]
			nn.weights2[i][j] += learningRate*change + momentum*nn.change2[i][j]
			nn.change2[i][j] = change
		}
	}
	// Update the weights of the hidden layer.
	for i := 0; i < inputNeurons; i++ {
		for j := 0; j < hiddenNeurons; j++ {
			change := hiddenError[j] * nn.inputLayer[i]
			nn.weights1[i][j] += learningRate*change + momentum*nn.change1[i][j]
			nn.change1[i][j] = change
		}
	}
	// Update the bias of the output layer.
	for i := 0; i < outputNeurons; i++ {
		nn.bias2[i] += learningRate * outputError[i]
	}
	// Update the bias of the hidden layer.
	for i := 0; i < hiddenNeurons; i++ {
		nn.bias1[i] += learningRate * hiddenError[i]
	}
}

// Train the neural network.
func (nn *NeuralNetwork) Train(inputs [][inputNeurons]float64, targets [][outputNeurons]float64) {
	for i := 0; i < iterations; i++ {
		for j := 0; j < len(inputs); j++ {
			nn.FeedForward(inputs[j])
			nn.BackPropagation(targets[j])
		}
	}
}

// Test the neural network.
func (nn *NeuralNetwork) Test(inputs [][inputNeurons]float64, targets [][outputNeurons]float64) {
	for i := 0; i < len(inputs); i++ {
		fmt.Println(nn.FeedForward(inputs[i]), "->", targets[i])
	}
}

func main() {
	var nn NeuralNetwork
	nn.Init()
	inputs := [][inputNeurons]float64{
		{0.0, 0.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{1.0, 1.0},
	}
	targets := [][outputNeurons]float64{
		{0.0, 0.0},
		{1.0, 1.0},
		{1.0, 1.0},
		{0.0, 0.0},
	}
	nn.Train(inputs, targets)
	nn.Test(inputs, targets)
}

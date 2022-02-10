package qnetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class NeuralNetwork {

	ArrayList<Neuron>[] layers;
	ArrayList<Neuron> allNeurons = new ArrayList<Neuron>();
	
	float[] answer;
	
	
	public NeuralNetwork(int numberOfHiddenLayers, int[] numberOfNeurons, int hiddenLayerActivationFunc,
							int outputLayerActivationFunc, boolean useMiniBatches)
	{
		
		layers = new ArrayList[numberOfHiddenLayers+2];
		
		for (int i = 0; i < numberOfHiddenLayers+2; i++)
		{
			layers[i] = new ArrayList<Neuron>();
			int id = 0;
			
			for (int k = 0; k < numberOfNeurons[i]; k++)
			{
				int input = i == 0 ? 1 : numberOfNeurons[i-1]; //Set input length equal to previous layer. Set to 1 if input layer
				int output = i == numberOfHiddenLayers+1 ? 1 : numberOfNeurons[i+1]; //Set output length to next layer. Set 1 if output
				int activationFunction;
				
				int type; //The type of neuron. 0 for input, 2 for output and 1 for hidden.
				if (i == 0) { type = 0; activationFunction = 0; }
				else if (i == numberOfHiddenLayers+1) { type = 2; activationFunction = outputLayerActivationFunc; }
				else { type = 1; activationFunction = hiddenLayerActivationFunc; }
				
				Neuron n = new Neuron(id, activationFunction, input, output, type, useMiniBatches);

				if (i != 0)
				{
					for (Neuron neu : layers[i-1]) { neu.addForwardConnection(n); }
				}
				layers[i].add(n);
				allNeurons.add(n);
				id++;
			}
				
		}
		
	}
	
	//Forward pass given an input.
	public float[] forwardPass(float[] input)
	{
		insertInput(input); //Actually pass the input.
		
		for (int i = 0; i < layers.length; i++)
		{
			for (Neuron n : layers[i])
			{
				n.forward();
			}
		}
		
		//Return the output from output layer neurons
		float[] output = new float[layers[layers.length-1].size()];
		for (int i = 0; i < layers[layers.length-1].size(); i++) { output[i] = layers[layers.length-1].get(i).output; }
		return output;
		
	}

	//Back Propagation given the expected answer. Apply Weights immediatly if applyWeights is true, otherwise
	//store them for meanBatch fitting.
	public float backPropagation(float[] answer, boolean applyWeights)
	{
		this.answer = answer;
		float output = 0;
		for (int i = 0; i < answer.length; i++) { output += Math.pow(answer[i] - layers[layers.length-1].get(i).output,2); }
		
		for (int i = layers.length-1; i > 0; i--)
		{
			for (int j = 0; j < layers[i].size(); j++)
			{
				float fixedAnswer = i == layers.length-1 ? answer[j] : 0;
				layers[i].get(j).backpropagate(fixedAnswer);
			}
		}
		
		if (applyWeights) { applyBatchMean(); }
		return output;
	}
	
	//Get an array List with all weights and biases
	public ArrayList<float[]> getWeights()
	{
		ArrayList<float[]> ws = new ArrayList<float[]>();
		for (int i = 1; i < layers.length; i++)
		{
			for (int j = 0; j < layers[i].size(); j++)
			{
				Neuron n = layers[i].get(j);
				float[] weightsAndBias = new float[n.weights.length+1];
				for (int f = 0; f < n.weights.length+1; f++)
				{
					if (f == n.weights.length) { weightsAndBias[f] = n.bias; break; }
					weightsAndBias[f] = n.weights[f];
				}
				ws.add(weightsAndBias);
			}
		}
		return ws;
	}
	
	//Set all weights and biases. Accepts input from getWeights function.
	public void setWeights(ArrayList<float[]> ws)
	{
		int counter = 0;
		for (int i = 1; i < layers.length; i++)
		{
			for (int j = 0; j < layers[i].size(); j++)
			{
				float[] weightsAndBias = ws.get(counter);
				Neuron n = layers[i].get(j);
				for (int f = 0; f < n.weights.length+1; f++)
				{
					if (f == n.weights.length) { n.bias = weightsAndBias[f]; break; }
					n.weights[f] = weightsAndBias[f];
				}
				counter++;
			}
		}
	}
	
	//Sets input at input layer
	public void insertInput(float[] input)
	{
		
		for (int i = 0; i < input.length; i++)
		{
			layers[0].get(i).inputs[0] = input[i];
		}
	}
	
	//Apply our mini-batch's weights
	public void applyBatchMean()
	{
		for (Neuron n : allNeurons)
		{
			n.applyBatchMean();
		}
	}
	
	
	
	
	
	
	
}

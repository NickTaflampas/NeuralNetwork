package qnetwork;

import java.util.ArrayList;
import java.util.Random;

public class Neuron {

	int id; //Unique ID, used for testing and weight matching
	float[] weights; //The weights
	ArrayList<Float>[] batchWeights; //All "weight fixes" from a batch. Used to apply their mean.
	float bias;// Our bias
	
	boolean usesBatches;
	int neuronType; //Numeric type. 0 for input, 1 for hidden, 2 for output
	int activationFunction; //Activation function used. 0: Linear, 1: RELU, 2: TANH, 3: Sigmoid
	ArrayList<Neuron> forwardConnections; //Forward reference of neurons in the next layer
	
	float[] inputs; //Input from previous layer (or normal input)
	float unactivatedOutput; //Output before activation function 
	float output; //Output after activation.
	
	float dCi; //Derivative of Cost Function from Input. Used for back propagation
	float learningStep = 0.01f; //The learning step. SHOULD BE SET IN MAIN CLASS NORMALLY, IGNORE THIS VALUE.
	
	public Neuron(int id, int activation, int inputSize, int outputSize, int type, boolean usesBatches)
	{
		this.id = id;
		this.usesBatches = usesBatches;
		neuronType = type;
		activationFunction = activation;
		forwardConnections = new ArrayList<Neuron>();
		
		weights = new float[inputSize];
		inputs = new float[inputSize];
		
		//Set-Up batch collection
		if (usesBatches) 
		{ 
			batchWeights = new ArrayList[inputSize+1]; 
			for (int i = 0; i < batchWeights.length; i++)
			{
				batchWeights[i] = new ArrayList<Float>();
			}
		}
		//Randomly init bias and weights between -1 and 1.
		Random r = new Random();
		for (int i = 0; i < inputSize; i++)
		{
			weights[i] = r.nextFloat()*2-1;
		}
		bias = r.nextFloat()*2-1;
		
	}
	
	
	//Forward function. Calculates output from input and passes it accordingly to the next neuron
	public void forward()
	{
		//Calculate output from weights and biases, unless we are in the input
		if (neuronType != 0)
		{
			output = 0;
			for (int i = 0; i < weights.length; i++)
			{
				output += inputs[i]*weights[i];
			}
			output += bias;
			unactivatedOutput = output;
		}
		else
		{
			output = inputs[0];
		}
		
		if (activationFunction == 1) { output = activationRELU(output); }
		if (activationFunction == 2) { output = activationTANH(output); }
		if (activationFunction == 3) { output = activationSigmoid(output); return; }
		
		
		//Each neuron's ID corresponds to their position on the layer. Thus used to pass forward data
		for (int i = 0; i < forwardConnections.size(); i++)
		{
			forwardConnections.get(i).inputs[id] = output;
		} 
			
	}
	
	
	//The back propagation function, given the actual values
	public void backpropagate(float answer)
	{
		if (neuronType == 2) //Different backpropagation for output layer neuron
		{
			for (int i = 0; i < weights.length; i++)
			{
				//dC/dW =          di/dW        *     do/di                     * dC/do
				float direction = inputs[i];
				
				dCi = 2 * (output - answer);
				if (activationFunction == 0) { dCi *= unactivatedOutput; }
				if (activationFunction == 1) { dCi *= derivativeRELU(unactivatedOutput); }
				if (activationFunction == 2) { dCi *= derivativeTANH(unactivatedOutput); }
				if (activationFunction == 3) { dCi *= derivativeSigmoid(unactivatedOutput); }
				
				
				direction *= dCi;
				if (usesBatches)
				{
					batchWeights[i].add(-learningStep*direction);
				}
				else
				{
					weights[i] -= learningStep*direction;	
				}
				
			}
			if (usesBatches) { batchWeights[batchWeights.length-1].add(-learningStep*dCi); }
			else { bias -= learningStep*dCi; }

			
		}
		else
		{
			for (int i = 0; i < weights.length; i++)
			{
				float direction = inputs[i];
				if (activationFunction == 0) { dCi = unactivatedOutput; }
				if (activationFunction == 1) { dCi = derivativeRELU(unactivatedOutput); }
				if (activationFunction == 2) { dCi = derivativeTANH(unactivatedOutput); }
				if (activationFunction == 3) { dCi = derivativeSigmoid(unactivatedOutput); }
				float dCo = 0;
				
				for (int j = 0; j < forwardConnections.size(); j++)
				{
					Neuron n = forwardConnections.get(j);

					dCo += forwardConnections.get(j).dCi*n.weights[id];
				}
				
				dCi *= dCo;
				direction *= dCi;
				
				if (usesBatches)
				{
					batchWeights[i].add(-learningStep*direction);
				}
				else
				{
					weights[i] -= learningStep*direction;	
				}
			}
			if (usesBatches) { batchWeights[batchWeights.length-1].add(-learningStep*dCi); }
			else { bias -= learningStep*dCi; }
		}
	}
	
	//Apply mean values of collected weights
	public void applyBatchMean()
	{
		if (!usesBatches) { return; }
		if (neuronType == 0) { return; }
		
		for (int i = 0; i < weights.length+1; i++)
		{
			float sum = 0;
			for (float f : batchWeights[i])
			{
				sum += f;
			}
			sum = sum / batchWeights[i].size();
			batchWeights[i].clear();
			if (i == weights.length) { bias += sum; break;}
			weights[i] += sum;
		}
	}
	
	
	//Connects this neuron with a neuron in front of it. Used to pass outputs to inputs
	public void addForwardConnection(Neuron n)
	{
		forwardConnections.add(n);
	}
	
	//Follows all functions used in the algorithm, as well as their derivatives.
	
	public float derivativeSigmoid(float n)
	{
		return activationSigmoid(n)*(1-activationSigmoid(n));
	}
	
	public float activationSigmoid(float n)
	{
		return (float) (1 / (1 + Math.exp(-n)));
	}
	

	public float activationRELU(float n)
	{
		return n > 0 ? n : 0;
	}
	
	public float derivativeRELU(float n)
	{
		return n > 0 ? 1 : 0;
	}
	
	
	public float activationTANH(float n)
	{
		return (float) Math.tanh(n);
	}
	
	public float derivativeTANH(float n)
	{
		return (float) (1 - Math.pow(Math.tanh(n),2));
	}
	
	
}

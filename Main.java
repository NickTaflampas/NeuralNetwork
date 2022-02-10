package qnetwork;

import java.util.Arrays;
import java.util.Random;

public class Main {

	//DEFINE VARIABLES
	final static int HIDDEN_LAYERS = 3; //Number of hidden layers
	final static int[] HIDDEN_LAYER_SIZES = {4,4,4}; //The number of neurons for each HIDDEN layer. MUST BE EQUAL LENGTH TO THE HIDDEN_LAYERS
	final static int INPUT_SIZE = 2; //Number of neurons of input layer
	final static int OUTPUT_SIZE = 4; //Number of neurons of output layer.
	final static int ACTIVATION_FUNC = 2; //The activation function for the hidden layers. 1 for RELU, 2 for TANH
	
	//MISC VARIABLES
	final static int OUTPUT_ACTIVATION_FUNC = 3; //The activation function for the output layer.
	final static boolean USE_MINIBATCHES = false; //Whether the algorithm utilizes mini-batch gradient descent. Set ALWAYS to true for this.
	final static int BATCH_SIZE = 4; //Batch size for MiniBatches. Used with USE_MINIBATCHES set to true.
	final static boolean USE_DIMINISHING_STEP = true; //Diminishign Step utillity. Set to false by default due to experimentation.
	final static float EPOCH_BREAK_POINT = 0.000001f; //The maximum difference required between 2 epochs for the training to continue.
	final static float LEARNING_STEP = 0.01f; //The learning step for all neurons.
	
	/*
	!!!! WARNING !!!!
	This Main class is a place holder. Many variables do nothing.
	!!!! WARNING !!!!
	*/
	
	public static void main(String[] args)
	{
		//Initialize our Neural Network
		int[] layerLengths = new int[HIDDEN_LAYER_SIZES.length+2];
		for (int i = 0; i < HIDDEN_LAYER_SIZES.length; i++) { layerLengths[i+1] = HIDDEN_LAYER_SIZES[i]; }
		layerLengths[0] = INPUT_SIZE;
		layerLengths[layerLengths.length-1] = OUTPUT_SIZE;

		NeuralNetwork QNN = new NeuralNetwork(HIDDEN_LAYERS, layerLengths, ACTIVATION_FUNC, 
															OUTPUT_ACTIVATION_FUNC, USE_MINIBATCHES);
		

		
	}
	

	
	
	
	
	
	
}

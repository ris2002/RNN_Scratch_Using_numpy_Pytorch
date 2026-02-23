# RNN_And_Transformer_From_Scratch
This is the practice repo for doing RNN and Transformers from scratch for my  basic intuition on the topics. For RNN, I have reffered this video by Dataquest(https://www.youtube.com/watch?v=4wuIOcD1LLI). I will also build RNN using vanilla RNN from pytorch on the same dataset. This project's quest is to understand the inner workings of RNN.

## What is RNN and clear distinction from FFNN
* A standard FFNN cannot understand a sequence of patterns(sentences). Many beginners will be surprised that it can't; many assume that patterns can be learnt using weights, so what's the problem?
* The problem is that the weights understand English grammer they would identify a noun, verb, adjective, etc. But when it comes to predicting a word, it fumbles. For example I tell the FFNN to fill in the blank of the following sentence- "Ferrari is ______ brand? I expect it to fill it with 'Italian', 'luxury',...words associated with Ferrari, but it randomly puts words like 'dog', 'cat', etc.
* This happens because weights are long-term memory; they learn only the grammar of the language, not the patterns.
* A feedforward neural network cannot model sequences because it processes each input independently and has no mechanism to remember previous tokens or maintain context across time
* The RNNs come to the rescue. The only difference between them and FFNN is that in RNN, along with the input, we provide a hidden state(it stores what the sentence has said so far(short-term memory)), basically, it stores the previous words in the sentence. The hiddenstate is also updated and produced during outputs and fed along with the next input.
*  Disadvantage- If the sentence is too long due to vanishing gradients, basically it requires it in passing the words layer by layer for n long layers..., this would result in the model not remembering the previous words. LSTMs are variants of RNNs, created to solve this issue, but they also after some pont suffer from the Vanishing Gradientproblem, hence both of these are replaced by the transformer. 
## Character-Level RNN Data Encoding (Summary)
* Objective: Train a model to predict the next token in a sequence (character or word).
*  X → Y Mapping
  * Characters: Each input X[t] is a character; target Y[t] is the next character.
  * | Time step | X (input) | Y (target) |
  * | --------- | --------- | ---------- |
  * | 0         | None      | r          |
  * | 1         | r         | e          |
  * | 2         | e         | x          |
  * | 3         | x         | \n (stop)  |
* Character Encoding Process
    * Build Vocabulary:
    * Extract all unique characters from the dataset.
    * Map each character to a unique index (char_to_ix) and vice versa (ix_to_char).
    * Convert Names/Sequences to Indices:
    * Input sequence (X) → [None] + [char_to_ix[ch] for ch in sequence]
    * Target sequence (Y) → [X[1:]] + [char_to_ix['\n']]
* Vector Representation:

    * One-hot: Sparse vector of size vocab_size, mostly zeros.
    * Embeddings (optional): Dense vector (embedding_dim << vocab_size) capturing semantic similarity.

* Result:
     * X[t] feeds into the RNN at each time step
     * Y[t] is used to compute softmax cross-entropy loss
     * This encoding process is dataset-agnostic: works for any text dataset, short or long sequences, and can be extended to word-level embeddings.
## Forward Layer Workings
* In a regular NN, we have forward pass as Inputs-->Hidden_Layer_1-->Activation_Layer_1-->>Hidden_Layer_2-->Activation_Layer_2-->Hidden_Layer_3-->Softmax_Function-->Loss Function, and the equation of Hidden Layer is $$output = \sum(w \cdot x) + b$$, where w is the weights of the number neurons present in the layer, x is the given input, b is the bias.
* In an RNN we pass a hidden state along with the input; the flow of the forward pass would be similar{Inputs-->Hidden_Layer_1-->tanh_Activation_Layer_1-->Hidden_Layer_2-->tanh_Activation_Layer_2-->Output_Layer-->Softmax_Function-->Loss }, but not the same and the equation used in RNN hidden layer would be:
  $$z(h_t)=\sum(W_xh\cdot x_t)+\sum(W_hh\cdot h_{t-1})+b$$, $$outputs=tanh(z)$$.
* To explain RNN we shall take an example problem, A=['r','e','x'] and compare how a forward pass occurs in NN,[Consider having 2 hidden layers in this example] 
         * In a standard FFNN, we start with input 'r', which goes through Hidden Layer, then calculating the output using the equation  $$z = \sum(w \cdot x) + b$$, then this is passed through the activation function, then to the next layer and through the softmax function to get the prediction after that loss is calculated using the predicted value and the true value then the backprop pass starts and then the next input 'e' is put as an input. In this step, we can see that only the weights (w) mentioned in the equation [ $$z = \sum(w \cdot x) + b$$ ] are updated. 
         * In RNN, at time_step=1, 'r' is put as input, when it first passes through hidden_layer 1 ($$h_0$$[a] as $$h_{t-1}$$) to compute $$h_1$$[a], then using tanh($$h_1$$[a])) , it finds $$h_1$$[b] inn second hidden layer, $$h_0$$[b] as the initial hidden state($$h_{t-1}$$).After this, it passes through the second tanh function, then to the output layer , then it computes softmax and the loss using the predictions from softmax function and the true values.
         * In RNN, at time_step=2, 'e' is put as input, when it first passes through hidden_layer 1 ($$h_1$$[a] as $$h_{t-1}$$)to compute $$h_2$$[a], then using tanh($$h_2$$[a]) as input to next layer and $$h_1$$[b] as $$h_{t-1}$$, it finds $$h_2$$[b] in the second hidden layer.After this, it passes through the second tanh function, then to the output layer , then it computes softmax and the loss using the predictions from softmax function and the true values.
         * In RNN, at time_step=3, 'x' is put as input, when it first passes through hidden_layer 1 ($$h_2$$[a] as $$h_{t-1}$$)to compute $$h_3$$[a], then using tanh($$h_3$$[a]) as input to next layer and $$h_2$$[b] as $$h_{t-1}$$, it finds $$h_3$$[b] in the second hidden layer.After this, it passes through the second tanh function, then to the output layer , then it computes softmax and the loss using the predictions from softmax function and the true values.
         * The losses at each time step are stored and summed up for backprop.
         * At each timestep, the hidden states, pre-activations, inputs and outputs are stored in a cache. This is essential because the backward pass needs to access these values to compute gradients. Without storing them, BPTT cannot be performed."
## Reason for using 'tanh' as the activation function.
* $$tanh$$ activation function values are bounded, i.e., from (-1 to 1), if we use Relu, i.e, Relu=max(0,x), it might result in exploding gradients which is not good for results.
* $$sigmoid$$ also has  bounded values, i.e., from (0 to 1), but it is ot used as when both are differentiated, its max is less than that of $$tanh$$ and to avoid vanishing gradient problem.
* $$tanh$$ derivative max = 1.0, $$sigmoid$$ derivative max = 0.25 
   
   
  

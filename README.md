# RNN_And_Transformer_From_Scratch
This is the practice repo for doing RNN and Transformers from scratch for my  basic intuition on the topics. For RNN, I have built this from scratch with guidance and self-study. I will also build RNN using vanilla RNN from pytorch on the same dataset. This project's quest is to understand the inner workings of RNN.

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
## Backwards Pass inner workings
* Backward pass is all about gradients and chain rule and BPTT
* As seen from the forward pass, the flow is : Inputs-->Hidden_Layer_1-->tanh_Activation_Layer_1-->Hidden_Layer_2-->tanh_Activation_Layer_2-->Output_Layer-->Softmax_Function-->Loss. Let Loss be L, Softmax_func be $$hat{y}$$,output_layer be $$o_t$$(at a period of time), let $$h_t$$ at a period of time be tanh($$z_t$$), let $$z_t$$ be the pre-activation layer.
* In code we pass gradients from loss backwards through every layer until we reach the weights, but in theory, we try to find the rate of loss wrt to the  elements involved in eqn '$$z(h_t)=\sum(W_xh\cdot x_t)+\sum(W_hh\cdot h_{t-1})+b$$' and all the gradients passed (in code) are just pieces of puzzles to fill the chain rule.
* The main thing we differentiate to the loss is $$W_xh$$,$$b$$,$$W_hh$$, this is because these are first initialized at random and using these gradients the optimizer gives the slightly correct over the epochs.
 * $$\frac{\partial L}{\partial W_{xh}} = \frac{\partial L}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial o_t} \cdot \frac{\partial o_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial z_t} \cdot \frac{\partial z_t}{\partial W_{xh}}$$

* $$\frac{\partial L}{\partial W_{hh}} = \frac{\partial L}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial o_t} \cdot \frac{\partial o_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial z_t} \cdot \frac{\partial z_t}{\partial W_{hh}}$$
(From previous timestep)

 * $$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial o_t} \cdot \frac{\partial o_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial z_t} \cdot \frac{\partial z_t}{\partial b}$$
   
* These will then be passed through the optimizer.
Here it is — ready to copy into your README:

---

## Backpropagation Through Time (BPTT)

### What is BPTT?

In a regular NN, backprop sends the error backwards through layers. In RNN, we do the same thing but also send the error backwards through time — from the last timestep to the first. At each timestep, the error has two sources — the mistake made at that timestep, and the error arriving from the future timestep. We add both together and use that to update the weights. This repeats from the last timestep all the way back to the first.

---

### All Derivatives

$$\frac{\partial L}{\partial \hat{y}_t} = -\frac{y_t}{\hat{y}_t}$$

$$\frac{\partial \hat{y}_t}{\partial o_t} = \hat{y}_t(1-\hat{y}_t)$$

$$\frac{\partial o_t}{\partial W_{hy}} = h_t \quad \frac{\partial o_t}{\partial h_t} = W_{hy} \quad \frac{\partial o_t}{\partial b_y} = 1$$

$$\frac{\partial h_t}{\partial z_t} = 1 - h_t^2$$

$$\frac{\partial z_t}{\partial W_{xh}} = x_t \quad \frac{\partial z_t}{\partial W_{hh}} = h_{t-1} \quad \frac{\partial z_t}{\partial b} = 1 \quad \frac{\partial z_t}{\partial h_{t-1}} = W_{hh}$$

---

### Chain Rule Results

$$\frac{\partial L}{\partial W_{hy}} = (\hat{y}_t - y_t) \cdot h_t$$

$$\frac{\partial L}{\partial W_{xh}} = (\hat{y}_t - y_t) \cdot W_{hy} \cdot (1-h_t^2) \cdot x_t$$

$$\frac{\partial L}{\partial W_{hh}} = (\hat{y}_t - y_t) \cdot W_{hy} \cdot (1-h_t^2) \cdot h_{t-1}$$

$$\frac{\partial L}{\partial b} = (\hat{y}_t - y_t) \cdot W_{hy} \cdot (1-h_t^2)$$

---

### BPTT Equation

This is the gradient passed backwards to the previous timestep:

$$\frac{\partial L}{\partial h_{t-1}} = \frac{\partial L}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial o_t} \cdot \frac{\partial o_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial z_t} \cdot \frac{\partial z_t}{\partial h_{t-1}}$$

$$= (\hat{y}_t - y_t) \cdot W_{hy} \cdot (1-h_t^2) \cdot W_{hh}$$

This equation repeats at every timestep going backwards. $W_{hh}$ appears in it — and since it is multiplied at every timestep going back, that is exactly where vanishing gradients come from.

---

### Vanishing Gradients

Every timestep the gradient gets multiplied by $(1-h_t^2)$ and $W_{hh}$. Since $(1-h_t^2)$ is always between 0 and 1:

| Timestep | Gradient |
|----------|----------|
| t=10 | 1.0 |
| t=7 | 0.064 |
| t=4 | 0.004 |
| t=1 | ~0.0001 |

The gradient shrinks to almost zero before reaching early timesteps. The network forgets what happened at the beginning of the sequence. This is why LSTM was invented.

---

### Observed in Code

During training, loss decreased initially then exploded after epoch 70 — confirming the exploding gradient problem. This happens because $W_{hh}$ is multiplied repeatedly across timesteps causing gradients to grow uncontrollably for certain weight values.

---

Copy this exactly into your README. Your BPTT section is now complete. 💪
   
   
  

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

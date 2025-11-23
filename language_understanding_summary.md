
# Language Understanding with Neural Networks — Beginner Summary

## 1. Converting Words to Features

### One-Hot Encoding
- Converts each word to a binary vector.
- Vector length = vocabulary size.
- Only one position is 1; rest are 0.
- Example:
  - Vocabulary: ["I", "like", "cats"]
  - "cats" → [0,0,1]

### Bag of Words (BoW)
- Combines one-hot vectors by summing them.
- Represents word frequency, not order.
- Example:
  - "I like cats" → [1,1,1]
  - "cats cats like" → [0,1,2]

### Word Embeddings
- Dense low-dimensional vectors that capture meaning.
- Similar words → similar embeddings.
- Example (dimension=3):
  - "cats" → [0.12, -0.33, 0.55]

### EmbeddingBag (PyTorch)
- Sums/averages embeddings efficiently.
- Output dimension = embedding dimension.
- Used for text classification.

---

## 2. Document Categorization — Prediction with TorchText

### Neural Network Basics
- Inputs: token indices.
- Layers:
  1. EmbeddingBag
  2. Linear (fully connected)
- Output: logits → probabilities → predicted class.

### Argmax
- Picks index with highest logit value.
- Example:
  - Logits: [1.2, 5.7, 0.9, 2.0]
  - Argmax = 1 → predicted class: Sports.

### Hyperparameters
- Embedding dimension
- Number of neurons/layers
- Vocabulary size
- Number of output classes

### Prediction Pipeline
Raw text → tokenization → indices → EmbeddingBag → Linear → logits → argmax

---

## 3. Document Categorization — Training with TorchText

### Learnable Parameters (θ)
- Includes embedding weights and linear layer weights.
- Model learns by adjusting θ.

### Loss Function — Cross Entropy
- Measures how wrong the prediction is.
- Low loss = model assigns high probability to correct class.
- Uses softmax on logits to compute probabilities.

### Gradient Descent
- Updates weights to reduce loss.
- Formula:  
  **new θ = old θ – learning_rate × gradient**

### Training Loop
For each epoch:
1. Set model to train mode.
2. For each batch:
   - Forward pass (logits)
   - Compute loss (cross entropy)
   - Backpropagation (`loss.backward()`)
   - Optimizer step
   - Zero gradients
3. Track accuracy & loss.
4. Save best model using validation accuracy.

---

## 4. Complete Flow (All Topics Together)

Text → tokenization → indices → embeddings → embedding bag → neural network → logits → loss → gradients → optimized weights.

This builds the foundation for language understanding using neural networks in PyTorch.

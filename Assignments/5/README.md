# Artificial Neural Networks - 5th Assignment

This repository contains the theoretical and practical components for the 5th assignment of the **Artificial Neural Networks** course. The assignment focuses on various aspects of **Recurrent Neural Networks (RNNs)**, including the use of **LSTM-based models** for **Persian language modeling**.

## Table of Contents

1. [Question 1 - Stateful RNN vs Stateless RNN](#question-1)
2. [Question 2 - Encoder-Decoder RNN vs Plain Sequence-to-Sequence RNN](#question-2)
3. [Question 3 - Designing Gated RNNs for Summing Inputs](#question-3)
4. [Question 5 - Persian Language Model Implementation](#question-5)
---

### Question 1 - Stateful RNN vs Stateless RNN

This question examines the difference between **Stateful** and **Stateless RNNs**, specifically how they manage hidden states across batches. A **Stateful RNN** retains the hidden state between batches, while a **Stateless RNN** resets the state after each batch.

### Question 2 - Encoder-Decoder RNN vs Plain Sequence-to-Sequence RNN

The second question compares **Encoder-Decoder RNNs** and **Plain Sequence-to-Sequence RNNs**. These models are often used in tasks like machine translation but differ in structure, where the encoder-decoder uses a more sophisticated mechanism to handle variable-length inputs and outputs.

### Question 3 - Designing Gated RNNs for Summing Inputs

In this task, the goal is to design a gated RNN cell that can sum inputs over time. The design leverages gating mechanisms to control how much of the input is carried forward and accumulated.

---

### Question 5 - Persian Language Model Implementation

This task focuses on implementing a **Persian language model** using **LSTM cells** for character-level text prediction. The model is trained on a subset of the **Persian Wikipedia dataset**, aiming to predict the next character in a sequence based on a given input sequence.

#### Implementation Steps:

1. **Dataset Loading**:
   - The **Persian Wikipedia text dataset** is loaded and preprocessed to remove unwanted characters (like digits and newlines) and normalize the text by converting all characters to lowercase.

2. **Character Indexing**:
   - The model works at the character level, so we create mappings from characters to integer indices (`char_to_int`) and vice versa (`int_to_char`).
   - The text is split into sequences of 60 characters, with the target being the next character in the sequence.

3. **Model Architecture**:
   - The model consists of two stacked **LSTM layers**, each with 128 units, followed by **dropout layers** to prevent overfitting.
   - A **dense layer** with a softmax activation predicts the probability distribution of the next character.

4. **Training**:
   - The model is trained for 50 epochs using the **RMSprop optimizer** and **categorical cross-entropy loss**.
   - We save the model weights during training for future use.

5. **Perplexity Calculation**:
   - **Perplexity** is used as a performance metric, measuring how well the model predicts the next character. Lower perplexity indicates better generalization.

6. **Text Generation**:
   - The trained model can generate text by taking a seed string of 3-5 words and predicting the next characters in the sequence using a sampling function.

#### Detailed Explanation:

- After loading the **Persian Wikipedia** dataset, preprocessing is performed to clean and normalize the data.
- Unique characters are identified, and a character-to-integer encoding system is created for input to the model.
- Training sequences are prepared by splitting the text into 60-character inputs and their next-character targets.
- The model architecture is based on **LSTMs** with dropout layers and is compiled with RMSprop optimizer and **categorical cross-entropy** as the loss function.
- The model is trained for 40-50 epochs, with checkpoints saved during training to track performance.
- Once trained, the model is used to generate new text sequences, showcasing its ability to form coherent sentences in Persian.
 
--- 

### Assignment Report Summary (based on Mehrdad Baradaran's submission)

- The **Persian Wikipedia dataset** was loaded and preprocessed to remove unnecessary characters and normalize text to lowercase.
- The unique characters were indexed to map each character to an integer.
- Training sequences of 60 characters were created, and the model was trained to predict the next character in each sequence.
- The model architecture consisted of two LSTM layers with 128 units, and it was compiled with **categorical cross-entropy loss**. After 40 epochs, the model achieved a loss of 0.9577.
- The trained model was able to generate coherent Persian text, maintaining logical sentence structure and predicting meaningful words.

The final model produced results with a good balance of coherence and generalization, as demonstrated in the text generation task.

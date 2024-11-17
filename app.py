#let import the model and initialise it

from decoder import Magic, TransDecoder
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import wandb
from get_indices import create_lookup_tables, tokenize_elements
from get_data import inputs, targets, vocab
import numpy

# Initialize the models
d_model = 40  # Dimension of embeddings, can be adjusted
n_heads = 2  # Number of attention heads
d_ff = 80  # Dimension of feed-forward network
batch_size = 1  # Number of sequences processed in parallel

# get data and iunitiate unk. data imported from get_data.py
# Add the '<UNK>' token with a unique index
vocab['<UNK>'] = len(vocab)
int_to_vocab = {idx: word for word, idx in vocab.items()}

vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
vocab_size = len(vocab_to_int)

print(len(inputs))
print("item in position ipnuts[:1]", inputs[:1])

model = TransDecoder(vocab_size=vocab_size, batch_size=batch_size, d_model=d_model, n_heads=n_heads, d_ff=d_ff)


# Define the batch size
batch_size = 1

model = TransDecoder(vocab_size, batch_size=batch_size)
#optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Initialize W&B
wandb.init(project="decoder", name="going mad")

epochs = 200  # Number of training epochs

# input = inputs[0].unsqueeze(0)
# target = targets[0].unsqueeze(0) #to add batch dimension

# print("input shape", input.shape)
# print("inputs lenght", len(inputs))
# print("input lenght", len(input))
# print("this is how input looks like", input)
# print("target shape", target.shape)

def train_epoch(model, inputs, targets, criterion, optimizer, epochs):
    for epoch in range(epochs):
        total_loss = 0
        
        # Iterate over each element in inputs
        for element_idx in range(len(inputs)):
            input_tensor = inputs[element_idx]  # Shape: [1002]
            target_tensor = targets[element_idx]
            
            # Reshape to [1, 1002] for batch_size and sequence_length
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            target_tensor = target_tensor.unsqueeze(0)
            
            # print(f"Processing element {element_idx}")
            # print(f"Input tensor shape after reshape: {input_tensor.shape}")
            
            # Forward pass
            predictions = model(input_tensor)
            
            # Compute loss
            loss = criterion(predictions.view(-1, predictions.size(-1)), 
                           target_tensor.view(-1))
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Calculate average loss
        avg_loss = total_loss / len(inputs)
        wandb.log({'average_loss': avg_loss})
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

train_epoch(model, inputs, targets, criterion, optimizer, epochs)

wandb.finish()

# def infer(model, input_sequence, vocab_to_int, int_to_vocab, temperature=1.0):
#     """
#     Inference function to predict the next sequence based on the model.
#     Added temperature sampling and better debugging.
#     """
#     model.eval()
    
#     # Convert input to tokens with debug printing
#     tokens = [vocab_to_int.get(char, vocab_to_int['<UNK>']) for char in input_sequence]
#     tokens = [vocab_to_int['<s>']] + tokens
#     print(f"\nInput sequence: {input_sequence}")
#     print(f"Tokenized (with start token): {tokens}")
    
#     # Create tensor and add batch dimension
#     tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(next(model.parameters()).device)
#     print(f"Input tensor shape: {tokens_tensor.shape}")
    
#     with torch.no_grad():
#         # Get model predictions
#         logits = model(tokens_tensor)
#         print(f"Logits shape: {logits.shape}")
#         print(f"Sample logits: {logits[0, 0, :5]}")  # Print first 5 logits of first position
        
#         # Apply temperature to logits
#         logits = logits / temperature
        
#         # Apply softmax to get probabilities
#         probabilities = F.softmax(logits, dim=-1)
#         print(f"Max probability: {probabilities.max().item():.4f}")
        
#         # Get top k probabilities and their indices for the last position
#         top_k = 5
#         top_probs, top_indices = torch.topk(probabilities[0, -1], top_k)
#         print("\nTop {} predictions for last position:".format(top_k))
#         for prob, idx in zip(top_probs, top_indices):
#             char = int_to_vocab.get(idx.item(), '<UNK>')
#             print(f"{char}: {prob.item():.4f}")
        
#         # Instead of just taking argmax, use multinomial sampling
#         predicted_tokens = torch.multinomial(probabilities.squeeze(0), num_samples=1)
#         predicted_tokens = predicted_tokens.squeeze(-1)
        
#     # Convert predictions to characters
#     output_sequence = []
#     for token in predicted_tokens.tolist():
#         if token in int_to_vocab:
#             output_sequence.append(int_to_vocab[token])
#         else:
#             print(f"Warning: Token {token} not in vocabulary")
#             output_sequence.append('<UNK>')
    
#     return ''.join(output_sequence)

# # Test with different temperatures
# temperatures = [1.0, 1.5]
# test_strings = ['ccgg', 'cgaf', 'eeggaBb', 'eegga']

# print("\nTesting with different temperatures:")
# for temp in temperatures:
#     print(f"\nTemperature: {temp}")
#     for test_string in test_strings:
#         result = infer(model, test_string, vocab_to_int, int_to_vocab, temperature=temp)
#         print(f"Input: {test_string:10} Output: {result}")




def infer(model, input_sequence):
    model.eval()
    # tokens = [tokenise(char) for char in input_sequence]
    # # add start token
    # tokens = [tokenise('<s>')] + tokens

    tokens = [vocab_to_int.get(char, vocab_to_int['<UNK>']) for char in input_sequence]
    tokens = [vocab_to_int['<s>']] + tokens
    print(vocab_to_int)
    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(next(model.parameters()).device)
#     print(f"Input tensor shape: {tokens_tensor.shape}")

    with torch.no_grad():
        logits = model(torch.tensor(tokens_tensor))

    probabilities = nn.functional.softmax(logits, dim=-1)

    predicted_tokens = torch.argmax(probabilities, dim=-1)
    print(predicted_tokens)

    # Convert predictions to characters
    output_sequence = []
    for token in predicted_tokens.squeeze(0).tolist():  # Flatten the tensor to a list   
        
        if token in int_to_vocab:
            output_sequence.append(int_to_vocab[token])
        else:
            print(f"Warning: Token {token} not in vocabulary")
            output_sequence.append('<UNK>')

    # items = [token.item() for token in predicted_tokens]
    # output_tokens = [chars[token] for token in items]
    return output_sequence



test_strings = ['RRR','BBB', 'PPP']
for str in test_strings:
   result = infer(model, test_strings)
   print(f"result for {str}: {result}")

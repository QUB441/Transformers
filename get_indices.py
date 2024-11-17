import torch
import collections

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_lookup_tables(elements: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """
    Create lookup tables to map words to indices and indices to words.

    Parameters:
        elements (list): List of elements (e.g., words) to create a vocabulary.

    Returns:
        vocab_to_int (dict): Mapping from word to index.
        int_to_vocab (dict): Mapping from index to word.
    """
    # Count unique elements and sort by frequency
    element_counts = collections.Counter(elements)
    vocab = sorted(element_counts, key=lambda k: element_counts[k], reverse=True)
    
    # Create int-to-word and word-to-int mappings with special tokens
    int_to_vocab = {ii + 1: word for ii, word in enumerate(vocab)}
    int_to_vocab[0] = '<PAD>'  # Add padding token at index 0
    int_to_vocab[len(int_to_vocab)] = '<UNK>'  # Add unknown token at the end
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    
    return vocab_to_int, int_to_vocab

def tokenize_elements(elements: list[str], vocab_to_int: dict) -> torch.Tensor:
    """
    Tokenize a list of elements into corresponding indices based on the vocab and return as a tensor.

    Parameters:
        elements (list): List of elements to be tokenized.
        vocab_to_int (dict): The lookup dictionary mapping from word to index.

    Returns:
        tokens (torch.Tensor): Tensor of tokenized elements.
    """
    tokens = []
    for element in elements:
        # If the element is in vocab, get its index; otherwise, return the index of '<UNK>'
        tokens.append([vocab_to_int.get(element, vocab_to_int['<UNK>'])])
    
    # Convert the list of tokenized elements into a tensor
    token_tensor = torch.tensor(tokens, dtype=torch.long).to(device)  # Move to device (GPU/CPU)
    
    return token_tensor

def main():
    # Example list of elements (could be words or any other tokens)
    elements = ['apple', 'banana', 'apple', 'orange', 'banana', 'grape', 'apple']

    # Creating the lookup tables
    vocab_to_int, int_to_vocab = create_lookup_tables(elements)

    # Print the lookup tables
    print("Vocabulary to Index:", vocab_to_int)
    print("Index to Vocabulary:", int_to_vocab)

    # Tokenize the elements into indices and get the tensor
    token_tensor = tokenize_elements(elements, vocab_to_int)

    # Print the tokenized elements as a tensor
    print("Tokenized Elements as Tensor:", token_tensor)
    print(type(token_tensor))

if __name__ == "__main__":
    main()



#sample usage
# from get_indices import create_lookup_tables, tokenize_elements

# new_elements = ['dog', 'cat', 'bird', 'dog', 'fish']

# vocab_to_int, int_to_vocab = create_lookup_tables(new_elements)
# token_tensor = tokenize_elements(new_elements, vocab_to_int)

# print("Tokenized Elements:", token_tensor)

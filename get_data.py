import torch


class NoteGenerator:
    def __init__(self, notes, length=8):
        self.notes = notes  # List of possible notes, like ['A', 'B', 'C', ...]
        self.store = []     # Stores generated sequences
        self.genre = "rock" # Default genre
        self.length = length  # Desired length of generated sequence
        
        # Define genre patterns
        self.patterns = {
            "rock": ["R", "R", "R", "R"],
            "pop": ["P", "P", "P", "P"],
            "blues": ["B", "B", "B", "B"]
        }
        
    def set_genre(self, genre):
        # Set the genre and validate if it exists in patterns
        if genre.lower() in self.patterns:
            self.genre = genre.lower()
        else:
            raise ValueError("Unsupported genre. Available genres: rock, pop, blues")
    
    def generate(self):
        # Fetch pattern based on the genre
        base_pattern = self.patterns.get(self.genre, [])
        
        if not base_pattern:
            raise ValueError(f"No pattern defined for genre '{self.genre}'")
        
        # Repeat the base pattern to reach the specified length
        generated_sequence = []
        for _ in range(self.length // len(base_pattern) + 1):
            generated_sequence.extend(base_pattern)
        
        # Trim the generated sequence to the specified length
        generated_sequence = generated_sequence[:self.length]
        
        # Store the generated sequence in `store`
        self.store.append(generated_sequence)
        
        return generated_sequence

    def add_genre(self, genre_name, pattern):
        """
        Adds a new genre with a specified note pattern.
        """
        self.patterns[genre_name.lower()] = pattern


# Usage
notes = ['R', 'P', 'B']  # Just a list of possible notes for initialization
note_generator = NoteGenerator(notes, length=10)


# Set genre to rock and generate a sequence
note_generator.set_genre("rock")
#print("Rock sequence:", note_generator.generate())

rock_notes = ['<rock>'] + note_generator.generate()

# Set genre to pop and generate a sequence
note_generator.set_genre("pop")
#print("Pop sequence:", note_generator.generate())
pop_notes = ['<pop>'] + note_generator.generate()

# # Adding and generating a sequence for a custom genre
# note_generator.add_genre("custom", ["D", "D", "G", "A", "C"])
# note_generator.set_genre("custom")
# print("Custom sequence:", note_generator.generate())

# Set genre to pop and generate a sequence
note_generator.set_genre("blues")
#print("Pop sequence:", note_generator.generate())
blues_notes = ['<blues>'] + note_generator.generate()

data = rock_notes + pop_notes + blues_notes

# Create the vocab dictionary, including start and end tokens
special_tokens = ['<s>', '</s>']
vocab = {char: idx for idx, char in enumerate(set(data).union(special_tokens))}
vocab_size = len(vocab)  # Total number of unique tokens

# Create the reversed int_to_vocab dictionary
int_to_vocab = {idx: word for word, idx in vocab.items()}

print(int_to_vocab)

print(vocab)

# Function to get indices of data
def get_indices(natural_data):
    return [vocab[char] for char in natural_data]

# Encode the data with start and end tokens
rock_train_data = [vocab['<s>']] + get_indices(rock_notes)
pop_train_data = [vocab['<s>']] + get_indices(pop_notes)
blues_train_data = [vocab['<s>']] + get_indices(blues_notes)

rock_target_data = get_indices(rock_notes) + [vocab['</s>']]
pop_target_data = get_indices(pop_notes) + [vocab['</s>']]
blues_target_data = get_indices(blues_notes) + [vocab['</s>']]

#turn into tensors

rock_input = torch.tensor(rock_train_data, dtype=torch.long)
rock_target = torch.tensor(rock_target_data, dtype=torch.long)

# For Pop genre
pop_input = torch.tensor(pop_train_data, dtype=torch.long)
pop_target = torch.tensor(pop_target_data, dtype=torch.long)

# For Blues genre
blues_input = torch.tensor(blues_train_data, dtype=torch.long)
blues_target = torch.tensor(blues_target_data, dtype=torch.long)

# Organize these as lists of inputs and targets per genre
inputs = [rock_input, pop_input, blues_input]
targets = [rock_target, pop_target, blues_target]

print(inputs)
print("input lenght", len(inputs[0]))

#  self.patterns = {
#             "rock": ["C", "C", "G", "G", "A", "A", "G"],
#             "pop": ["C", "G", "A", "F"],
#             "blues": ["E", "E", "G", "G", "A", "Bb", "B"]
#         }


import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        """
        Initialize the input embeddings layer

        Args:
            d_model: The dimensionality of the model (embedding size)
            vocab_size: The size of the vocabulary
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Use nn.Embedding to create a lookup table that maps integer indices (IDs) to dense vectors (corresponding embeddings)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        """
        Forward pass for input embeddings
        Args:
            x: Tensor of shape (batch_size, seq_length) containing input token IDs
        Returns:
            Tensor of shape (batch_size, seq_length, d_model) 
                containing the input embeddings scaled by sqrt(d_model) (according to the Transformer paper)
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    # Declare the buffer for the pe attribute as a class attribute with a type hint for better type checking
    pe: torch.Tensor

    def __init__(self, d_model: int, seq_length: int, dropout: float):
        """
        Initialize the positional encoding layer

        Args:
            d_model: The dimensionality of the model (embedding size)
            seq_length: The maximum sequence length
            dropout: The dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix of shape (seq_length, d_model) initialized to zeros
        pe = torch.zeros(seq_length, d_model)

        # Create a tensor to hold the position indices of shape (seq_length, 1)
        # unsqueeze is used to make it a column vector because the output of arange is 1D (seq_length,)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)

        # Compute the positional encodings
        # we use the formula: PE(pos, 2i) = sin(pos / (10000^(2i/d_model))) and PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))
        # we compute the div_term using exponential and logarithm for numerical stability and performance because it involves large powers
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        # Add a batch dimension by unsqueezing at dimension 0 so that pe shape becomes (1, seq_length, d_model)
        pe = pe.unsqueeze(0)

        # Register the pe tensor as a buffer so that it's saved and not considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass for positional encoding
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model) containing input embeddings
        Returns:
            Tensor of shape (batch_size, seq_length, d_model) containing the input embeddings with positional encodings added
        """
        # The positional encodings are fixed and not learned, so we use requires_grad_(False) to prevent gradients from being computed for them
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)

        # Apply dropout and return
        return self.dropout(x)

        
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SmallDenseModel(nn.Module):
    """A small fully connected neural network."""

    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SmallConv1DModel(nn.Module):
    """A small 1D convolutional neural network for sequence data."""

    def __init__(self, input_channels: int = 1, output_size: int = 10):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8, output_size)  # Assuming input length = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


class SmallConv2DModel(nn.Module):
    """A small 2D convolutional neural network."""

    def __init__(self, input_channels: int = 1, output_size: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 8 * 8, output_size)  # Assuming input is 8x8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


class SmallViTModel(nn.Module):
    """A small Vision Transformer (ViT) model."""

    def __init__(
        self,
        image_size: tuple[int, int] = (8, 8),
        patch_size: int = 2,
        num_heads: int = 2,
        num_layers: int = 2,
        output_size: int = 10,
        input_channels: int = 3,
    ):
        super().__init__()
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.patch_dim = (patch_size**2) * input_channels
        self.embedding = nn.Linear(self.patch_dim, 32)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=num_heads, dim_feedforward=64)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(32, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.unfold(2, 2, 2).unfold(3, 2, 2)  # Patchify
        x = x.reshape(batch_size, self.num_patches, self.patch_dim)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Mean pooling
        return self.fc(x)


class SmallRNNModel(nn.Module):
    """A small recurrent neural network model."""

    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 2, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.rnn(x)
        return self.fc(h_n[-1])  # Take the last hidden state


class SmallLSTMModel(nn.Module):
    """A small LSTM-based model for sequence processing."""

    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 2, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])  # Take the last hidden state


class SmallGRUModel(nn.Module):
    """A small GRU-based model for sequence processing."""

    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 2, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])  # Take the last hidden state


class SimpleTokenizer:
    """A simple tokenizer that maps words to unique indices (simulated)."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.word_to_idx = {f"word{i}": i for i in range(vocab_size)}
        self.unk_token = vocab_size - 1  # Unknown token index
        self.pad_token = 0  # Padding token index

    def encode(self, text: str, max_seq_len: int = 20) -> torch.Tensor:
        """
        Converts a string into a tensor of token indices.

        Args:
            text (str): Input text.
            max_seq_len (int): Maximum length of tokenized sequence.

        Returns:
            torch.Tensor: Tokenized text as a tensor of shape (max_seq_len,)
        """
        tokens = text.lower().split()[:max_seq_len]  # Simple whitespace-based tokenization
        token_ids = [self.word_to_idx.get(token, self.unk_token) for token in tokens]
        padding = [self.pad_token] * (max_seq_len - len(token_ids))  # Pad to fixed length
        return torch.tensor(token_ids + padding, dtype=torch.long)


class SmallTextClassifier(nn.Module):
    """A small Transformer model for text classification."""

    def __init__(
        self,
        vocab_size: int = 1000,
        embedding_dim: int = 32,
        num_heads: int = 2,
        num_layers: int = 2,
        max_seq_len: int = 20,
        output_size: int = 10,
    ):
        """
        Args:
            vocab_size (int): Number of unique tokens in vocabulary.
            embedding_dim (int): Size of word embeddings.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            max_seq_len (int): Maximum sequence length.
            output_size (int): Number of output classes.
        """
        super().__init__()
        self.tokenizer = SimpleTokenizer(vocab_size)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=64)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the text transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Mean pooling over sequence length
        return self.fc(x)


class SmallTextGenerator(nn.Module):
    """A small Transformer model for text generation."""

    def __init__(
        self,
        vocab_size: int = 1000,
        embedding_dim: int = 32,
        num_heads: int = 2,
        num_layers: int = 2,
        max_seq_len: int = 20,
    ):
        """
        Args:
            vocab_size (int): Number of unique tokens in vocabulary.
            embedding_dim (int): Size of word embeddings.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer decoder layers.
            max_seq_len (int): Maximum sequence length.
        """
        super().__init__()
        self.tokenizer = SimpleTokenizer(vocab_size)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=64)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)  # Predicts next token

    def forward(self, x: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for text generation.

        Args:
            x (torch.Tensor): Encoder input tensor of shape (batch_size, seq_len)
            tgt (torch.Tensor): Decoder input tensor of shape (batch_size, seq_len)

        Returns:
            torch.Tensor: Logits over vocabulary of shape (batch_size, seq_len, vocab_size)
        """
        x = self.embedding(x)
        tgt = self.embedding(tgt)
        x = self.transformer(tgt, x)
        return self.fc(x)

    def generate(self, input_text: str, max_new_tokens: int = 5) -> str:
        """
        Generates text auto-regressively.

        Args:
            input_text (str): The seed text.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            str: The generated text.
        """
        input_ids = self.tokenizer.encode(input_text).unsqueeze(0)  # Add batch dim
        generated_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            logits = self.forward(input_ids, generated_ids)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

        generated_words = [f"word{idx.item()}" for idx in generated_ids[0]]
        return " ".join(generated_words)

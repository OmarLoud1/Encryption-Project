import torch
import torch.nn as nn
from tokenizer import Tokenizer

class SimpleTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers):
        super(SimpleTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead), num_encoder_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x

def tokenize_and_transform(tokenizer, model, sentence):
    tokens = tokenizer.encode(sentence, bos=True, eos=True)
    tokens_tensor = torch.tensor(tokens).unsqueeze(1)
    output = model(tokens_tensor)
    return output

if __name__ == "__main__":
    # Load the tokenizer and model
    model_path = "/Users/omar/Documents/Research/llama/llama/tokenizer.model"
    tokenizer = Tokenizer(model_path=model_path)
    model = SimpleTransformerModel(tokenizer.n_words, 512, 8, 2)

    # Example sentences
    sentence_1 = "This is a sentence."
    sentence_2 = "This is another sentence."

    # Tokenize and transform
    output_1 = tokenize_and_transform(tokenizer, model, sentence_1)
    output_2 = tokenize_and_transform(tokenizer, model, sentence_2)

    print("Matrix multiplication out1:", output_1)
    # Ensure dimensions are correct for matrix multiplication
    output_1 = output_1.transpose(0, 1)
    output_2 = output_2.transpose(0, 1)

    # Perform matrix multiplication
    result = torch.matmul(output_1, output_2.transpose(1, 2))

    print("Matrix multiplication result:", result)

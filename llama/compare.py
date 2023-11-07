import torch
import torch.nn as nn
import time

# Assuming Tokenizer and CKKS-related classes are correctly imported
from tokenizer import Tokenizer
from ckks.ckks_decryptor import CKKSDecryptor
from ckks.ckks_encoder import CKKSEncoder
from ckks.ckks_encryptor import CKKSEncryptor
from ckks.ckks_evaluator import CKKSEvaluator
from ckks.ckks_key_generator import CKKSKeyGenerator
from ckks.ckks_parameters import CKKSParameters

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
    tokens = tokenizer.encode(sentence, True, True)  # Assuming True for both bos and eos
    tokens_tensor = torch.tensor(tokens).unsqueeze(1)
    output = model(tokens_tensor)
    return output

def main():
    # Load the tokenizer and model
    model_path = "/Users/omar/Documents/Research/llama/llama/tokenizer.model"
    tokenizer = Tokenizer(model_path=model_path)
    model = SimpleTransformerModel(tokenizer.n_words, 512, 8, 2)

    # Example sentences
    sentence_1 = "This is a sentence."
    sentence_2 = "This is another sentence."

    # Tokenize and transform
    start_time = time.time()
    print("Tokenizing and transforming sentences...")
    output_1 = tokenize_and_transform(tokenizer, model, sentence_1)
    output_2 = tokenize_and_transform(tokenizer, model, sentence_2)
    print("Done in {:.4f} seconds.".format(time.time() - start_time))

    # Ensure dimensions are correct for element-wise multiplication
    output_1 = output_1.view(-1)
    output_2 = output_2.view(-1)

    length = 4096
    output_1 = output_1[:length]
    output_2 = output_2[:length]

    # Perform element-wise multiplication
    token_product = output_1 * output_2
    print("Token product (first 10 elements):", token_product[:10].tolist())

    # Initialize CKKS components
    poly_degree = 8192
    ciph_modulus = 1 << 60
    big_modulus = 1 << 1200
    scaling_factor = 1 << 40
    params = CKKSParameters(poly_degree=poly_degree, ciph_modulus=ciph_modulus, big_modulus=big_modulus, scaling_factor=scaling_factor)
    key_generator = CKKSKeyGenerator(params)
    encoder = CKKSEncoder(params)
    encryptor = CKKSEncryptor(params, key_generator.public_key, key_generator.secret_key)
    decryptor = CKKSDecryptor(params, key_generator.secret_key)
    evaluator = CKKSEvaluator(params)

    # Encrypt, multiply, and decrypt
    start_time = time.time()
    print("Encrypting data...")
    output_1_list = output_1.flatten().tolist()
    output_2_list = output_2.flatten().tolist()
    plain_1 = encoder.encode(output_1_list, scaling_factor)
    plain_2 = encoder.encode(output_2_list, scaling_factor)
    ciph_1 = encryptor.encrypt(plain_1)
    ciph_2 = encryptor.encrypt(plain_2)
    print("Done in {:.4f} seconds.".format(time.time() - start_time))

    start_time = time.time()
    print("Multiplying encrypted data...")
    ciph_prod = evaluator.multiply(ciph_1, ciph_2, key_generator.relin_key)
    print("Done in {:.4f} seconds.".format(time.time() - start_time))

    start_time = time.time()
    print("Decrypting product...")
    decrypted_prod = decryptor.decrypt(ciph_prod)
    decoded_prod = encoder.decode(decrypted_prod)
    print("Done in {:.4f} seconds.".format(time.time() - start_time))

    print("Decrypted product after CKKS multiplication (first 10 elements):", decoded_prod[:10])

if __name__ == '__main__':
    main()

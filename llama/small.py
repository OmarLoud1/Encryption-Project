import torch
import time
from ckks.ckks_decryptor import CKKSDecryptor
from ckks.ckks_encoder import CKKSEncoder
from ckks.ckks_encryptor import CKKSEncryptor
from ckks.ckks_evaluator import CKKSEvaluator
from ckks.ckks_key_generator import CKKSKeyGenerator
from ckks.ckks_parameters import CKKSParameters

def main():
    # Initialize 2D matrices
    matrix_size = 4
    matrix_1 = torch.rand(matrix_size, matrix_size)
    matrix_2 = torch.rand(matrix_size, matrix_size)
    
    # Flatten the matrices for encryption
    vector_1 = matrix_1.view(-1)
    vector_2 = matrix_2.view(-1)
    
    # Initialize CKKS components
    poly_degree = 64
    ciph_modulus = 1 << 40
    big_modulus = 1 << 400  
    scaling_factor = 1 << 30    
    params = CKKSParameters(poly_degree=poly_degree, ciph_modulus=ciph_modulus, big_modulus=big_modulus, scaling_factor=scaling_factor)
    key_generator = CKKSKeyGenerator(params)
    encoder = CKKSEncoder(params)
    encryptor = CKKSEncryptor(params, key_generator.public_key, key_generator.secret_key)
    decryptor = CKKSDecryptor(params, key_generator.secret_key)
    evaluator = CKKSEvaluator(params)
    
    # Encrypt the vectors
    start_time = time.time()
    print("Encrypting matrices...")
    plain_1 = encoder.encode(vector_1.tolist(), scaling_factor)
    plain_2 = encoder.encode(vector_2.tolist(), scaling_factor)
    ciph_1 = encryptor.encrypt(plain_1)
    ciph_2 = encryptor.encrypt(plain_2)
    print("Encryption done in {:.4f} seconds.".format(time.time() - start_time))
    
    # Perform element-wise multiplication on encrypted vectors
    start_time = time.time()
    print("Multiplying encrypted matrices...")
    ciph_prod = evaluator.multiply(ciph_1, ciph_2, key_generator.relin_key)
    print("Multiplication done in {:.4f} seconds.".format(time.time() - start_time))
    
    # Decrypt the result
    start_time = time.time()
    print("Decrypting the product...")
    decrypted_prod = decryptor.decrypt(ciph_prod)
    decoded_prod = encoder.decode(decrypted_prod)
    print("Decryption done in {:.4f} seconds.".format(time.time() - start_time))
    
    # Reshape the result back to a 2D matrix
    result_matrix = torch.tensor(decoded_prod).view(matrix_size, matrix_size)
    
    print("Original Matrix 1:\n", matrix_1)
    print("Original Matrix 2:\n", matrix_2)
    print("Decrypted product after CKKS multiplication:\n", result_matrix)

if __name__ == '__main__':
    main()
s
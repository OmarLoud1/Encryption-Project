from ckks.ckks_decryptor import CKKSDecryptor
from ckks.ckks_encoder import CKKSEncoder
from ckks.ckks_encryptor import CKKSEncryptor
from ckks.ckks_evaluator import CKKSEvaluator
from ckks.ckks_key_generator import CKKSKeyGenerator
from ckks.ckks_parameters import CKKSParameters
import time
import numpy as np

def generate_random_complex_matrix(size):
    return np.random.rand(size, size) + 1j * np.random.rand(size, size)

def process_block(encoder, encryptor, evaluator, decryptor, block1, block2, relin_key, scaling_factor):
    encoded1 = encoder.encode(block1.flatten(), scaling_factor)
    encoded2 = encoder.encode(block2.flatten(), scaling_factor)
    encrypted1 = encryptor.encrypt(encoded1)
    encrypted2 = encryptor.encrypt(encoded2)
    encrypted_prod = evaluator.multiply(encrypted1, encrypted2, relin_key)
    decrypted_prod = decryptor.decrypt(encrypted_prod)
    decoded_prod = encoder.decode(decrypted_prod)
    return np.array(decoded_prod).reshape(block1.shape)

def block_matrix_multiply(matrix1, matrix2, block_size, process_func, params, keys, relin_key, scaling_factor):
    n = matrix1.shape[0]
    result_matrix = np.zeros((n, n), dtype=complex)
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            for k in range(0, n, block_size):
                block1 = matrix1[i:i + block_size, k:k + block_size]
                block2 = matrix2[k:k + block_size, j:j + block_size]
                result_matrix[i:i + block_size, j:j + block_size] += process_func(
                    keys['encoder'], keys['encryptor'], keys['evaluator'], keys['decryptor'], 
                    block1, block2, relin_key, scaling_factor)
    return result_matrix

def main():
    poly_degree = 64*2
    ciph_modulus = 1 << 600
    big_modulus = 1 << 1200
    scaling_factor = 1 << 30
    block_size = poly_degree // 2
    matrix_size = 8  # Assuming 2x2 matrices for simplicity

    params = CKKSParameters(poly_degree=poly_degree,
                            ciph_modulus=ciph_modulus,
                            big_modulus=big_modulus,
                            scaling_factor=scaling_factor)
    key_generator = CKKSKeyGenerator(params)
    keys = {
        'public_key': key_generator.public_key,
        'secret_key': key_generator.secret_key,
        'relin_key': key_generator.relin_key,
        'encoder': CKKSEncoder(params),
        'encryptor': CKKSEncryptor(params, key_generator.public_key, key_generator.secret_key),
        'decryptor': CKKSDecryptor(params, key_generator.secret_key),
        'evaluator': CKKSEvaluator(params)
    }

    matrix1 = generate_random_complex_matrix(matrix_size)
    matrix2 = generate_random_complex_matrix(matrix_size)

    start_enc = time.time()
    encrypted_matrix_product = block_matrix_multiply(matrix1, matrix2, block_size, process_block, 
                                                     params, keys, key_generator.relin_key, scaling_factor)
    end_enc = time.time()

    start_plain = time.time()
    plain_matrix_product = np.dot(matrix1, matrix2)
    end_plain = time.time()

    print("Input matrices, matrix 1:", matrix1, " , matrix2: ",matrix2)
    print("Plain matrix product:\n", plain_matrix_product)
    print("Encrypted matrix product:\n", encrypted_matrix_product)
    print("Encryption + Matrix Multiplication time:", end_enc - start_enc, "seconds")
    print("Plaintext Matrix Multiplication time:", end_plain - start_plain, "seconds")

if __name__ == '__main__':
    main()

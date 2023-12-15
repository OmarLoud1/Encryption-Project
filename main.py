import itertools
import argparse

from ckks.ckks_decryptor import CKKSDecryptor
from ckks.ckks_encoder import CKKSEncoder
from ckks.ckks_encryptor import CKKSEncryptor
from ckks.ckks_evaluator import CKKSEvaluator
from ckks.ckks_key_generator import CKKSKeyGenerator
from ckks.ckks_parameters import CKKSParameters

import time
import numpy as np
from multiprocessing import Pool
from functools import partial

NUM_WORKERS = 32

def generate_random_complex_matrix(size):
    return np.random.rand(size, size) + 1j * np.random.rand(size, size)


def block_matrix_multiply(matrix, message, keys, key_generator, scaling_factor, block_size):
    encoder = keys['encoder']
    encryptor = keys['encryptor']
    evaluator = keys['evaluator']
    decryptor = keys['decryptor']

    n = matrix.shape[0]

    # split message
    ciph = []
    for i in range(0, len(message), block_size):
        block = message[i:i + block_size]
        encoded = encoder.encode(block, scaling_factor)
        encrypted = encryptor.encrypt(encoded)
        ciph.append(encrypted)

    result = [0] * len(ciph)
    for bx in range(0, n // block_size):
        for by in range(0, n // block_size):
            mat = matrix[bx * block_size:(bx + 1) * block_size, by * block_size:(by + 1) * block_size]

            rot_keys = {}
            for i in range(len(mat)):
                rot_keys[i] = key_generator.generate_rot_key(i)

            ciph_prod = evaluator.multiply_matrix(ciph[by], mat, rot_keys, encoder)

            if by == 0:
                result[bx] = ciph_prod
            else:
                result[bx] = evaluator.add(result[bx], ciph_prod)

    decoded = []
    for chunk in result:
        decrypted_prod = decryptor.decrypt(chunk)
        decoded.extend(encoder.decode(decrypted_prod))

    return decoded


# Parallel over the output dimension only ("rows")
def row_bmm(n, block_size, matrix, key_generator, evaluator, ciph, encoder, decryptor, bx):
    for by in range(0, n // block_size):
        mat = matrix[bx * block_size:(bx + 1) * block_size, by * block_size:(by + 1) * block_size]

        rot_keys = {}
        for i in range(len(mat)):
            rot_keys[i] = key_generator.generate_rot_key(i)

        ciph_prod = evaluator.multiply_matrix(ciph[by], mat, rot_keys, encoder)

        if by == 0:
            result = ciph_prod
        else:
            result = evaluator.add(result, ciph_prod)

    decrypted_prod = decryptor.decrypt(result)
    return encoder.decode(decrypted_prod)


# Parallel over each multiplication
def bmm(n, block_size, matrix, key_generator, evaluator, ciph, encoder, idx):
    bx = idx // (n // block_size)
    by = idx % (n // block_size)

    mat = matrix[bx * block_size:(bx + 1) * block_size, by * block_size:(by + 1) * block_size]

    rot_keys = {}
    for i in range(len(mat)):
        rot_keys[i] = key_generator.generate_rot_key(i)

    ciph_prod = evaluator.multiply_matrix(ciph[by], mat, rot_keys, encoder)

    return ciph_prod


def row_parallel_matrix_multiply(matrix, message, keys, key_generator, scaling_factor, block_size):
    encoder = keys['encoder']
    encryptor = keys['encryptor']
    evaluator = keys['evaluator']
    decryptor = keys['decryptor']

    n = matrix.shape[0]

    # split message
    ciph = []
    for i in range(0, len(message), block_size):
        block = message[i:i + block_size]
        encoded = encoder.encode(block, scaling_factor)
        encrypted = encryptor.encrypt(encoded)
        ciph.append(encrypted)

    with Pool(NUM_WORKERS) as p:
        result = p.map(partial(row_bmm, n, block_size, matrix, key_generator, evaluator, ciph, encoder, decryptor),
                       range(n // block_size))

    # concatenate results
    return list(itertools.chain.from_iterable(result))


def parallel_matrix_multiply(matrix, message, keys, key_generator, scaling_factor, block_size):
    encoder = keys['encoder']
    encryptor = keys['encryptor']
    evaluator = keys['evaluator']
    decryptor = keys['decryptor']

    n = matrix.shape[0]

    # split message
    ciph = []
    for i in range(0, len(message), block_size):
        block = message[i:i + block_size]
        encoded = encoder.encode(block, scaling_factor)
        encrypted = encryptor.encrypt(encoded)
        ciph.append(encrypted)

    blocks = n ** 2 // block_size ** 2
    with Pool(NUM_WORKERS) as p:
        multiplied = p.map(partial(bmm, n, block_size, matrix, key_generator, evaluator, ciph, encoder),
                           range(blocks))

    decoded = []
    for bx in range(0, n // block_size):
        for by in range(0, n // block_size):
            if by == 0:
                result = multiplied[bx * (n // block_size) + by]
            else:
                result = evaluator.add(result, multiplied[bx * (n // block_size) + by])
        decrypted_prod = decryptor.decrypt(result)
        decoded.extend(encoder.decode(decrypted_prod))

    return decoded


def main():
    parser = argparse.ArgumentParser(prog='Encrypted Matrix Multiplication',
                                     description='Encrypted Matrix Multiplication')
    parser.add_argument("--block-size", help="block side length", type=int, default=16)
    parser.add_argument("--matrix-size", help="matrix side length", type=int, default=128)
    parser.add_argument("--algorithm", help="which algorithm", choices=['naive', 'blocked', 'row_parallel', 'parallel'],
                        default='row_parallel')
    args = parser.parse_args()
    print(f'Algorithm: {args.algorithm}\tMatrix Size: {args.matrix_size}\tBlock Size: {args.block_size}')

    ciph_modulus = 1 << 600
    big_modulus = 1 << 1200
    scaling_factor = 1 << 30
    block_size = args.block_size  # side length
    matrix_size = args.matrix_size  # side length
    poly_degree = 2 * block_size

    print("Initializing...")
    start_time = time.time()
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
    print("Done in {:.4f} seconds.".format(time.time() - start_time))

    matrix1 = generate_random_complex_matrix(matrix_size)
    matrix2 = generate_random_complex_matrix(matrix_size)

    vec = matrix2[0]

    start_plain = time.time()
    plain_matrix_product = matrix1 @ vec
    end_plain = time.time()
    print("Plaintext Matrix Multiplication time:", end_plain - start_plain, "seconds")

    algorithms = {'blocked': block_matrix_multiply, 'row_parallel': row_parallel_matrix_multiply,
                  'parallel': parallel_matrix_multiply}

    start_enc = time.time()
    algo_fn = algorithms[args.algorithm]
    encrypted_matrix_product = algo_fn(matrix1, vec, keys, key_generator, scaling_factor, block_size)

    end_enc = time.time()

    print("Encryption + Matrix Multiplication time:", end_enc - start_enc, "seconds")
    print(np.allclose(plain_matrix_product,
                      np.array(encrypted_matrix_product), atol=1e-3))

if __name__ == '__main__':
    main()

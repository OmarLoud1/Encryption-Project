import itertools
import math

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
import torch


def generate_random_complex_matrix(size):
    return np.random.rand(size, size) + 1j * np.random.rand(size, size)


def lib_matrix_multiply(mat, message, keys, key_generator, scaling_factor):
    encoder = keys['encoder']
    encryptor = keys['encryptor']
    evaluator = keys['evaluator']
    decryptor = keys['decryptor']

    plain = encoder.encode(message, scaling_factor)
    ciph = encryptor.encrypt(plain)

    rot_keys = {}
    for i in range(len(mat)):
        rot_keys[i] = key_generator.generate_rot_key(i)

    ciph_prod = evaluator.multiply_matrix(ciph, mat, rot_keys, encoder)
    decrypted_prod = decryptor.decrypt(ciph_prod)
    decoded_prod = encoder.decode(decrypted_prod)

    return decoded_prod


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

    with Pool(n // block_size) as p:
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
    with Pool(blocks) as p:
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
    # 128 with block 32, took 123 sec
    # 128 with block 16, took 115 sec
    # 128 with block 8, took 126 sec
    # 128 unblocked took 119

    # 128 row parallel with block 16, took 30
    # 128 all parallel with block 16, took 54
    # 128 all parallel with block 32, took 30

    ciph_modulus = 1 << 600
    big_modulus = 1 << 1200
    scaling_factor = 1 << 30
    block_size = 32  # side length
    matrix_size = 128  # side length
    poly_degree = 2 * block_size
    # poly_degree = 2 * matrix_size # if using lib_matrix_multiply instead of blocked

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

    start_enc = time.time()
    # encrypted_matrix_product = parallel_matrix_multiply(matrix1, vec, keys, key_generator, scaling_factor, block_size)
    encrypted_matrix_product = row_parallel_matrix_multiply(matrix1, vec, keys, key_generator, scaling_factor,
                                                            block_size)
    # encrypted_matrix_product = block_matrix_multiply(matrix1, vec, keys, key_generator, scaling_factor, block_size)
    # encrypted_matrix_product = lib_matrix_multiply(matrix1, vec, keys, key_generator, scaling_factor)
    end_enc = time.time()

    start_plain = time.time()
    plain_matrix_product = matrix1 @ vec
    end_plain = time.time()

    print("Input matrices, matrix 1:", matrix1, " , matrix2: ", matrix2)
    print("Plain matrix product:\n", plain_matrix_product)
    print("Encrypted matrix product:\n", encrypted_matrix_product)
    print("Encryption + Matrix Multiplication time:", end_enc - start_enc, "seconds")
    print("Plaintext Matrix Multiplication time:", end_plain - start_plain, "seconds")
    print(torch.allclose(torch.from_numpy(plain_matrix_product).type(torch.complex64),
                         torch.tensor(encrypted_matrix_product), atol=1e-3))


if __name__ == '__main__':
    main()

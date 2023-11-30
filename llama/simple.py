import numpy as np
from ckks.ckks_parameters import CKKSParameters
from ckks.ckks_key_generator import CKKSKeyGenerator
from ckks.ckks_encoder import CKKSEncoder
from ckks.ckks_encryptor import CKKSEncryptor
from ckks.ckks_decryptor import CKKSDecryptor

def main():
    # Initialize parameters and keys
    params = CKKSParameters(poly_degree=8192, ciph_modulus=1 << 60, big_modulus=1 << 1200, scaling_factor=1 << 40)
    key_generator = CKKSKeyGenerator(params)
    public_key = key_generator.public_key
    secret_key = key_generator.secret_key

    encoder = CKKSEncoder(params)
    encryptor = CKKSEncryptor(params, public_key)
    decryptor = CKKSDecryptor(params, secret_key)

    # Create a plaintext matrix
    plaintext_matrix = np.array([[1.5, 2.5], [3.5, 4.5]])

    # Encode and encrypt the matrix
    encoded_matrix = [encoder.encode([val], params.scaling_factor) for val in plaintext_matrix.flatten()]
    encrypted_matrix = [encryptor.encrypt(enc) for enc in encoded_matrix]

    # Decrypt and decode the matrix
    decrypted_encoded = [decryptor.decrypt(enc) for enc in encrypted_matrix]
    decrypted_matrix = np.array([encoder.decode(dec)[0] for dec in decrypted_encoded]).reshape(plaintext_matrix.shape)

    # Compare results
    print("Original Matrix:\n", plaintext_matrix)
    print("Decrypted Matrix:\n", decrypted_matrix)

if __name__ == "__main__":
    main()

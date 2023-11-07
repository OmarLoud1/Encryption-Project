import syft as sy
import torch

# Hook PyTorch to add extra functionalities like Federated and Encrypted Learning
hook = sy.TorchHook(torch) 

# Define PySyft's CKKS tensor, secret data to be encrypted
data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Define precision and base for CKKS encryption
precision_fractional=3
precision_integral=3

# Encrypt the data using CKKS
encrypted_data = data.fix_prec(precision_fractional=precision_fractional).ckks_tensor(precision_integral=precision_integral)

print("Encrypted data:", encrypted_data)

# Decrypt the data
decrypted_data = encrypted_data.decrypt().float_prec()
print("Decrypted data:", decrypted_data)

import numpy as np

f = np.loadtxt("features.txt", dtype=np.float32)

def to_bf16_and_back(x):
    bits = x.view(np.uint32)
    bits = (bits >> 16).astype(np.uint32) << 16
    return bits.view(np.float32)

f_bf16 = to_bf16_and_back(f)

for c in range(3):
    w = np.loadtxt(f"weights_{c}.txt", dtype=np.float32)
    w_bf16 = to_bf16_and_back(w)
    raw = np.dot(f, w)
    expected = np.dot(f_bf16.astype(np.float64), w_bf16.astype(np.float64))
    print(f"Class {c}: raw float32 = {raw:.6f}, bf16-expected = {expected:.6f}")

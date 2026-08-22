import numpy as np

f = np.loadtxt("features.txt", dtype=np.float32)
w = np.loadtxt("weights.txt", dtype=np.float32)

def to_bf16_and_back(x):
    bits = x.view(np.uint32)
    bits = (bits >> 16).astype(np.uint32) << 16
    return bits.view(np.float32)

f_bf16 = to_bf16_and_back(f)
w_bf16 = to_bf16_and_back(w)

print("Raw float32 dot:", np.dot(f, w))
print("bf16-truncated dot (expected NPU result):", np.dot(f_bf16.astype(np.float64), w_bf16.astype(np.float64)))

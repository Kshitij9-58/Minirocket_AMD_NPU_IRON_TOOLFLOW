import numpy as np, struct
def bf(f):
    b = struct.unpack('<I', struct.pack('<f', float(f)))[0] >> 16
    return struct.unpack('<f', struct.pack('<I', b << 16))[0]
w = np.array([bf(float(x)) for x in open('classifier_weights.txt')])
feat = np.array([bf(int(x)/1000.0) for x in open('cpu_reference_full420.txt')])
print("features:", len(feat), "weights:", len(w))
print("tile: hypothesis A (all tiles see features 0-69) | hypothesis B (correct distinct slices)")
for t in range(12):
    ws = w[t*70:(t+1)*70]
    a = float(np.sum(feat[0:70] * ws))
    b = float(np.sum(feat[t*70:(t+1)*70] * ws))
    print(f"{t:2d}: A={a: .6f}  B={b: .6f}")

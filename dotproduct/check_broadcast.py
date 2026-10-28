import numpy as np, struct
def bf(f):
    b = struct.unpack('<I', struct.pack('<f', float(f)))[0] >> 16
    return struct.unpack('<f', struct.pack('<I', b << 16))[0]
w = np.array([bf(float(x)) for x in open('classifier_weights.txt')])
ref = [float(x) for x in open('cpu_reference_fused_pipeline.txt')]
print("reference per-tile partials:", ref)
print("weights loaded:", len(w))

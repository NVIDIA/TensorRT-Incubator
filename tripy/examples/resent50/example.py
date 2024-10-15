import tripy as tp
class GEGLU(tp.Module):
    def __init__(self, dim_in, dim_out):
        self.proj = tp.Linear(dim_in, dim_out * 2)
        self.dim_out = dim_out

    def __call__(self, x):
        proj = self.proj(x)
        x, gate = tp.split(proj, 2, proj.rank - 1)
        return x * tp.gelu(gate)
    

layer = GEGLU(2, 8)
inp = tp.ones((1, 2))
out = layer(inp)

inp_info = tp.InputInfo(shape=(1, 2), dtype=tp.float32)

fast_geglu = tp.compile(layer, args=[inp_info])
fast_geglu(inp).eval()
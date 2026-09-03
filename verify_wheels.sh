#!/bin/bash
B=/workspace/wheelbuild
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
rm -rf $B/verify2 && uv venv -q $B/verify2 --python 3.12
export VIRTUAL_ENV=$B/verify2
uv pip install -q --index https://download.pytorch.org/whl/cu130 "torch==2.13.0" numpy einops 2>&1|tail -1
# TE core + meta come from Astral (no torch coupling), our torch bindings are local
uv pip install -q --index-strategy unsafe-best-match \
  --extra-index-url https://wheels.astral.sh/simple/cu130/ \
  "transformer-engine-cu13==2.16.0+cu.13.0" 2>&1|tail -2
uv pip install -q --no-deps $B/out/causal_conv1d-*.whl $B/out/flash_attn-*.whl \
  $B/out/mamba_ssm-*.whl $B/out/transformer_engine_torch-*.whl 2>&1|tail -2
$B/verify2/bin/python - <<'PY'
import torch
print("torch:", torch.__version__, "| device:", torch.cuda.get_device_name(0))
ok = {}

def check(name, fn):
    try:
        fn(); ok[name]="PASS"
    except Exception as e:
        ok[name]=f"FAIL {type(e).__name__}: {str(e)[:150]}"

def t_causal():
    import causal_conv1d_cuda
    from causal_conv1d import causal_conv1d_fn
    x=torch.randn(2,64,128,device="cuda",dtype=torch.float16); w=torch.randn(64,4,device="cuda",dtype=torch.float16)
    y=causal_conv1d_fn(x,w); assert torch.isfinite(y).all()

def t_fa():
    import flash_attn_2_cuda
    from flash_attn import flash_attn_func
    q,k,v=[torch.randn(1,128,8,64,device="cuda",dtype=torch.bfloat16) for _ in range(3)]
    o=flash_attn_func(q,k,v); assert torch.isfinite(o).all(), "non-finite"

def t_mamba():
    import selective_scan_cuda
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    B_,D,L,N=1,32,16,8
    u=torch.randn(B_,D,L,device="cuda",dtype=torch.float32)
    delta=torch.rand(B_,D,L,device="cuda",dtype=torch.float32)
    A=-torch.rand(D,N,device="cuda",dtype=torch.float32)
    Bm=torch.randn(B_,N,L,device="cuda",dtype=torch.float32)
    C=torch.randn(B_,N,L,device="cuda",dtype=torch.float32)
    y=selective_scan_fn(u,delta,A,Bm,C); 
    y=y[0] if isinstance(y,tuple) else y
    assert torch.isfinite(y).all()

def t_te():
    import transformer_engine_torch as tex
    import transformer_engine.pytorch as te
    print("   TE version:", __import__("transformer_engine").__version__)
    layer = te.Linear(64, 64, bias=True).cuda()
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        y = layer(x)
    assert torch.isfinite(y).all()

for n,f in [("causal-conv1d",t_causal),("flash-attn",t_fa),("mamba-ssm",t_mamba),("transformer-engine",t_te)]:
    check(n,f)
print()
for n,v in ok.items(): print(f"  {n:22s} {v}")
PY

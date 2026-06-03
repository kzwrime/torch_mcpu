import pathlib

import torch
import torch_mcpu  # noqa: F401


build_dir = pathlib.Path(__file__).resolve().parent / "build"
torch.ops.load_library(str(build_dir / "libexternal_mcpu_axpby.so"))

x_cpu = torch.arange(8, dtype=torch.float32).reshape(2, 4)
x = x_cpu.to("mcpu")
out = torch.empty_like(x)

torch.ops.mcpu_external.axpby(out, x, 2.0, 3.0)
torch.accelerator.synchronize()

expected = x_cpu * 2.0 + 3.0
torch.testing.assert_close(out.to("cpu"), expected)
print("external axpby kernel launch OK")

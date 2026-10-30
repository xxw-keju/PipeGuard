# PipeGuard

Experimental static hazard checking for SM90 kernels. Python 3.10+.

## Install

```bash
git clone https://github.com/xxw-keju/PipeGuard.git
cd PipeGuard
python -m pip install .
```

## Quick start

Check a PGIR file (maximal is the only engine):

```bash
python -m pipeguard check tests/fixtures/safe_handoff.json --pretty
```

Or use Python:

```python
from pipeguard import check_module, load_module

report = check_module(load_module("kernel.pgir.json"))
print(report["verdict"])
print(report["diagnostics"])
```

In the [TileLang fork](https://github.com/xxw-keju/tilelang-pipeguard), install
`./3rdparty/pipeguard`, then check a compiled kernel with its actual launch grid:

```python
report = kernel.check_pipeguard(launch_grid=(16, 8, 1), timeout=120)
```

See [frontends](docs/frontends.md) for IR capture and launch bindings,
[semantics](docs/semantics.md) for supported operations and result scope,
and [analysis notes](docs/analysis.md) for implementation and validation limits.

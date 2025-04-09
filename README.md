# ILP Decoder

An implementation of the ILP decoder for circuit-level noise.

Included is a `ilp_decoder.ILPDecoder` class that is configured using a `stim.DetectorErrorModel` and decodes shot data, directly outputting predicted observables (without sinter), as well as a `ilp_decoder.ILPSinterDecoder` class, which subclasses `sinter.Decoder`, for interfacing with sinter.

## Installation

To install from pypi, run:

```
pip install ilp_decoder
```

To install from source, run:

```
pip install -e .
```

from the root directory.

## Usage

Here is an example of how the decoder can be used directly with Stim:

```python
import stim
import numpy as np
from ilp_decoder import ILPDecoder

num_shots = 100
d = 3
p = 0.001
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    rounds=d,
    distance=d,
    before_round_data_depolarization=p,
    before_measure_flip_probability=p,
    after_reset_flip_probability=p,
    after_clifford_depolarization=p
)

sampler = circuit.compile_detector_sampler()
shots, observables = sampler.sample(num_shots, separate_observables=True)

decoder = ILPDecoder(circuit)

predicted_observables = decoder.decode_batch(shots)
num_mistakes = np.sum(np.any(predicted_observables != observables, axis=1))

print(f"{num_mistakes}/{num_shots}")
```

### Sinter integration

To integrate with [sinter](https://github.com/quantumlib/Stim/tree/main/glue/sample), you can use the
`ilp_decoder.ILPSinterDecoder` class, which inherits from `sinter.Decoder`.
To use it, you can use the `custom_decoders` argument when using `sinter.collect`:

```python
import sinter
from ilp_decoder import ILPSinterDecoder, sinter_decoders

samples = sinter.collect(
    num_workers=4,
    max_shots=1_000_000,
    max_errors=1000,
    tasks=generate_example_tasks(),
    decoders=['ilp'],
    custom_decoders=sinter_decoders()
    print_progress=True,
)
```

### Advanced ILP solvers

You can use whatever ILP solver supported by `cvxpy` instead of the default one `highs`, though
`highs` should be one of the fastest open-source solver available. If you have access to `gurobi`,
you may gain some extra speed boost (not tested).

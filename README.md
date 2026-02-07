# mnist-cvnn-jax-v1

Examples of complex-valued neural networks (CVNN) implemented with JAX.

## Files

- `main.py`: MNIST CVNN example that encodes image intensity into phase.
- `xor_cvnn_phase_example.py`: XOR CVNN example where binary inputs are encoded in the **phase** part of complex numbers.

## Evaluation scripts

Non-production evaluation and verification scripts live in `tests/`.

```bash
python tests/test_jax_grad_behavior.py
python tests/verify_cvnn_v2.py
```

## Run

```bash
python main.py
python xor_cvnn_phase_example.py
```

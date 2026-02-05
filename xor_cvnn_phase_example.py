"""XOR solved with phase-encoded complex-valued computation.

Encoding:
- bit 0 -> exp(i * 0)   = +1
- bit 1 -> exp(i * pi)  = -1

For two encoded inputs z1, z2 in {+1, -1}, XOR is determined by the phase relation:
- same bits:      z1 * z2 = +1
- different bits: z1 * z2 = -1

So XOR can be recovered with:
    xor = (1 - Re(z1 * z2)) / 2
"""

import jax.numpy as jnp


X_BITS = jnp.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ],
    dtype=jnp.float32,
)
Y_XOR = jnp.array([0, 1, 1, 0], dtype=jnp.int32)


def encode_phase(bits: jnp.ndarray) -> jnp.ndarray:
    """Encode binary values into the phase part of complex numbers."""
    theta = bits * jnp.pi
    return jnp.exp(1j * theta)


def cvnn_xor_predict(bits: jnp.ndarray) -> jnp.ndarray:
    """Predict XOR using phase interference in complex space."""
    z = encode_phase(bits)
    phase_relation = jnp.real(z[:, 0] * z[:, 1])
    xor_score = (1.0 - phase_relation) / 2.0
    return (xor_score > 0.5).astype(jnp.int32)


if __name__ == "__main__":
    preds = cvnn_xor_predict(X_BITS)
    accuracy = jnp.mean(preds == Y_XOR)

    print("XOR with phase-encoded CVNN computation")
    for x, y, p in zip(X_BITS, Y_XOR, preds):
        print(f"x={x.astype(jnp.int32).tolist()} y_true={int(y)} y_pred={int(p)}")

    print(f"Accuracy: {float(accuracy) * 100:.2f}%")

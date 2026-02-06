import jax
import jax.numpy as jnp
from jax import grad

def test_simple_grad():
    z = 1.0 + 2.0j
    f = lambda z: jnp.abs(z)**2
    g = grad(f)(z)
    print(f"z: {z}, grad(|z|^2): {g}")

if __name__ == "__main__":
    test_simple_grad()

import jax
import jax.numpy as jnp
from jax import grad
import cvnn_v1
import cvnn_v2

def test_sigmoid_grad():
    z_real = 0.5
    z_imag = 0.8
    
    # V1
    z1 = jnp.array([z_real, z_imag], dtype=jnp.float32)
    def f1(z):
        res = cvnn_v1.complex_sigmoid(z)
        return jnp.sum(res**2) # Real loss
    
    g1 = grad(f1)(z1)
    
    # V2
    z2 = jnp.array(z_real + 1j * z_imag, dtype=jnp.complex64)
    def f2(z):
        res = cvnn_v2.complex_sigmoid(z)
        return jnp.abs(res)**2 # Real loss
    
    g2 = grad(f2)(z2)
    
    print(f"V1 grad: {g1}")
    print(f"V2 grad: {g2}")
    print(f"V2 grad as real pair: {g2.real}, {g2.imag}")

if __name__ == "__main__":
    test_sigmoid_grad()

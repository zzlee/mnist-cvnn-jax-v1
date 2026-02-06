import jax
import jax.numpy as jnp

def f(z):
    return jnp.abs(z)**2

z = 1.0 + 2.0j
g = jax.grad(f)(z)
print(f"z = {z}")
print(f"f(z) = |z|^2 = x^2 + y^2")
print(f"grad(f)(z) = {g}")
print(f"Real grad: df/dx = 2x = {2*z.real}, df/dy = 2y = {2*z.imag}")

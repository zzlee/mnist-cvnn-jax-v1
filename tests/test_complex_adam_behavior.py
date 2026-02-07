import jax
import jax.numpy as jnp
import optax

def test_adam_complex():
    lr = 0.5
    optimizer = optax.adam(lr)
    
    # Complex parameter
    params = jnp.array([1.0 + 1.0j], dtype=jnp.complex64)
    state = optimizer.init(params)
    
    # Complex gradient
    grads = jnp.array([0.1 + 0.2j], dtype=jnp.complex64)
    
    updates, next_state = optimizer.update(grads, state, params)
    print(f"Gradient: {grads}")
    print(f"Update: {updates}")
    
    # Manual calculation if it was treating as real pairs
    # g_r = 0.1, g_i = 0.2
    # upd_r = -0.5 * sign(0.1) = -0.5
    # upd_i = -0.5 * sign(0.2) = -0.5
    # Expected if real-like: -0.5 - 0.5j
    
    # What if it uses g^2?
    # g^2 = (0.1+0.2j)^2 = 0.01 - 0.04 + 0.04j = -0.03 + 0.04j
    # sqrt(g^2) = g
    # update = -lr * g / g = -lr = -0.5
    
    # What if it uses |g|^2?
    # |g|^2 = 0.01 + 0.04 = 0.05
    # update = -lr * g / sqrt(|g|^2) = -lr * g / |g|
    # |g| = sqrt(0.05) = 0.2236
    # update = -0.5 * (0.1 + 0.2j) / 0.2236 = -0.2236 - 0.4472j

    # Let's see the actual output
    
if __name__ == "__main__":
    test_adam_complex()

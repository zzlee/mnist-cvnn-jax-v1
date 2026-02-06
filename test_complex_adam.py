import jax
import jax.numpy as jnp
import optax

def test_adam_comparison():
    lr = 0.1
    beta1 = 0.9
    beta2 = 0.999
    
    # Complex case
    w_c = jnp.array([1.0 + 2.0j], dtype=jnp.complex64)
    g_c = jnp.array([0.1 + 0.5j], dtype=jnp.complex64)
    
    opt_c = optax.adam(lr, b1=beta1, b2=beta2)
    state_c = opt_c.init(w_c)
    updates_c, next_state_c = opt_c.update(g_c, state_c, w_c)
    w_c_new = optax.apply_updates(w_c, updates_c)
    
    # Real case (V1 style)
    w_r = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
    g_r = jnp.array([[0.1, 0.5]], dtype=jnp.float32)
    
    opt_r = optax.adam(lr, b1=beta1, b2=beta2)
    state_r = opt_r.init(w_r)
    updates_r, next_state_r = opt_r.update(g_r, state_r, w_r)
    w_r_new = optax.apply_updates(w_r, updates_r)
    
    print(f"Complex result: {w_c_new}")
    print(f"Real result:    {w_r_new[0,0]} + {w_r_new[0,1]}j")
    
    # Check manual update for V2
    m = g_c * (1 - beta1)
    v = jnp.abs(g_c)**2 * (1 - beta2)
    m_hat = m / (1 - beta1)
    v_hat = v / (1 - beta2)
    expected_update_c = lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
    print(f"Expected update C: {expected_update_c}")

if __name__ == "__main__":
    test_adam_comparison()

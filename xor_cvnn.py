import jax
import jax.numpy as jnp
import optax
from jax import jit, value_and_grad

# 1. Hyperparameters
LEARNING_RATE = 0.005
EPOCHS = 3000
HIDDEN1_SIZE = 8
HIDDEN2_SIZE = 4

# 2. Activation Function (Hidden Layers)
def complex_tanh(z):
	"""
	Complex Tanh-like activation.
	Compresses magnitude to [0, 1) using tanh, preserves phase.
	"""
	r = jnp.abs(z)
	theta = jnp.angle(z)
	r_activated = jnp.tanh(r)
	return r_activated * jnp.exp(1j * theta)

# 3. Data Encoding (Phase Encoding)
def encode_xor(x):
	"""
	Encode data strictly into the phase.
	Input 0 -> Phase 0 (1+0j)
	Input 1 -> Phase pi (-1+0j)
	Magnitude is always 1 (Confidence = 1 for inputs).
	"""
	phases = x * jnp.pi
	return jnp.exp(1j * phases)

# 4. Initialization
def init_params(rng):
	k = jax.random.split(rng, 6)

	# Layer 1: 2 -> HIDDEN1
	w1 = (jax.random.normal(k[0], (2, HIDDEN1_SIZE)) +
		  1j * jax.random.normal(k[1], (2, HIDDEN1_SIZE))) * 0.5
	b1 = jnp.zeros((HIDDEN1_SIZE,), dtype=jnp.complex64)

	# Layer 2: HIDDEN1 -> HIDDEN2
	w2 = (jax.random.normal(k[2], (HIDDEN1_SIZE, HIDDEN2_SIZE)) +
		  1j * jax.random.normal(k[3], (HIDDEN1_SIZE, HIDDEN2_SIZE))) * 0.5
	b2 = jnp.zeros((HIDDEN2_SIZE,), dtype=jnp.complex64)

	# Layer 3: HIDDEN2 -> 1
	w3 = (jax.random.normal(k[4], (HIDDEN2_SIZE, 1)) +
		  1j * jax.random.normal(k[5], (HIDDEN2_SIZE, 1))) * 0.5
	b3 = jnp.zeros((1,), dtype=jnp.complex64)

	return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}

# 5. Model Forward
def model_forward(params, x):
	# L1
	z1 = jnp.dot(x, params["w1"]) + params["b1"]
	a1 = complex_tanh(z1)

	# L2
	z2 = jnp.dot(a1, params["w2"]) + params["b2"]
	a2 = complex_tanh(z2)

	# L3 (Output)
	z3 = jnp.dot(a2, params["w3"]) + params["b3"]
	return jnp.abs(z3).reshape(-1)

# 6. Loss and Update
def loss_fn(params, x, y):
	preds = model_forward(params, x)
	return jnp.mean((preds - y) ** 2)

@jit
def update_step(params, opt_state, x, y):
	loss, grads = value_and_grad(loss_fn)(params, x, y)
	# JAX returns the conjugate gradient for complex parameters, so we conjugate it back
	grads = jax.tree_util.tree_map(jnp.conj, grads)
	updates, next_opt_state = optimizer.update(grads, opt_state, params)
	next_params = optax.apply_updates(params, updates)
	return next_params, next_opt_state, loss

if __name__ == "__main__":
	# XOR Data
	X_raw = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
	Y = jnp.array([0, 1, 1, 0], dtype=jnp.float32)
	X_complex = encode_xor(X_raw)

	# Init
	rng = jax.random.PRNGKey(42)
	params = init_params(rng)
	optimizer = optax.adam(LEARNING_RATE)
	opt_state = optimizer.init(params)

	print(f"Training 3-layer CVNN ({HIDDEN1_SIZE}-{HIDDEN2_SIZE}-1)...")
	for step in range(EPOCHS):
		params, opt_state, loss = update_step(params, opt_state, X_complex, Y)
		if step % 500 == 0:
			print(f"Step {step}, Loss: {loss:.6f}")

	# Final Test
	def model_complex(params, x):
		z1 = jnp.dot(x, params["w1"]) + params["b1"]
		a1 = complex_tanh(z1)
		z2 = jnp.dot(a1, params["w2"]) + params["b2"]
		a2 = complex_tanh(z2)
		return jnp.dot(a2, params["w3"]) + params["b3"]

	z_out = model_complex(params, X_complex)

	print("\nResults (3-Layer Architecture):")
	print("-" * 65)
	for i in range(4):
		z = z_out[i, 0]
		mag = jnp.abs(z)
		phase = jnp.angle(z)
		decoded = 1 if jnp.cos(phase) < 0 else 0

		print(f"Input: {X_raw[i]} | Target: {Y[i]}")
		print(f"  Output Phase: {phase:6.2f} -> Decoded: {decoded}")
		print(f"  Confidence:   {mag:6.4f}")
		print("-" * 65)

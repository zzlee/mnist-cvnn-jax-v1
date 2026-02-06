import jax
import jax.numpy as jnp
import optax
from jax import jit, value_and_grad
from cvnn_v2 import *

def init_params(key, layer_sizes):
	params = []
	for i in range(len(layer_sizes) - 1):
		in_dim = layer_sizes[i]
		out_dim = layer_sizes[i+1]
		key, w_real_key, w_imag_key = jax.random.split(key, 3)
		stddev = jnp.sqrt(2. / (in_dim + out_dim))
		weights_real = jax.random.normal(w_real_key, (in_dim, out_dim)) * stddev
		weights_imag = jax.random.normal(w_imag_key, (in_dim, out_dim)) * stddev
		weights = weights_real + 1j * weights_imag
		biases = jnp.zeros((out_dim,), dtype=jnp.complex64)
		params.append({'weights': weights, 'biases': biases})
	return params

@jit
def forward_pass(params, x):
	activations = x
	for i, layer_param in enumerate(params[:-1]):
		weights = layer_param['weights']
		biases = layer_param['biases']
		linear = complex_add(complex_matmul(activations, weights), biases)
		activations = complex_sigmoid(linear)
	final_layer_param = params[-1]
	weights = final_layer_param['weights']
	biases = final_layer_param['biases']
	output = complex_add(complex_matmul(activations, weights), biases)
	return output

def loss_fn(params, x, y):
	y_pred = forward_pass(params, x)
	error = y_pred - y
	loss = jnp.mean(jnp.abs(error)**2)
	return loss;

def update_step(params, opt_state, x, y, optimizer):
	loss, grads = value_and_grad(loss_fn)(params, x, y)
	# Conjugate gradients for complex parameters
	grads = jax.tree_util.tree_map(jnp.conj, grads)
	updates, next_opt_state = optimizer.update(grads, opt_state, params)
	next_params = optax.apply_updates(params, updates)
	return next_params, next_opt_state, loss

def run_test(opt_name, lr):
	print(f"\nTesting {opt_name} with lr={lr} (CONJUGATED GRADS)")
	key = jax.random.PRNGKey(44)
	xor_layer_sizes = [2, 2, 2, 1]
	xor_input_data_theta = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
	xor_input_data = from_polar(jnp.ones_like(xor_input_data_theta), xor_input_data_theta)
	xor_labels_theta = jnp.array([[1, ], [0, ], [0, ], [1, ]], dtype=jnp.float32)
	xor_labels = from_polar(jnp.ones_like(xor_labels_theta), xor_labels_theta)

	params = init_params(key, xor_layer_sizes)
	if opt_name == "adam":
		optimizer = optax.adam(lr)
	else:
		optimizer = optax.sgd(lr)
	
	opt_state = optimizer.init(params)

	for step in range(301):
		params, opt_state, loss = update_step(params, opt_state, xor_input_data, xor_labels, optimizer)
		if step % 100 == 0:
			print(f"Step {step}, Loss: {loss:.6f}")

if __name__ == "__main__":
	run_test("adam", 0.5)
	run_test("sgd", 0.5)

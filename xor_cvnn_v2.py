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

		# 為實部、虛部權重和下一個key分割隨機種子
		key, w_real_key, w_imag_key = jax.random.split(key, 3)

		# 使用 Glorot/Xavier Normal 初始化權重，標準差為 sqrt(2 / (fan_in + fan_out))
		# 這有助於在訓練過程中穩定梯度，防止梯度消失或爆炸。
		stddev = jnp.sqrt(2. / (in_dim + out_dim))

		# 初始化實部和虛部權重，並組合成複數
		weights_real = jax.random.normal(w_real_key, (in_dim, out_dim)) * stddev
		weights_imag = jax.random.normal(w_imag_key, (in_dim, out_dim)) * stddev
		weights = weights_real + 1j * weights_imag

		# 偏置初始化為零，使用複數型態
		biases = jnp.zeros((out_dim,), dtype=jnp.complex64)
		params.append({'weights': weights, 'biases': biases})
	return params

@jit
def forward_pass(params, x):
	"""
	執行多層神經網路的前向傳播。

	Args:
		params: 網路的權重和偏置列表。
		x: 輸入資料。

	Returns:
		網路的最終輸出。
	"""
	activations = x
	# 遍歷所有層，除了最後一層 (輸出層)
	for i, layer_param in enumerate(params[:-1]):
		weights = layer_param['weights']
		biases = layer_param['biases']
		# 線性變換: activations @ weights + biases
		linear = complex_add(complex_matmul(activations, weights), biases)
		# 應用 激活函數
		activations = complex_sigmoid(linear)

	# 處理最後一層 (輸出層)，通常不應用激活函數
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

@jit
def update_step(params, opt_state, x, y):
	loss, grads = value_and_grad(loss_fn)(params, x, y)
	# JAX returns the conjugate gradient for complex parameters, so we conjugate it back
	grads = jax.tree_util.tree_map(jnp.conj, grads)
	updates, next_opt_state = optimizer.update(grads, opt_state, params)
	next_params = optax.apply_updates(params, updates)
	return next_params, next_opt_state, loss

key = jax.random.PRNGKey(44)

if 1:
	xor_layer_sizes = [2, 2, 2, 1]
	print(f"XOR network layer sizes: {xor_layer_sizes}")

	xor_input_data_theta = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
	xor_input_data = from_polar(jnp.ones_like(xor_input_data_theta), xor_input_data_theta)

	xor_labels_theta = jnp.array([[1, ], [0, ], [0, ], [1, ]], dtype=jnp.float32)
	xor_labels = from_polar(jnp.ones_like(xor_labels_theta), xor_labels_theta)

	# 3. Print the shape and content
	print("XOR Input Data Shape:", xor_input_data_theta.shape, xor_input_data.shape)
	print("XOR Input Data Theta:\n", xor_input_data_theta)
	print("XOR Input Data:\n", jnp.stack(to_polar(xor_input_data), axis=-1))
	print("\nXOR Labels  Shape:", xor_labels_theta.shape, xor_labels.shape)
	print("XOR Labels Data Theta:\n", xor_labels_theta)
	print("XOR Labels :\n", jnp.stack(to_polar(xor_labels), axis=-1))

	# 1. Define training hyperparameters
	learning_rate = 0.5
	num_epochs = 300

	params = init_params(key, xor_layer_sizes)
	print("\nXOR MLP Parameters Structure:")
	for i, layer_param in enumerate(params):
		print(f"  Layer {i+1}:")
		print(f"    Weights Shape: {layer_param['weights'].shape}")
		print(f"    Biases Shape: {layer_param['biases'].shape}")

	optimizer = optax.adam(learning_rate)
	opt_state = optimizer.init(params)

	print("Training...")
	for step in range(num_epochs):
		params, opt_state, loss = update_step(
			params,
			opt_state,
			xor_input_data,
			xor_labels)

		if step % 100 == 0:
			print(f"Step {step}, Loss: {loss:.6f}")

	print("Evaluating...");
	xor_logits = forward_pass(params, xor_input_data)
	xor_logits_r, xor_logits_theta = to_polar(xor_logits);

	with jnp.printoptions(precision=4, suppress=True):
		print("xor_logits_r:\n", xor_logits_r);
		print("xor_logits_theta:\n", xor_logits_theta);

import jax
import jax.numpy as jnp
import optax
from cvnn_v1 import complex_sigmoid as sigmoid_v1, from_polar as from_polar_v1, to_polar as to_polar_v1, complex_matmul as matmul_v1, complex_add as add_v1
from cvnn_v2 import complex_sigmoid as sigmoid_v2, from_polar as from_polar_v2, to_polar as to_polar_v2, complex_matmul as matmul_v2, complex_add as add_v2

def init_params_v1(key, layer_sizes):
    params = []
    for i in range(len(layer_sizes) - 1):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i+1]
        key, w_real_key, w_imag_key = jax.random.split(key, 3)
        stddev = jnp.sqrt(2. / (in_dim + out_dim))
        weights_real = jax.random.normal(w_real_key, (in_dim, out_dim)) * stddev
        weights_imag = jax.random.normal(w_imag_key, (in_dim, out_dim)) * stddev
        weights = jnp.stack([weights_real, weights_imag], axis=-1)
        biases = jnp.zeros((out_dim, 2))
        params.append({'weights': weights, 'biases': biases})
    return params

def forward_v1(params, x):
    activations = x
    for i, layer_param in enumerate(params[:-1]):
        weights = layer_param['weights']
        biases = layer_param['biases']
        linear = add_v1(matmul_v1(activations, weights), biases)
        activations = sigmoid_v1(linear)
    final_layer_param = params[-1]
    weights = final_layer_param['weights']
    biases = final_layer_param['biases']
    output = add_v1(matmul_v1(activations, weights), biases)
    return output

def loss_v1(params, x, y):
    y_pred = forward_v1(params, x)
    error = y_pred - y
    loss = jnp.mean(jnp.sum(error**2, axis=-1))
    return loss

def init_params_v2(key, layer_sizes):
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

def forward_v2(params, x):
    activations = x
    for i, layer_param in enumerate(params[:-1]):
        weights = layer_param['weights']
        biases = layer_param['biases']
        linear = add_v2(matmul_v2(activations, weights), biases)
        activations = sigmoid_v2(linear)
    final_layer_param = params[-1]
    weights = final_layer_param['weights']
    biases = final_layer_param['biases']
    output = add_v2(matmul_v2(activations, weights), biases)
    return output

def loss_v2(params, x, y):
    y_pred = forward_v2(params, x)
    error = y_pred - y
    loss = jnp.mean(jnp.abs(error)**2)
    return loss

key = jax.random.PRNGKey(44)
layer_sizes = [2, 2, 2, 1]

# Inputs
xor_input_data_theta = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
x_v1 = from_polar_v1(jnp.ones_like(xor_input_data_theta), xor_input_data_theta)
x_v2 = from_polar_v2(jnp.ones_like(xor_input_data_theta), xor_input_data_theta)

xor_labels_theta = jnp.array([[1, ], [0, ], [0, ], [1, ]], dtype=jnp.float32)
y_v1 = from_polar_v1(jnp.ones_like(xor_labels_theta), xor_labels_theta)
y_v2 = from_polar_v2(jnp.ones_like(xor_labels_theta), xor_labels_theta)

# Params
params_v1 = init_params_v1(key, layer_sizes)
params_v2 = init_params_v2(key, layer_sizes)

# Gradients
loss_v1_val, grads_v1 = jax.value_and_grad(loss_v1)(params_v1, x_v1, y_v1)
loss_v2_val, grads_v2 = jax.value_and_grad(loss_v2)(params_v2, x_v2, y_v2)

# Updates with SGD
lr = 0.5
opt_sgd_v1 = optax.sgd(lr)
opt_sgd_v2 = optax.sgd(lr)

state_sgd_v1 = opt_sgd_v1.init(params_v1)
state_sgd_v2 = opt_sgd_v2.init(params_v2)

grads_v2_conj = jax.tree_util.tree_map(jnp.conj, grads_v2)

upd_sgd_v1, _ = opt_sgd_v1.update(grads_v1, state_sgd_v1, params_v1)
upd_sgd_v2, _ = opt_sgd_v2.update(grads_v2_conj, state_sgd_v2, params_v2)

print(f"SGD Update V1 (First Layer Weights) Real:\n{upd_sgd_v1[0]['weights'][..., 0]}")
print(f"SGD Update V2 (First Layer Weights) Real:\n{upd_sgd_v2[0]['weights'].real}")

p_sgd_v1_next = optax.apply_updates(params_v1, upd_sgd_v1)
p_sgd_v2_next = optax.apply_updates(params_v2, upd_sgd_v2)

print(f"\nLoss V1 SGD after 1 step: {loss_v1(p_sgd_v1_next, x_v1, y_v1)}")
print(f"Loss V2 SGD after 1 step: {loss_v2(p_sgd_v2_next, x_v2, y_v2)}")

# Compare Adam again to be sure
opt_adam_v1 = optax.adam(lr)
opt_adam_v2 = optax.adam(lr)
state_adam_v1 = opt_adam_v1.init(params_v1)
state_adam_v2 = opt_adam_v2.init(params_v2)
upd_adam_v1, _ = opt_adam_v1.update(grads_v1, state_adam_v1, params_v1)
upd_adam_v2, _ = opt_adam_v2.update(grads_v2_conj, state_adam_v2, params_v2)

print(f"\nAdam Update V1 Real:\n{upd_adam_v1[0]['weights'][..., 0]}")
print(f"Adam Update V2 Real:\n{upd_adam_v2[0]['weights'].real}")

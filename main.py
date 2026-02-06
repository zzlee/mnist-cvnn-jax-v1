import jax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds
from cvnn_v1 import *

# 1. 超參數設定
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_CLASSES = 10

# 2. 資料編碼 (Phase Encoding, xor_cvnn_v1 風格)
def encode_data(images):
	"""編碼器：像素亮度 [0, 1] -> 相位 [0, pi], 振幅固定為 1"""
	# images 形狀為 (batch, 28, 28, 1)
	images = images.astype(jnp.float32) / 255.0
	theta = images * jnp.pi
	r = jnp.ones_like(images)
	# 返回形狀為 (batch, 28, 28, 1, 2) 的實部、虛部堆疊張量
	return from_polar(r, theta)

# 3. 權重初始化 (Glorot Normal, xor_cvnn_v1 風格)
def init_params(rng):
	k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 6)

	# 卷積層: [3, 3, 1, 16, 2]
	# 使用 Glorot 初始化，標準差為 sqrt(2 / (fan_in + fan_out))
	fan_in_conv = 3 * 3 * 1
	fan_out_conv = 16
	std_conv = jnp.sqrt(2. / (fan_in_conv + fan_out_conv))
	
	w_conv_real = jax.random.normal(k1, (3, 3, 1, 16)) * std_conv
	w_conv_imag = jax.random.normal(k2, (3, 3, 1, 16)) * std_conv
	w_conv = jnp.stack([w_conv_real, w_conv_imag], axis=-1)
	b_conv = jnp.zeros((16, 2))

	# 全連接層: [12544, 10, 2] (28x28x16 = 12544)
	fan_in_dense = 28 * 28 * 16
	fan_out_dense = NUM_CLASSES
	std_dense = jnp.sqrt(2. / (fan_in_dense + fan_out_dense))
	
	w_dense_real = jax.random.normal(k3, (fan_in_dense, NUM_CLASSES)) * std_dense
	w_dense_imag = jax.random.normal(k4, (fan_in_dense, NUM_CLASSES)) * std_dense
	w_dense = jnp.stack([w_dense_real, w_dense_imag], axis=-1)
	b_dense = jnp.zeros((NUM_CLASSES, 2))

	return {"w_conv": w_conv, "b_conv": b_conv, "w_dense": w_dense, "b_dense": b_dense}

# 4. 前向傳播 (使用 cvnn_v1 元件)
@jax.jit
def model_forward(params, x_complex):
	# 複數卷積
	z_conv = complex_conv2d(x_complex, params["w_conv"], (1, 1), 'SAME')
	# 加入偏置
	z_conv = complex_add(z_conv, params["b_conv"])
	# 複數激活函數
	a_conv = complex_sigmoid(z_conv)

	# Flatten
	# a_conv 形狀: (batch, 28, 28, 16, 2) -> (batch, 12544, 2)
	flat = a_conv.reshape((a_conv.shape[0], -1, 2))

	# 複數全連接
	z_out = complex_matmul(flat, params["w_dense"])
	z_out = complex_add(z_out, params["b_dense"])
	
	# 解碼：觀察 10 個類別神經元的振幅 (信心度)
	# 使用 cvnn_v1 的 complex_abs
	return complex_abs(z_out)

# 5. 損失函數與更新步
def loss_fn(params, x_complex, labels):
	magnitudes = model_forward(params, x_complex)
	# 使用 Log-Softmax 在振幅上進行交叉熵計算
	logits = jax.nn.log_softmax(magnitudes)
	one_hot = jax.nn.one_hot(labels, NUM_CLASSES)
	return -jnp.mean(jnp.sum(one_hot * logits, axis=-1))

@jax.jit
def update_step(params, opt_state, x_complex, labels):
	loss, grads = jax.value_and_grad(loss_fn)(params, x_complex, labels)
	updates, next_opt_state = optimizer.update(grads, opt_state, params)
	next_params = optax.apply_updates(params, updates)
	return next_params, next_opt_state, loss

# 6. 資料準備 (使用 TFDS)
def get_mnist_data():
	ds = tfds.load('mnist', split='train', as_supervised=True).batch(BATCH_SIZE)
	return tfds.as_numpy(ds)

# 7. 主訓練循環
if __name__ == "__main__":
	rng = jax.random.PRNGKey(42)
	params = init_params(rng)

	optimizer = optax.adam(LEARNING_RATE)
	opt_state = optimizer.init(params)

	print("開始訓練複數網路 (cvnn_v1 風格: Real-Split, Phase as Data)...")

	for epoch in range(EPOCHS):
		total_loss = 0
		count = 0
		for images, labels in get_mnist_data():
			# 轉換為複數格式 (實部、虛部堆疊)
			x_complex = encode_data(images)

			# 更新權重
			params, opt_state, loss = update_step(params, opt_state, x_complex, labels)

			total_loss += loss
			count += 1
			if count % 200 == 0:
				print(f"Batch {count}, Loss: {loss:.4f}")

		print(f"Epoch {epoch+1}, Average Loss: {total_loss/count:.4f}")

	# 測試準確度
	print("\n進行測試集評估...")
	test_ds = tfds.load('mnist', split='test', as_supervised=True).batch(1000)
	test_batch = next(iter(tfds.as_numpy(test_ds)))
	test_images, test_labels = test_batch

	magnitudes = model_forward(params, encode_data(test_images))
	accuracy = jnp.mean(jnp.argmax(magnitudes, axis=-1) == test_labels)
	print(f"測試集準確度: {accuracy*100:.2f}%")

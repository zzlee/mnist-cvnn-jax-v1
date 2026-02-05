import jax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds
import numpy as np
from jax import lax, jit, value_and_grad

# 1. 超參數設定
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_CLASSES = 10

# 2. 核心功能定義
def complex_sigmoid(z):
    """自定義激活函數：保持相位 theta，壓縮振幅 r"""
    r = jnp.abs(z)
    theta = jnp.angle(z)
    # 使用位移型 Sigmoid 確保 r=0 時輸出為 0 (信心度平滑門檻)
    r_activated = 2 / (1 + jnp.exp(-r)) - 1
    return r_activated * jnp.exp(1j * theta)

def encode_data(images):
    """編碼器：像素亮度 [0, 1] -> 相位 [0, pi]"""
    images = images.astype(jnp.float32) / 255.0
    theta = images * jnp.pi
    r = jnp.ones_like(images) # 初始觀測信心度設為 1
    return r * jnp.exp(1j * theta)

# 3. 權重初始化 (Complex Xavier)
def init_params(rng):
    k1, k2 = jax.random.split(rng)
    
    # 卷積核: [3, 3, 1, 16] (3x3 kernel, 1 input, 16 filters)
    std_conv = jnp.sqrt(1.0 / (3 * 3 * 1))
    w_conv = (jax.random.normal(k1, (3, 3, 1, 16)) + 
              1j * jax.random.normal(k2, (3, 3, 1, 16))) * std_conv
    
    # 全連接層: [12544, 10] (28x28x16 = 12544)
    # 注意：MNIST 'SAME' padding 後大小不變
    k3, k4 = jax.random.split(k1)
    fan_in_dense = 28 * 28 * 16
    std_dense = jnp.sqrt(1.0 / fan_in_dense)
    w_dense = (jax.random.normal(k3, (fan_in_dense, NUM_CLASSES)) + 
               1j * jax.random.normal(k4, (fan_in_dense, NUM_CLASSES))) * std_dense
    
    return {"w_conv": w_conv, "w_dense": w_dense}

# 4. 前向傳播
def model_forward(params, x_complex):
    # 複數卷積
    z_conv = lax.conv_general_dilated(
        x_complex, params["w_conv"], (1, 1), 'SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    a_conv = complex_sigmoid(z_conv)
    
    # Flatten
    flat = a_conv.reshape((a_conv.shape[0], -1))
    
    # 複數全連接
    z_out = jnp.dot(flat, params["w_dense"])
    
    # 輸出：觀察 10 個類別神經元的振幅 (信心度)
    # 如果某個數字對齊（相長干涉）得最好，其振幅會最大
    return jnp.abs(z_out)

# 5. 損失函數與更新步
def loss_fn(params, x_complex, labels):
    magnitudes = model_forward(params, x_complex)
    # 使用 Log-Softmax 在振幅上，這本質上是在尋找「信心度分布」的交叉熵
    logits = jax.nn.log_softmax(magnitudes)
    one_hot = jax.nn.one_hot(labels, NUM_CLASSES)
    return -jnp.mean(jnp.sum(one_hot * logits, axis=-1))

@jit
def update_step(params, opt_state, x_complex, labels):
    loss, grads = value_and_grad(loss_fn)(params, x_complex, labels)
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
    
    print("開始訓練複數網路 (Phase as Data, Magnitude as Confidence)...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        count = 0
        for images, labels in get_mnist_data():
            # 轉換為複數格式
            x_complex = encode_data(images)
            
            # 更新權重
            params, opt_state, loss = update_step(params, opt_state, x_complex, labels)
            
            total_loss += loss
            count += 1
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/count:.4f}")

    # 簡單測試準確度
    test_ds = tfds.load('mnist', split='test', as_supervised=True).batch(1000)
    test_batch = next(iter(tfds.as_numpy(test_ds)))
    test_images, test_labels = test_batch
    
    preds = model_forward(params, encode_data(test_images))
    accuracy = jnp.mean(jnp.argmax(preds, axis=-1) == test_labels)
    print(f"測試集準確度: {accuracy*100:.2f}%")
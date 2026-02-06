import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def complex_add(z1, z2):
    return z1 + z2

@jax.jit
def complex_sub(z1, z2):
    return z1 - z2

@jax.jit
def complex_mul(z1, z2):
    return z1 * z2

@jax.jit
def complex_div(z1, z2):
    return z1 / z2

@jax.jit
def complex_conj(z):
    return jnp.conj(z)

@jax.jit
def complex_abs(z):
    return jnp.abs(z)

@jax.jit
def complex_exp(z):
    return jnp.exp(z)

@jax.jit
def to_polar(z):
    """轉換為 (模長 r, 輻角 theta)"""
    return jnp.abs(z), jnp.angle(z)

@jax.jit
def from_polar(r, theta):
    """由 (r, theta) 轉回複數"""
    return r * jnp.exp(1j * theta)

def mse_loss(params, forward, x, y_true):
    # y_pred and y_true are now complex
    y_pred = forward(params, x)

    # 計算 |y_pred - y_true|^2 的平均值
    error = y_pred - y_true
    loss = jnp.mean(jnp.abs(error)**2)

    return loss

@jax.jit
def complex_matmul(A, B):
    """
    實作複數矩陣相乘
    A shape: (..., M, K)
    B shape: (..., K, N)
    Return shape: (..., M, N)
    """
    return jnp.matmul(A, B)

@jax.jit
def complex_sigmoid(z):
    r, theta = to_polar(z)
    # 使用 jax.nn.sigmoid 對振幅進行啟動
    r_activated = jax.nn.sigmoid(r)

    return from_polar(r_activated, theta)

def complex_conv1d(x, kernel, strides, padding):
    """
    實作複數 Conv1D 運算。

    Args:
        x: 輸入資料，形狀為 (batch, length, in_channels)。
        kernel: 卷積核，形狀為 (kernel_size, in_channels, out_channels)。
        strides: 步長，整數或長度為 1 的元組 (e.g., (1,)).
        padding: 填充方式，例如 'SAME' 或 'VALID'。

    Returns:
        卷積結果，形狀為 (batch, output_length, out_channels)。
    """
    # 直接使用 lax.conv_general_dilated，它支援複數
    return lax.conv_general_dilated(
        lhs=x,
        rhs=kernel,
        window_strides=strides,
        padding=padding,
        dimension_numbers=('NWC', 'WIO', 'NWC')
    )
complex_conv1d = jax.jit(complex_conv1d, static_argnums=(2, 3))

def complex_conv2d(x, kernel, strides, padding):
    """
    實作複數 Conv2D 運算。

    Args:
        x: 輸入資料，形狀為 (batch, height, width, in_channels)。
        kernel: 卷積核，形狀為 (kernel_height, kernel_width, in_channels, out_channels)。
        strides: 步長，長度為 2 的元組 (stride_h, stride_w)。
        padding: 填充方式，例如 'SAME' 或 'VALID'。

    Returns:
        卷積結果，形狀為 (batch, output_height, output_width, out_channels)。
    """
    # 直接使用 lax.conv_general_dilated，它支援複數
    return lax.conv_general_dilated(
        lhs=x,
        rhs=kernel,
        window_strides=strides,
        padding=padding,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
complex_conv2d = jax.jit(complex_conv2d, static_argnums=(2, 3))
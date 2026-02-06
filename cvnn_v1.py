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
    a, b = z1[..., 0], z1[..., 1]
    c, d = z2[..., 0], z2[..., 1]
    real = a * c - b * d
    imag = a * d + b * c
    return jnp.stack([real, imag], axis=-1)

@jax.jit
def complex_div(z1, z2):
    a, b = z1[..., 0], z1[..., 1]
    c, d = z2[..., 0], z2[..., 1]
    denom = c**2 + d**2
    real = (a * c + b * d) / denom
    imag = (b * c - a * d) / denom
    return jnp.stack([real, imag], axis=-1)

@jax.jit
def complex_conj(z):
    mask = jnp.array([1.0, -1.0])
    return z * mask

@jax.jit
def complex_abs(z):
    return jnp.sqrt(jnp.sum(z**2, axis=-1))

@jax.jit
def complex_exp(z):
    a, b = z[..., 0], z[..., 1]
    exp_a = jnp.exp(a)
    real = exp_a * jnp.cos(b)
    imag = exp_a * jnp.sin(b)
    return jnp.stack([real, imag], axis=-1)

@jax.jit
def to_polar(z):
    """轉換為 (模長 r, 輻角 theta)"""
    r = complex_abs(z)
    theta = jnp.arctan2(z[..., 1], z[..., 0])
    return r, theta

@jax.jit
def from_polar(r, theta):
    """由 (r, theta) 轉回直角座標 [real, imag]"""
    real = r * jnp.cos(theta)
    imag = r * jnp.sin(theta)
    return jnp.stack([real, imag], axis=-1)

def mse_loss(params, forward, x, y_true):
    # 假設 forward 是你定義的前向傳播函數
    y_pred = forward(params, x)

    # 計算 (實部差^2 + 虛部差^2)
    error = y_pred - y_true
    loss = jnp.mean(jnp.sum(error**2, axis=-1))

    return loss

@jax.jit
def complex_matmul(A, B):
    """
    實作複數矩陣相乘
    A shape: (M, K, 2)
    B shape: (K, N, 2)
    Return shape: (M, N, 2)
    """
    # 拆解實部與虛部
    Ra, Ia = A[..., 0], A[..., 1]
    Rb, Ib = B[..., 0], B[..., 1]

    # 根據公式計算：
    # 實部 = Ra*Rb - Ia*Ib
    real_part = jnp.matmul(Ra, Rb) - jnp.matmul(Ia, Ib)

    # 虛部 = Ra*Ib + Ia*Rb
    imag_part = jnp.matmul(Ra, Ib) + jnp.matmul(Ia, Rb)

    return jnp.stack([real_part, imag_part], axis=-1)

@jax.jit
def complex_sigmoid(z):
    r, theta = to_polar(z)
    r_activated = 1 / (1 + jnp.exp(-r))

    return from_polar(r_activated, theta)

def complex_conv1d(x, kernel, strides, padding):
    """
    實作複數 Conv1D 運算。

    Args:
        x: 輸入資料，形狀為 (batch, length, in_channels, 2)。
           最後一個維度 2 分別代表複數的實部和虛部。
        kernel: 卷積核，形狀為 (kernel_size, in_channels, out_channels, 2)。
                最後一個維度 2 分別代表複數的實部和虛部。
        strides: 步長，整數或長度為 1 的元組 (e.g., (1,)).
        padding: 填充方式，例如 'SAME' 或 'VALID'。

    Returns:
        卷積結果，形狀為 (batch, output_length, out_channels, 2)。
    """
    # 拆解實部與虛部
    x_r, x_i = x[..., 0], x[..., 1]
    k_r, k_i = kernel[..., 0], kernel[..., 1]

    # JAX conv_general_dilated 預設維度順序為 (N, C, spatial_dims) 對於 lhs (input)
    # 和 (O, I, spatial_kernel_dims) 對於 rhs (kernel)。
    # 我們的輸入 x 是 (batch, length, in_channels)，kernel 是 (kernel_size, in_channels, out_channels)。
    # 需要轉換維度。

    # 輸入轉換: (batch, length, in_channels) -> (batch, in_channels, length)
    x_r_trans = jnp.transpose(x_r, (0, 2, 1))
    x_i_trans = jnp.transpose(x_i, (0, 2, 1))

    # 卷積核轉換: (kernel_size, in_channels, out_channels) -> (out_channels, in_channels, kernel_size)
    k_r_trans = jnp.transpose(k_r, (2, 1, 0))
    k_i_trans = jnp.transpose(k_i, (2, 1, 0))

    # 執行四個實數卷積
    # 輸出實部 = conv(x_r, k_r) - conv(x_i, k_i)
    real_real = lax.conv_general_dilated(
        lhs=x_r_trans,
        rhs=k_r_trans,
        window_strides=strides,
        padding=padding,
        dimension_numbers=('NCW', 'OIW', 'NCW') # Input: N batch, C channel, W width. Kernel: O output, I input, W width.
    )
    imag_imag = lax.conv_general_dilated(
        lhs=x_i_trans,
        rhs=k_i_trans,
        window_strides=strides,
        padding=padding,
        dimension_numbers=('NCW', 'OIW', 'NCW')
    )
    real_out = real_real - imag_imag

    # 輸出虛部 = conv(x_r, k_i) + conv(x_i, k_r)
    real_imag = lax.conv_general_dilated(
        lhs=x_r_trans,
        rhs=k_i_trans,
        window_strides=strides,
        padding=padding,
        dimension_numbers=('NCW', 'OIW', 'NCW')
    )
    imag_real = lax.conv_general_dilated(
        lhs=x_i_trans,
        rhs=k_r_trans,
        window_strides=strides,
        padding=padding,
        dimension_numbers=('NCW', 'OIW', 'NCW')
    )
    imag_out = real_imag + imag_real

    # 轉置回 (batch, output_length, out_channels) 格式
    real_out_trans = jnp.transpose(real_out, (0, 2, 1))
    imag_out_trans = jnp.transpose(imag_out, (0, 2, 1))

    return jnp.stack([real_out_trans, imag_out_trans], axis=-1)
complex_conv1d = jax.jit(complex_conv1d, static_argnums=(2, 3))

def complex_conv2d(x, kernel, strides, padding):
    """
    實作複數 Conv2D 運算。

    Args:
        x: 輸入資料，形狀為 (batch, height, width, in_channels, 2)。
           最後一個維度 2 分別代表複數的實部和虛部。
        kernel: 卷積核，形狀為 (kernel_height, kernel_width, in_channels, out_channels, 2)。
                最後一個維度 2 分別代表複數的實部和虛部。
        strides: 步長，長度為 2 的元組 (stride_h, stride_w)。
        padding: 填充方式，例如 'SAME' 或 'VALID'。

    Returns:
        卷積結果，形狀為 (batch, output_height, output_width, out_channels, 2)。
    """
    # 拆解實部與虛部
    x_r, x_i = x[..., 0], x[..., 1]
    k_r, k_i = kernel[..., 0], kernel[..., 1]

    # JAX conv_general_dilated 預設維度順序為 (N, C, H, W) 對於 lhs (input)
    # 和 (O, I, H, W) 對於 rhs (kernel)。
    # 我們的輸入 x 是 (batch, H, W, in_channels)，kernel 是 (kH, kW, in_channels, out_channels)。
    # 需要轉換維度。

    # 輸入轉換: (batch, H, W, in_channels) -> (batch, in_channels, H, W)
    x_r_trans = jnp.transpose(x_r, (0, 3, 1, 2))
    x_i_trans = jnp.transpose(x_i, (0, 3, 1, 2))

    # 卷積核轉換: (kH, kW, in_channels, out_channels) -> (out_channels, in_channels, kH, kW)
    k_r_trans = jnp.transpose(k_r, (3, 2, 0, 1))
    k_i_trans = jnp.transpose(k_i, (3, 2, 0, 1))

    # 執行四個實數卷積
    # 輸出實部 = conv(x_r, k_r) - conv(x_i, k_i)
    real_real = lax.conv_general_dilated(
        lhs=x_r_trans,
        rhs=k_r_trans,
        window_strides=strides,
        padding=padding,
        dimension_numbers=('NCHW', 'OIHW', 'NCHW') # Input: N batch, C channel, H height, W width. Kernel: O output, I input, H height, W width.
    )
    imag_imag = lax.conv_general_dilated(
        lhs=x_i_trans,
        rhs=k_i_trans,
        window_strides=strides,
        padding=padding,
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
    )
    real_out = real_real - imag_imag

    # 輸出虛部 = conv(x_r, k_i) + conv(x_i, k_r)
    real_imag = lax.conv_general_dilated(
        lhs=x_r_trans,
        rhs=k_i_trans,
        window_strides=strides,
        padding=padding,
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
    )
    imag_real = lax.conv_general_dilated(
        lhs=x_i_trans,
        rhs=k_r_trans,
        window_strides=strides,
        padding=padding,
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
    )
    imag_out = real_imag + imag_real

    # 轉置回 (batch, output_height, output_width, out_channels) 格式
    real_out_trans = jnp.transpose(real_out, (0, 2, 3, 1))
    imag_out_trans = jnp.transpose(imag_out, (0, 2, 3, 1))

    return jnp.stack([real_out_trans, imag_out_trans], axis=-1)
complex_conv2d = jax.jit(complex_conv2d, static_argnums=(2, 3))

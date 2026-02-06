import jax
import jax.numpy as jnp
import cvnn_v2 as cvnn
import numpy as np

def test_complex_ops():
	z1 = 1.0 + 2.0j
	z2 = 3.0 - 4.0j

	# Basic ops
	assert cvnn.complex_add(z1, z2) == z1 + z2
	assert cvnn.complex_sub(z1, z2) == z1 - z2
	assert cvnn.complex_mul(z1, z2) == z1 * z2
	assert cvnn.complex_div(z1, z2) == z1 / z2
	assert cvnn.complex_conj(z1) == jnp.conj(z1)
	assert cvnn.complex_abs(z1) == jnp.abs(z1)
	assert cvnn.complex_exp(z1) == jnp.exp(z1)

	# Polar
	r, theta = cvnn.to_polar(z1)
	assert jnp.allclose(r, jnp.abs(z1))
	assert jnp.allclose(theta, jnp.angle(z1))
	assert jnp.allclose(cvnn.from_polar(r, theta), z1)

	print("Basic complex operations: PASSED")

def test_matmul():
	A = jnp.array([[1+1j, 2+2j], [3+3j, 4+4j]])
	B = jnp.array([[5+5j, 6+6j], [7+7j, 8+8j]])
	C = cvnn.complex_matmul(A, B)
	expected = jnp.matmul(A, B)
	assert jnp.allclose(C, expected)
	print("Complex matmul: PASSED")

def test_sigmoid():
	z = 1.0 + 1.0j
	res = cvnn.complex_sigmoid(z)
	r = jnp.abs(z)
	theta = jnp.angle(z)
	expected = jax.nn.sigmoid(r) * jnp.exp(1j * theta)
	assert jnp.allclose(res, expected)
	print("Complex sigmoid (polar): PASSED")

def test_conv():
	# 2D Conv test
	x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 8, 3)) + \
		1j * jax.random.normal(jax.random.PRNGKey(1), (2, 8, 8, 3))
	kernel = jax.random.normal(jax.random.PRNGKey(2), (3, 3, 3, 4)) + \
		1j * jax.random.normal(jax.random.PRNGKey(3), (3, 3, 3, 4))

	res = cvnn.complex_conv2d(x, kernel, (1, 1), 'SAME')
	assert res.shape == (2, 8, 8, 4)
	assert res.dtype == jnp.complex64
	print("Complex Conv2D: PASSED")

	# 1D Conv test
	x1d = jax.random.normal(jax.random.PRNGKey(4), (2, 10, 3)) + \
		1j * jax.random.normal(jax.random.PRNGKey(5), (2, 10, 3))
	kernel1d = jax.random.normal(jax.random.PRNGKey(6), (3, 3, 4)) + \
		1j * jax.random.normal(jax.random.PRNGKey(7), (3, 3, 4))

	res1d = cvnn.complex_conv1d(x1d, kernel1d, (1,), 'SAME')
	assert res1d.shape == (2, 10, 4)
	assert res1d.dtype == jnp.complex64
	print("Complex Conv1D: PASSED")

if __name__ == "__main__":
	test_complex_ops()
	test_matmul()
	test_sigmoid()
	test_conv()
	print("All tests passed!")

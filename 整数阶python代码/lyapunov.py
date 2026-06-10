import numpy as np
import matplotlib.pyplot as plt


# 1. 定义四维超混沌系统的导数方程
def system_derivatives(state, a, b, c, d):
    x, y, z, w = state
    dx = a * (y - x) + w
    dy = -x * z + c * y
    dz = x * y - b * z
    dw = x * z + d * w
    return np.array([dx, dy, dz, dw])


# 2. 定义雅可比矩阵
def jacobian_matrix(state, a, b, c, d):
    x, y, z, w = state
    return np.array([
        [-a, a, 0, 1],
        [-z, c, -x, 0],
        [y, x, -b, 0],
        [z, 0, x, d]
    ])


# 3. 第四阶龙格-库塔法 (RK4) 实现
def rk4_step(state, dt, a, b, c, d):
    k1 = system_derivatives(state, a, b, c, d)
    k2 = system_derivatives(state + 0.5 * dt * k1, a, b, c, d)
    k3 = system_derivatives(state + 0.5 * dt * k2, a, b, c, d)
    k4 = system_derivatives(state + dt * k3, a, b, c, d)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# 4. 计算李雅普诺夫指数谱
def calculate_lyapunov_spectrum(d_val):
    a, b, c = 36, 3, 20
    dt = 0.005  # 减小步长以提高精度
    steps = 10000

    state = np.array([1.0, 1.0, 1.0, 1.0])
    Q = np.eye(4)
    lexp = np.zeros(4)

    # 丢弃瞬态
    for _ in range(2000):
        state = rk4_step(state, dt, a, b, c, d_val)

    # 主演化循环
    for _ in range(steps):
        # 演化状态
        state = rk4_step(state, dt, a, b, c, d_val)

        # 演化切向量 (Linearized map)
        J = jacobian_matrix(state, a, b, c, d_val)
        M = np.eye(4) + J * dt
        Q = np.dot(M, Q)

        # QR 分解保持正交化并提取增长率
        Q, R = np.linalg.qr(Q)
        lexp += np.log(np.abs(np.diag(R)))

    return lexp / (steps * dt)


# 5. 执行参数扫描
print("正在计算李雅普诺夫指数谱，请稍候...")
d_range = np.linspace(-2.0, 1.5, 50)
results = []

for d in d_range:
    res = calculate_lyapunov_spectrum(d)
    results.append(res)
results = np.array(results)

# 6. 绘图
plt.figure(figsize=(12, 7))
colors = ['red', 'blue', 'green', 'orange']
labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$']

for i in range(4):
    plt.plot(d_range, results[:, i], label=labels[i], color=colors[i], lw=1.5)

plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
plt.title("Lyapunov Exponents Spectrum for 4D Hyperchaotic System", fontsize=14)
plt.xlabel("Scanning Parameter d", fontsize=12)
plt.ylabel("Lyapunov Exponents", fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, which='both', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('lyapunov.png', dpi=150, bbox_inches='tight')


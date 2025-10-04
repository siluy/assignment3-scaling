import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from collections import defaultdict

# 1. 定义并预加载数据
def power_law(C, a, b):
    return a * (C ** b)

with open('data/isoflops_curves.json', 'r') as f:
    runs = json.load(f)

# 2. 处理数据，寻找每个 budget 下最优值
runs_by_budget = defaultdict(list)
for run in runs:
    runs_by_budget[run['compute_budget']].append(run)
# 存每个预算 C_i 对应的最优模型大小和数据量
optimal_points = []
for budget, bedget_runs in sorted(runs_by_budget.items()):
    best_run = min(bedget_runs, key=lambda r: r['final_loss'])
    C_i = best_run['compute_budget']
    N_opt_i = best_run['parameters']
    D_opt_i = C_i / (6 * N_opt_i)
    optimal_points.append({'C': C_i, 'N_opt': N_opt_i, 'D_opt': D_opt_i})
# 将数据点分离成独立的 np 数组
C_vals = np.array([p['C'] for p in optimal_points])
N_opt_vals = np.array([p['N_opt'] for p in optimal_points])
D_opt_vals = np.array([p['D_opt'] for p in optimal_points])

# 3. 拟合模型大小N和数据量D
params_N, _ = curve_fit(power_law, C_vals, N_opt_vals)
a_N, b_N = params_N
print(f"N_opt: {a_N:.4e} * C^{b_N:.4f}")
params_D, _ = curve_fit(power_law, C_vals, D_opt_vals)
a_D, b_D = params_D
print(f"D_opt: {a_D:.4e} * C^{b_D:.4f}")

# 4. 外推预测
C_pred_23 = 1e23
N_pred_23 = power_law(C_pred_23, a_N, b_N)
D_pred_23 = power_law(C_pred_23, a_D, b_D)

C_pred_24 = 1e24
N_pred_24 = power_law(C_pred_24, a_N, b_N)
D_pred_24 = power_law(C_pred_24, a_D, b_D)
# 5. 绘图
C_plot = np.logspace(np.log10(C_vals.min()), 24, 100)
plt.figure(figsize=(10, 6))
plt.scatter(C_vals, N_opt_vals, label='Optimal Points from Data', color='red', zorder=5)
plt.plot(C_plot, power_law(C_plot, a_N, b_N), label=f'Fitted Law: $N \\propto C^{{{b_N:.3f}}}$')
plt.xscale('log')
plt.yscale('log')
plt.title('Scaling Law for Model Size (N) vs. Compute (C)')
plt.xlabel('Compute Budget (FLOPs)')
plt.ylabel('Optimal Model Parameters (N)')
plt.grid(True, which="both", ls="--")
plt.legend()
# plt.show()
plt.savefig('model_size_scaling_law.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(C_vals, D_opt_vals, label='Optimal Points from Data', color='blue', zorder=5)
plt.plot(C_plot, power_law(C_plot, a_D, b_D), label=f'Fitted Law: $D \\propto C^{{{b_D:.3f}}}$')
plt.xscale('log')
plt.yscale('log')
plt.title('Scaling Law for Dataset Size (D) vs. Compute (C)')
plt.xlabel('Compute Budget (FLOPs)')
plt.ylabel('Optimal Dataset Tokens (D)')
plt.grid(True, which="both", ls="--")
plt.legend()
# plt.show()
plt.savefig('dataset_size_scaling_law.png')
plt.close()

# 6. 输出最终预测结果
print("\n--- 最终预测结果 ---")
print(f"对于 1e23 FLOPs 预算，预测的最优模型大小约为 {N_pred_23:,.0f} 参数。")
print(f"对于 1e24 FLOPs 预算，预测的最优模型大小约为 {N_pred_24:,.0f} 参数。")
print(f"对于 1e23 FLOPs 预算，预测的最优数据集大小约为 {D_pred_23:,.0f} tokens。")
print(f"对于 1e24 FLOPs 预算，预测的最优数据集大小约为 {D_pred_24:,.0f} tokens。")
"""为什么两种方法的'退化中位数'分母不同 —— 用同一批数据演示。

退化中位数 = median(η | η<0)，而 {η<0} 这个集合由**该方法自己的预测**决定，
所以分母是方法相关的。本脚本给出"换用同一分母"的对照。
"""
import csv
import os
import statistics as st

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'data')


def load(name):
    with open(os.path.join(D, name), newline='', encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


rows = load('fig2_cdf.csv')
M = [float(r['eta_mse']) for r in rows]
S = [float(r['eta_soft']) for r in rows]
n = len(M)

setM = [k for k in range(n) if M[k] < 0]
setS = [k for k in range(n) if S[k] < 0]
both = [k for k in range(n) if M[k] < 0 and S[k] < 0]
onlyM = [k for k in range(n) if M[k] < 0 <= S[k]]     # 软掩码救回来
onlyS = [k for k in range(n) if S[k] < 0 <= M[k]]     # 软掩码新造成

print(f'总样本 n = {n}\n')
print('各方法**自己的**退化集合（= 论文口径的分母）：')
print(f'  标准MSE 的集合 |η_mse<0| = {len(setM):3d}   → 该集合内的 median(η_mse) = {st.median([M[k] for k in setM]):+7.2f}%')
print(f'  软掩码  的集合 |η_soft<0| = {len(setS):3d}   → 该集合内的 median(η_soft) = {st.median([S[k] for k in setS]):+7.2f}%')

print('\n【换用同一分母】在【软掩码的退化集合】(217 个) 上，两个方法各自的中位数：')
print(f'  median(η_soft | η_soft<0) = {st.median([S[k] for k in setS]):+7.2f}%')
print(f'  median(η_mse  | η_soft<0) = {st.median([M[k] for k in setS]):+7.2f}%   ← 同一批样本上，MSE 更差')

print('\n【换用同一分母】在【标准MSE 的退化集合】(282 个) 上：')
print(f'  median(η_mse  | η_mse<0) = {st.median([M[k] for k in setM]):+7.2f}%')
print(f'  median(η_soft | η_mse<0) = {st.median([S[k] for k in setM]):+7.2f}%   ← 同一批样本上，软掩码更好')

print(f'\n集合分解：')
print(f'  两者都退化   {len(both):3d}   中位: η_mse={st.median([M[k] for k in both]):+7.2f}%  η_soft={st.median([S[k] for k in both]):+7.2f}%')
print(f'  只有MSE退化  {len(onlyM):3d}   中位: η_mse={st.median([M[k] for k in onlyM]):+7.2f}%  η_soft={st.median([S[k] for k in onlyM]):+7.2f}%  ← 轻伤，被软掩码治好')
print(f'  只有软掩码退化 {len(onlyS):3d}   中位: η_mse={st.median([M[k] for k in onlyS]):+7.2f}%  η_soft={st.median([S[k] for k in onlyS]):+7.2f}%  ← 软掩码新造成的')
print()
print('一句话：软掩码的退化集合 = 183(共同的难样本) + 34(新造成)；')
print('        标准MSE 的退化集合 = 183(共同的难样本) + 99(轻伤样本)。')
print('        那 99 个轻伤样本（中位仅 -3.1%）把 MSE 的条件中位数往 0 拉，所以 MSE 看起来"更轻"。')

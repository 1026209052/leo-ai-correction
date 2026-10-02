"""collect_switch_a02.py -- **只用已训练好的模型**复算并打包 switch_a02_out.tar.gz（不训练）

用途：6 个 LSTM α=0.2 权重都已训好，但编排没走到 C/D（没生成 tar）时，用它补出结果包。

它做的事：
  1) 定位工作目录（含 eval_alpha_sweep.py 的目录，即 starlink_fixed/）
  2) 自检：依赖脚本在、含 SM_ALPHA / soft02 支持；6 个 LSTM α 权重存在且日志有收尾标记
  3) A 阶段（可跳过）：eval_alpha_sweep + 两条 boot_alpha_pairs（α vs 1.0 / vs 0.3）
  4) C 阶段：SM_ALPHA=α 复算 表5/表6/表8/表10/表11、图2 数据、§5.7
  5) D 阶段：收集 CSV + 打 switch_a02_out.tar.gz

用法：
    cd /u01/yk/satellite/starlink_fixed
    python collect_switch_a02.py              # 严格模式（推荐）
    python collect_switch_a02.py --yes        # 若有种子查不到日志标记，仍继续（会打印 mtime 供核对）
    python collect_switch_a02.py --skip-a     # 跳过 A 阶段（Transformer 侧已经跑过）
"""
import os
import glob
import shutil
import subprocess
import sys
import tarfile
import time

SEEDS = ['42', '123', '789', '101', '202', '303']
STAGE_A = [('eval_alpha_sweep.py', '10_alpha_sweep.txt')]
STAGE_C = [('verify_all_tables.py', '30_verify_all_tables.txt'),
           ('export_eta.py', '31_export_eta.txt'),
           ('export_eta_lstm.py', '32_export_eta_lstm.txt'),
           ('export_eta_perseed.py', '33_export_eta_perseed.txt'),
           ('validate_with_space_v5.py', '34_space_v5.txt'),
           ('c2_cluster_bootstrap.py', '35_cluster_bootstrap.txt')]
CSVS = ['verify_table4_inner_val_ckpt.csv', 'fig2_cdf.csv', 'fig2_cdf_all.csv',
        'per_sample_eta_lstm.csv', 'per_seed_metrics.csv', 'spacex_v5_verification.csv',
        'eval_out/alpha_sweep_metrics_inner_val_ckpt.csv']
NEED = ['eval_alpha_sweep.py', 'boot_alpha_pairs.py', 'verify_all_tables.py',
        'validate_with_space_v5.py', 'export_eta.py', 'export_eta_lstm.py',
        'export_eta_perseed.py']
MARK = 'lstm_training_log.csv'          # 训练脚本最后一行打印的纯 ASCII 收尾标记


def find_root():
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.getcwd(), here, os.path.dirname(here)):
        if os.path.isfile(os.path.join(d, 'eval_alpha_sweep.py')):
            return d
    raise SystemExit('[错误] 找不到 eval_alpha_sweep.py：请在 starlink_fixed/ 下运行本脚本')


def run(cmd, log, env=None):
    print('  $', ' '.join(cmd), flush=True)
    e = dict(os.environ)
    e.update(env or {})
    with open(log, 'w') as f:
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=e).returncode


ROOT = find_root()
os.chdir(ROOT)
YES = '--yes' in sys.argv
SKIP_A = '--skip-a' in sys.argv
SM = os.environ.get('SM_ALPHA', '0.2').strip()
TAG = SM.replace('.', '')
OUT = os.path.join(ROOT, 'switch_a02_out')
os.makedirs(OUT, exist_ok=True)


def has(fn, marker):
    try:
        with open(os.path.join(ROOT, fn), encoding='utf-8', errors='replace') as f:
            return marker in f.read()
    except OSError:
        return False


def main():
    print(f'[路径] {ROOT}\n[配置] 部署 α = {SM}（标签 {TAG}）\n[模式] '
          + ('宽松(--yes) ' if YES else '严格 ') + ('跳过A ' if SKIP_A else '含A ')
          + '\n' + '[时间] ' + time.strftime('%Y-%m-%d %H:%M:%S'))

    # ---------- 1) 依赖脚本自检 ----------
    miss = [f for f in NEED if not os.path.isfile(os.path.join(ROOT, f))]
    if miss:
        raise SystemExit('[错误] 当前目录缺少脚本: ' + ', '.join(miss))
    for fn, mk, msg in [('verify_all_tables.py', 'SM_ALPHA', '不含 SM_ALPHA 支持'),
                        ('validate_with_space_v5.py', 'SM_ALPHA', '不含 SM_ALPHA 支持'),
                        ('train_lstm_inner_val.py', 'soft02', '缺少 soft02 变体')]:
        if os.path.isfile(os.path.join(ROOT, fn)) and not has(fn, mk):
            raise SystemExit(f'[错误] {fn} {msg} —— 请把本地改好的版本拷到服务器')
    print('[自检] 依赖脚本 OK')

    # ---------- 2) 6 个 LSTM α 权重 + 训练日志收尾标记 ----------
    print(f'[检查] LSTM α={SM} 的 6 个权重与训练日志：')
    bad, warn = [], []
    for s in SEEDS:
        p = os.path.join(ROOT, f'inner_val_ckpt/lstm_softmask_seed{s}_innerval.pth')
        lg = os.path.join(OUT, f'21_lstm_a{TAG}_seed{s}.log')
        mt = time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(p))) if os.path.exists(p) else '-'
        # 收尾标记：先看本脚本约定的日志名，找不到再在该种子相关的其它日志里搜（兼容手动重定向）
        cands = [lg] if os.path.exists(lg) else []
        cands += [x for x in glob.glob(os.path.join(OUT, f'*seed{s}*.log'))
                  + glob.glob(os.path.join(ROOT, f'*seed{s}*.log')) if x not in cands]
        okmark, src = False, '-'
        for c in cands:
            try:
                with open(c, encoding='utf-8', errors='replace') as f:
                    if MARK in f.read():
                        okmark, src = True, os.path.basename(c)
                        break
            except OSError:
                continue
        state = 'OK' if (os.path.exists(p) and okmark) else ('无日志标记' if os.path.exists(p) else 'MISSING')
        print(f'    seed {s:>3}: {state:12s} 权重 mtime={mt}  标记来源={src}')
        if not os.path.exists(p):
            bad.append(s)
        elif not okmark:
            warn.append(s)
    if bad:
        raise SystemExit(f'[中止] 以下种子缺权重文件: {bad}（请先训练）')
    if warn and not YES:
        raise SystemExit(f'[中止] 以下种子查不到训练日志收尾标记: {warn}\n'
                         f'  若您确认它们已训练完成（例如手动重定向了日志），加 --yes 继续。')
    if warn:
        print(f'[警告] {warn} 无日志收尾标记，按 --yes 继续（请自行核对上面的 mtime）')

    env_sm = {'SM_ALPHA': SM}

    # ---------- 3) A 阶段 ----------
    if not SKIP_A:
        print('[A] Transformer 侧（无需训练）')
        for script, log in STAGE_A:
            run([sys.executable, script], os.path.join(OUT, log), env_sm)
        for a, b, log in [(SM, '1.0', f'11_boot_a{TAG}_vs_a1.0.txt'),
                          (SM, '0.3', f'12_boot_a{TAG}_vs_a0.3.txt')]:
            run([sys.executable, 'boot_alpha_pairs.py', '--a', a, '--b', b],
                os.path.join(OUT, log), env_sm)
        key = os.path.join(OUT, f'11_boot_a{TAG}_vs_a1.0.txt')
        if os.path.exists(key):
            txt = open(key, encoding='utf-8', errors='replace').read()
            i = txt.find('点估计')
            print('[A] 关键片段：')
            if i >= 0:
                for l in txt[i:].splitlines()[:14]:
                    print('    ' + l)
    else:
        print('[A] 已跳过（--skip-a）')

    # ---------- 4) C 阶段 ----------
    print('[C] 复算 表5/表6/表8/表10/表11、图2 数据、§5.7')
    for script, log in STAGE_C:
        run([sys.executable, script], os.path.join(OUT, log), env_sm)

    # 断言：复算确实在读 α=SM（而非静默落回 0.3）
    vtlog = os.path.join(OUT, '30_verify_all_tables.txt')
    if os.path.exists(vtlog):
        t = open(vtlog, encoding='utf-8', errors='replace').read()
        if f'α = {SM}' in t or f'alpha{SM}_seed' in t:
            print(f'[核对] verify_all_tables 确认使用 α={SM} OK')
        else:
            print(f'[警告] {vtlog} 里没看到 α={SM} 的痕迹，请打开确认（可能脚本版本不对）')

    # ---------- 5) D 阶段 ----------
    for f in CSVS:
        src = os.path.join(ROOT, f)
        if os.path.exists(src):
            shutil.copy2(src, OUT)
    tarname = os.path.join(ROOT, 'switch_a02_out.tar.gz')
    if os.path.exists(tarname):
        os.remove(tarname)
    with tarfile.open(tarname, 'w:gz') as t:
        t.add(OUT, arcname='switch_a02_out')
    size = os.path.getsize(tarname) / 1024.0
    print(f'\n==== DONE ====\n产物：{tarname}（{size:.0f} KB）')
    print('请把该文件发回；我据此同步论文文本并重画 图2/图3。')


if __name__ == '__main__':
    main()

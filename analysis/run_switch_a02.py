"""run_switch_a02.py -- 把部署配置切到规则选出的 α=0.2（一次性编排）。

A. 先跑 Transformer 侧（α=0.2 权重已存在）→ 立刻拿到最关键的显著性数字
B. LSTM+软掩码 α=0.2 重训：**每个可见 GPU 一种子**（6 种子并行；已完成的自动跳过）
C. 用 SM_ALPHA=0.2 复算全部表格 / 图数据 / §5.7
D. 收集 CSV 并打包 switch_a02_out.tar.gz

用法：
    cd /u01/yk/satellite/starlink_fixed
    python run_switch_a02.py --detach     # 后台跑（推荐：断线不影响）
    python run_switch_a02.py              # 前台跑（调试）
    python run_switch_a02.py --status     # 查看进度（不启动任何东西）
    python run_switch_a02.py --fresh      # 忽略断点，强制重训全部 6 个种子
    # 指定卡 / 只补某些种子：
    #   CUDA_VISIBLE_DEVICES=0,1,2,3 python run_switch_a02.py --detach
    #   ONLY_SEEDS="101 202" python run_switch_a02.py --detach
"""
import glob
import os
import shutil
import subprocess
import sys
import tarfile
import time


def _find_root():
    """找到内含 eval_alpha_sweep.py 的目录（服务器上脚本通常平铺在 starlink_fixed/）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.getcwd(), here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        if os.path.isfile(os.path.join(d, 'eval_alpha_sweep.py')):
            return d
    raise SystemExit('[错误] 找不到 eval_alpha_sweep.py：请在 starlink_fixed/ 下运行本脚本')


ROOT = _find_root()
OUT = os.path.join(ROOT, 'switch_a02_out')
BACKUP = os.path.join(ROOT, 'backup_alpha03')
SM = os.environ.get('SM_ALPHA', '0.2').strip()
TAG = SM.replace('.', '')
SEEDS = ['42', '123', '789', '101', '202', '303']
STAMP = os.path.join(OUT, '00_start.stamp')
RUNLOG = os.path.join(OUT, 'run.log')


def fresh(path):
    """文件是否比本次运行的起始时刻更新（用于断点续跑与完整性闸门）。"""
    try:
        return os.path.getmtime(path) >= os.path.getmtime(STAMP) - 1.0
    except OSError:
        return False


def inner_pth(seed):
    return os.path.join(ROOT, f'inner_val_ckpt/lstm_softmask_seed{seed}_innerval.pth')


def seed_log(seed):
    return os.path.join(OUT, f'21_lstm_a{TAG}_seed{seed}.log')


def completed(seed):
    """训练**真正跑完**才算完成：日志有完成标记 且 权重比本次起始时刻新。
    （只看 mtime 会把"中途崩溃但已写过最有权重"的种子误判为完成）"""
    p = seed_log(seed)
    if not os.path.exists(p):
        return False
    try:
        with open(p, encoding='utf-8', errors='replace') as f:
            txt = f.read()
    except OSError:
        return False
    # 主判据用纯 ASCII 的收尾标记（训练脚本最后一行打印），避免日志编码非 UTF-8 时匹配失效
    if ('lstm_training_log.csv' not in txt) and ('] 完成' not in txt) and ('全部完成' not in txt):
        return False
    return fresh(inner_pth(seed))


def seed_state(seed):
    ip = inner_pth(seed)
    if completed(seed):
        return 'OK(done)'
    if os.path.exists(ip) and fresh(ip):
        return 'PARTIAL(中途崩)'
    if os.path.exists(ip):
        return 'OLD(未更新)'
    return 'MISSING'


def show_status():
    print(f'[状态] 工作目录 = {ROOT}')
    if os.path.exists(STAMP):
        print('  起始时刻: ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(STAMP))))
    else:
        print('  起始时刻: 尚未创建（说明还没跑过本脚本）')
    print('  LSTM α 权重：')
    for s in SEEDS:
        p = inner_pth(s)
        mt = time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(p))) if os.path.exists(p) else '-'
        print(f'    seed {s:>3}: {seed_state(s):16s} {mt}')
    print('  阶段产物：')
    for f in ('10_alpha_sweep.txt', f'11_boot_a{TAG}_vs_a1.0.txt',
              '30_verify_all_tables.txt', '31_export_eta.txt', '34_space_v5.txt'):
        print(f'    {"有" if os.path.exists(os.path.join(OUT, f)) else "无"}  {f}')
    tg = os.path.join(ROOT, 'switch_a02_out.tar.gz')
    print(f'  打包: {"已生成" if os.path.exists(tg) else "未生成"}  switch_a02_out.tar.gz')


def sh(cmd, log, env=None):
    e = dict(os.environ)
    e.update(env or {})
    print('  $', ' '.join(cmd), flush=True)
    with open(os.path.join(OUT, log), 'w') as f:
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=e).returncode


def visible_gpus():
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if cvd:
        return [x.strip() for x in cvd.split(',') if x.strip()]
    try:
        out = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout
        n = len([l for l in out.splitlines() if l.strip()])
    except Exception:  # noqa: BLE001
        n = 0
    return [str(i) for i in range(max(1, n))]


def main():
    os.chdir(ROOT)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(BACKUP, exist_ok=True)

    # ---- 模式分发：--status / --detach ----
    if '--status' in sys.argv:
        show_status()
        return
    if '--detach' in sys.argv:
        log = open(RUNLOG, 'a')
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), '--bg'],
                             cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        print(f'[后台] 已启动 PID={p.pid}（脱离终端，断线不影响）')
        print(f'  日志: tail -f {RUNLOG}')
        print(f'  进度: python {os.path.basename(os.path.abspath(__file__))} --status')
        return
    if '--fresh' in sys.argv and os.path.exists(STAMP):
        os.remove(STAMP)
        print('[--fresh] 已重置起始时刻戳，将重训全部 6 个种子')
    if not os.path.exists(STAMP):
        with open(STAMP, 'w') as f:
            f.write(str(time.time()))
    print(f'[路径] {ROOT}\n[配置] 目标部署 α = {SM}（标签 {TAG}）')
    print('[起始] ' + time.strftime('%Y-%m-%d %H:%M:%S'))

    # ---- 依赖脚本自检：防止旧脚本被 SM_ALPHA 静默忽略，产出"假 α=0.2"结果 ----
    need = ['eval_alpha_sweep.py', 'boot_alpha_pairs.py', 'verify_all_tables.py',
            'validate_with_space_v5.py', 'train_lstm_inner_val.py',
            'export_eta.py', 'export_eta_lstm.py', 'export_eta_perseed.py']
    miss = [f for f in need if not os.path.isfile(os.path.join(ROOT, f))]
    if miss:
        raise SystemExit('[错误] 当前目录缺少脚本: ' + ', '.join(miss))

    def has(fn, marker):
        try:
            with open(os.path.join(ROOT, fn), encoding='utf-8', errors='replace') as f:
                return marker in f.read()
        except OSError:
            return False

    if not has('verify_all_tables.py', 'SM_ALPHA'):
        raise SystemExit('[错误] verify_all_tables.py 不含 SM_ALPHA 支持 —— 请把本地改好的版本拷到服务器')
    if not has('validate_with_space_v5.py', 'SM_ALPHA'):
        raise SystemExit('[错误] validate_with_space_v5.py 不含 SM_ALPHA 支持 —— 请拷贝改好的版本')
    if not has('train_lstm_inner_val.py', 'soft02'):
        raise SystemExit('[错误] train_lstm_inner_val.py 缺少 soft02 变体（α=0.2）—— 请拷贝改好的版本')
    print('[自检] 依赖脚本均包含 α 切换支持 OK')

    # ---- 0) 备份 α=0.3 的 LSTM 权重（已存在则不覆盖，防止二次运行冲掉备份）----
    copied = []
    for pat in ('lstm_softmask_seed*.pth', 'inner_val_ckpt/lstm_softmask_seed*_innerval.pth'):
        for p in glob.glob(os.path.join(ROOT, pat)):
            dst = os.path.join(BACKUP, os.path.basename(p))
            if not os.path.exists(dst):
                shutil.copy2(p, dst)
                copied.append(os.path.basename(p) + ' (新备份)')
            else:
                copied.append(os.path.basename(p) + ' (已存在，跳过)')
    with open(os.path.join(OUT, '00_backup_list.txt'), 'w') as f:
        f.write('\n'.join(copied) + '\n')
    print(f'[0] 备份检查完成（backup_alpha03/ 共 {len(os.listdir(BACKUP))} 个文件）')

    env_sm = {'SM_ALPHA': SM}

    # ---- A) Transformer 侧 ----
    print('[A] Transformer 侧（无需训练）')
    sh([sys.executable, 'eval_alpha_sweep.py'], '10_alpha_sweep.txt', env_sm)
    sh([sys.executable, 'boot_alpha_pairs.py', '--a', SM, '--b', '1.0'],
       f'11_boot_a{TAG}_vs_a1.0.txt', env_sm)
    sh([sys.executable, 'boot_alpha_pairs.py', '--a', SM, '--b', '0.3'],
       f'12_boot_a{TAG}_vs_a0.3.txt', env_sm)
    key = os.path.join(OUT, f'11_boot_a{TAG}_vs_a1.0.txt')
    if os.path.exists(key):
        txt = open(key, encoding='utf-8', errors='replace').read()
        i = txt.find('点估计')
        print('[A] 关键片段：')
        if i >= 0:
            for l in txt[i:].splitlines()[:14]:
                print('    ' + l)

    # ---- B) LSTM α=0.2：每卡一种子（已完成的跳过）----
    gpus = visible_gpus()
    only = os.environ.get('ONLY_SEEDS', '').split()
    if only:
        todo = [s for s in SEEDS if s in only]      # ONLY_SEEDS 指定的：强制重训（不看是否已完成）
    else:
        todo = [s for s in SEEDS if not completed(s)]
    skip = [s for s in SEEDS if s not in todo]
    print(f'[调度] 可见 GPU = {gpus}（{len(gpus)} 张）')
    if skip:
        print(f'[B] 跳过: {skip}')
    print(f'[B] 本次要训练: {todo if todo else "（无）"}')
    if todo:
        sh([sys.executable, 'train_lstm_inner_val.py', '--check-split'], '20_lstm_check_split.txt')
    procs = []
    for i, s in enumerate(todo):
        dev = gpus[i % len(gpus)]
        e = dict(os.environ)
        e.update({'CUDA_VISIBLE_DEVICES': dev, 'TRAIN_DEVICE': 'cuda:0',
                  'VARIANTS': 'soft02', 'SEEDS': s,
                  # 缓冲碎片导致的 OOM 兜底（torch≥2.0 支持；旧版会忽略）
                  'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'})
        log = open(os.path.join(OUT, f'21_lstm_a{TAG}_seed{s}.log'), 'w')
        print(f'  [启动] seed={s} -> physical GPU {dev}', flush=True)
        procs.append((s, subprocess.Popen([sys.executable, 'train_lstm_inner_val.py'],
                                          cwd=ROOT, stdout=log,
                                          stderr=subprocess.STDOUT, env=e), log))
    for s, p, log in procs:
        rc = p.wait()
        log.close()
        print(f'  [完成] seed={s} 返回码 {rc}')
    with open(os.path.join(OUT, '22_lstm_ckpt_list.txt'), 'w') as f:
        files = sorted(glob.glob(os.path.join(ROOT, 'lstm_softmask_seed*.pth'))) + \
            sorted(glob.glob(os.path.join(ROOT, 'inner_val_ckpt/lstm_softmask_seed*_innerval.pth')))
        for p in files:
            f.write(f'{os.path.getmtime(p):.0f}  {p}\n')

    # ---- 完整性闸门：6 个种子必须**真正训练完成**，否则中止（避免用半成品/旧口径出表）----
    bad = [s for s in SEEDS if not completed(s)]
    if bad:
        print(f'\n[中止] 以下种子未完成 α={SM} 训练（或权重早于本次起始时刻）: {bad}')
        for s in bad:
            print(f'        seed {s} -> {seed_state(s)}')
        print('  详见 switch_a02_out/21_lstm_a{TAG}_seed*.log；补跑示例：')
        print(f'    ONLY_SEEDS="{" ".join(bad)}" python {os.path.basename(os.path.abspath(__file__))} --detach')
        sys.exit(3)
    print('[B] 6 个种子均已训练完成 OK')

    # ---- C) 复算 ----
    print('[C] 复算全部表格 / 图数据 / §5.7')
    for script, log in [('verify_all_tables.py', '30_verify_all_tables.txt'),
                        ('export_eta.py', '31_export_eta.txt'),
                        ('export_eta_lstm.py', '32_export_eta_lstm.txt'),
                        ('export_eta_perseed.py', '33_export_eta_perseed.txt'),
                        ('validate_with_space_v5.py', '34_space_v5.txt'),
                        ('c2_cluster_bootstrap.py', '35_cluster_bootstrap.txt')]:
        sh([sys.executable, script], log, env_sm)

    # ---- D) 收集 + 打包 ----
    for f in ('verify_table4_inner_val_ckpt.csv', 'fig2_cdf.csv', 'fig2_cdf_all.csv',
              'per_sample_eta_lstm.csv', 'per_seed_metrics.csv', 'spacex_v5_verification.csv'):
        src = os.path.join(ROOT, f)
        if os.path.exists(src):
            shutil.copy2(src, OUT)
    p = os.path.join(ROOT, 'eval_out/alpha_sweep_metrics_inner_val_ckpt.csv')
    if os.path.exists(p):
        shutil.copy2(p, OUT)
    tarname = os.path.join(ROOT, 'switch_a02_out.tar.gz')
    with tarfile.open(tarname, 'w:gz') as t:
        t.add(OUT, arcname='switch_a02_out')
    print(f'\n==== DONE ====\n产物：{tarname}\n回退：backup_alpha03/ 有 α=0.3 的 LSTM 权重。')


if __name__ == '__main__':
    main()

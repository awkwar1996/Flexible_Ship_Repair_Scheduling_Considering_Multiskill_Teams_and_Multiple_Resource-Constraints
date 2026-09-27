"""
================================================================================
修船厂调度 — 人力资源动态调配与修船坞资源协同调度优化（Gurobi 版本）
================================================================================

论文模型基础：《人力资源动态调配与修船坞资源协同调度优化研究》

问题描述：
  多艘船舶进入修船厂，每艘船有一组按顺序执行的维修任务。资源包括：
  - 泊位 (Berth)  B = {1,...,N_B}     : 空间资源，可容纳普通维修任务
  - 干船坞 (Dock) D = {1,...,N_D}     : 空间资源，水下/船体维修必须在船坞进行
  - 维修团队 (Team) K = {1,...,N_K}   : 多技能复合型团队，每个团队掌握多种核心技能

模型核心特征（论文 §3.1）：
  1. 多技能复合型团队：团队掌握多种技能，在不同技能上有差异化作业时间
  2. 技能等级级联机制：高等级团队可执行低等级任务（δ^skill_kq ≥ δ^req_jq，数值越小等级越高）
  3. 柔性工艺路线：任务可根据空间资源可用性选择泊位或干船坞（δ^B_j, δ^D_j）
  4. 空间-技能-团队-时间四维匹配：任务→空间资源→团队→处理时间的联合决策

数学符号（论文 §3.2）：

  集合与索引：
    i ∈ I              船舶编号，I = {1,...,N}
    j ∈ J_i            船舶 i 的任务集合，全局任务编号
    b ∈ B              泊位编号，B = {1,...,N_B}
    d ∈ D              干船坞编号，D = {1,...,N_D}
    k ∈ K              维修团队编号，K = {1,...,N_K}
    q ∈ Q              技能类型，Q = {weld(焊接), pipe(管道), elec(电气)}

  参数：
    a_i                船舶 i 的到港时间 (arrival time)
    p_jk               团队 k 执行任务 j 的处理时间
    δ^B_j ∈ {0,1}      任务 j 是否可在泊位执行 (1=可, 0=不可)
    δ^D_j ∈ {0,1}      任务 j 是否可在干船坞执行 (1=可, 0=不可)
    δ^skill_kq ∈ N     团队 k 在技能 q 上的资质等级 (1=最高级, 数值越小等级越高)
    δ^req_jq  ∈ N      任务 j 对技能 q 的最低资质要求
    M                   大M常数（足够大的正数）

  决策变量：
    S_j ∈ R+           任务 j 的开始时间
    C_j ∈ R+           任务 j 的完工时间
    x_jb ∈ {0,1}       任务 j 分配到泊位 b (=1 表示分配)
    y_jd ∈ {0,1}       任务 j 分配到干船坞 d (=1 表示分配)
    z_jk ∈ {0,1}       任务 j 分配到团队 k (=1 表示分配)
    u^B_{j1,j2,b}      泊位 b 上任务 j1 先于 j2 开始 (0-1)
    u^D_{j1,j2,d}      干船坞 d 上任务 j1 先于 j2 开始 (0-1)
    u^K_{j1,j2,k}      团队 k 上任务 j1 先于 j2 开始 (0-1)
    C^ship_i ∈ R+      船舶 i 的完工时间

  目标函数（Min total flow time）：
    Min  Σ_{i∈I} (C^ship_i - a_i)

  约束条件（详见 _build_model 方法中的注释）：
    (1)  空间分配约束       Σ_b x_jb + Σ_d y_jd = 1           ∀j    — 每个任务恰好一个空间
    (2)  空间-任务兼容性    x_jb ≤ δ^B_j,  y_jd ≤ δ^D_j       ∀j,b,d — 仅在兼容空间执行
    (3)  团队分配约束       Σ_k z_jk = 1                      ∀j    — 每个任务恰好一个团队
    (4)  团队-技能资质级联   z_jk = 0  if ∃q: δ^skill_kq > δ^req_jq  — 级联：高等级覆盖低等级
    (5)  完工时间计算       C_j = S_j + Σ_k z_jk · p_jk       ∀j    — 开始+处理=完工
    (6)  泊位容量互斥(Big-M) 同泊位任务不重叠
    (7)  干船坞容量互斥(Big-M) 同船坞任务不重叠
    (8)  团队容量互斥(Big-M) 同团队任务不重叠
    (9)  同船任务顺序约束   C_j ≤ S_{j+1}                    ∀i,∀j∈J_i — 船内任务按序
    (10) 到港时间约束       S_j ≥ a_i                        ∀i,∀j∈J_i — 到港后方可开工
    (11) 船舶完工时间定义   C^ship_i ≥ C_j                   ∀i,∀j∈J_i

实验设计（论文 §4-5）：
  阶段一（小规模）：30 组随机算例（15 基准+15 增强），1-10 艘船，Gurobi (TL=120s) vs KG-ALNS (TL=45s) 解质量对标
    - 基准配置: 适度紧张（泊位≈N/3+1, 船坞≈N/5+1, 团队≈N/3+3, 每船1-2任务）
    - 增强配置: 适度宽松（泊位≈N/2+1, 船坞≈N/4+1, 团队≈N/2+2），测试 Gap 改善上限
  阶段二（大规模）：60 组随机算例，30-60 艘船，3 梯度 × 20 组；同等计算预算下 KG-ALNS vs GA vs ALNS
    主报告口径：各算法独立重复运行的 **均值±标准差**（SCI 推荐）；辅报告 **best-of-runs**
    重复次数：KG-ALNS×3，GA×5，ALNS×5（随机基线更多次以稳定估计）
    跨算例统计：配对 t 检验 + Wilcoxon 符号秩检验（基于均值）
  目标报告口径（SCI）：Gurobi OPTIMAL 取 ObjVal；所有启发式最终解统一经 report_objective_from_schedule 后报告
  关键约束: 泊位≤0.25N(≤8), 船坞≤0.15N(≤4), 团队≤0.15T(≤12), 利用率>60%
================================================================================
"""

GUROBI_AVAILABLE = False
_gp_import_error = None
try:
    import gurobipy as gp  # Gurobi Python API
    from gurobipy import GRB  # Gurobi 常量（GRB.BINARY, GRB.CONTINUOUS, GRB.OPTIMAL 等）
    _gurobi_test_env = gp.Env(empty=True)
    _gurobi_test_env.setParam('OutputFlag', 0)
    _gurobi_test_env.start()
    _gurobi_test_env.dispose()
    GUROBI_AVAILABLE = True
except ImportError as _e:
    gp = None  # type: ignore[assignment]
    GRB = None  # type: ignore[assignment]
    _gp_import_error = _e
except Exception as _e:
    gp = None  # type: ignore[assignment]
    GRB = None  # type: ignore[assignment]
    _gp_import_error = _e
if not GUROBI_AVAILABLE:
    print(f"[警告] Gurobi 不可用 ({_gp_import_error})，精确求解/MIP 子问题将跳过；启发式仍可运行")

import random  # 随机算例生成
import copy
import math
import re  # 正则表达式（解析 Gurobi 日志）
import pandas as pd  # 数据表格处理（CSV读写、DataFrame操作）
import numpy as np  # 数值计算（np.mean, np.polyfit 等）
import time  # 求解计时
import cProfile  # 性能分析
import pstats  # 性能统计
from io import StringIO  # 性能报告字符串缓冲
import os  # 文件路径操作
import glob  # 文件名模式匹配
import shutil  # 文件复制（Gurobi 日志转移）
from datetime import datetime  # 实验时间戳
import json  # JSON 序列化（保存元数据和报告）
import sys  # 平台检测（Windows 并行基线内存提示）
import gc  # 释放 Gurobi/大对象内存，避免 Windows 并行子进程 OOM
from concurrent.futures import (
    ProcessPoolExecutor, BrokenExecutor, as_completed, wait, FIRST_COMPLETED,
)
try:
    import openpyxl  # Excel 读写（生成格式化报告）
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False  # 如果未安装 openpyxl，跳过 Excel 导出
# ============================================================
# 固定参数（全局默认值）
# 当随机生成的 instance 中没有 team_config 时，求解器回退使用这些默认值
# 论文 §3.1：团队为多技能复合型战略单元，技能等级 1=最高, 数值越大等级越低
# ============================================================

# 可用资源列表（默认值，实际算例会随机生成覆盖）
berths = [1, 2]  # B: 泊位集合，2个泊位
docks = [1]      # D: 干船坞集合，1个干船坞（水下船体维修必须在船坞进行）
teams = [1, 2, 3]  # K: 维修团队集合，3个多技能复合型团队
skills = ['weld', 'pipe', 'elec']  # Q: 技能集合（焊接、管道、电气）

# 团队技能矩阵 δ^skill_kq：团队 k 在技能 q 上的资质等级
# 等级 1=高级(最高), 2=中级, 3=初级 — 高等级可覆盖低等级任务（级联机制）
team_skills = {
    1: {'weld': 1, 'pipe': 1, 'elec': 1},  # T1: 全能高级团队
    2: {'weld': 1, 'pipe': 2},              # T2: 焊接高级+管道中级
    3: {'weld': 2, 'elec': 1},              # T3: 焊接中级+电气高级
}

# 团队处理时间 p_jk：团队 k 执行各技能任务的作业时间（天）
# 论文 §3.1：不同团队在同一技能上的作业时间存在差异，体现团队的专业化程度
processing_time = {
    1: {'weld': 2.0, 'pipe': 1.5, 'elec': 1.0},
    2: {'weld': 2.0, 'pipe': 2.0},
    3: {'weld': 2.5, 'elec': 1.5},
}

# 论文 §3.1 / 公式(4)：工种 q 与空间类型 s 的兼容性 c_qs（1=允许，0=禁止）
SKILL_SPACE_COMPAT = {
    'weld': {'berth': 0, 'dock': 1},
    'pipe': {'berth': 1, 'dock': 1},
    'elec': {'berth': 1, 'dock': 1},
}


def _safe_dataframe_to_csv(df, path, *, encoding='utf-8-sig', index=False,
                           max_retries=5, **kwargs):
    """Windows 友好 CSV 写入：临时文件 + os.replace，遇占用则重试/改存。"""
    path = os.path.normpath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    last_err = None
    for attempt in range(max_retries):
        try:
            df.to_csv(tmp_path, index=index, encoding=encoding, **kwargs)
            os.replace(tmp_path, path)
            return path
        except PermissionError as e:
            last_err = e
            if os.path.isfile(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if attempt < max_retries - 1:
                time.sleep(0.4 * (attempt + 1))
        except Exception:
            if os.path.isfile(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
    base, ext = os.path.splitext(path)
    alt_path = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
    df.to_csv(alt_path, index=index, encoding=encoding, **kwargs)
    print(f"[警告] {path} 被占用，已改存: {alt_path} ({last_err})", flush=True)
    return alt_path


def merge_large_scale_results_df(new_df, csv_path):
    """按 instance_id 与磁盘 CSV 合并（多进程/多梯度并行写同一文件时用）。"""
    if new_df is None or new_df.empty:
        return new_df
    if not os.path.isfile(csv_path):
        return new_df
    try:
        old_df = pd.read_csv(csv_path, encoding='utf-8-sig')
        if old_df.empty or 'instance_id' not in old_df.columns:
            return new_df
        new_ids = set(int(x) for x in new_df['instance_id'])
        old_keep = old_df[~old_df['instance_id'].astype(int).isin(new_ids)]
        if old_keep.empty:
            return new_df
        return pd.concat([old_keep, new_df], ignore_index=True)
    except Exception:
        return new_df


def _safe_rmtree(path, max_retries=5):
    """删除目录；Windows 上 Excel 占用 CSV 时重试。"""
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            return True
        except PermissionError:
            gc.collect()
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
    print(f"[警告] 无法删除目录（请关闭 Excel/预览窗口后重试）: {path}", flush=True)
    return False


def task_allows_space(task, space_type):
    """任务是否可在 space_type ('berth'/'dock') 执行"""
    for skill in task.get('skill_req', {}):
        compat = SKILL_SPACE_COMPAT.get(skill, {'berth': 1, 'dock': 1})
        if compat.get(space_type, 1) == 0:
            return False
    return True


def task_requires_dock_space(task):
    """任务是否必须分配干船坞（含仅船坞工种或显式 require_dock）"""
    return bool(task.get('require_dock', False)) or not task_allows_space(task, 'berth')


def feasible_spaces_for_task(task, berths_list, docks_list):
    """返回任务可行空间列表 [(space_type, space_id), ...]"""
    spaces = []
    if task_allows_space(task, 'berth'):
        spaces.extend(('berth', b) for b in berths_list)
    if task_allows_space(task, 'dock'):
        spaces.extend(('dock', d) for d in docks_list)
    return spaces


def add_skill_space_compat_constraints(model, x, y, tasks_data, berths, docks, task_ids=None):
    """论文公式(4)：工种-空间兼容性 → 禁止不兼容的 x_jb / y_jd"""
    allowed = set(task_ids) if task_ids is not None else None
    for task in tasks_data:
        t = task['task_id']
        if allowed is not None and t not in allowed:
            continue
        for skill in task.get('skill_req', {}):
            compat = SKILL_SPACE_COMPAT.get(skill, {'berth': 1, 'dock': 1})
            if compat.get('berth', 1) == 0:
                for b in berths:
                    model.addConstr(x[t, b] == 0, name=f"C4_skill_berth_{t}_{skill}_{b}")
            if compat.get('dock', 1) == 0:
                for d in docks:
                    model.addConstr(y[t, d] == 0, name=f"C4_skill_dock_{t}_{skill}_{d}")


def compute_big_M(tasks_data, arrival_times, teams_list, processing_time_dict,
                  team_skills_dict=None, buffer=100):
    """主模型 Big-M：max_arrival + max_duration * |J| + buffer"""
    max_duration = 0.0
    for task in tasks_data:
        for k in teams_list:
            if team_skills_dict is not None and k not in team_skills_dict:
                continue
            duration = 0.0
            for skill in task.get('skill_req', {}):
                if skill in processing_time_dict.get(k, {}):
                    duration += processing_time_dict[k][skill]
            max_duration = max(max_duration, duration)
    max_arrival = max(arrival_times.values()) if arrival_times else 0.0
    return max_arrival + max_duration * len(tasks_data) + buffer


# ============================================================
# 随机算例生成器
# ============================================================

class RandomInstanceGenerator:
    """随机生成修船厂调度算例"""

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_ship_arrival_times(self, num_ships, min_arrival=0, max_arrival=20):
        arrivals = [min_arrival]
        for i in range(1, num_ships):
            gap = random.uniform(2, 8)
            arrivals.append(arrivals[-1] + gap)
        arrivals = [min(a, max_arrival) for a in arrivals]
        return {i + 1: arrivals[i] for i in range(num_ships)}

    def generate_ship_due_dates(self, num_ships, arrival_times, tasks_data, team_config, slack_range=(2.0, 4.0)):
        """按 EDD 规则生成交付期：到达时间 + 估算总工时 * 松弛系数"""
        ship_tasks = {}
        for t in tasks_data:
            ship_tasks.setdefault(t['ship_id'], []).append(t)

        processing_time = team_config['processing_time']
        due_dates = {}
        for ship_id in range(1, num_ships + 1):
            arrival = arrival_times.get(ship_id, 0)
            total_work = 0.0
            for task in ship_tasks.get(ship_id, []):
                min_dur = float('inf')
                for team_id in team_config['teams']:
                    dur = 0
                    feasible = True
                    for skill in task['skill_req']:
                        if skill in processing_time.get(team_id, {}):
                            dur += processing_time[team_id][skill]
                        else:
                            feasible = False
                            break
                    if feasible and dur < min_dur:
                        min_dur = dur
                total_work += min_dur if min_dur < float('inf') else 2.0

            slack = random.uniform(slack_range[0], slack_range[1])
            due_dates[ship_id] = round(arrival + total_work * slack, 2)

        return due_dates

    def generate_task_skills_enhanced(self, num_tasks_per_ship, allow_dock=True):
        skill_patterns = [
            ({'weld': 1}, 1, 1.0),
            ({'pipe': 1}, 0, 0.8),
            ({'elec': 1}, 0, 0.7),
            ({'weld': 2}, 0, 0.6),
            ({'pipe': 2}, 0, 0.5),
            ({'elec': 2}, 0, 0.5),
            ({'weld': 1, 'pipe': 1}, 0, 1.5),
            ({'weld': 1, 'elec': 1}, 1, 1.6),
            ({'pipe': 1, 'elec': 1}, 0, 1.3),
            ({'weld': 2, 'pipe': 2}, 0, 1.0),
            ({'weld': 2, 'elec': 2}, 0, 1.0),
            ({'pipe': 2, 'elec': 2}, 0, 0.9),
            ({'weld': 1, 'pipe': 2}, 0, 1.2),
            ({'weld': 2, 'pipe': 1}, 0, 1.1),
            ({'weld': 1, 'elec': 2}, 1, 1.3),
            ({'weld': 2, 'elec': 1}, 0, 1.2),
            ({'pipe': 1, 'elec': 2}, 0, 1.0),
            ({'pipe': 2, 'elec': 1}, 0, 0.9),
            ({'weld': 1, 'pipe': 1, 'elec': 1}, 1, 2.0),
            ({'weld': 2, 'pipe': 2, 'elec': 2}, 0, 1.5),
        ]

        if not allow_dock:
            effective_patterns = [(s, d, c) for (s, d, c) in skill_patterns if d == 0]
        else:
            effective_patterns = skill_patterns

        tasks = []
        task_id = 1
        for ship_id, num_tasks in enumerate(num_tasks_per_ship, 1):
            for _ in range(num_tasks):
                skill_req, require_dock, complexity = random.choice(effective_patterns)
                task_payload = {
                    'skill_req': skill_req.copy(),
                    'require_dock': require_dock,
                }
                tasks.append({
                    'task_id': task_id,
                    'ship_id': ship_id,
                    'skill_req': task_payload['skill_req'],
                    'require_dock': task_requires_dock_space(task_payload),
                    'min_qual': max(skill_req.values()) if skill_req else 1,
                    'complexity': complexity
                })
                task_id += 1

        return tasks

    def generate_team_config(self, config=None):
        if config is None:
            config = {}

        force_team_count = config.get('force_team_count', None)
        if force_team_count is not None:
            num_teams = force_team_count
        else:
            team_count_range = config.get('team_count_range', (2, 5))
            num_teams = random.randint(team_count_range[0], team_count_range[1])

        skill_coverage_prob = config.get('skill_coverage_prob', 0.7)
        force_narrow_skills = config.get('force_narrow_skills', False)

        skill_time_ranges = {
            'weld': (2.0, 4.0),
            'pipe': (1.0, 2.5),
            'elec': (0.5, 1.5)
        }

        team_ids = list(range(1, num_teams + 1))
        team_skills_dict = {}
        processing_time_dict = {}

        for tid in team_ids:
            team_skills_dict[tid] = {}
            processing_time_dict[tid] = {}

            if tid == 1:
                for skill in skills:
                    team_skills_dict[tid][skill] = 1
                    skill_range = skill_time_ranges.get(skill, (1.0, 3.0))
                    processing_time_dict[tid][skill] = round(
                        random.uniform(skill_range[0], skill_range[1]), 2
                    )
            elif force_narrow_skills:
                assigned_skills = [random.choice(skills)]
                for skill in assigned_skills:
                    team_skills_dict[tid][skill] = random.randint(1, 3)
                    skill_range = skill_time_ranges.get(skill, (1.0, 3.0))
                    processing_time_dict[tid][skill] = round(
                        random.uniform(skill_range[0], skill_range[1]), 2
                    )
            else:
                assigned_skills = []
                for skill in skills:
                    if random.random() < skill_coverage_prob:
                        assigned_skills.append(skill)
                if not assigned_skills:
                    assigned_skills = [random.choice(skills)]

                for skill in assigned_skills:
                    team_skills_dict[tid][skill] = random.randint(1, 3)
                    skill_range = skill_time_ranges.get(skill, (1.0, 3.0))
                    processing_time_dict[tid][skill] = round(
                        random.uniform(skill_range[0], skill_range[1]), 2
                    )

        return {
            'teams': team_ids,
            'team_skills': team_skills_dict,
            'processing_time': processing_time_dict
        }

    def generate_instance(self, instance_id, config=None):
        if config is None:
            config = {}

        num_ships = config.get('num_ships')
        if num_ships is None and 'num_ships_range' in config:
            num_ships = random.randint(*config['num_ships_range'])
        if num_ships is None:
            num_ships = random.randint(3, 5)

        tasks_per_ship_range = config.get('tasks_per_ship_range', (1, 2))
        arrival_max = config.get('arrival_max', 20)
        team_count_range = config.get('team_count_range', (2, 4))
        allow_dock = config.get('allow_dock', True)

        berth_count = config.get('num_berths')
        if berth_count is None:
            berth_count = random.randint(1, 2)
        dock_count = config.get('num_docks')
        if dock_count is None:
            dock_count = random.randint(1, 2)

        # 生成每船任务数（移除原 total_tasks <= 10 的硬限制，改为按船舶数动态限制）
        num_tasks_per_ship = [
            random.randint(tasks_per_ship_range[0], tasks_per_ship_range[1])
            for _ in range(num_ships)
        ]
        total_tasks = sum(num_tasks_per_ship)

        arrival_times = self.generate_ship_arrival_times(num_ships, 0, arrival_max)
        tasks_data = self.generate_task_skills_enhanced(num_tasks_per_ship, allow_dock=allow_dock)

        team_config = self.generate_team_config({
            'team_count_range': team_count_range,
            'skill_coverage_prob': config.get('skill_coverage_prob', 0.7),
            'proc_time_range': config.get('proc_time_range', (1.0, 3.0)),
            'force_team_count': config.get('force_team_count'),
            'force_narrow_skills': config.get('force_narrow_skills', False)
        })

        due_dates = self.generate_ship_due_dates(num_ships, arrival_times, tasks_data, team_config)

        return {
            'instance_id': instance_id,
            'num_ships': num_ships,
            'num_tasks': total_tasks,
            'num_berths': berth_count,
            'num_docks': dock_count,
            'num_teams': len(team_config['teams']),
            'berths': list(range(1, berth_count + 1)),
            'docks': list(range(1, dock_count + 1)),
            'arrival_times': arrival_times,
            'due_dates': due_dates,
            'tasks_data': tasks_data,
            'team_config': team_config,
            'config': config
        }


# ============================================================
# 调度模型求解器（Gurobi 版本）
# ============================================================

MISALLOCATION_PENALTY_COEFF = 10  # λ：高级团队承担低级任务的惩罚系数
# 小规模对标：Gurobi 用 config.time_limit（通常 120s）求精确最优；
# KG-ALNS 用较短时限做质量验证（SCI 允许精确求解器更长时限）
SMALL_SCALE_BENCH_TIME_LIMIT = 60   # config.time_limit ≥ 此值时启用对标模式
SMALL_SCALE_HEU_TIME_LIMIT = 45     # 小规模 KG-ALNS 单次时限（秒）

class ShipyardSchedulerGurobi:
    """修船厂调度求解器（Gurobi）"""

    def __init__(self, time_limit=600, mip_gap=0.05):
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.model = None
        self._berths = None
        self._docks = None
        self._teams = None
        self._team_skills = None
        self._processing_time = None

    def _load_team_config(self, instance):
        team_config = instance.get('team_config')
        if team_config is not None:
            self._teams = team_config['teams']
            self._team_skills = team_config['team_skills']
            self._processing_time = team_config['processing_time']
        else:
            self._teams = teams
            self._team_skills = team_skills
            self._processing_time = processing_time

    def _load_resource_config(self, instance):
        self._berths = instance.get('berths', berths)
        self._docks = instance.get('docks', docks)

    def _compute_M(self, tasks_data, arrival_times):
        return compute_big_M(
            tasks_data, arrival_times, self._teams, self._processing_time,
            team_skills_dict=self._team_skills)

    def _precompute_durations(self, tasks_data):
        task_durations = {}
        for task in tasks_data:
            t_id = task['task_id']
            task_durations[t_id] = {}
            for k in self._teams:
                duration = 0
                feasible = True
                for skill in task['skill_req']:
                    if skill in self._processing_time.get(k, {}):
                        duration += self._processing_time[k][skill]
                    else:
                        feasible = False
                        break
                task_durations[t_id][k] = duration if feasible else None
        return task_durations

    def _compute_eligible_teams(self, tasks_data, task_durations):
        eligible_teams = {}
        for task in tasks_data:
            t_id = task['task_id']
            skill_req = task['skill_req']
            eligible_teams[t_id] = []
            for k in self._teams:
                team_skill_dict = self._team_skills.get(k, {})
                eligible = True
                for skill, min_qual in skill_req.items():
                    if skill not in team_skill_dict:
                        eligible = False
                        break
                    if team_skill_dict[skill] > min_qual:
                        eligible = False
                        break
                if eligible and task_durations[t_id][k] is not None:
                    eligible_teams[t_id].append(k)
            if not eligible_teams[t_id]:
                eligible_teams[t_id] = self._teams.copy()
        return eligible_teams

    def _build_model(self, instance):
        """构建 Gurobi MIP 模型"""
        if self.model is not None:
            try:
                self.model.dispose()
            except Exception:
                pass
            self.model = None

        tasks_data = instance['tasks_data']
        arrival_times = instance['arrival_times']
        num_ships = instance['num_ships']

        tasks = [t['task_id'] for t in tasks_data]
        ships = list(range(1, num_ships + 1))
        task_to_ship = {t['task_id']: t['ship_id'] for t in tasks_data}
        task_requires_dock = {t['task_id']: task_requires_dock_space(t) for t in tasks_data}

        self._load_team_config(instance)
        self._load_resource_config(instance)

        ship_tasks = {}
        for ship in ships:
            ship_tasks[ship] = [t['task_id'] for t in tasks_data if t['ship_id'] == ship]
            ship_tasks[ship].sort()

        M = self._compute_M(tasks_data, arrival_times)
        task_durations = self._precompute_durations(tasks_data)
        eligible_teams = self._compute_eligible_teams(tasks_data, task_durations)

        # ========== 创建 Gurobi 模型（静默环境）==========
        env = gp.Env(params={"OutputFlag": 0})
        model = gp.Model('Shipyard_Scheduling_S', env=env)

        # ====================================================================
        # 决策变量（论文 §3.2）
        # ====================================================================
        # S_j ∈ R+: 任务 j 的开始时间
        S = model.addVars(tasks, vtype=GRB.CONTINUOUS, lb=0, name="S_j")
        # C_j ∈ R+: 任务 j 的完工时间
        C = model.addVars(tasks, vtype=GRB.CONTINUOUS, lb=0, name="C_j")
        # C^ship_i ∈ R+: 船舶 i 的完工时间
        ship_completion = model.addVars(ships, vtype=GRB.CONTINUOUS, lb=0, name="C_ship_i")

        # x_jb ∈ {0,1}: 任务 j 分配到泊位 b
        x = model.addVars(
            [(t, b) for t in tasks for b in self._berths],
            vtype=GRB.BINARY, name="x_jb"
        )
        # y_jd ∈ {0,1}: 任务 j 分配到干船坞 d
        y = model.addVars(
            [(t, d) for t in tasks for d in self._docks],
            vtype=GRB.BINARY, name="y_jd"
        )
        # z_jk ∈ {0,1}: 任务 j 分配到团队 k
        z = model.addVars(
            [(t, k) for t in tasks for k in self._teams],
            vtype=GRB.BINARY, name="z_jk"
        )

        # u^B_{j1,j2,b} ∈ {0,1}: 泊位 b 上 j1 先于 j2 (仅 j1<j2)
        u_berth = model.addVars(
            [(t1, t2, b) for t1 in tasks for t2 in tasks if t1 < t2 for b in self._berths],
            vtype=GRB.BINARY, name="uB"
        )
        # u^D_{j1,j2,d} ∈ {0,1}: 干船坞 d 上 j1 先于 j2 (仅 j1<j2)
        u_dock = model.addVars(
            [(t1, t2, d) for t1 in tasks for t2 in tasks if t1 < t2 for d in self._docks],
            vtype=GRB.BINARY, name="uD"
        )
        # u^K_{j1,j2,k} ∈ {0,1}: 团队 k 上 j1 先于 j2 (仅 j1<j2)
        u_team = model.addVars(
            [(t1, t2, k) for t1 in tasks for t2 in tasks if t1 < t2 for k in self._teams],
            vtype=GRB.BINARY, name="uK"
        )

        # ====================================================================
        # 约束条件（论文 §3.2 公式 1-11）
        # 按论文顺序逐一实现，每条标注对应的数学公式
        # ====================================================================

        # 约束(1): 空间分配约束 — 每个任务恰好分配到一个空间资源
        # Σ_{b∈B} x_jb + Σ_{d∈D} y_jd = 1   ∀j∈J
        model.addConstrs(
            (gp.quicksum(x[t, b] for b in self._berths) +
             gp.quicksum(y[t, d] for d in self._docks) == 1
             for t in tasks),
            name="C1_space_alloc"
        )

        # 约束(2): 空间-任务兼容性约束 — 需干船坞的任务不能分配泊位
        # x_jb ≤ δ^B_j (δ^B_j=0时强制x_jb=0), y_jd ≤ δ^D_j
        # 水下船体维修(焊接为主)必须在干船坞执行（论文 §3.1 第39行）
        dock_only_tasks = [t for t in tasks if task_requires_dock[t]]
        if dock_only_tasks:
            model.addConstrs(
                (x[t, b] == 0 for t in dock_only_tasks for b in self._berths),
                name="C2_dock_compat"
            )

        # 约束(4): 工种-空间资源兼容性 — Σ_b c_{mb}·x_jb + Σ_d c_{md}·y_jd ≥ 1, ∀j, ∀m∈M_j
        add_skill_space_compat_constraints(
            model, x, y, tasks_data, self._berths, self._docks)

        # 约束(3): 团队分配约束 — 每个任务恰好分配一个多技能复合型团队
        # Σ_{k∈K} z_jk = 1   ∀j∈J
        model.addConstrs(
            (gp.quicksum(z[t, k] for k in self._teams) == 1
             for t in tasks),
            name="C3_team_alloc"
        )

        # 约束(4): 团队-技能资质级联约束 — 高等级可覆盖低等级（论文核心创新点）
        # z_jk = 0 if ∃q∈Q: δ^skill_kq > δ^req_jq
        # 级联机制：团队资质等级(1=最高)≤任务要求等级时方可指派
        # 即高等级(数值小)可执行低等级(数值大)任务，反之不可
        model.addConstrs(
            (z[t, k] == 0
             for t in tasks for k in self._teams if k not in eligible_teams[t]),
            name="C4_skill_cascade"
        )

        # 约束(5): 完工时间计算 — 开始时间 + 团队相关处理时间 = 完工时间
        # C_j = S_j + Σ_{k∈K} z_jk · p_jk   ∀j∈J
        # p_jk 为团队k执行任务j的处理时间，体现不同团队的差异化作业效率
        model.addConstrs(
            (C[t] == S[t] + gp.quicksum(
                z[t, k] * task_durations[t][k]
                for k in eligible_teams[t]
                if task_durations[t][k] is not None)
             for t in tasks),
            name="C5_completion"
        )

        # 约束(6): 泊位容量互斥约束（Big-M法）— 同一泊位同时只能服务一个任务
        # C_j1 ≤ S_j2 + M(1 - u^B_{j1,j2,b}) + M(2 - x_{j1,b} - x_{j2,b})   ∀j1<j2, ∀b
        # 当 j1,j2 都分配到泊位b (x_{j1,b}=x_{j2,b}=1) 且 j1先于j2 (u=1) 时:
        #   C_j1 ≤ S_j2 (有效)  否则Big-M松弛
        model.addConstrs(
            (C[t1] <= S[t2] +
             M * (1 - u_berth[t1, t2, b]) +
             M * (2 - x[t1, b] - x[t2, b])
             for b in self._berths for t1 in tasks for t2 in tasks if t1 < t2),
            name="C6_berth_seq1"
        )
        model.addConstrs(
            (C[t2] <= S[t1] +
             M * u_berth[t1, t2, b] +
             M * (2 - x[t1, b] - x[t2, b])
             for b in self._berths for t1 in tasks for t2 in tasks if t1 < t2),
            name="C6_berth_seq2"
        )

        # 约束(7): 干船坞容量互斥约束（Big-M法）— 同一船坞同时只能服务一个任务
        # C_j1 ≤ S_j2 + M(1 - u^D_{j1,j2,d}) + M(2 - y_{j1,d} - y_{j2,d})   ∀j1<j2, ∀d
        model.addConstrs(
            (C[t1] <= S[t2] +
             M * (1 - u_dock[t1, t2, d]) +
             M * (2 - y[t1, d] - y[t2, d])
             for d in self._docks for t1 in tasks for t2 in tasks if t1 < t2),
            name="C7_dock_seq1"
        )
        model.addConstrs(
            (C[t2] <= S[t1] +
             M * u_dock[t1, t2, d] +
             M * (2 - y[t1, d] - y[t2, d])
             for d in self._docks for t1 in tasks for t2 in tasks if t1 < t2),
            name="C7_dock_seq2"
        )

        # 约束(8): 团队容量互斥约束（Big-M法）— 同一团队同时只能服务一个任务
        # C_j1 ≤ S_j2 + M(1 - u^K_{j1,j2,k}) + M(2 - z_{j1,k} - z_{j2,k})   ∀j1<j2, ∀k
        # 体现人力资源的排他性：多技能团队虽可做多种任务，但同一时刻只能做一个
        model.addConstrs(
            (C[t1] <= S[t2] +
             M * (1 - u_team[t1, t2, k]) +
             M * (2 - z[t1, k] - z[t2, k])
             for k in self._teams for t1 in tasks for t2 in tasks if t1 < t2),
            name="C8_team_seq1"
        )
        model.addConstrs(
            (C[t2] <= S[t1] +
             M * u_team[t1, t2, k] +
             M * (2 - z[t1, k] - z[t2, k])
             for k in self._teams for t1 in tasks for t2 in tasks if t1 < t2),
            name="C8_team_seq2"
        )

        # 约束(9): 同船任务顺序约束 — 船舶i的任务必须按指定顺序执行
        # C_j ≤ S_{j+1}   ∀i∈I, ∀j∈J_i\{last}
        # 体现船舶维修的工艺流程约束（如先焊后检）
        model.addConstrs(
            (C[ship_tasks[ship][i]] <= S[ship_tasks[ship][i + 1]]
             for ship in ships for i in range(len(ship_tasks[ship]) - 1)),
            name="C9_ship_seq"
        )

        # 约束(10): 到港时间约束 — 任务开始时间不早于船舶到港时间
        # S_j ≥ a_i   ∀i∈I, ∀j∈J_i
        model.addConstrs(
            (S[t] >= arrival_times[ship]
             for ship in ships for t in ship_tasks[ship]),
            name="C10_arrival"
        )

        # 约束(11): 船舶完工时间定义 — 取该船最后一个任务的完工时间
        # C^ship_i ≥ C_j   ∀i∈I, ∀j∈J_i
        # 与目标函数配合：Min Σ(C^ship_i - a_i)
        model.addConstrs(
            (ship_completion[ship] >= C[t]
             for ship in ships for t in ship_tasks[ship]),
            name="C11_ship_comp"
        )

        # 注：不在主模型中加入 C12 对称性破缺约束。
        # C12 会缩小可行域，导致 Gurobi“最优”高于启发式真实可行解，产生负 Gap。
        # 启发式与 MIP 子问题均不施加 C12，主模型与之保持一致以便公平对比。

        # ====================================================================
        # 目标函数（论文 §3.2 公式 + 资质匹配软约束）
        # Min Σ_{i∈I} (C^ship_i - a_i) + λ · Σ(高级团队←低级任务)
        # λ = MISALLOCATION_PENALTY_COEFF (默认 10)
        # ====================================================================
        flow_obj = gp.quicksum(
            ship_completion[ship] - arrival_times[ship] for ship in ships)

        elite_teams = [k for k in self._teams
                       if any(lv == 1 for lv in self._team_skills.get(k, {}).values())]
        penalty_terms = []
        for t in tasks:
            task_info = next((ti for ti in tasks_data if ti['task_id'] == t), None)
            if task_info is None:
                continue
            skill_req = task_info.get('skill_req', {})
            if skill_req and all(req >= 2 for req in skill_req.values()):
                for k in elite_teams:
                    if k in eligible_teams[t]:
                        penalty_terms.append(z[t, k])

        misallocation_obj = (MISALLOCATION_PENALTY_COEFF * gp.quicksum(penalty_terms)
                             if penalty_terms else 0)
        model.setObjective(flow_obj + misallocation_obj, GRB.MINIMIZE)

        self.model = model

        # 返回变量字典供提取解时使用
        return {
            'task_to_ship': task_to_ship,
            'ship_tasks': ship_tasks,
            'eligible_teams': eligible_teams,
            'task_durations': task_durations,
            'M': M,
            'S': S,
            'C': C,
            'ship_completion': ship_completion,
            'x': x,
            'y': y,
            'z': z,
        }

    def _apply_warm_start(self, model_data, warm_start_schedule):
        """将启发式调度解注入 Gurobi 作为 MIP 热启动。

        设置二进制分配变量 (x, y, z) 和连续时间变量 (S, C) 的初始值。
        Gurobi 会从这些值出发进行分支定界，显著加速收敛。
        """
        x = model_data['x']
        y = model_data['y']
        z = model_data['z']
        S = model_data['S']
        C = model_data['C']

        for entry in warm_start_schedule:
            t_id = entry['task_id']
            start_val = entry['start']
            comp_val = entry['completion']
            space_type = entry['space_type']
            space_id = entry['space']
            team_id = entry['team']

            # 连续变量：开始和完工时间
            S[t_id].Start = start_val
            C[t_id].Start = comp_val

            # 二进制变量：空间分配 — 先全部设 0，再设目标为 1
            for b in self._berths:
                x[t_id, b].Start = 1 if (space_type == 'berth' and b == space_id) else 0
            for d in self._docks:
                y[t_id, d].Start = 1 if (space_type == 'dock' and d == space_id) else 0

            # 二进制变量：团队分配
            for k in self._teams:
                z[t_id, k].Start = 1 if k == team_id else 0

        # 船舶完工时间：从调度中推算每艘船最晚任务完工时间
        ship_completion = model_data['ship_completion']
        ship_max_completion = {}
        for entry in warm_start_schedule:
            ship_id = entry['ship_id']
            ship_max_completion[ship_id] = max(
                ship_max_completion.get(ship_id, 0), entry['completion']
            )
        for ship_id, comp_val in ship_max_completion.items():
            ship_completion[ship_id].Start = comp_val

    def solve(self, instance, log_dir=None, warm_start_schedule=None):
        """求解算例（论文 §4 精确求解）

        参数:
            instance: 算例字典
            log_dir: Gurobi 日志输出目录（用于收敛曲线），None 则不保存日志
            warm_start_schedule: 启发式调度结果，用于 Gurobi MIP 热启动
        """
        if not GUROBI_AVAILABLE:
            return {
                'status': 'GUROBI_UNAVAILABLE',
                'objective': None,
                'solve_time': 0.0,
                'num_variables': 0,
                'num_constraints': 0,
                'num_binary': 0,
                'num_continuous': 0,
                'node_count': 0,
                'iteration_count': 0,
                'best_bound': None,
                'mip_gap': None,
                'feasible': False,
                'schedule': None,
                'log_file': None,
            }

        model_data = self._build_model(instance)

        model = self.model
        model.Params.TimeLimit = self.time_limit
        model.Params.MIPGap = self.mip_gap
        model.Params.OutputFlag = 0  # 关闭控制台输出，日志写入文件
        if GUROBI_SUB_THREADS > 0:
            model.Params.Threads = GUROBI_SUB_THREADS

        # 热启动：将启发式解注入 Gurobi 作为初始可行解
        if warm_start_schedule is not None:
            self._apply_warm_start(model_data, warm_start_schedule)

        # 保存 Gurobi 日志文件（收敛曲线数据源）
        # Gurobi C 库在 Windows 上可能无法处理非 ASCII 路径，先写入安全临时位置
        log_file = None
        tmp_log = None
        if log_dir is not None:
            log_dir = os.path.normpath(log_dir)
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.normpath(os.path.join(log_dir, f"gurobi_log_instance_{instance['instance_id']}.log"))
            # 使用 ASCII-safe 临时目录（Python 的 os.getcwd() 和 tempfile 均返回 ASCII 路径）
            import tempfile
            tmp_log = os.path.normpath(os.path.join(
                tempfile.gettempdir(), f"_gurobi_log_{instance['instance_id']}_{os.getpid()}.log"))
            model.Params.LogFile = tmp_log
            model.Params.LogToConsole = 0

        solve_start = time.time()
        model.optimize()
        solve_time = time.time() - solve_start

        # 关闭日志文件句柄，将临时日志转移到目标目录
        if tmp_log is not None:
            model.Params.LogFile = ""
            if os.path.exists(tmp_log):
                try:
                    shutil.move(tmp_log, log_file)
                except Exception:
                    try:
                        shutil.copy(tmp_log, log_file)
                    except Exception as e:
                        print(f"  警告: 无法复制 Gurobi 日志到 {log_file}: {e}")

        # 统计变量
        total_vars = model.NumVars
        bin_vars = sum(1 for v in model.getVars() if v.VType == GRB.BINARY)
        cont_vars = sum(1 for v in model.getVars() if v.VType == GRB.CONTINUOUS)

        # 获取状态信息
        status_map = {
            GRB.OPTIMAL: 'OPTIMAL',
            GRB.TIME_LIMIT: 'TIME_LIMIT',
            GRB.INF_OR_UNBD: 'INF_OR_UNBD',
            GRB.INFEASIBLE: 'INFEASIBLE',
            GRB.UNBOUNDED: 'UNBOUNDED',
            GRB.SUBOPTIMAL: 'SUBOPTIMAL',
        }
        status_str = status_map.get(model.Status, f'CODE_{model.Status}')

        # 提取目标值和 Gap
        objective_value = None
        best_bound = None
        mip_gap = None

        if model.SolCount > 0:
            objective_value = model.ObjVal
            mip_gap = model.MIPGap
        if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            try:
                best_bound = model.ObjBound
            except Exception:
                best_bound = None

        node_count = int(model.NodeCount) if hasattr(model, 'NodeCount') else 0
        iteration_count = int(model.IterCount) if hasattr(model, 'IterCount') else 0

        feasible = model.SolCount > 0

        result = {
            'status': status_str,
            'objective': objective_value,
            'solve_time': solve_time,
            'num_variables': total_vars,
            'num_constraints': model.NumConstrs,
            'num_binary': bin_vars,
            'num_continuous': cont_vars,
            'node_count': node_count,
            'iteration_count': iteration_count,
            'best_bound': best_bound,
            'mip_gap': mip_gap,
            'feasible': feasible,
            'log_file': log_file,  # Gurobi 日志路径（收敛曲线数据源）
        }

        if feasible:
            schedule = self._extract_schedule(instance, model_data)
            result['schedule'] = schedule
            if model.Status == GRB.OPTIMAL:
                # SCI：已证最优 → 直接取 MIP 目标值（精确基准）
                result['objective'] = objective_value
            else:
                _, result['objective'], _ = report_objective_from_schedule(
                    schedule, instance, self._team_skills)
        else:
            result['schedule'] = None

        return result

    def _extract_schedule(self, instance, model_data):
        """提取调度结果（使用 Gurobi .X 属性）"""
        tasks_data = instance['tasks_data']

        S = model_data['S']
        C = model_data['C']
        x = model_data['x']
        y = model_data['y']
        z = model_data['z']

        schedule = []
        for task in tasks_data:
            t_id = task['task_id']
            start = S[t_id].X
            comp = C[t_id].X

            # 找分配的空间资源
            space = None
            space_type = None
            for b in self._berths:
                if x[t_id, b].X > 0.5:
                    space = b
                    space_type = 'berth'
                    break
            if space is None:
                for d in self._docks:
                    if y[t_id, d].X > 0.5:
                        space = d
                        space_type = 'dock'
                        break

            # 找分配的团队
            assigned_team = None
            for k in self._teams:
                if z[t_id, k].X > 0.5:
                    assigned_team = k
                    break

            schedule.append({
                'task_id': t_id,
                'ship_id': task['ship_id'],
                'start': start,
                'completion': comp,
                'duration': comp - start,
                'space': space,
                'space_type': space_type,
                'team': assigned_team,
                'skill_req': task['skill_req']
            })

        return schedule


# ============================================================
# 启发式调度求解器（GRASP 算法）
# ============================================================


# ============================================================
# 统一目标函数（原始单目标形式）
# ============================================================

def compute_flow_from_schedule(schedule, instance):
    """总在厂时间：Σ(C^ship_i - a_i)，对所有船舶求和（与 Gurobi 模型一致）"""
    arrival_times = instance['arrival_times']
    num_ships = instance.get('num_ships', 0)
    ship_completion = {}
    for entry in schedule:
        ship_id = entry['ship_id']
        completion = entry['completion']
        if ship_id not in ship_completion or completion > ship_completion[ship_id]:
            ship_completion[ship_id] = completion
    return sum(
        ship_completion.get(s, 0.0) - arrival_times.get(s, 0.0)
        for s in range(1, num_ships + 1)
    )


def compute_misallocation_count_from_schedule(schedule, team_skills):
    """高级团队承担低级任务次数（大材小用计数，不含系数）"""
    if not team_skills:
        return 0
    elite_teams = {k for k in team_skills
                   if any(lv == 1 for lv in team_skills[k].values())}
    count = 0
    for entry in schedule:
        if entry['team'] not in elite_teams:
            continue
        skill_req = entry.get('skill_req', {})
        if skill_req and all(req >= 2 for req in skill_req.values()):
            count += 1
    return count


def sa_accept_move(delta, temperature, improvement_tol=0.01):
    """模拟退火接受准则（防止 T→0 时 math.exp 溢出）"""
    if delta < -improvement_tol:
        return True
    exp_arg = -delta / max(temperature, 1e-6)
    if exp_arg >= 700.0:
        return True
    if exp_arg <= -700.0:
        return False
    return random.random() < math.exp(exp_arg)


def normalize_team_config_keys(team_config):
    """将 JSON 反序列化后的团队配置键统一为 int。

    json.load 会把字典键变成 str；若保持 str，则 _processing_time.get(team_id:int)
    全部 miss，导致消融/敏感性目标严重偏低、与大规模不可比。
    """
    if not team_config or not isinstance(team_config, dict):
        return team_config

    def _as_int_key_dict(d):
        if not isinstance(d, dict):
            return d
        out = {}
        for k, v in d.items():
            try:
                ik = int(k)
            except (TypeError, ValueError):
                ik = k
            if isinstance(v, dict):
                # processing_time[team][skill] 的内层 skill 保持 str
                out[ik] = v
            else:
                out[ik] = v
        return out

    fixed = dict(team_config)
    if 'teams' in fixed and isinstance(fixed['teams'], list):
        fixed['teams'] = [int(t) for t in fixed['teams']]
    if 'team_skills' in fixed:
        fixed['team_skills'] = _as_int_key_dict(fixed['team_skills'])
    if 'processing_time' in fixed:
        fixed['processing_time'] = _as_int_key_dict(fixed['processing_time'])
    return fixed


def evaluate_objective(total_flow, misallocation_count, penalty_coeff=None):
    """统一目标函数：total_flow + λ × misallocation_count"""
    coeff = MISALLOCATION_PENALTY_COEFF if penalty_coeff is None else penalty_coeff
    return total_flow + coeff * misallocation_count


def compute_objective_from_schedule(schedule, instance, team_skills=None,
                                    penalty_coeff=None):
    """从调度方案计算统一目标值（所有求解器共用）"""
    if team_skills is None:
        utils = HeuristicScheduler()
        utils._load_team_config(instance)
        team_skills = utils._team_skills
    if penalty_coeff is None:
        penalty_coeff = instance.get('_penalty_coeff', MISALLOCATION_PENALTY_COEFF)
    total_flow = compute_flow_from_schedule(schedule, instance)
    misallocation_count = compute_misallocation_count_from_schedule(schedule, team_skills)
    return evaluate_objective(total_flow, misallocation_count, penalty_coeff)


def decompose_schedule_objective(schedule, instance, penalty_coeff=None):
    """分解目标：流动时间、技能过度配置次数/惩罚、干船坞误用次数。"""
    if not schedule:
        return {
            'total_flow': None,
            'misallocation_count': None,
            'misallocation_penalty': None,
            'dock_misuse_count': None,
        }
    utils = HeuristicScheduler()
    utils._load_team_config(instance)
    coeff = MISALLOCATION_PENALTY_COEFF if penalty_coeff is None else penalty_coeff
    if penalty_coeff is None:
        coeff = instance.get('_penalty_coeff', MISALLOCATION_PENALTY_COEFF)
    total_flow = compute_flow_from_schedule(schedule, instance)
    misc = compute_misallocation_count_from_schedule(schedule, utils._team_skills)
    dock_misuse = 0
    task_map = {t.get('task_id'): t for t in instance.get('tasks_data', [])}
    for entry in schedule:
        if entry.get('space_type') == 'dock':
            task = task_map.get(entry.get('task_id'))
            if task is not None and not task_requires_dock_space(task):
                dock_misuse += 1
    return {
        'total_flow': float(total_flow),
        'misallocation_count': int(misc),
        'misallocation_penalty': float(coeff * misc),
        'dock_misuse_count': int(dock_misuse),
    }


def compute_instance_knowledge_intensity(instance):
    """实例级知识需求强度 H_i = mean_j (P̂_j + R̂_j + Ŝ_j)，与算法结果无关。"""
    tasks = instance.get('tasks_data') or []
    if not tasks:
        return None
    solver = MatheuristicSolver(verbose=False)
    solver._ensure_config_loaded(instance)
    ps, rs, ss = [], [], []
    for task in tasks:
        P, R, S = solver._task_knowledge_components(
            task, team_id=None, space_type=None)
        ps.append(float(P))
        rs.append(float(R))
        ss.append(float(S))

    def _norm01(vals):
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-12:
            return [0.0] * len(vals)
        return [(v - lo) / (hi - lo) for v in vals]

    pn, rn, sn = _norm01(ps), _norm01(rs), _norm01(ss)
    h_vals = [pn[i] + rn[i] + sn[i] for i in range(len(tasks))]
    return float(np.mean(h_vals)) if h_vals else None


def sort_schedule_for_rebuild(schedule, instance):
    """按到港时间→船舶→任务排序，供时间线重建使用（与 Gurobi C9 船序一致）"""
    arrival_times = instance.get('arrival_times', {})
    return sorted(
        schedule,
        key=lambda e: (arrival_times.get(e['ship_id'], 0.0), e['ship_id'], e['task_id']),
    )


def report_objective_from_schedule(schedule, instance, team_skills=None,
                                  penalty_coeff=None):
    """SCI 论文对比用统一报告口径：排序 → 重建时间线 → 算目标 → 可行性校验"""
    if not schedule:
        return None, None, False
    sched, obj = evaluate_schedule_consistently(
        schedule, instance, team_skills, penalty_coeff)
    feasible = validate_schedule_feasibility(sched, instance)
    return sched, obj, feasible


def evaluate_schedule_consistently(schedule, instance, team_skills=None,
                                   penalty_coeff=None):
    """统一评估：排序 → 重建时间线 → 计算目标（启发式/Gurobi 非最优解共用）"""
    utils = HeuristicScheduler()
    utils._load_team_config(instance)
    utils._load_resource_config(instance)
    sched = [dict(e) for e in sort_schedule_for_rebuild(schedule, instance)]
    sched = utils._rebuild_schedule_times(sched, instance)
    if team_skills is None:
        team_skills = utils._team_skills
    if penalty_coeff is None:
        penalty_coeff = instance.get('_penalty_coeff', MISALLOCATION_PENALTY_COEFF)
    obj = compute_objective_from_schedule(
        sched, instance, team_skills, penalty_coeff)
    return sched, obj


# ---------------------------------------------------------------------------
# SCI 批量实验：墙钟收敛轨迹观测钩子（仅记录，不改变搜索决策）
# 由 sci_experiments 在运行前设置 SCI_RECORD_WALL_TRACE / SCI_TRACE_CHECKPOINTS
# ---------------------------------------------------------------------------
SCI_RECORD_WALL_TRACE = False
SCI_TRACE_CHECKPOINTS = (0, 1, 2, 5, 10, 20, 30, 60, 90, 120, 180, 240, 300)


class SciWallClockTracer:
    """固定墙钟检查点记录 best-so-far（观测器，不影响接受/拒绝决策）。"""

    def __init__(self, checkpoints, t0, time_limit=None):
        self.checkpoints = [float(t) for t in checkpoints]
        self.t0 = float(t0)
        self.time_limit = float(time_limit) if time_limit is not None else None
        self.best = float('inf')
        self.best_time = None
        self.current = float('inf')
        self.iteration = 0
        self.feasible = True
        self.next_idx = 0
        self.snapshot = {}  # elapsed -> row dict

    def update(self, best_obj, current_obj=None, iteration=None, feasible=True, now=None):
        if not globals().get('SCI_RECORD_WALL_TRACE', False):
            return
        try:
            bo = float(best_obj)
        except (TypeError, ValueError):
            return
        if not math.isfinite(bo):
            return
        now = time.time() if now is None else float(now)
        elapsed = now - self.t0
        if current_obj is not None:
            try:
                co = float(current_obj)
                if math.isfinite(co):
                    self.current = co
            except (TypeError, ValueError):
                pass
        if iteration is not None:
            self.iteration = int(iteration)
        self.feasible = bool(feasible)
        if bo < self.best - 1e-12:
            self.best = bo
            self.best_time = elapsed
        else:
            self.best = min(self.best, bo)
        while self.next_idx < len(self.checkpoints) and elapsed + 1e-12 >= self.checkpoints[self.next_idx]:
            cp = self.checkpoints[self.next_idx]
            self.snapshot[cp] = {
                'elapsed_seconds': float(cp),
                'best_objective': float(self.best),
                'current_objective': float(self.current if math.isfinite(self.current) else self.best),
                'iteration': int(self.iteration),
                'feasible': bool(self.feasible),
            }
            self.next_idx += 1

    def finalize(self, best_obj=None, current_obj=None, iteration=None, feasible=True):
        if not globals().get('SCI_RECORD_WALL_TRACE', False):
            return []
        end_t = self.t0 + (self.time_limit if self.time_limit is not None else 0.0)
        if best_obj is not None:
            self.update(best_obj, current_obj=current_obj, iteration=iteration,
                        feasible=feasible, now=max(time.time(), end_t))
        else:
            self.update(self.best, current_obj=current_obj, iteration=iteration,
                        feasible=feasible, now=max(time.time(), end_t))
        last_best = float(self.best) if math.isfinite(self.best) else None
        last_cur = float(self.current) if math.isfinite(self.current) else last_best
        last_iter = int(self.iteration)
        last_feas = bool(self.feasible)
        rows = []
        for cp in self.checkpoints:
            row = self.snapshot.get(cp)
            if row is None:
                row = {
                    'elapsed_seconds': float(cp),
                    'best_objective': last_best,
                    'current_objective': last_cur if last_cur is not None else last_best,
                    'iteration': last_iter,
                    'feasible': last_feas,
                }
                self.snapshot[cp] = row
            else:
                last_best = row['best_objective']
                last_cur = row['current_objective']
                last_iter = row['iteration']
                last_feas = row['feasible']
            rows.append(dict(row))
        return rows


def validate_schedule_feasibility(schedule, instance, eps=1e-4):
    """校验调度是否满足到港、船序、资源互斥等基本硬约束"""
    if not schedule:
        return False

    expected_task_ids = [int(t['task_id']) for t in instance.get('tasks_data', [])]
    scheduled_task_ids = [int(e['task_id']) for e in schedule]
    if len(scheduled_task_ids) != len(expected_task_ids):
        return False
    if len(set(scheduled_task_ids)) != len(scheduled_task_ids):
        return False
    if set(scheduled_task_ids) != set(expected_task_ids):
        return False

    arrival_times = instance.get('arrival_times', {})
    tasks_data = {int(t['task_id']): t for t in instance.get('tasks_data', [])}

    # 技能可行性（数值越小等级越高：team_level > required → 不合格）
    team_config = instance.get('team_config') or {}
    team_skills = team_config.get('team_skills') or {}
    if team_skills:
        for entry in schedule:
            task = tasks_data.get(int(entry['task_id']))
            if task is None:
                return False
            team_id = entry.get('team')
            skills = team_skills.get(team_id)
            if skills is None:
                try:
                    skills = team_skills.get(int(team_id))
                except (TypeError, ValueError):
                    skills = None
            if skills is None:
                return False
            for skill, required_level in (task.get('skill_req') or {}).items():
                actual_level = skills.get(skill)
                if actual_level is None:
                    return False
                if int(actual_level) > int(required_level):
                    return False

    for entry in schedule:
        if entry['start'] < arrival_times.get(entry['ship_id'], 0.0) - eps:
            return False
        if entry['completion'] < entry['start'] - eps:
            return False
        task_info = tasks_data.get(int(entry['task_id']))
        if task_info and not task_allows_space(task_info, entry.get('space_type')):
            return False

    ship_tasks = {}
    for entry in schedule:
        ship_tasks.setdefault(entry['ship_id'], []).append(entry)
    for tasks in ship_tasks.values():
        tasks.sort(key=lambda e: e['task_id'])
        for i in range(len(tasks) - 1):
            if tasks[i]['completion'] > tasks[i + 1]['start'] + eps:
                return False

    resource_intervals = {}
    for entry in schedule:
        for key in ((entry['space_type'], entry['space']), ('team', entry['team'])):
            resource_intervals.setdefault(key, []).append(
                (entry['start'], entry['completion']))
    for intervals in resource_intervals.values():
        intervals.sort(key=lambda x: x[0])
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0] + eps:
                return False
    return True


# ============================================================
# 简化版启发式工具类（仅保留基础工具函数）
# ============================================================

class HeuristicScheduler:
    """简化版启发式工具类 — 仅保留基础工具函数（配置加载、技能匹配、关键路径计算等）"""

    def __init__(self, alpha=0.3, max_iterations=50, max_ls_evals=3000, time_limit_per_instance=30.0):
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.max_ls_evals = max_ls_evals
        self.time_limit_per_instance = time_limit_per_instance
        self._berths = None
        self._docks = None
        self._teams = None
        self._team_skills = None
        self._processing_time = None

    # ================================================================
    # 配置加载
    # ================================================================

    def _load_team_config(self, instance):
        team_config = instance.get('team_config')
        if team_config is not None:
            self._teams = team_config['teams']
            self._team_skills = team_config['team_skills']
            self._processing_time = team_config['processing_time']
        else:
            self._teams = teams
            self._team_skills = team_skills
            self._processing_time = processing_time

    def _load_resource_config(self, instance):
        self._berths = instance.get('berths', berths)
        self._docks = instance.get('docks', docks)

    # ================================================================
    # 任务-团队匹配辅助
    # ================================================================

    def _get_eligible_teams_for_task(self, task):
        """获取可执行该任务的团队列表（级联机制：高等级覆盖低等级）"""
        skill_req = task.get('skill_req', {})
        eligible = []
        for k in self._teams:
            team_skill_dict = self._team_skills.get(k, {})
            ok = True
            for skill, min_qual in skill_req.items():
                if skill not in team_skill_dict:
                    ok = False
                    break
                if team_skill_dict[skill] > min_qual:
                    ok = False
                    break
            if ok:
                eligible.append(k)
        return eligible if eligible else self._teams.copy()

    def _get_task_duration(self, task, team_id):
        """获取任务在指定团队上的处理时间"""
        skill_req = task.get('skill_req', {})
        duration = 0.0
        for skill in skill_req:
            duration += self._processing_time.get(team_id, {}).get(skill, 1.0)
        return max(duration, 0.01)

    # ================================================================
    # 时间线重建
    # ================================================================

    def _rebuild_schedule_times(self, schedule, instance):
        """根据任务顺序重新计算所有开始/完工时间"""
        arrival_times = instance['arrival_times']
        space_available = {}
        for b in self._berths:
            space_available[('berth', b)] = 0.0
        for d in self._docks:
            space_available[('dock', d)] = 0.0
        team_available = {k: 0.0 for k in self._teams}
        ship_prev_end = {}

        for task in schedule:
            ship_id = task['ship_id']
            space_key = (task['space_type'], task['space'])
            team_id = task['team']
            duration = task['duration']
            arrival = arrival_times.get(ship_id, 0.0)
            prev_end = ship_prev_end.get(ship_id, arrival)

            new_start = max(arrival, prev_end,
                           space_available.get(space_key, 0.0),
                           team_available.get(team_id, 0.0))
            new_completion = new_start + duration

            task['start'] = new_start
            task['completion'] = new_completion
            space_available[space_key] = new_completion
            team_available[team_id] = new_completion
            ship_prev_end[ship_id] = new_completion

        return schedule

    # ================================================================
    # 关键路径计算
    # ================================================================

    def _get_prev_ship_task(self, schedule, idx):
        ship_id = schedule[idx]['ship_id']
        for j in range(idx - 1, -1, -1):
            if schedule[j]['ship_id'] == ship_id:
                return j
        return None

    def _get_next_ship_task(self, schedule, idx):
        ship_id = schedule[idx]['ship_id']
        for j in range(idx + 1, len(schedule)):
            if schedule[j]['ship_id'] == ship_id:
                return j
        return None

    def _get_prev_resource_task(self, schedule, idx):
        task = schedule[idx]
        space_key = (task['space_type'], task['space'])
        team_id = task['team']
        for j in range(idx - 1, -1, -1):
            other = schedule[j]
            if (other['space_type'], other['space']) == space_key:
                return j
            if other['team'] == team_id:
                return j
        return None

    def _get_next_resource_task(self, schedule, idx):
        task = schedule[idx]
        space_key = (task['space_type'], task['space'])
        team_id = task['team']
        for j in range(idx + 1, len(schedule)):
            other = schedule[j]
            if (other['space_type'], other['space']) == space_key:
                return j
            if other['team'] == team_id:
                return j
        return None

    def _compute_critical_path(self, schedule):
        """计算关键路径任务索引集合（ES == LS 的任务）"""
        n = len(schedule)
        es = [0.0] * n
        for i in range(n):
            ship_prev = self._get_prev_ship_task(schedule, i)
            if ship_prev is not None:
                es[i] = max(es[i], es[ship_prev] + schedule[ship_prev]['duration'])
            res_prev = self._get_prev_resource_task(schedule, i)
            if res_prev is not None:
                es[i] = max(es[i], es[res_prev] + schedule[res_prev]['duration'])

        makespan = max(es[i] + schedule[i]['duration'] for i in range(n))

        ls = [makespan] * n
        for i in range(n - 1, -1, -1):
            ls[i] = min(ls[i], makespan - schedule[i]['duration'])
            ship_next = self._get_next_ship_task(schedule, i)
            if ship_next is not None:
                ls[i] = min(ls[i], ls[ship_next] - schedule[i]['duration'])
            res_next = self._get_next_resource_task(schedule, i)
            if res_next is not None:
                ls[i] = min(ls[i], ls[res_next] - schedule[i]['duration'])

        return [i for i in range(n) if abs(es[i] - ls[i]) < 0.001]

    def _compute_objective(self, schedule, instance):
        """计算目标值：total_flow + 10 × misallocation_count"""
        return compute_objective_from_schedule(schedule, instance, self._team_skills)

    def _format_result(self, schedule, objective, solve_time):
        """格式化输出"""
        return {
            'status': 'HEURISTIC_GRASP',
            'objective': objective,
            'solve_time': solve_time,
            'feasible': True,
            'schedule': schedule,
            'num_variables': 0,
            'num_constraints': 0,
            'num_binary': 0,
            'num_continuous': 0,
            'node_count': 0,
            'iteration_count': 0,
            'best_bound': None,
            'mip_gap': None,
        }

    def solve(self, instance, profile=False):
        """保持兼容的 solve 接口 — 委托给贪心构造"""
        schedule = generate_greedy_schedule(instance)
        obj = self._compute_objective(schedule, instance)
        return self._format_result(schedule, obj, 0.0)


# ============================================================
# 多起点贪心初始解构造器
# ============================================================

def _greedy_construct_single(instance, random_seed):
    """单次贪心构造（内部函数）

    通过打乱同批到达船舶的顺序引入多样性。
    船舶内部按 task_id，任务选择最早可用空间+团队，平局按种子随机。
    """
    random.seed(random_seed)

    tasks_data = instance['tasks_data']
    arrival_times = instance['arrival_times']
    num_ships = instance['num_ships']

    team_config = instance.get('team_config')
    if team_config is not None:
        _teams_list = team_config['teams']
        _team_skills_dict = team_config['team_skills']
        _processing_time_dict = team_config['processing_time']
    else:
        _teams_list = teams
        _team_skills_dict = team_skills
        _processing_time_dict = processing_time

    _berths_list = instance.get('berths', berths)
    _docks_list = instance.get('docks', docks)

    ship_tasks = {}
    for ship_id in range(1, num_ships + 1):
        ship_tasks[ship_id] = sorted(
            [t for t in tasks_data if t['ship_id'] == ship_id],
            key=lambda x: x['task_id'])

    # 按到港时间分组，同组内随机打乱顺序引入多样性
    ship_groups = {}
    for s in range(1, num_ships + 1):
        arr = arrival_times.get(s, 0.0)
        ship_groups.setdefault(arr, []).append(s)
    ship_order = []
    for arr in sorted(ship_groups.keys()):
        group = ship_groups[arr]
        random.shuffle(group)
        ship_order.extend(group)

    space_available = {}
    for b in _berths_list:
        space_available[('berth', b)] = 0.0
    for d in _docks_list:
        space_available[('dock', d)] = 0.0
    team_available = {k: 0.0 for k in _teams_list}

    schedule = []

    def _feasible_spaces(task):
        return feasible_spaces_for_task(task, _berths_list, _docks_list)

    def _feasible_teams(task):
        skill_req = task.get('skill_req', {})
        eligible = []
        for k in _teams_list:
            t_skills = _team_skills_dict.get(k, {})
            ok = True
            for skill, min_qual in skill_req.items():
                if skill not in t_skills or t_skills[skill] > min_qual:
                    ok = False
                    break
            if ok:
                eligible.append(k)
        return eligible if eligible else _teams_list.copy()

    def _task_duration(task, team_id):
        d = 0.0
        for skill in task.get('skill_req', {}):
            d += _processing_time_dict.get(team_id, {}).get(skill, 1.0)
        return max(d, 0.01)

    for ship_id in ship_order:
        ship_prev_end = arrival_times.get(ship_id, 0.0)
        for task in ship_tasks[ship_id]:
            candidates = []
            for space_type, space_id in _feasible_spaces(task):
                space_ready = space_available.get((space_type, space_id), 0.0)
                for team_id in _feasible_teams(task):
                    team_ready = team_available.get(team_id, 0.0)
                    dur = _task_duration(task, team_id)
                    actual_start = max(ship_prev_end, space_ready, team_ready)
                    candidates.append((actual_start, dur, space_type, space_id, team_id))

            if not candidates:
                continue

            candidates.sort(key=lambda x: (x[0], random.random() * 0.001))
            best_start, dur, space_type, space_id, team_id = candidates[0]
            end = best_start + dur

            schedule.append({
                'task_id': task['task_id'],
                'ship_id': ship_id,
                'start': best_start,
                'completion': end,
                'duration': dur,
                'space': space_id,
                'space_type': space_type,
                'team': team_id,
                'skill_req': task.get('skill_req', {})
            })

            space_available[(space_type, space_id)] = end
            team_available[team_id] = end
            ship_prev_end = end

    return schedule


def generate_multistart_greedy_schedule(instance, num_starts=5):
    """多起点贪心构造：生成 num_starts 个贪心解，返回目标值最优的 schedule"""
    best_schedule = None
    best_obj = float('inf')

    utils = HeuristicScheduler()
    utils._load_team_config(instance)
    utils._load_resource_config(instance)

    for seed in range(num_starts):
        schedule = _greedy_construct_single(instance, random_seed=seed * 137 + 42)
        schedule = utils._rebuild_schedule_times(schedule, instance)
        obj = utils._compute_objective(schedule, instance)
        if obj < best_obj:
            best_obj = obj
            best_schedule = schedule

    return best_schedule, best_obj


def generate_greedy_schedule(instance, random_seed=None):
    """兼容旧接口：单次贪心构造"""
    return _greedy_construct_single(instance, random_seed=random_seed or 42)


# ============================================================
# ALNS 求解器（章嘉文等 2025 修船厂场景适配版）
# ============================================================

class ALNSSolver:
    """对比基线：通用 ALNS + 模拟退火

    参考：章嘉文等，《考虑潮汐和船舶到港时间不确定性的泊位岸桥联合调度》
    破坏/修复为通用算子集；主算法为 KG-ALNS（领域破坏）；基线为本通用 ALNS。
    目标函数与 Gurobi / KG-ALNS 完全一致：
    total_flow + 10 × misallocation_count
    """

    DESTROY_OPS = [
        'random', 'time_based', 'worst_cost',
    ]
    REPAIR_OPS = [
        'random', 'greedy', 'earliest_time', 'regret',
    ]

    def __init__(self, time_limit=600, max_iter=None, T0=100.0, alpha=0.995,
                 num_runs=10, verbose=False):
        self.time_limit = time_limit
        self.max_iter = max_iter
        self.T0 = T0
        self.alpha = alpha
        self.num_runs = num_runs
        self.verbose = verbose
        self._utils = HeuristicScheduler()
        self.last_archive = []  # [{obj, sched, sig}, ...] 供 Final Polish 多靶

    @staticmethod
    def space_assign_signature(schedule):
        """Φ(S)：任务→空间分配签名（含 task_id，避免仅多重集碰撞）。"""
        if not schedule:
            return 0
        key = tuple(sorted(
            (int(e.get('task_id', -1)),
             str(e.get('space_type', '')),
             e.get('space'))
            for e in schedule
        ))
        return hash(key)

    @staticmethod
    def _archive_prune(archive, best_obj, eta):
        """剔除与当前最优差距超过 η 的条目。"""
        if not archive or best_obj is None:
            return []
        if best_obj <= 0:
            thr = best_obj + max(1.0, abs(best_obj) * eta + 1e-6)
        else:
            thr = (1.0 + eta) * best_obj
        return [e for e in archive if e['obj'] <= thr + 1e-9]

    @staticmethod
    def _archive_try_add(archive, sched, obj, best_obj, eta, capacity):
        """SA 接受的近优劣解：签名去重后插入，再截断为最优 K 个。"""
        if sched is None or capacity <= 0:
            return archive
        if not (best_obj < obj - 0.01):
            return archive
        if best_obj > 0:
            if obj > (1.0 + eta) * best_obj + 1e-9:
                return archive
        elif obj > best_obj + max(1.0, abs(best_obj) * eta) + 1e-9:
            return archive
        sig = ALNSSolver.space_assign_signature(sched)
        entry = {
            'obj': float(obj),
            'sched': [dict(e) for e in sched],
            'sig': sig,
        }
        replaced = False
        for i, old in enumerate(archive):
            if old['sig'] == sig:
                if obj < old['obj'] - 0.01:
                    archive[i] = entry
                replaced = True
                break
        if not replaced:
            archive.append(entry)
        archive.sort(key=lambda e: e['obj'])
        if len(archive) > capacity:
            del archive[capacity:]
        return archive

    def _setup(self, instance):
        self._utils._load_team_config(instance)
        self._utils._load_resource_config(instance)
        self._arrival_times = instance['arrival_times']
        self._tasks_by_id = {t['task_id']: t for t in instance['tasks_data']}
        self._tasks_by_ship = {}
        for t in instance['tasks_data']:
            self._tasks_by_ship.setdefault(t['ship_id'], []).append(t)
        for sid in self._tasks_by_ship:
            self._tasks_by_ship[sid].sort(key=lambda x: x['task_id'])

    def _adaptive_max_iter(self, n_tasks):
        """迭代上限仅作安全阀，实际由 time_limit 控制"""
        if self.max_iter is not None:
            return self.max_iter
        return 50000

    def _rho(self, n):
        return max(3, min(int(0.1 * n), 20))

    @staticmethod
    def _copy_schedule(schedule):
        return copy.deepcopy(schedule)

    def _eval(self, schedule, instance):
        sched = [dict(e) for e in schedule]
        sched = self._utils._rebuild_schedule_times(sched, instance)
        obj = compute_objective_from_schedule(sched, instance, self._utils._team_skills)
        return obj, sched

    def _sort_schedule(self, schedule):
        return sorted(schedule,
                        key=lambda e: (self._arrival_times.get(e['ship_id'], 0.0),
                                       e['ship_id'], e['task_id']))

    def _feasible_spaces(self, task):
        return feasible_spaces_for_task(
            task, self._utils._berths, self._utils._docks)

    def _feasible_teams(self, task):
        return self._utils._get_eligible_teams_for_task(task)

    def _make_entry(self, task, team_id, space_type, space_id):
        dur = self._utils._get_task_duration(task, team_id)
        return {
            'task_id': task['task_id'],
            'ship_id': task['ship_id'],
            'team': team_id,
            'space_type': space_type,
            'space': space_id,
            'duration': dur,
            'skill_req': task.get('skill_req', {}),
            'start': 0.0,
            'completion': 0.0,
        }

    def _insert_assignment(self, schedule, task, team_id, space_type, space_id, instance):
        entry = self._make_entry(task, team_id, space_type, space_id)
        new_sched = self._sort_schedule(schedule + [entry])
        return self._eval(new_sched, instance)

    def _enumerate_insertions(self, schedule, task, instance, limit=15):
        spaces = self._feasible_spaces(task)
        teams = self._feasible_teams(task)
        combos = [(st, sid, tm) for st, sid in spaces for tm in teams]
        if len(combos) > limit:
            combos = random.sample(combos, limit)
        options = []
        for space_type, space_id, team_id in combos:
            obj, sched = self._insert_assignment(
                schedule, task, team_id, space_type, space_id, instance)
            options.append((obj, sched, team_id, space_type, space_id))
        options.sort(key=lambda x: x[0])
        return options

    def _team_loads(self, schedule):
        loads = {k: 0.0 for k in self._utils._teams}
        for e in schedule:
            loads[e['team']] = loads.get(e['team'], 0.0) + e['duration']
        return loads

    def _resource_loads(self, schedule):
        loads = {}
        for e in schedule:
            key = (e['space_type'], e['space'])
            loads[key] = loads.get(key, 0.0) + e['duration']
            tk = ('team', e['team'])
            loads[tk] = loads.get(tk, 0.0) + e['duration']
        return loads

    def _destroy(self, op, schedule, instance):
        n = len(schedule)
        if n <= 1:
            return self._copy_schedule(schedule), []
        rho = min(self._rho(n), n - 1)
        indices = list(range(n))

        if op == 'random':
            remove_idx = random.sample(indices, rho)

        elif op == 'time_based':
            scores = []
            for i, e in enumerate(schedule):
                arrival = self._arrival_times.get(e['ship_id'], 0.0)
                wait = max(0.0, e['start'] - arrival)
                delay = max(0.0, e['completion'] - arrival - e['duration'])
                scores.append((wait + delay + e['duration'], i))
            scores.sort(reverse=True)
            remove_idx = [i for _, i in scores[:rho]]

        elif op == 'resource_bottleneck':
            loads = self._resource_loads(schedule)
            if not loads:
                remove_idx = random.sample(indices, rho)
            else:
                bottleneck = max(loads, key=loads.get)
                if bottleneck[0] == 'team':
                    remove_idx = [i for i, e in enumerate(schedule) if e['team'] == bottleneck[1]]
                else:
                    st, sid = bottleneck
                    remove_idx = [i for i, e in enumerate(schedule)
                                  if e['space_type'] == st and e['space'] == sid]
                if len(remove_idx) > rho:
                    remove_idx = random.sample(remove_idx, rho)
                if len(remove_idx) < rho:
                    extra = [i for i in indices if i not in remove_idx]
                    remove_idx += random.sample(extra, min(rho - len(remove_idx), len(extra)))

        elif op == 'critical_path':
            crit = self._utils._compute_critical_path(schedule)
            if len(crit) >= 2:
                seg = max(1, rho // 2)
                start = random.randint(0, max(0, len(crit) - seg))
                remove_idx = crit[start:start + seg]
            else:
                remove_idx = crit[:rho] if crit else random.sample(indices, rho)

        elif op == 'worst_cost':
            base_obj, _ = self._eval(schedule, instance)
            contributions = []
            sample_idx = indices if n <= 80 else random.sample(indices, 80)
            for i in sample_idx:
                partial = [schedule[j] for j in range(n) if j != i]
                if not partial:
                    continue
                obj, _ = self._eval(partial, instance)
                contributions.append((base_obj - obj, i))
            contributions.sort(reverse=True)
            remove_idx = [i for _, i in contributions[:rho]]
            if len(remove_idx) < rho:
                extra = [i for i in indices if i not in remove_idx]
                remove_idx += random.sample(extra, min(rho - len(remove_idx), len(extra)))

        elif op == 'ship_consecutive':
            ships = list(self._tasks_by_ship.keys())
            ship_id = random.choice(ships)
            ship_tasks = self._tasks_by_ship[ship_id]
            start_pos = random.randint(0, len(ship_tasks) - 1)
            remove_ids = {t['task_id'] for t in ship_tasks[start_pos:]}
            remove_idx = [i for i, e in enumerate(schedule) if e['task_id'] in remove_ids]
            if len(remove_idx) > rho:
                remove_idx = remove_idx[:rho]
            if not remove_idx:
                remove_idx = random.sample(indices, rho)

        else:
            remove_idx = random.sample(indices, rho)

        remove_set = set(remove_idx)
        partial = [schedule[i] for i in range(n) if i not in remove_set]
        removed = [schedule[i] for i in remove_idx]
        removed_tasks = [self._tasks_by_id[e['task_id']] for e in removed]
        return partial, removed_tasks

    def _repair(self, op, partial, removed_tasks, instance):
        if not removed_tasks:
            return self._eval(partial, instance)

        schedule = self._copy_schedule(partial)
        remaining = list(removed_tasks)

        if op == 'random':
            random.shuffle(remaining)
            for task in remaining:
                teams = self._feasible_teams(task)
                spaces = self._feasible_spaces(task)
                if not teams or not spaces:
                    continue
                team_id = random.choice(teams)
                space_type, space_id = random.choice(spaces)
                _, schedule = self._insert_assignment(
                    schedule, task, team_id, space_type, space_id, instance)

        elif op == 'greedy':
            remaining.sort(key=lambda t: (self._arrival_times.get(t['ship_id'], 0.0), t['task_id']))
            for task in remaining:
                options = self._enumerate_insertions(schedule, task, instance)
                if options:
                    _, schedule, _, _, _ = options[0]
                else:
                    team_id = random.choice(self._feasible_teams(task))
                    st, sid = random.choice(self._feasible_spaces(task))
                    _, schedule = self._insert_assignment(schedule, task, team_id, st, sid, instance)

        elif op == 'earliest_time':
            remaining.sort(key=lambda t: (self._arrival_times.get(t['ship_id'], 0.0), t['task_id']))
            for task in remaining:
                best = None
                for space_type, space_id in self._feasible_spaces(task):
                    for team_id in self._feasible_teams(task):
                        obj, sched = self._insert_assignment(
                            schedule, task, team_id, space_type, space_id, instance)
                        start = next(e['start'] for e in sched if e['task_id'] == task['task_id'])
                        key = (start, obj)
                        if best is None or key < best[0]:
                            best = (key, sched)
                schedule = best[1] if best else schedule

        elif op == 'regret':
            unplaced = list(remaining)
            while unplaced:
                best_task, best_sched, best_reg = None, None, -1.0
                for task in unplaced:
                    options = self._enumerate_insertions(schedule, task, instance, limit=2)
                    if not options:
                        teams = self._feasible_teams(task)
                        spaces = self._feasible_spaces(task)
                        if not teams or not spaces:
                            continue
                        team_id = random.choice(teams)
                        st, sid = random.choice(spaces)
                        _, sched = self._insert_assignment(
                            schedule, task, team_id, st, sid, instance)
                        regret = 1e6
                    elif len(options) == 1:
                        regret = options[0][0]
                        sched = options[0][1]
                    else:
                        regret = options[1][0] - options[0][0]
                        sched = options[0][1]
                    if regret > best_reg:
                        best_reg, best_task, best_sched = regret, task, sched
                if best_task is None or best_sched is None:
                    break
                schedule = best_sched
                unplaced.remove(best_task)

        elif op == 'resource_aware':
            loads = self._team_loads(schedule)
            remaining.sort(key=lambda t: (self._arrival_times.get(t['ship_id'], 0.0), t['task_id']))
            for task in remaining:
                teams = sorted(self._feasible_teams(task), key=lambda k: loads.get(k, 0.0))
                best = None
                for team_id in teams[:3]:
                    for space_type, space_id in self._feasible_spaces(task):
                        obj, sched = self._insert_assignment(
                            schedule, task, team_id, space_type, space_id, instance)
                        if best is None or obj < best[0]:
                            best = (obj, sched)
                            loads[team_id] = loads.get(team_id, 0.0) + self._utils._get_task_duration(task, team_id)
                if best:
                    schedule = best[1]

        else:
            return self._repair('greedy', partial, removed_tasks, instance)

        return self._eval(schedule, instance)

    def _select_operator(self, ops, weights):
        total = sum(weights)
        r = random.random() * total
        cum = 0.0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                return i
        return len(ops) - 1

    def _update_weights(self, weights, idx, outcome):
        if outcome == 'improved':
            weights[idx] += 2.0
        elif outcome == 'accepted':
            weights[idx] += 1.0
        else:
            weights[idx] = max(0.2, weights[idx] - 0.5)

    def _solve_single(self, instance, seed, warm_start_schedule=None,
                      wall_deadline=None, intensify_fn=None, stall_threshold=4,
                      collect_archive=False, archive_eta=0.10, archive_capacity=5):
        """单次 ALNS 运行。

        wall_deadline: 绝对墙钟截止（优先于 self.time_limit）
        intensify_fn(best_sched, best_obj, since_best=...) -> (obj, sched)|None
            方案B：停滞时调用；仅当返回解严格更优时写回 best/current。
            since_best 为距上次改进的代数，供强化侧决定是否升级到 MIP。
        collect_archive: 若 True，维护近优劣解档案（空间签名去重），写入 self.last_archive。
        """
        random.seed(seed)
        self._setup(instance)
        n_tasks = len(instance['tasks_data'])
        max_iter = self._adaptive_max_iter(n_tasks)

        if warm_start_schedule is not None:
            current_obj, current_sched = self._eval(warm_start_schedule, instance)
        else:
            init_sched, init_obj = generate_multistart_greedy_schedule(instance, num_starts=3)
            current_obj, current_sched = self._eval(init_sched, instance)
        best_sched, best_obj = current_sched, current_obj

        destroy_w = [1.0] * len(self.DESTROY_OPS)
        repair_w = [1.0] * len(self.REPAIR_OPS)
        T = self.T0
        solve_start = time.time()
        if wall_deadline is None:
            wall_deadline = solve_start + self.time_limit
        convergence = []
        iters_done = 0
        since_best = 0
        intensify_count = 0
        last_intensify_iter = -999
        archive = []
        self.last_archive = []
        eta = float(archive_eta) if archive_eta is not None else 0.10
        cap = int(archive_capacity) if archive_capacity else 5
        # SCI 观测钩子：仅记录墙钟轨迹，不参与接受准则
        _sci_tracer = None
        self.last_wall_trace = []
        self.last_best_solution_time = None
        if globals().get('SCI_RECORD_WALL_TRACE', False):
            _cps = globals().get('SCI_TRACE_CHECKPOINTS', SCI_TRACE_CHECKPOINTS)
            _sci_tracer = SciWallClockTracer(
                _cps, solve_start, time_limit=self.time_limit)
            _sci_tracer.update(best_obj, current_obj=current_obj, iteration=0)

        for iteration in range(max_iter):
            if time.time() >= wall_deadline:
                break
            iters_done = iteration + 1

            d_idx = self._select_operator(self.DESTROY_OPS, destroy_w)
            r_idx = self._select_operator(self.REPAIR_OPS, repair_w)
            partial, removed = self._destroy(self.DESTROY_OPS[d_idx], current_sched, instance)
            new_obj, new_sched = self._repair(self.REPAIR_OPS[r_idx], partial, removed, instance)

            delta = new_obj - current_obj
            accept = sa_accept_move(delta, T)
            outcome = 'rejected'
            if accept:
                current_sched, current_obj = new_sched, new_obj
                outcome = 'accepted'
                if current_obj < best_obj - 0.01:
                    best_sched, best_obj = current_sched, current_obj
                    outcome = 'improved'
                    since_best = 0
                    if collect_archive:
                        archive = self._archive_prune(archive, best_obj, eta)
                else:
                    since_best += 1
                    if collect_archive:
                        archive = self._archive_try_add(
                            archive, current_sched, current_obj, best_obj, eta, cap)
                self._update_weights(destroy_w, d_idx, outcome)
                self._update_weights(repair_w, r_idx, outcome)
            else:
                since_best += 1
                self._update_weights(destroy_w, d_idx, 'rejected')
                self._update_weights(repair_w, r_idx, 'rejected')

            # 方案B：停滞时 improve-only 强化（领域/MIP），失败则不影响主链
            if (intensify_fn is not None
                    and since_best >= stall_threshold
                    and (iteration - last_intensify_iter) >= stall_threshold
                    and time.time() < wall_deadline - 1.0):
                last_intensify_iter = iteration
                try:
                    cand = intensify_fn(best_sched, best_obj, since_best)
                except TypeError:
                    try:
                        cand = intensify_fn(best_sched, best_obj)
                    except Exception:
                        cand = None
                except Exception:
                    cand = None
                if cand is not None:
                    c_obj, c_sched = cand
                    if c_obj < best_obj - 0.01:
                        best_sched, best_obj = c_sched, c_obj
                        current_sched, current_obj = c_sched, c_obj
                        since_best = 0
                        intensify_count += 1
                        if collect_archive:
                            archive = self._archive_prune(archive, best_obj, eta)

            T *= self.alpha
            if (iteration + 1) % 100 == 0:
                destroy_w = [max(0.2, w * 0.7 + 0.3) for w in destroy_w]
                repair_w = [max(0.2, w * 0.7 + 0.3) for w in repair_w]
            if (iteration + 1) % 50 == 0:
                convergence.append({'iteration': iteration + 1, 'objective': best_obj})
            if _sci_tracer is not None:
                _sci_tracer.update(
                    best_obj, current_obj=current_obj, iteration=iters_done)

        solve_time = time.time() - solve_start
        if best_sched is not None:
            best_obj = compute_objective_from_schedule(
                best_sched, instance, self._utils._team_skills)
        if collect_archive:
            archive = self._archive_prune(archive, best_obj, eta)
            self.last_archive = archive
        else:
            self.last_archive = []
        if _sci_tracer is not None:
            self.last_wall_trace = _sci_tracer.finalize(
                best_obj, current_obj=current_obj, iteration=iters_done)
            self.last_best_solution_time = _sci_tracer.best_time
        return best_sched, best_obj, solve_time, iters_done, convergence, intensify_count

    def solve(self, instance):
        self._setup(instance)
        objs, times, all_conv = [], [], []
        best_sched, best_obj = None, float('inf')
        shared = instance.get('_shared_init_schedule')
        for run in range(self.num_runs):
            seed = instance.get('instance_id', 0) * 1000 + run * 17 + 91
            warm = shared if shared is not None else None
            sched, obj, t, iters, conv, _ = self._solve_single(
                instance, seed, warm_start_schedule=warm)
            if sched is not None:
                sched, obj, _ = report_objective_from_schedule(
                    sched, instance, self._utils._team_skills)
            objs.append(obj)
            times.append(t)
            all_conv.append(conv)
            if sched is not None and obj < best_obj:
                best_obj = obj
                best_sched = sched

        return {
            'status': 'ALNS',
            'objective': float(np.mean(objs)),
            'objective_std': float(np.std(objs)) if len(objs) > 1 else 0.0,
            'objective_min': float(min(objs)) if objs else None,
            'objective_max': float(max(objs)) if objs else None,
            'objective_runs': objs,
            'solve_time': float(np.mean(times)),
            'solve_time_std': float(np.std(times)) if len(objs) > 1 else 0.0,
            'feasible': best_sched is not None,
            'schedule': best_sched,
            'num_variables': 0,
            'num_constraints': 0,
            'num_binary': 0,
            'num_continuous': 0,
            'node_count': 0,
            'iteration_count': iters if self.num_runs == 1 else 0,
            'best_bound': None,
            'mip_gap': None,
            'convergence_history': all_conv[0] if all_conv else [],
            'wall_clock_trace': list(getattr(self, 'last_wall_trace', []) or []),
            'best_solution_time': getattr(self, 'last_best_solution_time', None),
            'num_runs': self.num_runs,
        }


class MatheuristicSolver:
    """KG-ALNS+：修船厂知识引导的自适应大邻域搜索

    机制链：知识评价 → 知识引导破坏 → 弱知识修复(近成本 Score=ΔC-α·Ki)
            → 技能感知终局精修 → 停滞 elite restart；保留 V1 minimum weight protection。
    修复骨架与基线 ALNS 同构；仅在成本近似相等时用 Ki，不强制高知识插入。
    diversity_escape 有上限与奖励衰减，避免退化为 Diversity-ALNS。

    对比基线 pure ALNS = ALNSSolver（仅 random/time_based/worst_cost）。
    """

    DOMAIN_DESTROY_LAYERS = (
        'L1_critical_path', 'L2_time_resource', 'L3_skill_dock_chain',
        'L4_bottleneck', 'diversity_escape',
    )
    DOMAIN_L3_SUB_OPS = ('skill_cascade', 'dock_dependency', 'ship_chain')
    DOMAIN_LAYER_L1 = 0
    DOMAIN_LAYER_L2 = 1
    DOMAIN_LAYER_L3 = 2
    DOMAIN_LAYER_L4 = 3
    DOMAIN_LAYER_DIVERSITY = 4
    DOMAIN_REPAIR_MODES = ('skill_dock_aware', 'mip_exact')
    # destroy 自适应权重下限：防止长时间搜索中知识算子被 diversity 挤出
    # 顺序同 DOMAIN_DESTROY_LAYERS；保留原 reward(+2/+1/-0.5) 与周期衰减
    DESTROY_LAYER_MIN_WEIGHTS = (5.0, 5.0, 3.0, 3.0, 10.0)
    # diversity_escape 上限：辅助逃逸，不得长期主导（V1_kg_plus）
    DESTROY_DIVERSITY_MAX_WEIGHT = 50.0
    DIVERSITY_REWARD_SCALE = 0.3          # diversity 正向奖励衰减
    # 知识引导 repair：增量成本近乎相等时才用 Score=ΔC-α·Ki（不牺牲目标）
    KNOWLEDGE_REPAIR_COST_EPS = 0.05      # ε∈[0.05,0.1]；消融可调
    KNOWLEDGE_REPAIR_COST_EPS_REL = 0.0005  # 另取 0.05%·|C| 与绝对阈值较大者
    KNOWLEDGE_REPAIR_ALPHA = 0.05         # α∈[0.05,0.1]；Score=ΔC-α·Ki
    # elite restart：连续无提升 best 时，从 elite 邻域重破坏（仅停滞触发）
    ELITE_RESTART_NO_IMPROVE = 1000       # since_best >= N 触发
    ELITE_RESTART_DESTROY_RATIO = (0.20, 0.30)

    # 消融实验默认开关（KG-ALNS+：V1 权重保护 + 弱知识 repair + 技能精修 + elite restart）
    DEFAULT_ABLATION_FLAGS = {
        'use_mip_repair': False,
        'use_l1_destroy': True,
        'use_l3_destroy': True,
        'use_l4_destroy': True,
        'use_adaptive_weights': True,
        'use_domain_repair': False,    # 保持通用修复骨架；知识仅作 tie-break
        'use_knowledge_eval': True,    # 任务知识评价 Ki=α1P+α2R+α3S
        'use_knowledge_repair': True,  # Knowledge-guided feasible insertion
        'use_knowledge_priority': True,  # False: L1/L3/L4 槽位保留但用随机破坏
        'use_polish': False,           # 不用 MIP polish（公平性）
        'use_team_swap': False,
        'use_skill_refine': True,      # 轻量 skill-aware 局部精修（终局）
        'use_elite_restart': True,     # 停滞时 elite restart
    }

    def __init__(self, time_limit=600, mip_sub_time_limit=90, max_iters=150,
                 neighborhood_size=None, verbose=True, ablation_flags=None,
                 ablation_variant='full', penalty_coeff=None,
                 phase_ratios_override=None, knowledge_weights=None):
        self.time_limit = time_limit
        self.mip_sub_time_limit = mip_sub_time_limit
        self.max_iters = max_iters
        self.neighborhood_size = neighborhood_size
        self.verbose = verbose
        self.ablation_variant = ablation_variant
        self.ablation_flags = dict(self.DEFAULT_ABLATION_FLAGS)
        if ablation_flags:
            self.ablation_flags.update(ablation_flags)
        # 消融可覆盖知识修复 α（置 0 时仍走同一 repair 骨架，仅知识贡献为 0）
        if 'knowledge_repair_alpha' in self.ablation_flags:
            try:
                self.KNOWLEDGE_REPAIR_ALPHA = float(
                    self.ablation_flags['knowledge_repair_alpha'])
            except (TypeError, ValueError):
                pass
        self.penalty_coeff = (penalty_coeff if penalty_coeff is not None
                              else MISALLOCATION_PENALTY_COEFF)
        self.phase_ratios_override = phase_ratios_override
        # 知识权重 (α1,α2,α3)：路径/关键性 P、资源竞争 R、技能复杂 S；默认均衡
        if knowledge_weights is not None and len(knowledge_weights) == 3:
            self.knowledge_weights = tuple(float(x) for x in knowledge_weights)
        else:
            self.knowledge_weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        self._destroy_ratio_fixed = None  # 敏感性：固定破坏比例；None=自适应
        self._selection_counts = {}  # task_id → 被选入邻域的次数（用于多样化加权）
        self._time_series = []
        self._series_t0 = None
        self._init_kg_plus_stats()

    def _init_kg_plus_stats(self):
        """KG+ 终局强化统计（知识引导 / skill refine / elite restart）。"""
        self._kg_plus_stats = {
            'knowledge_guide_count': 0,
            'knowledge_candidate_count': 0,
            'knowledge_changed_choice_count': 0,
            'knowledge_improvement_count': 0,
            'knowledge_destroy_selected_count': 0,
            'skill_refine_improve': 0,
            'elite_restart_count': 0,
            'obj_at_init': None,
            'obj_final': None,
        }
        self._last_repair_knowledge_changed = False

    def _init_destroy_operator_stats(self):
        """按 destroy 层累计：选择/接受/改进/新最优、改善量、最终权重。"""
        self._destroy_operator_stats = {}
        for name in self.DOMAIN_DESTROY_LAYERS:
            self._destroy_operator_stats[name] = {
                'selected': 0,
                'accepted': 0,
                'improved': 0,
                'new_best': 0,
                'sum_improve': 0.0,   # 改进时 (old-new)= -delta，越大越好
                'final_weight': None,
            }

    def _record_destroy_operator(self, layer_idx, outcome, delta=None):
        """记录一次 destroy 算子使用结果（仅诊断模式；默认关闭，不影响 Full）。"""
        if not getattr(self, '_collect_op_stats', False):
            return
        stats = getattr(self, '_destroy_operator_stats', None)
        if stats is None:
            self._init_destroy_operator_stats()
            stats = self._destroy_operator_stats
        layers = self.DOMAIN_DESTROY_LAYERS
        if layer_idx is None or layer_idx < 0 or layer_idx >= len(layers):
            return
        name = layers[layer_idx]
        row = stats[name]
        row['selected'] = int(row.get('selected', 0) or 0) + 1
        if outcome in ('accepted', 'improved'):
            row['accepted'] = int(row.get('accepted', 0) or 0) + 1
        if outcome == 'improved':
            row['new_best'] = int(row.get('new_best', 0) or 0) + 1
        if delta is not None and float(delta) < -1e-9:
            row['improved'] = int(row.get('improved', 0) or 0) + 1
            row['sum_improve'] = float(row.get('sum_improve', 0.0) or 0.0) + (
                -float(delta))

    def _finalize_destroy_operator_weights(self, destroy_w):
        """写入各层最终自适应权重（仅诊断模式）。"""
        if not getattr(self, '_collect_op_stats', False):
            return
        stats = getattr(self, '_destroy_operator_stats', None)
        if stats is None:
            return
        for i, name in enumerate(self.DOMAIN_DESTROY_LAYERS):
            if i < len(destroy_w) and name in stats:
                stats[name]['final_weight'] = float(destroy_w[i])

    def _export_destroy_operator_stats(self):
        """导出可 JSON 序列化的算子诊断（含平均改善量）。"""
        stats = getattr(self, '_destroy_operator_stats', None) or {}
        out = {}
        for name, row in stats.items():
            selected = int(row.get('selected', 0) or 0)
            improved = int(row.get('improved', 0) or 0)
            sum_imp = float(row.get('sum_improve', 0.0) or 0.0)
            out[name] = {
                'selected': selected,
                'accepted': int(row.get('accepted', 0) or 0),
                'improved': improved,
                'new_best': int(row.get('new_best', 0) or 0),
                'sum_improve': sum_imp,
                'avg_improve': (sum_imp / improved) if improved > 0 else 0.0,
                'final_weight': row.get('final_weight'),
            }
        return out

    def _bump_kg_stat(self, key, n=1):
        stats = getattr(self, '_kg_plus_stats', None)
        if stats is None:
            self._init_kg_plus_stats()
            stats = self._kg_plus_stats
        stats[key] = int(stats.get(key, 0) or 0) + int(n)

    def _log_kg_plus_stats(self, force=False):
        """输出 KG+ 模块计数与目标变化（不改 CSV 列）。"""
        stats = getattr(self, '_kg_plus_stats', None) or {}
        if not (force or self.verbose):
            return
        o0 = stats.get('obj_at_init')
        o1 = stats.get('obj_final')
        if o0 is not None and o1 is not None:
            delta = float(o0) - float(o1)
            obj_part = (f"obj {float(o0):.1f}->{float(o1):.1f} "
                        f"(Δ={delta:+.1f})")
        elif o1 is not None:
            obj_part = f"obj={float(o1):.1f}"
        else:
            obj_part = "obj=?"
        print(
            f"  KG+统计: knowledge_guide={stats.get('knowledge_guide_count', 0)}, "
            f"skill_refine_imp={stats.get('skill_refine_improve', 0)}, "
            f"elite_restart={stats.get('elite_restart_count', 0)}, "
            f"{obj_part}",
            flush=True,
        )

    def _reset_time_series(self):
        self._time_series = []
        self._series_t0 = None

    def _snap_time_series(self, best_obj, force=False, min_interval=5.0):
        """记录 (墙钟秒, 最优目标) 供论文收敛曲线使用。"""
        if self._series_t0 is None:
            return
        elapsed = time.time() - self._series_t0
        if force or not self._time_series or elapsed - self._time_series[-1][0] >= min_interval:
            self._time_series.append((round(elapsed, 1), float(best_obj)))

    def _reset_convergence_trace(self, solve_start, time_limit):
        """固定墙钟检查点（30/60/120/180/300s）记录当前最优目标。"""
        self._record_convergence = True
        self._series_t0 = float(solve_start)
        self._convergence_checkpoints = list(
            globals().get('CONVERGENCE_CHECKPOINT_SEC', (30, 60, 120, 180, 300)))
        self._convergence_time_limit = float(time_limit)
        self._convergence_snapshot = {}
        self._convergence_next_cp_idx = 0
        self._trace_running_best = float('inf')

    def _update_convergence_trace(self, best_obj, now=None):
        """在墙钟到达检查点时快照 best-so-far 目标（越小越好）。"""
        if not getattr(self, '_record_convergence', False):
            return
        if self._series_t0 is None:
            return
        try:
            bo = float(best_obj)
        except (TypeError, ValueError):
            return
        if not math.isfinite(bo):
            return
        self._trace_running_best = min(
            getattr(self, '_trace_running_best', bo), bo)
        now = now or time.time()
        elapsed = now - self._series_t0
        idx = int(getattr(self, '_convergence_next_cp_idx', 0) or 0)
        cps = getattr(self, '_convergence_checkpoints', ())
        while idx < len(cps) and elapsed >= float(cps[idx]):
            self._convergence_snapshot[int(cps[idx])] = float(
                self._trace_running_best)
            idx += 1
        self._convergence_next_cp_idx = idx

    def _finalize_convergence_trace(self, best_obj, time_limit=None):
        """求解结束前补齐未触发的检查点（用最终 best-so-far）。"""
        if not getattr(self, '_record_convergence', False):
            return
        tl = float(time_limit or getattr(self, '_convergence_time_limit', 300) or 300)
        self._update_convergence_trace(best_obj, now=self._series_t0 + tl)
        last = float(self._trace_running_best)
        if not math.isfinite(last):
            last = float(best_obj)
        for t in getattr(self, '_convergence_checkpoints', ()):
            t = int(t)
            if t not in self._convergence_snapshot:
                self._convergence_snapshot[t] = last
            else:
                last = min(last, self._convergence_snapshot[t])

    def _export_convergence_trace(self):
        """导出 {检查点秒: 最优目标} 字典（键为 int）。"""
        if not getattr(self, '_record_convergence', False):
            return {}
        return dict(getattr(self, '_convergence_snapshot', {}) or {})

    def _maybe_trace_best(self, best_obj):
        """若启用收敛记录，则更新检查点快照。"""
        if getattr(self, '_record_convergence', False):
            self._update_convergence_trace(best_obj)

    def set_ablation(self, variant='full', flags=None):
        """运行时切换消融配置"""
        self.ablation_variant = variant
        self.ablation_flags = dict(self.DEFAULT_ABLATION_FLAGS)
        if flags:
            self.ablation_flags.update(flags)
        if 'knowledge_repair_alpha' in self.ablation_flags:
            try:
                self.KNOWLEDGE_REPAIR_ALPHA = float(
                    self.ablation_flags['knowledge_repair_alpha'])
            except (TypeError, ValueError):
                pass
        else:
            self.KNOWLEDGE_REPAIR_ALPHA = type(self).KNOWLEDGE_REPAIR_ALPHA

    def _mip_enabled(self):
        """KG 主表默认 False；打开 use_mip_repair 时日志才展示 MIP 字段。"""
        return bool(self.ablation_flags.get('use_mip_repair', False))

    def _initial_destroy_weights(self):
        """Destroy 自适应选择的初始权重（由 DESTROY_PROFILE 切换，不改算子本身）。

        DOMAIN_DESTROY_LAYERS 映射到论文四类 destroy：
          L1_critical_path   → critical task destroy
          L2_time_resource   → random destroy
          L3_skill_dock_chain→ skill complexity destroy
          L4_bottleneck      → resource conflict destroy
          diversity_escape   → 逃逸（baseline 等权；其它 profile 初始为 0）
        返回前施加 DESTROY_LAYER_MIN_WEIGHTS 下限保护。
        """
        n = len(self.DOMAIN_DESTROY_LAYERS)
        profile = str(globals().get('DESTROY_PROFILE', 'baseline') or 'baseline')
        table = globals().get('DESTROY_PROFILE_INIT_WEIGHTS') or {}
        weights = table.get(profile)
        if not weights or len(weights) != n:
            weights = [1.0] * n
        else:
            weights = [max(0.0, float(w)) for w in weights]
        return self._enforce_destroy_weight_floors(weights)

    def _enforce_destroy_weight_floors(self, destroy_w):
        """Minimum weight protection + diversity 上限。

        不改 adaptive reward 规则，仅在更新/衰减后夹紧：
          - 知识/通用 destroy 权重 ≥ DESTROY_LAYER_MIN_WEIGHTS
          - diversity_escape ≤ min(MAX, 1.5×知识算子峰值)，避免退化为 Diversity-ALNS
        """
        mins = self.DESTROY_LAYER_MIN_WEIGHTS
        for i in range(min(len(destroy_w), len(mins))):
            destroy_w[i] = max(float(mins[i]), float(destroy_w[i]))
        di = self.DOMAIN_LAYER_DIVERSITY
        if di < len(destroy_w):
            knowledge_peak = max(
                float(destroy_w[self.DOMAIN_LAYER_L1]),
                float(destroy_w[self.DOMAIN_LAYER_L3]),
                float(destroy_w[self.DOMAIN_LAYER_L4]),
            )
            hard_cap = float(getattr(self, 'DESTROY_DIVERSITY_MAX_WEIGHT', 50.0) or 50.0)
            soft_cap = max(float(mins[di]), knowledge_peak * 1.5)
            destroy_w[di] = min(float(destroy_w[di]), hard_cap, soft_cap)
            destroy_w[di] = max(float(mins[di]), float(destroy_w[di]))
        return destroy_w

    def _apply_destroy_outcome(self, destroy_w, layer_idx, outcome):
        """更新 destroy 权重；diversity 正向奖励按 DIVERSITY_REWARD_SCALE 衰减后再夹紧。"""
        prev = float(destroy_w[layer_idx])
        self._update_layer_weights(destroy_w, layer_idx, outcome)
        if (layer_idx == self.DOMAIN_LAYER_DIVERSITY
                and outcome in ('improved', 'accepted')):
            scale = float(getattr(self, 'DIVERSITY_REWARD_SCALE', 0.3) or 0.3)
            destroy_w[layer_idx] = prev + (float(destroy_w[layer_idx]) - prev) * scale
        self._enforce_destroy_weight_floors(destroy_w)

    def _fixed_layer_weights(self, stall_count):
        """无自适应权重时的固定层级概率（实验前锁定，不随 stall 改）。"""
        locked = globals().get('ABLATION_FIXED_DESTROY_WEIGHTS', None)
        if locked is not None and len(locked) >= 5:
            weights = [float(x) for x in locked[:5]]
        elif stall_count >= 5:
            weights = [0.20, 0.15, 0.15, 0.25, 0.25]
        elif stall_count >= 2:
            weights = [0.30, 0.20, 0.20, 0.30, 0.00]
        else:
            weights = [0.40, 0.15, 0.15, 0.30, 0.00]
        ab = self.ablation_flags
        if not ab.get('use_l1_destroy', True):
            weights[self.DOMAIN_LAYER_L1] = 0.0
        if not ab.get('use_l3_destroy', True):
            weights[self.DOMAIN_LAYER_L3] = 0.0
        if not ab.get('use_l4_destroy', True):
            weights[self.DOMAIN_LAYER_L4] = 0.0
        total = sum(weights)
        if total <= 0:
            # knowledge_off：仅 L2 + diversity
            return [0.0, 0.5, 0.0, 0.0, 0.5]
        return [w / total for w in weights]

    def _apply_instance_scale(self, instance):
        """按算例规模返回求解参数（小规模快速 / 大规模高质量）"""
        num_tasks = instance.get('num_tasks', len(instance.get('tasks_data', [])))
        num_ships = instance.get('num_ships', 0)
        config = instance.get('config', {})
        time_limit = config.get('time_limit', self.time_limit)

        if num_ships <= 10 or num_tasks <= 35:
            bench_mode = (min(time_limit, self.time_limit)
                          >= SMALL_SCALE_BENCH_TIME_LIMIT)
            if bench_mode:
                effective_tl = min(
                    SMALL_SCALE_HEU_TIME_LIMIT, time_limit, self.time_limit)
                return {
                    'time_limit': effective_tl,
                    'quick_mode': True,
                    'hybrid_max_sec': effective_tl,
                    'max_hybrid_iters': min(50, max(20, num_tasks * 3)),
                    'stall_early_exit': 12,
                    'max_iters': min(50, self.max_iters),
                    'sub_time': min(12, self.mip_sub_time_limit),
                    'iter_sub_cap': 15,
                    'neighborhood_cap': 12,
                    'prefer_greedy_init': True,
                    'num_starts': 4,
                    'polish': False,
                    'mip_alns_iters': 50000,
                    'mip_repair_prob': 0.30,
                    'alns_bootstrap_ratio': 0.0,
                    'alns_refine_ratio': 0.0,
                    'alns_init_sec': 0,
                    'alns_warm_sec': 0,
                    'init_max_sec': 6,
                }
            if num_tasks <= 5:
                cap = 10
            elif num_tasks <= 15:
                cap = 20
            else:
                cap = 30
            effective_tl = min(time_limit, self.time_limit, cap)
            return {
                'time_limit': effective_tl,
                'quick_mode': True,
                'hybrid_max_sec': effective_tl,
                'max_hybrid_iters': min(25, max(8, num_tasks * 2)),
                'stall_early_exit': 8,
                'max_iters': min(25, self.max_iters),
                'sub_time': min(10, self.mip_sub_time_limit),
                'iter_sub_cap': 15,
                'neighborhood_cap': 12,
                'prefer_greedy_init': True,
                'num_starts': 5,
                'polish': False,
                'mip_alns_iters': 50000,
                'mip_repair_prob': 0.35,
                'alns_bootstrap_ratio': 0.0,
                'alns_refine_ratio': 0.0,
                'alns_init_sec': 0,
            }
        if num_tasks <= 100:
            tier = config.get('tier', '')
            scale = self._large_scale_profile(
                time_limit=min(time_limit, self.time_limit),
                max_iters=min(50, self.max_iters),
                heavy=False,
                tier=tier,
            )
            return self._finalize_instance_scale(scale, instance)
        scale = self._large_scale_profile(
            time_limit=self.time_limit,
            max_iters=self.max_iters,
            heavy=True,
        )
        return self._finalize_instance_scale(scale, instance)

    def _large_scale_profile(self, time_limit, max_iters, heavy=False, tier=None):
        """大规模算例参数；MIP_ALNS_USE_TUNED_PROFILE=True 时启用调优配置"""
        scale = {
            'time_limit': time_limit,
            'max_iters': max_iters,
            'sub_time': min(15, self.mip_sub_time_limit),
            'iter_sub_cap': 28 if heavy else 25,
            'neighborhood_cap': 18 if heavy else 16,
            'prefer_greedy_init': True,
            'num_starts': 10 if heavy else 8,
            'polish': True,
            'polish_sub_cap': 40 if heavy else 30,
            'mip_alns_iters': 50000,
            'mip_repair_prob': 0.10 if heavy else 0.12,
            'stall_mip_threshold': 3,
            'alns_bootstrap_ratio': 0.20,
            'alns_refine_ratio': 0.30,
            'bootstrap_mip_prob': 0.0,
            'refine_mip_prob': 0.0,
            'alns_init_sec': 18 if heavy else 15,
            'init_max_sec': 32 if heavy else 28,
            'stall_early_exit': 100 if heavy else 80,
            'domain_insert_limit': 12,
            'destroy_ratio_mult': 1.0,
            'use_full_sub_time': False,
            'max_mip_nei': 20,
            'mip_fix_space': True,
            'scale_aware_hybrid': True,
            'scale_aware_threshold': 120,
        }
        if globals().get('LARGE_EQUAL_COMPUTE_BUDGET', True):
            # 定稿主表：同等墙钟统一走方案B参数（不依赖 TIER_APPLY 开关）
            scale.update(MIP_ALNS_EQUAL_BUDGET_120)
        elif globals().get('LARGE_TIER_APPLY_DOMINANCE_PROFILE') and globals().get(
                'MIP_ALNS_TIER1_DOMINANCE', None):
            scale.update(MIP_ALNS_QUALITY_FIRST)
            if globals().get('LARGE_MIP_TIME_LIMIT', 120) >= 300:
                scale.update(MIP_ALNS_QUALITY_FIRST_LONG)
        elif MIP_ALNS_USE_TUNED_PROFILE:
            profile = globals().get('LARGE_MIP_ALNS_PROFILE', 'quality')
            if profile == 'quality':
                tuned = (MIP_ALNS_LARGE_QUALITY_HEAVY if heavy
                         else MIP_ALNS_LARGE_QUALITY)
            else:
                tuned = (MIP_ALNS_LARGE_TUNED_HEAVY if heavy
                         else MIP_ALNS_LARGE_TUNED)
            scale.update(tuned)
        return scale

    def _finalize_instance_scale(self, scale, instance=None):
        """合并阶段预算覆盖与 MIP 子时限敏感性设置"""
        if self.phase_ratios_override:
            scale['alns_bootstrap_ratio'] = self.phase_ratios_override['bootstrap']
            scale['alns_refine_ratio'] = self.phase_ratios_override['refine']
        sub_override = getattr(self, '_sub_time_override', None)
        if sub_override is not None:
            scale['sub_time'] = sub_override
        elif scale.get('use_full_sub_time'):
            scale['sub_time'] = self.mip_sub_time_limit
        if instance is not None and not scale.get('lock_neighborhood'):
            num_tasks = instance.get(
                'num_tasks', len(instance.get('tasks_data', [])))
            if num_tasks >= 60:
                # 动态邻域：约 18% 任务，夹在 [18, 28]
                # 80→18, 100→18, 120→22, 150→27
                cap = min(28, max(18, round(num_tasks * 0.18)))
                scale['neighborhood_cap'] = cap
                scale['domain_insert_limit'] = min(36, cap + 8)
        return scale

    @staticmethod
    def _get_iter_sub_time_limit(scale, n_sel, sub_time):
        """MIP 子问题时限。方案A同等墙钟：硬顶 ≤1s（或 iter_sub_cap）。"""
        floor = scale.get('mip_sub_time_floor', 6)
        cap = scale.get('iter_sub_cap', 8)
        sub_limit = max(floor, int(sub_time or floor))
        if scale.get('equal_budget_tight_mip'):
            return min(cap, max(floor, 1), sub_limit)
        return min(cap, max(floor, int(n_sel * 0.7)), sub_limit)

    def _build_schedule_load_maps(self, schedule):
        """团队/空间负荷及最大值，供邻域打分复用。"""
        team_load, space_load = {}, {}
        for e in schedule:
            team_load[e['team']] = team_load.get(e['team'], 0.0) + e.get('duration', 0.0)
            sk = (e['space_type'], e['space'])
            space_load[sk] = space_load.get(sk, 0.0) + e.get('duration', 0.0)
        max_team = max(team_load.values()) if team_load else 1.0
        max_space = max(space_load.values()) if space_load else 1.0
        return team_load, space_load, max_team, max_space

    def _rank_schedule_entry_score(self, schedule, entry, crit_ids, team_load, space_load,
                                   max_team, max_space, misalloc_bonus=0.0):
        tid = entry['task_id']
        score = 3.0 if tid in crit_ids else 0.0
        score += 1.5 * (team_load.get(entry['team'], 0.0) / max_team)
        score += 1.2 * (space_load.get(
            (entry['space_type'], entry['space']), 0.0) / max_space)
        score += 0.05 * float(entry.get('duration', 0.0) or 0.0)
        score += misalloc_bonus
        return score

    def _select_mip_core_task_ids(self, schedule, candidate_ids, instance, max_nei):
        """从候选破坏任务中挑最重要的 ≤max_nei 个交给 MIP。

        优先：关键路径 → 资源拥挤（团队/泊位/船坞负荷）→ 任务时长。
        其余候选由启发式插入，避免 O(n²) 排序变量爆炸。
        """
        cand = list(candidate_ids)
        if max_nei is None or max_nei <= 0 or len(cand) <= max_nei:
            return set(cand)

        by_id = {e['task_id']: e for e in schedule}
        utils = HeuristicScheduler()
        utils._load_team_config(instance)
        utils._load_resource_config(instance)
        crit_idx = set(utils._compute_critical_path(schedule))
        crit_ids = {schedule[i]['task_id'] for i in crit_idx if i < len(schedule)}
        team_load, space_load, max_team, max_space = self._build_schedule_load_maps(schedule)

        scored = []
        for tid in cand:
            e = by_id.get(tid)
            if e is None:
                scored.append((0.0, tid))
                continue
            score = self._rank_schedule_entry_score(
                schedule, e, crit_ids, team_load, space_load, max_team, max_space)
            scored.append((score, tid))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return {tid for _, tid in scored[:max_nei]}

    def _select_connected_high_score_ids(self, schedule, instance, max_nei,
                                          seed_candidates=None):
        """选「高分且资源/船舶耦合」的连通邻域，供领域强化与短 MIP。

        以高分种子 BFS 扩展：同船、同团队、同空间、时间重叠者优先。
        避免关键路径随机截断导致邻域松散、MIP 1–数秒内无法证明最优。
        """
        if not schedule or max_nei is None or max_nei <= 0:
            return set()
        max_nei = int(max_nei)
        utils = HeuristicScheduler()
        utils._load_team_config(instance)
        utils._load_resource_config(instance)
        crit_idx = set(utils._compute_critical_path(schedule))
        crit_ids = {schedule[i]['task_id'] for i in crit_idx if i < len(schedule)}
        team_load, space_load, max_team, max_space = self._build_schedule_load_maps(schedule)

        # 错配（大材小用）加分：精英队做低技能任务
        elite = set()
        if utils._team_skills:
            elite = {k for k in utils._teams
                     if any(lv == 1 for lv in utils._team_skills.get(k, {}).values())}

        scores = {}
        by_id = {}
        for e in schedule:
            tid = e['task_id']
            by_id[tid] = e
            mis = 0.0
            if e.get('team') in elite:
                req = e.get('skill_req') or {}
                if req and all(int(v) >= 2 for v in req.values()):
                    mis = 1.5
            scores[tid] = self._rank_schedule_entry_score(
                schedule, e, crit_ids, team_load, space_load, max_team, max_space, mis)

        pool = set(seed_candidates) if seed_candidates else set(by_id.keys())
        pool = {t for t in pool if t in by_id}
        if not pool:
            pool = set(by_id.keys())
        if len(pool) <= max_nei:
            return set(pool)

        # 种子：候选中分数最高；无关键则从全表高分起
        seed_pool = pool & crit_ids if (pool & crit_ids) else pool
        seed = max(seed_pool, key=lambda t: (scores.get(t, 0.0), t))
        selected = {seed}
        # 邻接：同船 / 同团队 / 同空间 / 时间重叠
        def _neighbors(tid):
            e = by_id[tid]
            t0, t1 = float(e.get('start', 0.0) or 0.0), float(e.get('completion', 0.0) or 0.0)
            out = []
            for o in schedule:
                oid = o['task_id']
                if oid == tid or oid in selected:
                    continue
                if oid not in pool:
                    # 池外仅吸入强耦合点，保持邻域连通
                    if not (o['ship_id'] == e['ship_id']
                            or o['team'] == e['team']
                            or (o['space_type'], o['space']) == (e['space_type'], e['space'])):
                        continue
                couple = 0.0
                if o['ship_id'] == e['ship_id']:
                    couple += 3.0
                if o['team'] == e['team']:
                    couple += 2.0
                if (o['space_type'], o['space']) == (e['space_type'], e['space']):
                    couple += 2.0
                o0, o1 = float(o.get('start', 0.0) or 0.0), float(o.get('completion', 0.0) or 0.0)
                if t0 < o1 and o0 < t1:
                    couple += 1.5
                if couple <= 0 and oid not in pool:
                    continue
                if couple <= 0 and oid in pool:
                    # 池内弱耦合同样可加，靠分数排序
                    couple = 0.2
                out.append((couple + 0.35 * scores.get(oid, 0.0), oid))
            out.sort(key=lambda x: (-x[0], x[1]))
            return [oid for _, oid in out]

        frontier = list(_neighbors(seed))
        while len(selected) < max_nei and frontier:
            # 取当前前沿最高分，并刷新
            frontier = [t for t in frontier if t not in selected]
            if not frontier:
                break
            frontier.sort(key=lambda t: (-scores.get(t, 0.0), t))
            nxt = frontier.pop(0)
            selected.add(nxt)
            for nb in _neighbors(nxt):
                if nb not in selected and nb not in frontier:
                    frontier.append(nb)

        if len(selected) < min(2, max_nei):
            # 退化：纯高分 top-k
            ranked = sorted(pool, key=lambda t: (-scores.get(t, 0.0), t))
            selected = set(ranked[:max_nei])
        return selected

    def _mip_budget_remaining(self):
        return float(getattr(self, '_mip_budget_left', 0.0) or 0.0)

    def _charge_mip_budget(self, seconds):
        left = self._mip_budget_remaining() - float(seconds)
        self._mip_budget_left = max(0.0, left)
        return self._mip_budget_left

    def _note_mip_call(self, seconds, improved=False, kind='mid'):
        """累计 MIP 调用统计（中途强化 / Final Polish 共用，便于日志名实相符）。"""
        stats = getattr(self, '_last_mip_stats', None)
        if stats is None:
            stats = {'mip_count': 0, 'mip_sec_sum': 0.0, 'iters': 0,
                     'mip_improve': 0, 'polish_mip_count': 0}
            self._last_mip_stats = stats
        stats['mip_count'] = int(stats.get('mip_count', 0) or 0) + 1
        stats['mip_sec_sum'] = float(stats.get('mip_sec_sum', 0.0) or 0.0) + float(seconds)
        if kind == 'polish':
            stats['polish_mip_count'] = int(stats.get('polish_mip_count', 0) or 0) + 1
        if improved:
            stats['mip_improve'] = int(stats.get('mip_improve', 0) or 0) + 1
        stats['mip_budget_left'] = self._mip_budget_remaining()
        return stats

    def _init_plan_b_mip_budget(self, scale, solve_start, deadline):
        """中途 MIP 累计墙钟帽（不含 Final Polish 专属预算）。"""
        tl = float(scale.get('time_limit', max(1.0, deadline - solve_start)))
        if scale.get('mip_budget_cap_sec') is not None:
            cap = float(scale['mip_budget_cap_sec'])
        else:
            frac = float(scale.get('mip_budget_frac', 0.12) or 0.12)
            cap = frac * tl
        self._mip_budget_left = max(0.0, cap)
        self._mip_budget_cap = self._mip_budget_left
        self._next_mip_allowed_at = 0.0
        self._late_unfix_mip_used = False
        return self._mip_budget_left

    def _select_neighborhood(self, schedule, instance, iteration, max_iters, stall_count=0):
        """选择 LNS 邻域 — 自适应破坏比例 + 关键路径聚焦

        主算法：自适应比例 + neighborhood_cap 截断。
        敏感性固定 d（_destroy_ratio_fixed）：严格按比例选邻域，不再被 cap 截断。
        """
        n = len(schedule)
        utils = HeuristicScheduler()
        utils._load_team_config(instance)
        utils._load_resource_config(instance)

        critical_indices = utils._compute_critical_path(schedule)

        fixed_ratio = getattr(self, '_destroy_ratio_fixed', None)
        if fixed_ratio is not None:
            # 敏感性旁路：严格实现设定破坏比例（固定比例策略对比，非主算法自适应）
            ratio = min(0.95, max(0.01, float(fixed_ratio)))
            n_select = int(round(n * ratio))
            n_select = max(1, min(n - 1, n_select)) if n >= 2 else max(1, n)
        else:
            progress = iteration / max(max_iters - 1, 1)
            base_ratio = 0.15 + 0.40 * progress
            if stall_count >= 2:
                base_ratio = min(0.55, base_ratio * 1.4)
            if stall_count >= 5:
                base_ratio = min(0.60, base_ratio * 1.6)
            ratio_mult = getattr(self, '_destroy_ratio_mult', 1.0)
            base_ratio = min(0.65, base_ratio * ratio_mult)
            n_select = max(5, int(round(n * base_ratio)))
            max_cap = getattr(
                self, '_neighborhood_cap',
                max(20, min(50, int(n * 0.50))))
            n_select = min(n_select, max_cap)

        # 70% 来自关键路径
        n_critical = min(len(critical_indices), max(1, int(n_select * 0.7)))
        selected_indices = set(random.sample(
            critical_indices, min(n_critical, len(critical_indices))))

        # 30% 随机补充
        others = [i for i in range(n) if i not in selected_indices]
        n_rest = n_select - len(selected_indices)
        if n_rest > 0 and others:
            selected_indices.update(random.sample(others, min(n_rest, len(others))))

        removed = {schedule[i]['task_id'] for i in selected_indices}
        self._last_actual_destroy_count = int(len(removed))
        self._last_actual_destroy_ratio = float(len(removed) / max(n, 1))
        return removed

    # ====================================================================
    # L2 辅助：时间窗随机破坏（内部函数，被 _random_structure_destroy 调用）
    # ====================================================================

    def _select_time_window(self, schedule):
        """在 makespan 上随机截取 10%-40% 连续时间段内的所有任务"""
        if not schedule:
            return set()
        makespan = max(t['completion'] for t in schedule)
        if makespan <= 0:
            return {schedule[0]['task_id']} if schedule else set()
        t1 = random.uniform(0, makespan * 0.7)
        t2 = t1 + random.uniform(makespan * 0.1, makespan * 0.4)
        return {t['task_id'] for t in schedule
                if t['start'] <= t2 and t['completion'] >= t1}

    # ====================================================================
    # L2 辅助：随机资源破坏（内部函数，被 _random_structure_destroy 调用）
    # ====================================================================

    def _select_random_resource(self, schedule):
        """随机选一个团队的全部任务，邻域太小时追加一个泊位"""
        selected = set()
        if not self._teams:
            return selected
        k = random.choice(self._teams)
        for t in schedule:
            if t['team'] == k:
                selected.add(t['task_id'])
        if len(selected) < 3 and self._berths:
            b = random.choice(self._berths)
            for t in schedule:
                if t['space_type'] == 'berth' and t['space'] == b:
                    selected.add(t['task_id'])
        return selected

    # ====================================================================
    # L2 入口：混合随机结构破坏（合并时间窗 + 资源破坏）
    # ====================================================================

    def _random_structure_destroy(self, schedule):
        """L2 混合随机破坏：等概率调用时间窗破坏或随机资源破坏

        设计思想：随机打乱当前调度的时间结构和资源分配结构。
        两种子策略互补 —— 时间窗打乱连续时间段，资源破坏重置某一资源的所有任务。
        合并为一个入口便于上层按比例调度。
        """
        if random.random() < 0.5:
            return self._select_time_window(schedule)
        else:
            return self._select_random_resource(schedule)

    # ====================================================================
    # L3-E：资质级联破坏
    # ====================================================================

    def _skill_cascade_destroy(self, schedule):
        """E: 资质级联破坏

        设计思想：利用模型核心特征"技能等级级联机制"（高等级可覆盖低等级）。
        选择一个持有最高等级（level=1）技能的团队，找出该团队当前承担的、
        实际只需低等级技能（要求≥2）即可完成的任务 —— 这些任务"浪费"了
        高级团队的能力。破坏这些任务的团队分配，迫使 MIP 将它们重新分配给
        低等级团队，释放高级团队去处理真正的硬任务。
        """
        selected = set()
        if not self._team_skills:
            return selected

        # 找出所有在某个技能上持有最高资质（level=1）的团队
        elite_teams = [
            k for k in self._teams
            if any(level == 1 for level in self._team_skills.get(k, {}).values())
        ]
        if not elite_teams:
            return selected

        k = random.choice(elite_teams)
        team_skill_dict = self._team_skills.get(k, {})

        for t in schedule:
            if t['team'] != k:
                continue
            skill_req = t.get('skill_req', {})
            # 该任务的所有技能要求都 ≥2 → 属于"低级任务"，高级团队在做低端活
            if skill_req and all(req >= 2 for req in skill_req.values()):
                selected.add(t['task_id'])

        return selected

    # ====================================================================
    # L3-F：空间-技能依赖破坏
    # ====================================================================

    def _dock_dependency_destroy(self, schedule):
        """F: 空间-技能依赖破坏

        设计思想：利用模型约束"水下/船体维修必须在干船坞执行"。
        干船坞是系统中最稀缺的空间资源（数量远少于泊位），船坞任务的调度
        质量直接影响 makespan。随机选取 30%-50% 的船坞强制任务，连带破坏
        与其共享同一船坞的其他任务，迫使 MIP 重新优化船坞使用序列。
        """
        selected = set()
        if not self._docks:
            return selected

        # 找出所有必须在干船坞执行的任务
        dock_tasks = [t for t in schedule if t['space_type'] == 'dock']
        if not dock_tasks:
            return selected

        # 随机选取 30%-50%
        ratio = random.uniform(0.3, 0.5)
        n_pick = max(1, int(len(dock_tasks) * ratio))
        picked = random.sample(dock_tasks, min(n_pick, len(dock_tasks)))

        for pt in picked:
            selected.add(pt['task_id'])
            # 连带同船坞的其他任务
            dock_id = pt['space']
            for t in schedule:
                if t['space_type'] == 'dock' and t['space'] == dock_id:
                    selected.add(t['task_id'])

        return selected

    # ====================================================================
    # L4：资源瓶颈破坏
    # ====================================================================

    def _resource_bottleneck_destroy(self, schedule):
        """L4: 资源瓶颈破坏

        找出占用时长最高的 2 个泊位、1 个团队，将其上全部任务纳入邻域；
        若不足 10 个任务，则从关键路径补充。
        """
        selected = set()
        if not schedule:
            return selected

        berth_load = {}
        for t in schedule:
            if t.get('space_type') == 'berth' and t.get('space') is not None:
                b = t['space']
                berth_load[b] = berth_load.get(b, 0.0) + t.get('duration', 0.0)

        if berth_load:
            top_berths = sorted(berth_load, key=lambda b: berth_load[b], reverse=True)[:2]
            for b in top_berths:
                for t in schedule:
                    if t.get('space_type') == 'berth' and t.get('space') == b:
                        selected.add(t['task_id'])

        team_load = {}
        for t in schedule:
            team = t.get('team')
            if team is not None:
                team_load[team] = team_load.get(team, 0.0) + t.get('duration', 0.0)

        if team_load:
            top_team = max(team_load, key=lambda k: team_load[k])
            for t in schedule:
                if t.get('team') == top_team:
                    selected.add(t['task_id'])

        if len(selected) < 10:
            utils = HeuristicScheduler()
            critical_indices = utils._compute_critical_path(schedule)
            for idx in critical_indices:
                if 0 <= idx < len(schedule):
                    selected.add(schedule[idx]['task_id'])
                if len(selected) >= 10:
                    break

        return selected

    # ====================================================================
    # L3-G：任务-船舶关键链破坏
    # ====================================================================

    def _ship_chain_destroy(self, schedule, instance):
        """G: 任务-船舶关键链破坏

        设计思想：船舶维修任务是按顺序执行的（约束9），同船任务构成一条链。
        对每艘船计算每个任务的"松弛时间" = LS - ES（利用关键路径分析），
        松弛越小的任务越紧张。选择松弛最小的连续 2-4 个任务链加入邻域，
        集中 MIP 算力优化最紧张的工序段。
        """
        selected = set()
        arrival_times = instance.get('arrival_times', {})

        # 按 ship_id 分组
        ship_groups = {}
        for t in schedule:
            ship_groups.setdefault(t['ship_id'], []).append(t)
        for ship_tasks in ship_groups.values():
            ship_tasks.sort(key=lambda x: x['task_id'])

        for ship_id, ship_tasks in ship_groups.items():
            if len(ship_tasks) < 2:
                continue

            # 计算每个任务的 ES（最早开始）和 LS（最晚开始）的简化版
            # ES[i] = max(arrival, prev_ES + prev_dur)
            # LS[i] = min(next_LS - dur_i, ...)
            n = len(ship_tasks)
            es = [0.0] * n
            arrival = arrival_times.get(ship_id, 0.0)
            for i in range(n):
                if i == 0:
                    es[i] = max(arrival, ship_tasks[i]['start'])
                else:
                    es[i] = max(ship_tasks[i]['start'],
                                es[i - 1] + ship_tasks[i - 1]['duration'])

            # 简化 LS：从后往前推
            ls = [float('inf')] * n
            makespan_ship = max(t['completion'] for t in ship_tasks)
            for i in range(n - 1, -1, -1):
                if i == n - 1:
                    ls[i] = makespan_ship - ship_tasks[i]['duration']
                else:
                    ls[i] = min(makespan_ship - ship_tasks[i]['duration'],
                                ls[i + 1] - ship_tasks[i]['duration'])

            # 计算每个任务的松弛
            slacks = [max(0.0, ls[i] - es[i]) for i in range(n)]

            # 找到松弛之和最小的连续 2-4 个任务的窗口
            chain_len = random.randint(2, min(4, n))
            best_sum = float('inf')
            best_start = 0
            for start in range(n - chain_len + 1):
                chain_sum = sum(slacks[start:start + chain_len])
                if chain_sum < best_sum:
                    best_sum = chain_sum
                    best_start = start

            for i in range(best_start, best_start + chain_len):
                selected.add(ship_tasks[i]['task_id'])

        return selected

    # ====================================================================
    # L3 入口：领域知识破坏（统一调度 E / F / G）
    # ====================================================================

    def _domain_knowledge_destroy(self, schedule, instance, sub_weights=None):
        """L3 领域知识破坏：E/F/G 子算子自适应加权选择

        E 资质级联 | F 船坞依赖 | G 船舶关键链
        """
        if sub_weights is None:
            sub_weights = [1.0, 1.0, 1.0]
        sub_idx = self._select_weighted_index(sub_weights)
        if sub_idx == 0:
            return self._skill_cascade_destroy(schedule), 0
        if sub_idx == 1:
            return self._dock_dependency_destroy(schedule), 1
        return self._ship_chain_destroy(schedule, instance), 2

    # ====================================================================
    # 停滞逃逸：多样化加权随机破坏（最高优先级）
    # ====================================================================

    def _diversity_destroy(self, schedule):
        """停滞逃逸：加权随机抽样，历史被选次数越少的任务权重越高

        破坏比例 40%，权重 = 1/(count+1)，保证长期未被选中的任务获得曝光。
        仅在 stall_count ≥ 5 时触发，优先级高于 L1/L2/L3。
        """
        n = len(schedule)
        k = max(5, int(n * 0.4))
        weights = [
            1.0 / (self._selection_counts.get(t['task_id'], 0) + 1.0)
            for t in schedule
        ]
        chosen_ids = random.choices(
            [t['task_id'] for t in schedule], weights=weights, k=min(k, n))
        return set(chosen_ids)

    # ====================================================================
    # 配置懒加载
    # ====================================================================

    def _ensure_config_loaded(self, instance):
        """懒加载团队和资源配置到 self，供 L2/L3 破坏算子使用"""
        team_config = instance.get('team_config')
        if team_config is not None:
            self._teams = team_config['teams']
            self._team_skills = team_config['team_skills']
            self._processing_time = team_config['processing_time']
        else:
            # 回退到模块级全局默认值
            self._teams = teams
            self._team_skills = team_skills
            self._processing_time = processing_time
        self._berths = instance.get('berths', berths)
        self._docks = instance.get('docks', docks)

    # ====================================================================
    # 领域破坏算子 — 自适应权重调度
    # ====================================================================

    @staticmethod
    def _select_weighted_index(weights):
        total = sum(weights)
        if total <= 0:
            return 0
        r = random.random() * total
        cum = 0.0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                return i
        return len(weights) - 1

    @staticmethod
    def _update_layer_weights(weights, idx, outcome):
        """与 ALNS 基线一致的得分/惩罚规则（destroy 下限在更新后单独 enforce）"""
        if outcome == 'improved':
            weights[idx] += 2.0
        elif outcome == 'accepted':
            weights[idx] += 1.0
        else:
            weights[idx] = max(0.2, weights[idx] - 0.5)

    def _select_domain_layer(self, destroy_w, stall_count):
        """按自适应权重或固定概率选择 L1–L4 / diversity"""
        ab = self.ablation_flags
        if ab.get('use_adaptive_weights', True):
            w = list(destroy_w)
            if not ab.get('use_l1_destroy', True):
                w[self.DOMAIN_LAYER_L1] = 0.0
            if not ab.get('use_l3_destroy', True):
                w[self.DOMAIN_LAYER_L3] = 0.0
            if not ab.get('use_l4_destroy', True):
                w[self.DOMAIN_LAYER_L4] = 0.0
            if stall_count >= 5:
                w[self.DOMAIN_LAYER_DIVERSITY] *= 2.5
            if stall_count >= 2 and ab.get('use_l4_destroy', True):
                w[self.DOMAIN_LAYER_L4] *= 1.3
            if stall_count >= 2 and ab.get('use_l3_destroy', True):
                w[self.DOMAIN_LAYER_L3] *= 1.2
            if sum(w) <= 0:
                return self.DOMAIN_LAYER_L2
            return self._select_weighted_index(w)
        fixed = self._fixed_layer_weights(stall_count)
        return self._select_weighted_index(fixed)

    def _destroy_by_layer(self, layer_idx, schedule, instance, iteration, max_iters,
                          stall_count, l3_sub_w):
        """执行指定层级的领域破坏，返回 (task_ids, l3_sub_idx|None)"""
        self._ensure_config_loaded(instance)
        l3_sub_idx = None

        ab = self.ablation_flags
        use_kp = ab.get('use_knowledge_priority', True)
        if layer_idx == self.DOMAIN_LAYER_DIVERSITY:
            removed_ids = self._diversity_destroy(schedule)
        elif layer_idx == self.DOMAIN_LAYER_L1 and ab.get('use_l1_destroy', True):
            if use_kp:
                removed_ids = self._select_neighborhood(
                    schedule, instance, iteration, max_iters, stall_count)
            else:
                removed_ids = self._random_structure_destroy(schedule)
        elif layer_idx == self.DOMAIN_LAYER_L2:
            removed_ids = self._random_structure_destroy(schedule)
        elif layer_idx == self.DOMAIN_LAYER_L3:
            if ab.get('use_l3_destroy', True):
                if use_kp:
                    removed_ids, l3_sub_idx = self._domain_knowledge_destroy(
                        schedule, instance, l3_sub_w)
                else:
                    removed_ids = self._random_structure_destroy(schedule)
            else:
                removed_ids = self._random_structure_destroy(schedule)
        elif layer_idx == self.DOMAIN_LAYER_L4 and ab.get('use_l4_destroy', True):
            if use_kp:
                removed_ids = self._resource_bottleneck_destroy(schedule)
            else:
                removed_ids = self._random_structure_destroy(schedule)
        else:
            removed_ids = self._random_structure_destroy(schedule)

        if len(removed_ids) < 2:
            # 回退：知识层关闭时用 L2，避免偷偷退回 L1 关键路径
            if ab.get('use_l1_destroy', True):
                removed_ids = self._select_neighborhood(
                    schedule, instance, iteration, max_iters, stall_count)
            else:
                removed_ids = self._random_structure_destroy(schedule)
            l3_sub_idx = None
        return removed_ids, l3_sub_idx

    def _select_neighborhood_multi(self, schedule, instance, iteration, max_iters,
                                  stall_count, destroy_w, l3_sub_w):
        """兼容入口：按自适应权重选择层级并破坏（替代原固定概率表）"""
        layer_idx = self._select_domain_layer(destroy_w, stall_count)
        removed_ids, _ = self._destroy_by_layer(
            layer_idx, schedule, instance, iteration, max_iters, stall_count, l3_sub_w)
        return removed_ids, layer_idx

    # ================================================================
    # 固定非邻域任务
    # ================================================================

    def _fix_other_tasks(self, schedule, selected_task_ids):
        """生成固定任务信息字典

        对于非选中任务，记录其资源分配和时间安排，作为子问题 MIP 的硬约束。
        """
        fixed = {}
        for entry in schedule:
            t_id = entry['task_id']
            if t_id not in selected_task_ids:
                fixed[t_id] = {
                    'team': entry['team'],
                    'space_type': entry['space_type'],
                    'space': entry['space'],
                    'start': entry['start'],
                    'completion': entry['completion'],
                    'duration': entry['duration'],
                    'ship_id': entry['ship_id'],
                }
        return fixed

    # ================================================================
    # 构建并求解受限 MIP 子问题
    # ================================================================

    def _solve_mip_subproblem(self, instance, selected_task_ids, fixed_assignments,
                              warm_start_schedule=None, fix_space=None):
        """构建受限 MIP：仅对选中任务重新优化，固定其余任务。

        warm_start_schedule: 当前全局调度，用于 Gurobi Start 热启动。
        fix_space: True 时固定空间资源，只优化团队与时间顺序（大幅削减二元变量）。
        """
        if not GUROBI_AVAILABLE:
            return None

        selected_tasks = sorted(selected_task_ids)
        if len(selected_tasks) < 2:
            return None

        active_scale = getattr(self, '_active_scale', {}) or {}
        if fix_space is None:
            fix_space = bool(active_scale.get('mip_fix_space', True))

        # 硬上限二次保险
        max_nei = int(active_scale.get('max_mip_nei', 20) or 20)
        if max_nei > 0 and len(selected_tasks) > max_nei:
            if warm_start_schedule:
                core = self._select_mip_core_task_ids(
                    warm_start_schedule, selected_tasks, instance, max_nei)
            else:
                core = set(selected_tasks[:max_nei])
            # 多余任务并入固定（若有暖解）
            for tid in selected_tasks:
                if tid not in core and warm_start_schedule is not None:
                    for e in warm_start_schedule:
                        if e['task_id'] == tid and tid not in fixed_assignments:
                            fixed_assignments[tid] = {
                                'ship_id': e['ship_id'], 'start': e['start'],
                                'completion': e['completion'],
                                'duration': e['duration'],
                                'space': e['space'], 'space_type': e['space_type'],
                                'team': e['team'],
                            }
            selected_tasks = sorted(core)

        # 固定空间模式：只要暖解覆盖所选任务，就走轻量模型
        if fix_space and warm_start_schedule is not None:
            by_id = {e['task_id']: e for e in warm_start_schedule}
            if all(t in by_id for t in selected_tasks):
                return self._solve_mip_subproblem_fix_space(
                    instance, selected_tasks, fixed_assignments, warm_start_schedule)

        return self._solve_mip_subproblem_full(
            instance, selected_tasks, fixed_assignments, warm_start_schedule)

    def _apply_mip_warm_start(self, model, selected_tasks, warm_by_id, S, C, x, y, z,
                              _berths, _docks, _teams, u_berth=None, u_dock=None,
                              u_team=None, fixed_space=None):
        """将破坏前的分配写入 Start，让 Gurobi 跳过找可行解阶段。"""
        if not warm_by_id:
            return
        for t in selected_tasks:
            e = warm_by_id.get(t)
            if e is None:
                continue
            try:
                S[t].Start = float(e.get('start', 0.0))
                C[t].Start = float(e.get('completion', 0.0))
            except Exception:
                pass
            if fixed_space is None:
                for b in _berths:
                    try:
                        x[t, b].Start = 1.0 if (
                            e.get('space_type') == 'berth' and e.get('space') == b) else 0.0
                    except Exception:
                        pass
                for d in _docks:
                    try:
                        y[t, d].Start = 1.0 if (
                            e.get('space_type') == 'dock' and e.get('space') == d) else 0.0
                    except Exception:
                        pass
            for k in _teams:
                try:
                    z[t, k].Start = 1.0 if e.get('team') == k else 0.0
                except Exception:
                    pass

        # 排序变量：按暖解开始时间给初值
        if u_berth is not None:
            for (t1, t2, b), var in u_berth.items():
                e1, e2 = warm_by_id.get(t1), warm_by_id.get(t2)
                if e1 and e2:
                    try:
                        var.Start = 1.0 if e1.get('start', 0) <= e2.get('start', 0) else 0.0
                    except Exception:
                        pass
        if u_dock is not None:
            for (t1, t2, d), var in u_dock.items():
                e1, e2 = warm_by_id.get(t1), warm_by_id.get(t2)
                if e1 and e2:
                    try:
                        var.Start = 1.0 if e1.get('start', 0) <= e2.get('start', 0) else 0.0
                    except Exception:
                        pass
        if u_team is not None:
            for (t1, t2, k), var in u_team.items():
                e1, e2 = warm_by_id.get(t1), warm_by_id.get(t2)
                if e1 and e2:
                    try:
                        var.Start = 1.0 if e1.get('start', 0) <= e2.get('start', 0) else 0.0
                    except Exception:
                        pass

    def _solve_mip_subproblem_fix_space(self, instance, selected_tasks, fixed_assignments,
                                         warm_start_schedule):
        """轻量子问题：固定空间，只优化团队选择与开始/完工时间 + 必要排序。

        相对 full 模型：去掉 x/y 二元变量及跨泊位/船坞的冗余 Big-M，
        排序变量仅在「同空间」或「潜在同团队」的选中任务对上创建。
        """
        tasks_data = instance['tasks_data']
        arrival_times = instance['arrival_times']
        num_ships = instance['num_ships']
        ships = list(range(1, num_ships + 1))
        team_config = instance.get('team_config')
        if team_config is not None:
            _teams = team_config['teams']
            _team_skills = team_config['team_skills']
            _processing_time = team_config['processing_time']
        else:
            _teams = teams
            _team_skills = team_skills
            _processing_time = processing_time

        warm_by_id = {e['task_id']: e for e in warm_start_schedule}
        fixed_space = {}
        for t in selected_tasks:
            e = warm_by_id[t]
            fixed_space[t] = (e['space_type'], e['space'])

        task_durations, eligible_teams_dict = {}, {}
        for t_id in selected_tasks:
            task_info = next((t for t in tasks_data if t['task_id'] == t_id), None)
            if task_info is None:
                continue
            task_durations[t_id] = {}
            for k in _teams:
                dur, feasible = 0.0, True
                for skill in task_info.get('skill_req', {}):
                    if skill in _processing_time.get(k, {}):
                        dur += _processing_time[k][skill]
                    else:
                        feasible = False
                        break
                task_durations[t_id][k] = dur if feasible else None
            skill_req = task_info.get('skill_req', {})
            eligible = []
            for k in _teams:
                t_skills = _team_skills.get(k, {})
                ok = all(skill in t_skills and t_skills[skill] <= min_qual
                         for skill, min_qual in skill_req.items())
                if ok and task_durations[t_id].get(k) is not None:
                    eligible.append(k)
            eligible_teams_dict[t_id] = eligible if eligible else list(_teams)

        task_to_ship = {t['task_id']: t['ship_id'] for t in tasks_data}
        ship_tasks = {ship: sorted(
            [t['task_id'] for t in tasks_data if t['ship_id'] == ship]) for ship in ships}

        max_dur = max(
            (d for td in task_durations.values() for d in td.values() if d is not None),
            default=5.0)
        max_arrival = max(arrival_times.values()) if arrival_times else 20.0
        fixed_max_time = max(
            (f['completion'] for f in fixed_assignments.values()), default=0.0)
        M = max_arrival + fixed_max_time + max_dur * (len(selected_tasks) + 5) + 50

        env = gp.Env(params={"OutputFlag": 0})
        model = gp.Model('MIP_LNS_SubFixSpace', env=env)
        S, C, z = {}, {}, {}
        for t in selected_tasks:
            S[t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"S_{t}")
            C[t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"C_{t}")
            for k in eligible_teams_dict.get(t, _teams):
                z[t, k] = model.addVar(vtype=GRB.BINARY, name=f"z_{t}_{k}")

        ship_completion = {
            ship: model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"C_ship_{ship}")
            for ship in ships
        }

        # 仅同空间任务对需要空间排序变量；团队排序仅对有资格重叠的团队
        u_space, u_team = {}, {}
        for i, t1 in enumerate(selected_tasks):
            for t2 in selected_tasks[i + 1:]:
                if fixed_space[t1] == fixed_space[t2]:
                    u_space[t1, t2] = model.addVar(vtype=GRB.BINARY)
                common = set(eligible_teams_dict.get(t1, [])).intersection(
                    eligible_teams_dict.get(t2, []))
                for k in common:
                    u_team[t1, t2, k] = model.addVar(vtype=GRB.BINARY)
        model.update()

        for t in selected_tasks:
            elig = eligible_teams_dict.get(t, _teams)
            model.addConstr(gp.quicksum(z[t, k] for k in elig) == 1, name=f"C3_{t}")
            model.addConstr(
                C[t] == S[t] + gp.quicksum(
                    z[t, k] * task_durations[t][k]
                    for k in elig if task_durations[t].get(k) is not None),
                name=f"C5_{t}")
            ship = task_to_ship.get(t, 1)
            model.addConstr(S[t] >= arrival_times.get(ship, 0.0), name=f"C10_{t}")

        # 同空间互斥
        for (t1, t2), u in u_space.items():
            model.addConstr(C[t1] <= S[t2] + M * (1 - u), name=f"Sp_a_{t1}_{t2}")
            model.addConstr(C[t2] <= S[t1] + M * u, name=f"Sp_b_{t1}_{t2}")

        # 同团队互斥
        for (t1, t2, k), u in u_team.items():
            model.addConstr(
                C[t1] <= S[t2] + M * (1 - u) + M * (2 - z[t1, k] - z[t2, k]),
                name=f"T_a_{t1}_{t2}_{k}")
            model.addConstr(
                C[t2] <= S[t1] + M * u + M * (2 - z[t1, k] - z[t2, k]),
                name=f"T_b_{t1}_{t2}_{k}")

        # 与固定任务：仅同空间 / 可能同团队
        for t_sel in selected_tasks:
            stype, sid = fixed_space[t_sel]
            for t_fix, finfo in fixed_assignments.items():
                if finfo['space_type'] == stype and finfo['space'] == sid:
                    u = model.addVar(vtype=GRB.BINARY)
                    model.addConstr(
                        C[t_sel] <= finfo['start'] + M * (1 - u))
                    model.addConstr(
                        S[t_sel] >= finfo['completion'] - M * u)
            elig = set(eligible_teams_dict.get(t_sel, []))
            for t_fix, finfo in fixed_assignments.items():
                if finfo['team'] in elig:
                    u = model.addVar(vtype=GRB.BINARY)
                    k = finfo['team']
                    model.addConstr(
                        C[t_sel] <= finfo['start'] + M * (1 - u) + M * (1 - z[t_sel, k]))
                    model.addConstr(
                        S[t_sel] >= finfo['completion'] - M * u + M * (1 - z[t_sel, k]))

        for ship in ships:
            task_list = ship_tasks[ship]
            for i in range(len(task_list) - 1):
                prev_t, next_t = task_list[i], task_list[i + 1]
                if prev_t in selected_tasks and next_t in selected_tasks:
                    model.addConstr(C[prev_t] <= S[next_t])
                elif prev_t in fixed_assignments and next_t in selected_tasks:
                    model.addConstr(S[next_t] >= fixed_assignments[prev_t]['completion'])
                elif prev_t in selected_tasks and next_t in fixed_assignments:
                    model.addConstr(C[prev_t] <= fixed_assignments[next_t]['start'])

        for ship in ships:
            max_fixed_comp = max(
                (f['completion'] for f in fixed_assignments.values()
                 if f['ship_id'] == ship), default=0.0)
            if max_fixed_comp > 0:
                model.addConstr(ship_completion[ship] >= max_fixed_comp)
            for t in selected_tasks:
                if task_to_ship.get(t, 0) == ship:
                    model.addConstr(ship_completion[ship] >= C[t])

        flow_obj = gp.quicksum(
            ship_completion[ship] - arrival_times.get(ship, 0.0) for ship in ships)
        elite_teams = [k for k in _teams
                       if any(lv == 1 for lv in _team_skills.get(k, {}).values())]
        pen = []
        for t in selected_tasks:
            task_info = next((ti for ti in tasks_data if ti['task_id'] == t), None)
            if not task_info:
                continue
            skill_req = task_info.get('skill_req', {})
            if skill_req and all(req >= 2 for req in skill_req.values()):
                for k in elite_teams:
                    if (t, k) in z:
                        pen.append(z[t, k])
        model.setObjective(
            flow_obj + (self.penalty_coeff * gp.quicksum(pen) if pen else 0), GRB.MINIMIZE)

        self._apply_mip_warm_start(
            model, selected_tasks, warm_by_id, S, C, None, None, z,
            [], [], _teams, u_team=u_team, fixed_space=fixed_space)
        # 空间序 Start
        for (t1, t2), u in u_space.items():
            e1, e2 = warm_by_id.get(t1), warm_by_id.get(t2)
            if e1 and e2:
                try:
                    u.Start = 1.0 if e1.get('start', 0) <= e2.get('start', 0) else 0.0
                except Exception:
                    pass

        active_scale = getattr(self, '_active_scale', {}) or {}
        try:
            model.Params.TimeLimit = self.mip_sub_time_limit
            model.Params.MIPGap = active_scale.get('sub_mip_gap', 0.015)
            model.Params.MIPFocus = 1
            model.Params.Presolve = 2
            model.Params.Heuristics = 0.85
            model.Params.OutputFlag = 0
            model.Params.Threads = GUROBI_SUB_THREADS
            model.optimize()
        except Exception as e:
            if self.verbose:
                print(f"    MIP子问题(fix_space)异常: {e}")
            try:
                model.dispose(); env.dispose()
            except Exception:
                pass
            return None

        if model.SolCount == 0:
            try:
                model.dispose(); env.dispose()
            except Exception:
                pass
            return None

        new_schedule = list(self._rebuild_full_schedule_from_fixed(fixed_assignments, instance))
        for t in selected_tasks:
            task_info = next((ti for ti in tasks_data if ti['task_id'] == t), None)
            if task_info is None:
                continue
            stype, sid = fixed_space[t]
            assigned_team = None
            for k in eligible_teams_dict.get(t, _teams):
                if (t, k) in z and z[t, k].X > 0.5:
                    assigned_team = k
                    break
            start_val, comp_val = S[t].X, C[t].X
            new_schedule.append({
                'task_id': t, 'ship_id': task_info['ship_id'],
                'start': start_val, 'completion': comp_val,
                'duration': comp_val - start_val,
                'space': sid, 'space_type': stype, 'team': assigned_team,
                'skill_req': task_info.get('skill_req', {}),
            })

        new_schedule.sort(key=lambda x: (x['completion'], x['ship_id'], x['task_id']))
        utils = HeuristicScheduler()
        utils._load_team_config(instance)
        utils._load_resource_config(instance)
        new_schedule = utils._rebuild_schedule_times(new_schedule, instance)
        new_obj = utils._compute_objective(new_schedule, instance)
        try:
            model.dispose(); env.dispose()
        except Exception:
            pass
        return new_schedule, new_obj

    def _solve_mip_subproblem_full(self, instance, selected_tasks, fixed_assignments,
                                   warm_start_schedule=None):
        """完整受限 MIP（保留空间二元变量）；对小邻域 + Warm Start 仍可用。"""
        if not GUROBI_AVAILABLE:
            return None

        tasks_data = instance['tasks_data']
        arrival_times = instance['arrival_times']
        num_ships = instance['num_ships']
        all_tasks = [t['task_id'] for t in tasks_data]
        ships = list(range(1, num_ships + 1))

        team_config = instance.get('team_config')
        if team_config is not None:
            _teams = team_config['teams']
            _team_skills = team_config['team_skills']
            _processing_time = team_config['processing_time']
        else:
            _teams = teams
            _team_skills = team_skills
            _processing_time = processing_time

        _berths = instance.get('berths', berths)
        _docks = instance.get('docks', docks)
        selected_task_ids = set(selected_tasks)

        # 预计算处理时间
        task_durations = {}
        eligible_teams_dict = {}
        for t_id in all_tasks:
            task_info = next((t for t in tasks_data if t['task_id'] == t_id), None)
            if task_info is None:
                continue
            task_durations[t_id] = {}
            for k in _teams:
                dur = 0.0
                feasible = True
                for skill in task_info.get('skill_req', {}):
                    if skill in _processing_time.get(k, {}):
                        dur += _processing_time[k][skill]
                    else:
                        feasible = False
                        break
                task_durations[t_id][k] = dur if feasible else None

            # 资格团队
            skill_req = task_info.get('skill_req', {})
            eligible = []
            for k in _teams:
                t_skills = _team_skills.get(k, {})
                ok = True
                for skill, min_qual in skill_req.items():
                    if skill not in t_skills or t_skills[skill] > min_qual:
                        ok = False
                        break
                if ok and task_durations[t_id].get(k) is not None:
                    eligible.append(k)
            eligible_teams_dict[t_id] = eligible if eligible else _teams.copy()

        # 计算 ship_tasks 映射
        task_to_ship = {t['task_id']: t['ship_id'] for t in tasks_data}
        task_requires_dock = {t['task_id']: task_requires_dock_space(t) for t in tasks_data}
        ship_tasks = {}
        for ship in ships:
            ship_tasks[ship] = sorted(
                [t['task_id'] for t in tasks_data if t['ship_id'] == ship])

        # Big-M：子问题局部估计 + 主模型 horizon 取 max，保证析取约束不失效
        max_dur = max(
            (d for td in task_durations.values() for d in td.values() if d is not None),
            default=5.0)
        max_arrival = max(arrival_times.values()) if arrival_times else 20.0
        fixed_max_time = max(
            (f['completion'] for f in fixed_assignments.values()), default=0.0)
        sel_total_dur = sum(
            max((task_durations[t].get(k, 0) or 0 for k in eligible_teams_dict.get(t, [])),
                default=max_dur)
            for t in selected_tasks)
        M_main = compute_big_M(
            tasks_data, arrival_times, _teams, _processing_time,
            team_skills_dict=_team_skills)
        M_sub = (max_arrival + fixed_max_time + sel_total_dur
                 + max_dur * max(len(selected_tasks), 1) + 100)
        M = max(M_main, M_sub)

        # ========== 创建 Gurobi 受限模型（静默环境）==========
        env = gp.Env(params={"OutputFlag": 0})
        model = gp.Model('MIP_LNS_Sub', env=env)

        # ---- 决策变量（仅为选中任务创建）----
        S = {}
        C = {}
        x = {}
        y = {}
        z = {}

        for t in selected_tasks:
            S[t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"S_{t}")
            C[t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"C_{t}")
            for b in _berths:
                x[t, b] = model.addVar(vtype=GRB.BINARY, name=f"x_{t}_{b}")
            for d in _docks:
                y[t, d] = model.addVar(vtype=GRB.BINARY, name=f"y_{t}_{d}")
            for k in _teams:
                z[t, k] = model.addVar(vtype=GRB.BINARY, name=f"z_{t}_{k}")

        # 船舶完工时间变量（所有船舶）
        ship_completion = {}
        for ship in ships:
            ship_completion[ship] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"C_ship_{ship}")

        # 顺序变量（选中任务之间 + 选中与固定之间）
        u_berth = {}
        u_dock = {}
        u_team = {}
        for t1 in selected_tasks:
            for t2 in selected_tasks:
                if t1 >= t2:
                    continue
                for b in _berths:
                    u_berth[t1, t2, b] = model.addVar(vtype=GRB.BINARY)
                for d in _docks:
                    u_dock[t1, t2, d] = model.addVar(vtype=GRB.BINARY)
                for k in _teams:
                    u_team[t1, t2, k] = model.addVar(vtype=GRB.BINARY)

        model.update()

        # ---- 约束(1): 空间分配 ----
        for t in selected_tasks:
            model.addConstr(
                gp.quicksum(x[t, b] for b in _berths) +
                gp.quicksum(y[t, d] for d in _docks) == 1,
                name=f"C1_{t}")

        # ---- 约束(2): 空间兼容性 ----
        for t in selected_tasks:
            if task_requires_dock.get(t, False):
                for b in _berths:
                    model.addConstr(x[t, b] == 0, name=f"C2_{t}_{b}")

        # ---- 约束(4): 工种-空间兼容性（与主模型一致）----
        add_skill_space_compat_constraints(
            model, x, y, tasks_data, _berths, _docks, task_ids=selected_tasks)

        # ---- 约束(3): 团队分配 ----
        for t in selected_tasks:
            model.addConstr(
                gp.quicksum(z[t, k] for k in _teams) == 1,
                name=f"C3_{t}")

        # ---- 约束(4): 技能级联 ----
        for t in selected_tasks:
            for k in _teams:
                if k not in eligible_teams_dict.get(t, []):
                    model.addConstr(z[t, k] == 0, name=f"C4_{t}_{k}")

        # ---- 约束(5): 完工时间计算 ----
        for t in selected_tasks:
            model.addConstr(
                C[t] == S[t] + gp.quicksum(
                    z[t, k] * task_durations[t][k]
                    for k in eligible_teams_dict.get(t, [])
                    if task_durations[t].get(k) is not None),
                name=f"C5_{t}")

        # ---- 约束(6-8): 资源容量互斥 ----
        # 选中任务之间的互斥
        for t1 in selected_tasks:
            for t2 in selected_tasks:
                if t1 >= t2:
                    continue
                for b in _berths:
                    model.addConstr(
                        C[t1] <= S[t2] + M * (1 - u_berth[t1, t2, b]) +
                        M * (2 - x[t1, b] - x[t2, b]),
                        name=f"C6a_{t1}_{t2}_{b}")
                    model.addConstr(
                        C[t2] <= S[t1] + M * u_berth[t1, t2, b] +
                        M * (2 - x[t1, b] - x[t2, b]),
                        name=f"C6b_{t1}_{t2}_{b}")
                for d in _docks:
                    model.addConstr(
                        C[t1] <= S[t2] + M * (1 - u_dock[t1, t2, d]) +
                        M * (2 - y[t1, d] - y[t2, d]),
                        name=f"C7a_{t1}_{t2}_{d}")
                    model.addConstr(
                        C[t2] <= S[t1] + M * u_dock[t1, t2, d] +
                        M * (2 - y[t1, d] - y[t2, d]),
                        name=f"C7b_{t1}_{t2}_{d}")
                for k in _teams:
                    model.addConstr(
                        C[t1] <= S[t2] + M * (1 - u_team[t1, t2, k]) +
                        M * (2 - z[t1, k] - z[t2, k]),
                        name=f"C8a_{t1}_{t2}_{k}")
                    model.addConstr(
                        C[t2] <= S[t1] + M * u_team[t1, t2, k] +
                        M * (2 - z[t1, k] - z[t2, k]),
                        name=f"C8b_{t1}_{t2}_{k}")

        # ---- 选中任务与固定任务之间的互斥（析取：排在固定任务前或后）----
        berth_intervals = {b: [] for b in _berths}
        dock_intervals = {d: [] for d in _docks}
        team_intervals = {k: [] for k in _teams}

        for t_fix, finfo in fixed_assignments.items():
            if finfo['space_type'] == 'berth':
                berth_intervals[finfo['space']].append((finfo['start'], finfo['completion']))
            else:
                dock_intervals[finfo['space']].append((finfo['start'], finfo['completion']))
            team_intervals[finfo['team']].append((finfo['start'], finfo['completion']))

        for t_sel in selected_tasks:
            for b in _berths:
                for idx, (f_start, f_end) in enumerate(berth_intervals[b]):
                    u = model.addVar(vtype=GRB.BINARY, name=f"uFb_{t_sel}_{b}_{idx}")
                    model.addConstr(
                        C[t_sel] <= f_start + M * (1 - u) + M * (1 - x[t_sel, b]),
                        name=f"FX_B_{t_sel}_{b}_{idx}")
                    model.addConstr(
                        S[t_sel] >= f_end - M * u + M * (1 - x[t_sel, b]),
                        name=f"FX_B2_{t_sel}_{b}_{idx}")
            for d in _docks:
                for idx, (f_start, f_end) in enumerate(dock_intervals[d]):
                    u = model.addVar(vtype=GRB.BINARY, name=f"uFd_{t_sel}_{d}_{idx}")
                    model.addConstr(
                        C[t_sel] <= f_start + M * (1 - u) + M * (1 - y[t_sel, d]),
                        name=f"FX_D_{t_sel}_{d}_{idx}")
                    model.addConstr(
                        S[t_sel] >= f_end - M * u + M * (1 - y[t_sel, d]),
                        name=f"FX_D2_{t_sel}_{d}_{idx}")
            for k in _teams:
                for idx, (f_start, f_end) in enumerate(team_intervals[k]):
                    u = model.addVar(vtype=GRB.BINARY, name=f"uFt_{t_sel}_{k}_{idx}")
                    model.addConstr(
                        C[t_sel] <= f_start + M * (1 - u) + M * (1 - z[t_sel, k]),
                        name=f"FX_T_{t_sel}_{k}_{idx}")
                    model.addConstr(
                        S[t_sel] >= f_end - M * u + M * (1 - z[t_sel, k]),
                        name=f"FX_T2_{t_sel}_{k}_{idx}")

        # ---- 约束(9): 同船任务顺序 ----
        for ship in ships:
            task_list = ship_tasks[ship]
            for i in range(len(task_list) - 1):
                prev_t = task_list[i]
                next_t = task_list[i + 1]
                # 两种情况：前驱固定/选中 × 后继固定/选中
                if prev_t in selected_tasks and next_t in selected_tasks:
                    model.addConstr(C[prev_t] <= S[next_t], name=f"C9_ss_{prev_t}_{next_t}")
                elif prev_t in fixed_assignments and next_t in selected_tasks:
                    model.addConstr(S[next_t] >= fixed_assignments[prev_t]['completion'],
                                    name=f"C9_fs_{prev_t}_{next_t}")
                elif prev_t in selected_tasks and next_t in fixed_assignments:
                    model.addConstr(C[prev_t] <= fixed_assignments[next_t]['start'],
                                    name=f"C9_sf_{prev_t}_{next_t}")
                # 两者都固定：自动满足，无需约束

        # ---- 约束(10): 到港时间 ----
        for t in selected_tasks:
            ship = task_to_ship.get(t, 1)
            model.addConstr(S[t] >= arrival_times.get(ship, 0.0), name=f"C10_{t}")

        # ---- 约束(11): 船舶完工时间 ----
        for ship in ships:
            # 固定任务对船舶完工时间的贡献
            max_fixed_comp = 0.0
            for t_fix, finfo in fixed_assignments.items():
                if finfo['ship_id'] == ship:
                    max_fixed_comp = max(max_fixed_comp, finfo['completion'])
            if max_fixed_comp > 0:
                model.addConstr(ship_completion[ship] >= max_fixed_comp,
                                name=f"C11_fixed_{ship}")

            # 选中任务对船舶完工时间的贡献
            for t in selected_tasks:
                if task_to_ship.get(t, 0) == ship:
                    model.addConstr(ship_completion[ship] >= C[t],
                                    name=f"C11_sel_{ship}_{t}")

        # ---- 目标函数（与全局 Gurobi 模型一致）----
        flow_obj = gp.quicksum(
            ship_completion[ship] - arrival_times.get(ship, 0.0) for ship in ships)

        elite_teams = [k for k in _teams
                       if any(lv == 1 for lv in _team_skills.get(k, {}).values())]
        sub_penalty_terms = []
        for t in selected_tasks:
            task_info = next((ti for ti in tasks_data if ti['task_id'] == t), None)
            if task_info is None:
                continue
            skill_req = task_info.get('skill_req', {})
            if skill_req and all(req >= 2 for req in skill_req.values()):
                for k in elite_teams:
                    if k in eligible_teams_dict.get(t, []):
                        sub_penalty_terms.append(z[t, k])

        misallocation_obj = (self.penalty_coeff * gp.quicksum(sub_penalty_terms)
                             if sub_penalty_terms else 0)
        model.setObjective(flow_obj + misallocation_obj, GRB.MINIMIZE)

        # ---- Warm Start：破坏前分配作为初始可行解 ----
        if warm_start_schedule is not None:
            warm_by_id = {e['task_id']: e for e in warm_start_schedule}
            self._apply_mip_warm_start(
                model, selected_tasks, warm_by_id, S, C, x, y, z,
                _berths, _docks, _teams, u_berth=u_berth, u_dock=u_dock,
                u_team=u_team)

        # ---- 求解（带错误保护）----
        n_sel = len(selected_tasks)
        active_scale = getattr(self, '_active_scale', {}) or {}
        gap_small = active_scale.get('sub_mip_gap', 0.012)
        gap_large = active_scale.get('sub_mip_gap_large', 0.028)
        try:
            model.Params.TimeLimit = self.mip_sub_time_limit
            if n_sel <= 12:
                model.Params.MIPGap = gap_small
                model.Params.MIPFocus = 1
            else:
                model.Params.MIPGap = gap_large
                model.Params.MIPFocus = 1
            model.Params.Presolve = 2
            model.Params.Heuristics = 0.90
            model.Params.Cuts = 0 if n_sel <= 8 else 1
            model.Params.OutputFlag = 0
            model.Params.Threads = GUROBI_SUB_THREADS
            model.optimize()
        except Exception as e:
            if self.verbose:
                print(f"    MIP子问题异常: {e}")
            try:
                model.dispose()
                env.dispose()
            except Exception:
                pass
            return None

        # ---- 提取结果 ----
        if model.SolCount == 0:
            try:
                model.dispose()
                env.dispose()
            except Exception:
                pass
            return None

        # 构建完整调度（固定任务不变 + 选中任务用新值）
        new_schedule = []
        for entry in [e for e in self._rebuild_full_schedule_from_fixed(fixed_assignments, instance)]:
            new_schedule.append(entry)

        # 添加选中任务的新调度
        for t in selected_tasks:
            task_info = next((ti for ti in tasks_data if ti['task_id'] == t), None)
            if task_info is None:
                continue

            start_val = S[t].X
            comp_val = C[t].X

            # 找分配的空间
            space_type = None
            space_id = None
            for b in _berths:
                if x[t, b].X > 0.5:
                    space_type = 'berth'
                    space_id = b
                    break
            if space_type is None:
                for d in _docks:
                    if y[t, d].X > 0.5:
                        space_type = 'dock'
                        space_id = d
                        break

            # 找分配的团队
            assigned_team = None
            for k in _teams:
                if z[t, k].X > 0.5:
                    assigned_team = k
                    break

            new_schedule.append({
                'task_id': t,
                'ship_id': task_info['ship_id'],
                'start': start_val,
                'completion': comp_val,
                'duration': comp_val - start_val,
                'space': space_id,
                'space_type': space_type,
                'team': assigned_team,
                'skill_req': task_info.get('skill_req', {})
            })

        # 按开始时间排序并重新计算时间线
        new_schedule.sort(key=lambda x: (x['completion'], x['ship_id'], x['task_id']))
        utils = HeuristicScheduler()
        utils._load_team_config(instance)
        utils._load_resource_config(instance)
        new_schedule = utils._rebuild_schedule_times(new_schedule, instance)
        new_obj = utils._compute_objective(new_schedule, instance)

        try:
            model.dispose()
            env.dispose()
        except Exception:
            pass
        return new_schedule, new_obj

    def _rebuild_full_schedule_from_fixed(self, fixed_assignments, instance):
        """从固定任务信息构建初始调度列表"""
        tasks_data = instance['tasks_data']
        schedule = []
        for t_id, finfo in fixed_assignments.items():
            task_info = next((t for t in tasks_data if t['task_id'] == t_id), None)
            schedule.append({
                'task_id': t_id,
                'ship_id': finfo['ship_id'],
                'start': finfo['start'],
                'completion': finfo['completion'],
                'duration': finfo['duration'],
                'space': finfo['space'],
                'space_type': finfo['space_type'],
                'team': finfo['team'],
                'skill_req': task_info.get('skill_req', {}) if task_info else {}
            })
        return schedule

    # ================================================================
    # 后优化：关键路径再优化 (Final Polish)
    # ================================================================

    def _polish_one_target(self, schedule, instance, time_budget, fix_space=True):
        """对单个靶点做一轮连通核 MIP 精修。返回 (obj, sched) 或 None。"""
        if not schedule or time_budget is None or time_budget < 1.2:
            return None
        utils = HeuristicScheduler()
        utils._load_team_config(instance)
        utils._load_resource_config(instance)
        scale = getattr(self, '_active_scale', {}) or {}
        max_nei = int(scale.get('polish_max_nei',
                                scale.get('max_mip_nei', 10)) or 10)
        best_sched = [dict(e) for e in schedule]
        best_obj = utils._compute_objective(best_sched, instance)
        crit_indices = utils._compute_critical_path(best_sched)
        if len(crit_indices) < 2:
            return None
        seed = {best_sched[i]['task_id']
                for i in crit_indices if i < len(best_sched)}
        crit_ids = self._select_connected_high_score_ids(
            best_sched, instance, max(2, max_nei), seed_candidates=seed)
        if len(crit_ids) < 2:
            return None
        fixed = self._fix_other_tasks(best_sched, crit_ids)
        saved_limit = self.mip_sub_time_limit
        self.mip_sub_time_limit = max(1, int(time_budget))
        t0 = time.time()
        result = self._solve_mip_subproblem(
            instance, crit_ids, fixed,
            warm_start_schedule=best_sched,
            fix_space=bool(fix_space))
        if result is None and not fix_space:
            self.mip_sub_time_limit = max(1, int(time_budget))
            result = self._solve_mip_subproblem(
                instance, crit_ids, fixed,
                warm_start_schedule=best_sched,
                fix_space=True)
        self.mip_sub_time_limit = saved_limit
        dt = time.time() - t0
        improved = False
        out = None
        if result is not None:
            cand_sched = result[0]
            cand_obj = utils._compute_objective(cand_sched, instance)
            if cand_obj < best_obj - 0.01:
                improved = True
                out = (cand_obj, cand_sched)
        self._note_mip_call(dt, improved=improved, kind='polish')
        return out

    def _allocate_archive_polish_budgets(self, n_targets, t_post, tau_min, primary_frac):
        """非均匀分摊：首靶 ~primary_frac，其余平分剩余；不足 τ_min 则截断。

        仅 1 个靶点时把全部预算给该靶，避免 40% 截断浪费。
        """
        if n_targets <= 0 or t_post < 1.5:
            return []
        if n_targets == 1:
            return [float(t_post)]
        tau_min = float(tau_min)
        primary_frac = float(primary_frac)
        budgets = []
        remaining = float(t_post)
        for i in range(n_targets):
            left_others = n_targets - i - 1
            if i == 0:
                reserve = left_others * tau_min
                ti = min(t_post * primary_frac, max(1.0, remaining - reserve))
            else:
                ti = max(tau_min, remaining / max(1, n_targets - i))
            ti = max(1.0, min(ti, remaining))
            budgets.append(ti)
            remaining -= ti
            if remaining < tau_min - 1e-9:
                break
        return budgets

    def _final_polish_from_archive(self, incumbent_schedule, instance, time_budget,
                                     archive):
        """多靶点 Final Polish：incumbent + 档案，预算非均匀分摊。"""
        utils = HeuristicScheduler()
        utils._load_team_config(instance)
        utils._load_resource_config(instance)
        scale = getattr(self, '_active_scale', {}) or {}
        eta = float(scale.get('archive_eta', 0.10) or 0.10)
        best_sched = [dict(e) for e in incumbent_schedule]
        best_obj = utils._compute_objective(best_sched, instance)

        # 构造 C = {best} ∪ Filter(A)，签名去重
        targets = [{'obj': best_obj, 'sched': best_sched,
                    'sig': ALNSSolver.space_assign_signature(best_sched)}]
        thr = ((1.0 + eta) * best_obj) if best_obj > 0 else (
            best_obj + max(1.0, abs(best_obj) * eta + 1e-6))
        for e in (archive or []):
            obj = float(e.get('obj', float('inf')))
            sched = e.get('sched')
            if not sched or obj > thr + 1e-9:
                continue
            sig = e.get('sig', ALNSSolver.space_assign_signature(sched))
            dup = next((t for t in targets if t['sig'] == sig), None)
            if dup is not None:
                if obj < dup['obj'] - 0.01:
                    dup['obj'], dup['sched'] = obj, [dict(x) for x in sched]
                continue
            targets.append({
                'obj': obj,
                'sched': [dict(x) for x in sched],
                'sig': sig,
            })
        targets.sort(key=lambda t: t['obj'])

        tau_min = float(scale.get('polish_archive_tau_min', 2.0) or 2.0)
        primary_frac = float(scale.get('polish_archive_primary_frac', 0.40) or 0.40)
        budgets = self._allocate_archive_polish_budgets(
            len(targets), float(time_budget), tau_min, primary_frac)

        s_star, f_star = best_sched, best_obj
        n_tried = 0
        for i, ti in enumerate(budgets):
            if ti < 1.5:
                continue
            fix_space = (i != 0)  # 首靶放开空间
            rem = time_budget  # wall checked by caller; slice is ti
            polished = self._polish_one_target(
                targets[i]['sched'], instance, ti, fix_space=fix_space)
            n_tried += 1
            if polished is None:
                continue
            f_tmp, s_tmp = polished
            if f_tmp < f_star - 0.01:
                s_star, f_star = s_tmp, f_tmp

        stats = getattr(self, '_last_mip_stats', None)
        if stats is not None:
            stats['archive_polish_targets'] = n_tried
            stats['archive_size'] = max(0, len(targets) - 1)
        return s_star

    def _final_polish(self, incumbent_schedule, instance, time_budget=None,
                      archive=None):
        """对当前最优解做精确 MIP 磨光（连通高分核，可选放开空间 / 多轮换核）。

        若 scale['polish_from_archive'] 且提供 archive，则走多靶点分摊精修。
        仅放开选中任务，其余固定。time_budget 纳入总墙钟。
        """
        polish_cap = getattr(self, '_polish_sub_cap', 120)
        if polish_cap <= 0:
            return incumbent_schedule
        if time_budget is not None and time_budget <= 1.0:
            return incumbent_schedule

        scale = getattr(self, '_active_scale', {}) or {}
        # 开关开启即走多靶入口（档案可空 → 退化为单靶吃满预算）
        if (scale.get('polish_from_archive')
                and time_budget is not None and time_budget >= 3.0):
            return self._final_polish_from_archive(
                incumbent_schedule, instance, time_budget, archive or [])

        utils = HeuristicScheduler()
        utils._load_team_config(instance)
        utils._load_resource_config(instance)
        max_nei = int(scale.get('polish_max_nei',
                                scale.get('max_mip_nei', 10)) or 10)
        rounds = 1
        if scale.get('multi_polish', False):
            rounds = max(1, int(scale.get('polish_rounds', 2) or 2))
        deadline = time.time() + (float(time_budget) if time_budget is not None
                                  else float(polish_cap))
        best_sched = [dict(e) for e in incumbent_schedule]
        best_obj = utils._compute_objective(best_sched, instance)
        used_cores = []

        for round_i in range(rounds):
            rem = deadline - time.time()
            if rem < 1.2:
                break
            crit_indices = utils._compute_critical_path(best_sched)
            if len(crit_indices) < 2:
                break
            seed = {best_sched[i]['task_id']
                    for i in crit_indices if i < len(best_sched)}
            # 多轮时避开已用过的核，换一块连通高分区
            if used_cores and len(seed) > max_nei:
                seed = {t for t in seed if t not in used_cores[-1]} or seed
            crit_ids = self._select_connected_high_score_ids(
                best_sched, instance, max(2, max_nei), seed_candidates=seed)
            if len(crit_ids) < 2:
                break
            used_cores.append(set(crit_ids))
            fixed = self._fix_other_tasks(best_sched, crit_ids)

            saved_limit = self.mip_sub_time_limit
            slice_tl = min(polish_cap if rounds == 1 else max(2, int(rem / (rounds - round_i))),
                           max(1, int(rem - 0.3)))
            self.mip_sub_time_limit = slice_tl
            unfix = bool(scale.get('polish_unfix_space', False)) and round_i == 0
            fix_space = not unfix
            t0 = time.time()
            result = self._solve_mip_subproblem(
                instance, crit_ids, fixed,
                warm_start_schedule=best_sched,
                fix_space=fix_space)
            if result is None and not fix_space:
                self.mip_sub_time_limit = max(1, int(min(slice_tl, deadline - time.time())))
                result = self._solve_mip_subproblem(
                    instance, crit_ids, fixed,
                    warm_start_schedule=best_sched,
                    fix_space=True)
            self.mip_sub_time_limit = saved_limit
            dt = time.time() - t0
            improved = False
            if result is not None:
                cand_sched = result[0]
                cand_obj = utils._compute_objective(cand_sched, instance)
                if cand_obj < best_obj - 0.01:
                    best_sched, best_obj = cand_sched, cand_obj
                    improved = True
            self._note_mip_call(dt, improved=improved, kind='polish')

        return best_sched

    # ================================================================
    # 团队交换后处理（局部搜索）
    # ================================================================

    def _team_swap_postprocess(self, schedule, instance, deadline=None):
        """扫描精英团队的低级任务，尝试与低等级团队交换任务分配

        原理：精英团队（任意技能 level=1）承担低级任务（所有技能需求 ≥2）
        属于"大材小用"。尝试将这些任务与低等级团队承担的任务交换，
        前提是双方技能覆盖允许。若交换后目标值降低则保留。
        deadline: Unix 时间戳，到达后停止（纳入总墙钟预算）。
        """
        utils = HeuristicScheduler()
        utils._load_team_config(instance)
        utils._load_resource_config(instance)

        if not utils._team_skills:
            return schedule

        # 精英团队：任意技能 level=1
        elite = {k for k in utils._teams
                 if any(lv == 1 for lv in utils._team_skills.get(k, {}).values())}
        non_elite = [k for k in utils._teams if k not in elite]
        if not elite or not non_elite:
            return schedule

        best_schedule = schedule
        best_obj = utils._compute_objective(best_schedule, instance)

        def _can_do(team_id, task_entry):
            """检查团队是否技能覆盖该任务"""
            t_skills = utils._team_skills.get(team_id, {})
            for skill, req in task_entry.get('skill_req', {}).items():
                if skill not in t_skills or t_skills[skill] > req:
                    return False
            return True

        # 迭代改进，直到无改进或墙钟耗尽
        while True:
            if deadline is not None and time.time() >= deadline:
                break
            improved = False
            # 精英团队的低级任务索引
            elite_low = [
                i for i, e in enumerate(best_schedule)
                if e['team'] in elite
                and (sr := e.get('skill_req', {}))
                and all(r >= 2 for r in sr.values())
            ]
            # 非精英团队的任务索引
            non_elite_tasks = [
                i for i, e in enumerate(best_schedule)
                if e['team'] in non_elite
            ]

            for i in elite_low:
                if deadline is not None and time.time() >= deadline:
                    break
                ta = best_schedule[i]
                for j in non_elite_tasks:
                    tb = best_schedule[j]
                    team_a = ta['team']  # elite
                    team_b = tb['team']  # non-elite
                    # 交换可行性检查
                    if not _can_do(team_b, ta):
                        continue
                    if not _can_do(team_a, tb):
                        continue

                    # 尝试交换
                    test = [dict(e) for e in best_schedule]
                    test[i]['team'] = team_b
                    test[j]['team'] = team_a
                    test = utils._rebuild_schedule_times(test, instance)
                    test_obj = utils._compute_objective(test, instance)

                    if test_obj < best_obj - 0.01:
                        best_schedule = test
                        best_obj = test_obj
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                break

        return best_schedule

    # ================================================================
    # 主求解入口
    # ================================================================
    # 滚动时域初始解生成
    # ================================================================

    def _rolling_horizon_initial_solution(self, instance, deadline=None):
        """滚动时域法生成高质量初始解

        将任务按到港时间排序后分窗口依次求解 MIP 子问题，前序窗口结果
        固定后作为后续窗口的约束。相比多起点贪心，滚动时域能获得更接近
        全局最优的初始解。小规模(N<=10): 任务>=8 且 任务/队>=2.5；
        大规模: 任务>=15 且 任务/队>=3.0。

        deadline: 初始解阶段时间上限（Unix 时间戳），超时则中止并返回 None
        """
        self._ensure_config_loaded(instance)
        tasks_data = instance['tasks_data']
        arrival_times = instance['arrival_times']
        total_tasks = len(tasks_data)

        # 窗口参数：较小窗口 + 更高重叠，按船/任务序排列
        if total_tasks <= 50:
            window_size = max(8, min(20, total_tasks // 4))
            overlap_ratio = 0.35
        else:
            window_size = max(10, min(22, total_tasks // 8))
            overlap_ratio = 0.45
        step_size = max(1, int(window_size * (1 - overlap_ratio)))

        all_sorted = sorted(tasks_data,
                            key=lambda t: (arrival_times.get(t['ship_id'], 0),
                                           t['ship_id'], t['task_id']))

        fixed_assignments = {}  # task_id → 固定信息
        start_idx = 0

        while start_idx < len(all_sorted):
            if deadline is not None and time.time() >= deadline:
                if self.verbose:
                    print(f"  滚动时域超时 ({len(fixed_assignments)}/{total_tasks} 任务已固定)")
                return None, None

            # 取窗口任务
            window_tasks = sorted(
                all_sorted[start_idx:start_idx + window_size],
                key=lambda t: (t['ship_id'], t['task_id']))
            window_ids = {t['task_id'] for t in window_tasks}

            if not window_ids:
                break

            saved_win_limit = self.mip_sub_time_limit
            self.mip_sub_time_limit = min(120, max(saved_win_limit, len(window_ids) * 3))
            result = self._solve_mip_subproblem(instance, window_ids, fixed_assignments)
            self.mip_sub_time_limit = saved_win_limit

            if result is None:
                if self.verbose:
                    print(f"  窗口 MIP 失败，窗口内回退贪心 ({len(window_ids)} 任务)")
                utils = HeuristicScheduler()
                utils._load_team_config(instance)
                utils._load_resource_config(instance)
                merged = self._rebuild_full_schedule_from_fixed(fixed_assignments, instance)
                greedy_full = _greedy_construct_single(instance, random_seed=start_idx * 17 + 42)
                greedy_full = utils._rebuild_schedule_times(greedy_full, instance)
                gmap = {e['task_id']: e for e in greedy_full}
                for tid in window_ids:
                    if tid in gmap:
                        merged.append(gmap[tid])
                merged = utils._rebuild_schedule_times(merged, instance)
                win_schedule = [e for e in merged if e['task_id'] in window_ids]
                if not win_schedule:
                    return None, None
            else:
                win_schedule, _ = result

            # 提取窗口任务的调度结果，加入 fixed_assignments
            for entry in win_schedule:
                t_id = entry['task_id']
                if t_id in window_ids:
                    fixed_assignments[t_id] = {
                        'team': entry['team'],
                        'space_type': entry['space_type'],
                        'space': entry['space'],
                        'start': entry['start'],
                        'completion': entry['completion'],
                        'duration': entry['duration'],
                        'ship_id': entry['ship_id'],
                    }

            start_idx += step_size

        # 从 fixed_assignments 构建完整调度
        schedule = []
        for t in tasks_data:
            t_id = t['task_id']
            f = fixed_assignments.get(t_id)
            if f is None:
                return None, None
            schedule.append({
                'task_id': t_id,
                'ship_id': f['ship_id'],
                'start': f['start'],
                'completion': f['completion'],
                'duration': f['duration'],
                'space': f['space'],
                'space_type': f['space_type'],
                'team': f['team'],
                'skill_req': t.get('skill_req', {})
            })

        # 重建时间线确保一致性
        utils = HeuristicScheduler()
        utils._load_team_config(instance)
        utils._load_resource_config(instance)
        schedule = utils._rebuild_schedule_times(schedule, instance)
        obj = utils._compute_objective(schedule, instance)

        return schedule, obj

    def _trim_neighborhood(self, schedule, removed_ids, instance):
        """将邻域裁剪到 cap 以内，优先保留关键路径任务（使 MIP 子问题更快）

        固定破坏比例敏感性（_destroy_ratio_fixed）时跳过 cap，保持严格比例。
        """
        n = len(schedule) if schedule else 0
        if getattr(self, '_destroy_ratio_fixed', None) is not None:
            kept = set(removed_ids)
            self._last_actual_destroy_count = int(len(kept))
            self._last_actual_destroy_ratio = float(len(kept) / max(n, 1))
            return kept

        cap = getattr(self, '_neighborhood_cap', 12)
        if len(removed_ids) <= cap:
            kept = set(removed_ids)
            self._last_actual_destroy_count = int(len(kept))
            self._last_actual_destroy_ratio = float(len(kept) / max(n, 1))
            return kept
        utils = HeuristicScheduler()
        utils._load_team_config(instance)
        crit_ids = {schedule[i]['task_id'] for i in utils._compute_critical_path(schedule)}
        # 关键+随机混合，避免裁剪后几乎全是关键路径
        priority = [tid for tid in removed_ids if tid in crit_ids]
        rest = list(set(removed_ids) - set(priority))
        random.shuffle(rest)
        n_crit = min(len(priority), max(1, int(cap * 0.7)))
        kept = set(priority[:n_crit])
        if len(kept) < cap:
            kept.update(rest[:cap - len(kept)])
        if len(kept) < cap:
            leftover = [tid for tid in priority[n_crit:] if tid not in kept]
            kept.update(leftover[:cap - len(kept)])
        self._last_actual_destroy_count = int(len(kept))
        self._last_actual_destroy_ratio = float(len(kept) / max(n, 1))
        return kept

    def _apply_smart_destroy(self, schedule, instance, iteration, max_iters, stall_count,
                             destroy_w, l3_sub_w):
        """领域破坏 + 邻域裁剪（必须先 trim，再构造 partial/removed_tasks）"""
        layer_idx = self._select_domain_layer(destroy_w, stall_count)
        removed_ids, l3_sub_idx = self._destroy_by_layer(
            layer_idx, schedule, instance, iteration, max_iters, stall_count, l3_sub_w)
        # ★ 先裁剪，再构造 partial —— 否则 MIP 收到的仍是大邻域
        removed_ids = self._trim_neighborhood(schedule, removed_ids, instance)
        for tid in removed_ids:
            self._selection_counts[tid] = self._selection_counts.get(tid, 0) + 1
        remove_set = set(removed_ids)
        partial = [e for e in schedule if e['task_id'] not in remove_set]
        removed_tasks = [e for e in schedule if e['task_id'] in remove_set]
        return partial, removed_tasks, removed_ids, layer_idx, l3_sub_idx

    # ================================================================
    # KG-ALNS 辅助
    # ================================================================

    def _get_insertion_engine(self, instance):
        """任务插入/评估引擎（仅复用可行性检查与时间重建，非 ALNS 算子集）"""
        if getattr(self, '_insertion_engine', None) is None:
            self._insertion_engine = ALNSSolver(verbose=False)
        self._insertion_engine._setup(instance)
        return self._insertion_engine

    def _is_misallocated(self, task, team_id):
        """任务是否被「大材小用」：低级任务分配给持有 level=1 技能的团队"""
        skill_req = task.get('skill_req', {})
        if not skill_req or not all(req >= 2 for req in skill_req.values()):
            return False
        team_skills = self._team_skills.get(team_id, {})
        return any(team_skills.get(skill, 99) == 1 for skill in skill_req)

    def _task_knowledge_components(self, task, team_id=None, space_type=None):
        """分解知识分三因子（论文 Ki=α1·P+α2·R+α3·S）。

        P — 路径/结构关键性（干船坞必需、复杂度）
        R — 资源竞争/空间匹配
        S — 技能复杂程度与团队贴合
        """
        skill_req = task.get('skill_req') or {}
        # P: path / structural criticality
        P = 0.0
        if task_requires_dock_space(task):
            P += 2.0
        try:
            P += 0.3 * float(task.get('complexity', 0) or 0)
        except (TypeError, ValueError):
            pass

        # R: resource contention / space fit
        R = 0.0
        if space_type == 'dock' and task_requires_dock_space(task):
            R += 1.5
        elif space_type == 'berth' and not task_requires_dock_space(task):
            R += 0.8
        else:
            if task_requires_dock_space(task):
                R += 1.0
            R += 0.3 * len(skill_req)

        # S: skill complexity / team fit
        S = 0.0
        S += 0.5 * len(skill_req)
        for req in skill_req.values():
            try:
                S += 0.4 * max(0, 4 - int(req))
            except (TypeError, ValueError):
                pass
        if team_id is not None:
            tskills = getattr(self, '_team_skills', None) or {}
            ts = tskills.get(team_id, {})
            for skill, req in skill_req.items():
                lv = ts.get(skill)
                if lv is None:
                    continue
                try:
                    gap = int(req) - int(lv)
                except (TypeError, ValueError):
                    continue
                if gap == 0:
                    S += 1.2
                elif gap > 0:
                    S -= 0.4
                else:
                    S += max(0.15, 1.0 + 0.25 * gap)
            if self._is_misallocated(task, team_id):
                S -= 1.5
        return P, R, S

    def _task_knowledge_score(self, task, team_id=None, space_type=None):
        """任务知识分 Ki=α1·P+α2·R+α3·S（论文知识评价 → 修复 tie-break）。

        消融 use_knowledge_eval=False 时返回 0（等价于取消知识评价）。
        仅用于成本近似相等时的次序，不直接强制插入。
        """
        if not self.ablation_flags.get('use_knowledge_eval', True):
            return 0.0
        P, R, S = self._task_knowledge_components(task, team_id, space_type)
        w = getattr(self, 'knowledge_weights', None) or (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        a1, a2, a3 = float(w[0]), float(w[1]), float(w[2])
        return a1 * P + a2 * R + a3 * S

    def _knowledge_repair_eps(self, ref_cost):
        abs_eps = float(getattr(self, 'KNOWLEDGE_REPAIR_COST_EPS', 0.05) or 0.05)
        rel = float(getattr(self, 'KNOWLEDGE_REPAIR_COST_EPS_REL', 0.0005) or 0.0)
        return max(abs_eps, rel * abs(float(ref_cost)))

    def _knowledge_repair_contribution(self, task, team_id=None, space_type=None):
        """知识修复贡献 α·Ki；α=0 时贡献恒为 0（仍走同一 repair 骨架）。"""
        alpha = float(getattr(self, 'KNOWLEDGE_REPAIR_ALPHA', 0.05))
        ki = self._task_knowledge_score(
            task, team_id=team_id, space_type=space_type)
        return float(alpha) * float(ki)

    def _pick_insertion_with_knowledge_tiebreak(self, options, task):
        """在最小增量成本原则下，对近成本候选用 Score=ΔC-α·Ki 选择。

        仅当 |ΔC_i - ΔC_j| < ε（相对同一 partial，即 |obj_i-obj_j|<ε）时引入 Ki；
        否则仍选纯最小成本。不直接按知识分选任务。
        options: [(cost, sched, team_id, space_type, space_id), ...]
        """
        if not options:
            return None
        options = sorted(options, key=lambda x: x[0])
        best_cost = float(options[0][0])
        eps = self._knowledge_repair_eps(best_cost)
        near = [o for o in options if float(o[0]) <= best_cost + eps]
        use_ki = (
            self.ablation_flags.get('use_knowledge_repair', True)
            and self.ablation_flags.get('use_knowledge_eval', True)
            and float(getattr(self, 'KNOWLEDGE_REPAIR_ALPHA', 0.05) or 0.0) > 1e-12)
        if len(options) > 1:
            self._bump_kg_stat('knowledge_candidate_count')
        if len(near) == 1:
            return near[0]

        def _score(o):
            contribution = self._knowledge_repair_contribution(
                task, team_id=o[2], space_type=o[3])
            return float(o[0]) - contribution

        cost_pick = min(near, key=lambda o: float(o[0]))
        picked = min(near, key=_score) if use_ki else cost_pick
        if use_ki and len(near) > 1:
            changed = (
                picked[2] != cost_pick[2]
                or picked[3] != cost_pick[3]
                or abs(float(picked[0]) - float(cost_pick[0])) > 1e-9)
            if changed:
                self._bump_kg_stat('knowledge_changed_choice_count')
                self._last_repair_knowledge_changed = True
            self._bump_kg_stat('knowledge_guide_count')
        return picked

    def _enumerate_domain_insertions(self, engine, schedule, task, instance, limit=None):
        """枚举插入方案，优先技能匹配（降低 misallocation）

        组合过多时：80% 保留领域评分最优，20% 随机多样性（避免纯随机漏掉高质量组合）。
        返回 [(cost, sched, team_id, space_type, space_id), ...] 按 cost 升序。
        """
        if limit is None:
            limit = getattr(self, '_domain_insert_limit', 12)
        spaces = engine._feasible_spaces(task)
        teams = engine._feasible_teams(task)
        combos = [(st, sid, tm) for st, sid in spaces for tm in teams]
        if len(combos) > limit:
            def combo_score(combo):
                space_type, space_id, team_id = combo
                duration = engine._utils._get_task_duration(task, team_id)
                misallocation = 1 if self._is_misallocated(task, team_id) else 0
                dock_penalty = (
                    1 if space_type == 'dock'
                    and not task_requires_dock_space(task)
                    else 0)
                return (10 * misallocation + duration + 0.5 * dock_penalty)

            combos.sort(key=combo_score)
            elite_count = max(1, int(limit * 0.8))
            elite = combos[:elite_count]
            remaining = combos[elite_count:]
            random_count = limit - elite_count
            diverse = (random.sample(remaining, min(random_count, len(remaining)))
                       if remaining and random_count > 0 else [])
            combos = elite + diverse
        options = []
        for space_type, space_id, team_id in combos:
            obj, sched = engine._insert_assignment(
                schedule, task, team_id, space_type, space_id, instance)
            tie = 1 if self._is_misallocated(task, team_id) else 0
            options.append((obj + tie * 0.01, sched, team_id, space_type, space_id))
        options.sort(key=lambda x: x[0])
        return options

    def _domain_repair(self, partial, removed_tasks, instance):
        """修船厂领域修复 skill_dock_aware：船坞任务优先 + 技能匹配 + 后悔值插入

        启用 use_knowledge_repair 时：最小成本插入保留，近成本候选用 Ki tie-break。
        """
        self._ensure_config_loaded(instance)
        engine = self._get_insertion_engine(instance)
        use_ki = self.ablation_flags.get('use_knowledge_repair', True)
        if not removed_tasks:
            return engine._eval(partial, instance)

        schedule = engine._copy_schedule(partial)
        remaining = sorted(
            removed_tasks,
            key=lambda t: (
                not task_requires_dock_space(t),
                engine._arrival_times.get(t['ship_id'], 0.0),
                t['task_id'],
            ),
        )
        unplaced = list(remaining)
        while unplaced:
            best_task, best_sched, best_reg, best_contrib = None, None, -1.0, -1e18
            for task in unplaced:
                options = self._enumerate_domain_insertions(engine, schedule, task, instance)
                if not options:
                    teams = engine._feasible_teams(task)
                    spaces = engine._feasible_spaces(task)
                    if not teams or not spaces:
                        continue
                    team_id = random.choice(teams)
                    st, sid = random.choice(spaces)
                    _, sched = engine._insert_assignment(
                        schedule, task, team_id, st, sid, instance)
                    regret = 1e6
                    contrib = self._knowledge_repair_contribution(
                        task, team_id=team_id, space_type=st)
                elif len(options) == 1:
                    regret = 1e6
                    picked = options[0]
                    sched = picked[1]
                    contrib = self._knowledge_repair_contribution(
                        task, team_id=picked[2], space_type=picked[3])
                else:
                    if use_ki:
                        picked = self._pick_insertion_with_knowledge_tiebreak(options, task)
                    else:
                        picked = options[0]
                    # regret 仍相对次优成本，避免 tie-break 扭曲后悔值语义
                    regret = options[1][0] - options[0][0]
                    sched = picked[1]
                    contrib = self._knowledge_repair_contribution(
                        task, team_id=picked[2], space_type=picked[3])
                # 后悔值优先；接近时用 α·Ki 打破任务间平局
                if best_task is None:
                    better = True
                elif use_ki:
                    eps = self._knowledge_repair_eps(options[0][0]) if options else 0.05
                    better = (regret > best_reg + eps) or (
                        abs(regret - best_reg) <= eps and contrib > best_contrib)
                else:
                    better = regret > best_reg
                if better:
                    best_reg, best_task, best_sched, best_contrib = (
                        regret, task, sched, contrib)
            if best_task is None or best_sched is None:
                break
            schedule = best_sched
            unplaced.remove(best_task)
        return engine._eval(schedule, instance)

    def _knowledge_guided_repair(self, engine, op, partial, removed_tasks, instance):
        """Knowledge-guided feasible insertion（弱知识 repair）。

        保持最小增量成本；仅当候选成本差 < ε 时用 Task Knowledge Score Ki 打破平局。
        不修改基线 ALNSSolver._repair，避免污染对比算法。
        """
        if not removed_tasks:
            return engine._eval(partial, instance)
        self._ensure_config_loaded(instance)
        schedule = engine._copy_schedule(partial)
        remaining = list(removed_tasks)

        def _best_insert(sched, task, limit=15):
            options = engine._enumerate_insertions(sched, task, instance, limit=limit)
            if not options:
                return None
            # engine options: (obj, sched, team_id, space_type, space_id)
            return self._pick_insertion_with_knowledge_tiebreak(options, task)

        if op == 'random':
            random.shuffle(remaining)
            for task in remaining:
                picked = _best_insert(schedule, task)
                if picked is not None:
                    schedule = picked[1]
                    continue
                teams = engine._feasible_teams(task)
                spaces = engine._feasible_spaces(task)
                if not teams or not spaces:
                    continue
                team_id = random.choice(teams)
                space_type, space_id = random.choice(spaces)
                _, schedule = engine._insert_assignment(
                    schedule, task, team_id, space_type, space_id, instance)

        elif op in ('greedy', 'resource_aware', 'earliest_time'):
            remaining.sort(
                key=lambda t: (engine._arrival_times.get(t['ship_id'], 0.0), t['task_id']))
            for task in remaining:
                if op == 'earliest_time':
                    best = None
                    for space_type, space_id in engine._feasible_spaces(task):
                        for team_id in engine._feasible_teams(task):
                            obj, sched = engine._insert_assignment(
                                schedule, task, team_id, space_type, space_id, instance)
                            start = next(
                                e['start'] for e in sched if e['task_id'] == task['task_id'])
                            cand = (start, obj, sched, team_id, space_type)
                            if best is None or cand[0] < best[0] - 1e-9 or (
                                    abs(cand[0] - best[0]) <= 1e-9
                                    and cand[1] < best[1] - self._knowledge_repair_eps(cand[1])):
                                best = cand
                            elif (best is not None
                                  and abs(cand[0] - best[0]) <= 1e-9
                                  and abs(cand[1] - best[1]) <= self._knowledge_repair_eps(cand[1])
                                  and self._knowledge_repair_contribution(
                                      task, team_id=team_id, space_type=space_type)
                                  > self._knowledge_repair_contribution(
                                      task, team_id=best[3], space_type=best[4])):
                                best = cand
                    if best:
                        schedule = best[2]
                elif op == 'resource_aware':
                    loads = engine._team_loads(schedule)
                    teams = sorted(
                        engine._feasible_teams(task), key=lambda k: loads.get(k, 0.0))
                    options = []
                    for team_id in teams[:3]:
                        for space_type, space_id in engine._feasible_spaces(task):
                            obj, sched = engine._insert_assignment(
                                schedule, task, team_id, space_type, space_id, instance)
                            options.append((obj, sched, team_id, space_type, space_id))
                    picked = self._pick_insertion_with_knowledge_tiebreak(options, task)
                    if picked:
                        schedule = picked[1]
                        loads[picked[2]] = (
                            loads.get(picked[2], 0.0)
                            + engine._utils._get_task_duration(task, picked[2]))
                else:
                    picked = _best_insert(schedule, task)
                    if picked is not None:
                        schedule = picked[1]
                    else:
                        teams = engine._feasible_teams(task)
                        spaces = engine._feasible_spaces(task)
                        if teams and spaces:
                            team_id = random.choice(teams)
                            st, sid = random.choice(spaces)
                            _, schedule = engine._insert_assignment(
                                schedule, task, team_id, st, sid, instance)

        else:  # regret（默认）及未知算子
            unplaced = list(remaining)
            while unplaced:
                best_task, best_sched, best_reg, best_contrib = (
                    None, None, -1.0, -1e18)
                for task in unplaced:
                    options = engine._enumerate_insertions(
                        schedule, task, instance, limit=8)
                    if not options:
                        teams = engine._feasible_teams(task)
                        spaces = engine._feasible_spaces(task)
                        if not teams or not spaces:
                            continue
                        team_id = random.choice(teams)
                        st, sid = random.choice(spaces)
                        _, sched = engine._insert_assignment(
                            schedule, task, team_id, st, sid, instance)
                        regret = 1e6
                        contrib = self._knowledge_repair_contribution(
                            task, team_id=team_id, space_type=st)
                    elif len(options) == 1:
                        picked = options[0]
                        regret = picked[0]
                        sched = picked[1]
                        contrib = self._knowledge_repair_contribution(
                            task, team_id=picked[2], space_type=picked[3])
                    else:
                        picked = self._pick_insertion_with_knowledge_tiebreak(options, task)
                        regret = options[1][0] - options[0][0]
                        sched = picked[1]
                        contrib = self._knowledge_repair_contribution(
                            task, team_id=picked[2], space_type=picked[3])
                    eps = self._knowledge_repair_eps(options[0][0]) if options else 0.05
                    better = (regret > best_reg + eps) or (
                        abs(regret - best_reg) <= eps and contrib > best_contrib)
                    if better or best_task is None:
                        best_reg, best_task, best_sched, best_contrib = (
                            regret, task, sched, contrib)
                if best_task is None or best_sched is None:
                    break
                schedule = best_sched
                unplaced.remove(best_task)

        return engine._eval(schedule, instance)

    def _generic_repair(self, partial, removed_tasks, instance, repair_op=None):
        """与基线 ALNS 同构的通用修复；KG 路径可叠加弱知识 tie-break。"""
        engine = self._get_insertion_engine(instance)
        op = repair_op if repair_op in ALNSSolver.REPAIR_OPS else 'regret'
        if self.ablation_flags.get('use_knowledge_repair', True):
            return self._knowledge_guided_repair(
                engine, op, partial, removed_tasks, instance)
        return engine._repair(op, partial, removed_tasks, instance)

    def _skill_match_gap(self, task_or_entry, team_id):
        """技能匹配间隙：0=完全贴合；越大表示越不贴合（大材小用或等级差）。"""
        skill_req = task_or_entry.get('skill_req') or {}
        if not skill_req:
            return 0.0
        tskills = getattr(self, '_team_skills', None) or {}
        ts = tskills.get(team_id, {})
        gap = 0.0
        for skill, req in skill_req.items():
            lv = ts.get(skill)
            if lv is None:
                gap += 3.0
                continue
            try:
                gap += abs(int(lv) - int(req))
            except (TypeError, ValueError):
                gap += 1.0
        return gap

    def _skill_aware_local_refine(self, schedule, instance, deadline=None, max_passes=2):
        """轻量 skill-aware 局部精修（终局 best-solution refinement）。

        操作：
          1) 任务→更贴合可行团队重分配
          2) 两任务团队交换（技能兼容）
        接受准则：技能可行 + 时间重建后目标 ≤ 当前目标。
        不调用 MIP，不扩大主搜索；仅 refinement。
        返回 (obj, sched) 或 None（无改进）。
        """
        if not schedule or not self.ablation_flags.get('use_skill_refine', True):
            return None
        self._ensure_config_loaded(instance)
        utils = HeuristicScheduler()
        utils._load_team_config(instance)
        utils._load_resource_config(instance)
        best_schedule = [dict(e) for e in schedule]
        best_obj = utils._compute_objective(best_schedule, instance)
        improved_moves = 0

        def _eligible(team_id, entry):
            t_skills = utils._team_skills.get(team_id, {})
            for skill, req in (entry.get('skill_req') or {}).items():
                if skill not in t_skills or t_skills[skill] > req:
                    return False
            return True

        def _apply_team(sched, idx, new_team):
            test = [dict(e) for e in sched]
            test[idx] = dict(test[idx])
            test[idx]['team'] = new_team
            entry = test[idx]
            task = {
                'task_id': entry['task_id'],
                'ship_id': entry['ship_id'],
                'skill_req': entry.get('skill_req', {}),
            }
            test[idx]['duration'] = utils._get_task_duration(task, new_team)
            test = utils._rebuild_schedule_times(test, instance)
            return test, utils._compute_objective(test, instance)

        # ---- 1) 任务-团队重新匹配 ----
        for _ in range(max(1, int(max_passes))):
            if deadline is not None and time.time() >= deadline:
                break
            pass_gain = False
            order = sorted(
                range(len(best_schedule)),
                key=lambda i: -self._skill_match_gap(
                    best_schedule[i], best_schedule[i]['team']))
            for i in order:
                if deadline is not None and time.time() >= deadline:
                    break
                entry = best_schedule[i]
                cur_team = entry['team']
                cur_gap = self._skill_match_gap(entry, cur_team)
                if cur_gap <= 0:
                    continue
                candidates = [
                    k for k in utils._teams
                    if k != cur_team and _eligible(k, entry)
                ]
                candidates.sort(key=lambda k: self._skill_match_gap(entry, k))
                for new_team in candidates[:6]:
                    new_gap = self._skill_match_gap(entry, new_team)
                    if new_gap >= cur_gap - 1e-9:
                        continue
                    test, test_obj = _apply_team(best_schedule, i, new_team)
                    if test_obj <= best_obj + 1e-9:
                        best_schedule, best_obj = test, test_obj
                        improved_moves += 1
                        pass_gain = True
                        break
                if pass_gain:
                    break
            if not pass_gain:
                break

        # ---- 2) 轻量团队交换（限次数，终局用）----
        n = len(best_schedule)
        swap_budget = min(40, max(8, n // 3))
        swap_tries = 0
        if n >= 2:
            # 优先：错配任务 × 其他任务
            mismatch = [
                i for i in range(n)
                if self._skill_match_gap(best_schedule[i], best_schedule[i]['team']) > 0
            ]
            random.shuffle(mismatch)
            for i in mismatch[:min(12, len(mismatch))]:
                if deadline is not None and time.time() >= deadline:
                    break
                if swap_tries >= swap_budget:
                    break
                ta = best_schedule[i]
                partners = list(range(n))
                random.shuffle(partners)
                for j in partners[:8]:
                    if j == i:
                        continue
                    if deadline is not None and time.time() >= deadline:
                        break
                    if swap_tries >= swap_budget:
                        break
                    swap_tries += 1
                    tb = best_schedule[j]
                    team_a, team_b = ta['team'], tb['team']
                    if team_a == team_b:
                        continue
                    if not _eligible(team_b, ta) or not _eligible(team_a, tb):
                        continue
                    # 交换后技能间隙应不恶化（至少一方改善）
                    gap_before = (
                        self._skill_match_gap(ta, team_a)
                        + self._skill_match_gap(tb, team_b))
                    gap_after = (
                        self._skill_match_gap(ta, team_b)
                        + self._skill_match_gap(tb, team_a))
                    if gap_after >= gap_before - 1e-9:
                        continue
                    test = [dict(e) for e in best_schedule]
                    test[i] = dict(test[i])
                    test[j] = dict(test[j])
                    test[i]['team'], test[j]['team'] = team_b, team_a
                    for idx, team in ((i, team_b), (j, team_a)):
                        ent = test[idx]
                        task = {
                            'task_id': ent['task_id'],
                            'ship_id': ent['ship_id'],
                            'skill_req': ent.get('skill_req', {}),
                        }
                        ent['duration'] = utils._get_task_duration(task, team)
                    test = utils._rebuild_schedule_times(test, instance)
                    test_obj = utils._compute_objective(test, instance)
                    if test_obj <= best_obj + 1e-9:
                        best_schedule, best_obj = test, test_obj
                        improved_moves += 1
                        ta = best_schedule[i]
                        break

        if improved_moves <= 0:
            return None
        self._bump_kg_stat('skill_refine_improve', improved_moves)
        return best_obj, best_schedule

    def _elite_restart_from_best(self, best_schedule, instance, repair_op='regret'):
        """从当前 best 附近大比例破坏后重新 repair（仅搜索停滞时调用）。

        destroy 比例 20%–30%；不改变正常 destroy 算子集合。
        返回 (obj, sched) 或 None。
        """
        if (not best_schedule
                or not self.ablation_flags.get('use_elite_restart', True)):
            return None
        n = len(best_schedule)
        if n < 4:
            return None
        lo, hi = getattr(self, 'ELITE_RESTART_DESTROY_RATIO', (0.20, 0.30))
        ratio = random.uniform(float(lo), float(hi))
        k = max(2, min(n - 1, int(round(n * ratio))))
        remove_idx = set(random.sample(range(n), k))
        removed_ids = {best_schedule[i]['task_id'] for i in remove_idx}
        partial = [dict(e) for e in best_schedule if e['task_id'] not in removed_ids]
        task_map = {t['task_id']: t for t in instance.get('tasks_data', [])}
        removed_tasks = [task_map[tid] for tid in removed_ids if tid in task_map]
        if not removed_tasks:
            return None
        return self._generic_repair(
            partial, removed_tasks, instance, repair_op=repair_op)

    @staticmethod
    def _normalize_repair_result(result):
        """统一修复结果为 (objective, schedule)；MIP 子问题返回 (schedule, objective)"""
        if result is None:
            return None
        first, second = result
        if isinstance(first, (list, tuple)) and isinstance(second, (int, float)):
            return second, first
        return first, second

    def _intensify_critical_path_domain(self, schedule, instance, scale):
        """关键路径连通邻域多轮破坏–领域修复（不含 Gurobi）。

        返回 (objective, schedule)；无改进时返回 None。
        V2 主表：停滞时有限预算启用（critical_path_intensify=True）。
        """
        if (not schedule or not scale
                or not self.ablation_flags.get('use_domain_repair', False)):
            return None
        rounds = int(scale.get('critical_intensify_rounds', 3) or 3)
        if rounds <= 0:
            return None
        call_cap = float(scale.get('domain_intensify_max_sec', 0.5) or 0.5)
        mid_left = float(getattr(self, '_mid_intensify_budget_left', call_cap) or 0.0)
        if mid_left < 0.05 or call_cap < 0.05:
            return None
        hard_deadline = time.time() + min(call_cap, mid_left)
        max_nei = int(scale.get('neighborhood_cap', 12) or 12)
        utils = HeuristicScheduler()
        utils._load_team_config(instance)
        utils._load_resource_config(instance)
        best_sched = [dict(e) for e in schedule]
        best_obj = utils._compute_objective(best_sched, instance)
        improved_any = False
        t0 = time.time()
        for _ in range(rounds):
            if time.time() >= hard_deadline:
                break
            crit_idx = list(utils._compute_critical_path(best_sched))
            if len(crit_idx) < 2:
                break
            seed = {best_sched[i]['task_id'] for i in crit_idx if i < len(best_sched)}
            remove_ids = self._select_connected_high_score_ids(
                best_sched, instance, max(2, min(max_nei, len(best_sched))),
                seed_candidates=seed)
            if len(remove_ids) < 2:
                break
            partial = [dict(e) for e in best_sched if e['task_id'] not in remove_ids]
            removed_tasks = [dict(e) for e in best_sched if e['task_id'] in remove_ids]
            repaired = self._normalize_repair_result(
                self._domain_repair(partial, removed_tasks, instance))
            if repaired is None:
                continue
            new_obj, new_sched = repaired
            if new_obj < best_obj - 0.01:
                best_obj, best_sched = new_obj, new_sched
                improved_any = True
        dt = time.time() - t0
        self._mid_intensify_budget_left = max(
            0.0, float(getattr(self, '_mid_intensify_budget_left', 0.0)) - dt)
        if not improved_any:
            return None
        return best_obj, best_sched

    def _repair_removed_tasks(self, instance, partial, removed_tasks, current_sched,
                              removed_ids, scale, sub_time, force_mip=False,
                              repair_op=None, dual_repair=False):
        """按消融开关选择修复方式。

        KG-ALNS V2：use_domain_repair=True → 领域修复；
        dual_repair=True 时同时跑领域修复与通用 regret，取目标更优者。
        """
        ab = self.ablation_flags
        use_mip = force_mip and ab.get('use_mip_repair', True)
        if use_mip:
            return self._normalize_repair_result(self._mip_repair(
                instance, current_sched, partial, removed_tasks, removed_ids, scale, sub_time))
        if ab.get('use_domain_repair', False):
            domain_result = self._normalize_repair_result(
                self._domain_repair(partial, removed_tasks, instance))
            if dual_repair:
                generic_result = self._normalize_repair_result(
                    self._generic_repair(
                        partial, removed_tasks, instance, repair_op='regret'))
                candidates = [
                    x for x in (domain_result, generic_result) if x is not None]
                if candidates:
                    return min(candidates, key=lambda x: x[0])
            return domain_result
        return self._normalize_repair_result(
            self._generic_repair(partial, removed_tasks, instance, repair_op=repair_op))

    def _mip_repair(self, instance, current_sched, partial, removed_tasks,
                    removed_ids, scale, sub_time):
        """mip_exact：领域修复与短时 MIP 竞标（谁更好用谁）。

        核心任务 ≤ max_mip_nei；非核心先启发式插回；子问题时限由 equal_budget 硬顶约 1s。
        """
        if not self.ablation_flags.get('use_mip_repair', True):
            if self.ablation_flags.get('use_domain_repair', True):
                return self._domain_repair(partial, removed_tasks, instance)
            return self._generic_repair(partial, removed_tasks, instance)

        if len(removed_ids) < 2:
            return self._repair_removed_tasks(
                instance, partial, removed_tasks, current_sched, removed_ids,
                scale, sub_time, force_mip=False)

        if self.ablation_flags.get('use_domain_repair', True):
            domain_raw = self._domain_repair(partial, removed_tasks, instance)
        else:
            domain_raw = self._generic_repair(partial, removed_tasks, instance)
        best_result = self._normalize_repair_result(domain_raw)

        max_nei = int(scale.get('max_mip_nei', 20) if scale else 20)
        mip_core = self._select_mip_core_task_ids(
            current_sched, removed_ids, instance, max_nei)
        if len(mip_core) < 2:
            return best_result

        greedy_removed = [t for t in removed_tasks if t['task_id'] not in mip_core]
        seed_partial = [dict(e) for e in partial]
        if greedy_removed:
            if self.ablation_flags.get('use_domain_repair', True):
                g_raw = self._domain_repair(seed_partial, greedy_removed, instance)
            else:
                g_raw = self._generic_repair(seed_partial, greedy_removed, instance)
            g_res = self._normalize_repair_result(g_raw)
            if g_res is not None:
                seed_partial = [dict(e) for e in g_res[1]]

        fixed_assignments = {}
        for e in current_sched:
            if e['task_id'] not in removed_ids:
                fixed_assignments[e['task_id']] = {
                    'ship_id': e['ship_id'], 'start': e['start'],
                    'completion': e['completion'], 'duration': e['duration'],
                    'space': e['space'], 'space_type': e['space_type'],
                    'team': e['team'],
                }
        for e in seed_partial:
            if e['task_id'] not in mip_core:
                fixed_assignments[e['task_id']] = {
                    'ship_id': e['ship_id'], 'start': e.get('start', 0.0),
                    'completion': e.get('completion', 0.0),
                    'duration': e.get('duration', 0.0),
                    'space': e['space'], 'space_type': e['space_type'],
                    'team': e['team'],
                }

        n_sel = len(mip_core)
        iter_sub_time = self._get_iter_sub_time_limit(scale, n_sel, sub_time)
        # 方案A：硬顶 ≤1s，避免 Gurobi 吞噬迭代
        hard_cap = int(scale.get('iter_sub_cap', 1) or 1) if scale else 1
        if scale and scale.get('equal_budget_tight_mip'):
            iter_sub_time = min(iter_sub_time, max(1, hard_cap))
        prev_sub = self.mip_sub_time_limit
        self.mip_sub_time_limit = iter_sub_time
        mip_result = self._solve_mip_subproblem(
            instance, mip_core, fixed_assignments,
            warm_start_schedule=current_sched,
            fix_space=bool(scale.get('mip_fix_space', True) if scale else True))
        self.mip_sub_time_limit = prev_sub

        if mip_result is not None:
            mip_sched, mip_obj = mip_result
            domain_obj = best_result[0] if best_result is not None else float('inf')
            if best_result is None or mip_obj < domain_obj - 0.01:
                best_result = (mip_obj, mip_sched)

        return best_result
    @staticmethod
    def _sa_accept(delta, temperature):
        return sa_accept_move(delta, temperature)

    def _try_domain_warmstart(self, instance, scale, init_deadline, start_schedule, start_obj):
        """短时领域搜索预热（不使用对比基线 ALNS 算子）"""
        warm_sec = scale.get('domain_warm_sec', scale.get('alns_init_sec', 0))
        if warm_sec <= 0 or time.time() + warm_sec >= init_deadline:
            return start_schedule, start_obj
        engine = self._get_insertion_engine(instance)
        warm_deadline = time.time() + warm_sec
        try:
            sched, obj, _, _, _, _ = self._run_domain_search_loop(
                instance, start_schedule, start_obj, warm_deadline,
                engine, 100.0, 0.995, mip_repair_prob=0.0, scale=scale,
                label='领域预热')
            if self.verbose:
                print(f"  领域预热 ({warm_sec}s): obj={obj:.1f}")
            return sched, obj
        except Exception as e:
            if self.verbose:
                print(f"  领域预热失败 ({type(e).__name__}: {e})")
        return start_schedule, start_obj

    def _try_alns_warmstart(self, instance, scale, start_schedule, start_obj, phase_deadline):
        """外部 ALNS 热启动：用对比基线算子短时搜索，结果注入主搜索"""
        warm_sec = scale.get('alns_warm_sec', 0)
        if warm_sec <= 0 or time.time() + warm_sec >= phase_deadline - 1:
            return start_schedule, start_obj
        try:
            seed = instance.get('instance_id', 0) * 1000 + 42
            alns = ALNSSolver(time_limit=warm_sec, verbose=False, num_runs=1)
            sched, obj, _, _, _, _ = alns._solve_single(
                instance, seed, warm_start_schedule=start_schedule)
            if sched is not None and obj < start_obj - 0.01:
                if self.verbose:
                    print(f"  ALNS热启动 ({warm_sec}s): {start_obj:.1f} -> {obj:.1f}")
                return sched, obj
            if self.verbose:
                print(f"  ALNS热启动 ({warm_sec}s): obj={obj:.1f} "
                      f"(未改进初始解 {start_obj:.1f})")
            if sched is not None and obj < start_obj + 0.01:
                return sched, obj
        except Exception as e:
            if self.verbose:
                print(f"  ALNS热启动失败 ({type(e).__name__}: {e})")
        return start_schedule, start_obj

    def _run_domain_search_loop(self, instance, start_schedule, start_obj, loop_deadline,
                                engine, T0, alpha, mip_repair_prob=0.0, scale=None,
                                sub_time=15, stall_mip_threshold=5, max_iters=None,
                                stall_early_exit=None, label='领域搜索'):
        """KG-ALNS 主循环：领域定制破坏 + 领域/通用修复；可选停滞关键路径强化。"""
        if time.time() >= loop_deadline - 1:
            return start_schedule, start_obj, 0, 0, 0, 0

        current_sched, current_obj = start_schedule, start_obj
        best_sched, best_obj = start_schedule, start_obj
        destroy_w = self._initial_destroy_weights()
        l3_sub_w = [1.0] * len(self.DOMAIN_L3_SUB_OPS)
        repair_ops = list(ALNSSolver.REPAIR_OPS)
        repair_w = [1.0] * len(repair_ops)
        use_generic_repair = not self.ablation_flags.get('use_domain_repair', False)
        T = T0
        iters = improved = accepted = mip_count = 0
        stall_count = 0
        since_best = 0
        mip_times = []
        mip_nsels = []
        nb_cap = getattr(self, '_neighborhood_cap', 12)
        loop_max_iters = (max_iters if max_iters is not None
                          else scale.get('mip_alns_iters', 50000) if scale else 50000)
        loop_start = time.time()
        loop_span = max(1.0, loop_deadline - loop_start)
        last_intensify_at = -999
        last_elite_restart_at = -10**9
        elite_thr = int(getattr(self, 'ELITE_RESTART_NO_IMPROVE', 1000) or 1000)

        while time.time() < loop_deadline:
            if max_iters is not None and iters >= max_iters:
                break
            if stall_early_exit is not None and since_best >= stall_early_exit:
                break

            time_left = loop_deadline - time.time()
            if time_left <= 0.5:
                break

            iters += 1
            self._last_repair_knowledge_changed = False

            # elite restart：仅长期无提升 best 时，从 best 邻域大比例破坏–repair
            if (self.ablation_flags.get('use_elite_restart', True)
                    and since_best >= elite_thr
                    and (iters - last_elite_restart_at) >= elite_thr
                    and time_left > 1.0
                    and best_sched):
                er = self._elite_restart_from_best(best_sched, instance)
                last_elite_restart_at = iters
                self._bump_kg_stat('elite_restart_count')
                if er is not None:
                    e_obj, e_sched = er
                    # 跳转到 elite 邻域，恢复探索；不改 SA/destroy 正常路径
                    current_sched, current_obj = e_sched, e_obj
                    if e_obj < best_obj - 0.01:
                        best_sched, best_obj = e_sched, e_obj
                        improved += 1
                    stall_count = 0
                    since_best = 0
                    if self.verbose or (scale and scale.get('log_mip_stats')):
                        print(f"  EliteRestart#{self._kg_plus_stats.get('elite_restart_count', 0)}: "
                              f"obj={e_obj:.1f} (best={best_obj:.1f})",
                              flush=True)
                T *= alpha
                continue

            # 可选：停滞时关键路径强化（V2：critical_path_intensify=True，有限预算）
            stall_thr = stall_mip_threshold if scale else 5
            if (scale and scale.get('critical_path_intensify', False)
                    and since_best >= stall_thr
                    and (iters - last_intensify_at) >= max(2, stall_thr)
                    and time_left > 2.0):
                inten = self._intensify_critical_path_domain(
                    current_sched, instance, scale)
                last_intensify_at = iters
                if inten is not None:
                    i_obj, i_sched = inten
                    delta_i = i_obj - current_obj
                    if self._sa_accept(delta_i, T):
                        current_sched, current_obj = i_sched, i_obj
                        accepted += 1
                        if current_obj < best_obj - 0.01:
                            best_sched, best_obj = current_sched, current_obj
                            improved += 1
                            stall_count = 0
                            since_best = 0
                        else:
                            stall_count += 1
                            since_best += 1
                    else:
                        stall_count += 1
                        since_best += 1
                    T *= alpha
                    time_left = loop_deadline - time.time()
                    if time_left <= 0.5:
                        break

            partial, removed_tasks, removed_ids, layer_idx, l3_sub_idx = (
                self._apply_smart_destroy(
                    current_sched, instance, iters, loop_max_iters, stall_count,
                    destroy_w, l3_sub_w))
            if layer_idx in (
                    self.DOMAIN_LAYER_L1, self.DOMAIN_LAYER_L3, self.DOMAIN_LAYER_L4):
                ab = self.ablation_flags
                if ((layer_idx == self.DOMAIN_LAYER_L1 and ab.get('use_l1_destroy', True))
                        or (layer_idx == self.DOMAIN_LAYER_L3
                            and ab.get('use_l3_destroy', True))
                        or (layer_idx == self.DOMAIN_LAYER_L4
                            and ab.get('use_l4_destroy', True))):
                    self._bump_kg_stat('knowledge_destroy_selected_count')

            effective_sub = min(sub_time, max(1, int(time_left)))
            if scale and scale.get('equal_budget_tight_mip'):
                effective_sub = min(
                    effective_sub, int(scale.get('iter_sub_cap', 1) or 1))
            min_time_for_mip = (scale.get('mip_min_time_left', 12)
                                  if scale is not None else 12)
            use_mip = False
            n_sel = len(removed_ids)
            mip_n_min = scale.get('mip_repair_n_min', 3) if scale else 3
            mip_n_max = scale.get('mip_repair_n_max', nb_cap) if scale else nb_cap
            mip_size_ok = mip_n_min <= n_sel <= mip_n_max
            elapsed_frac = min(1.0, (time.time() - loop_start) / loop_span)
            late_boost = scale.get('mip_late_boost', 1.0) if scale else 1.0
            if elapsed_frac > 0.45 and late_boost > 1.0:
                phase_mult = 1.0 + (late_boost - 1.0) * ((elapsed_frac - 0.45) / 0.55)
            else:
                phase_mult = 1.0
            eff_mip_prob = min(0.50, mip_repair_prob * phase_mult)
            eff_stall_thr = max(2, stall_mip_threshold - (1 if elapsed_frac > 0.6 else 0))
            only_stall = bool(scale.get('mip_only_on_stall', True)) if scale else True
            if (self.ablation_flags.get('use_mip_repair', False)
                    and scale is not None and mip_repair_prob >= 0
                    and mip_size_ok
                    and time_left > min_time_for_mip):
                if stall_count >= eff_stall_thr:
                    use_mip = True
                elif (scale.get('always_compare_mip_on_stall')
                      and stall_count >= scale.get('mip_stall_force_after', 3)):
                    use_mip = True
                elif not only_stall and mip_repair_prob > 0:
                    if since_best >= 8:
                        use_mip = random.random() < min(0.20, max(eff_mip_prob, 1e-9))
                    else:
                        use_mip = random.random() < eff_mip_prob

            repair_op = None
            r_idx = None
            if use_generic_repair and not use_mip:
                r_idx = self._select_weighted_index(repair_w)
                repair_op = repair_ops[r_idx]

            # 停滞或领域层破坏时：领域修复 vs 通用 regret 竞标
            dual_repair = (
                (not use_mip)
                and self.ablation_flags.get('use_domain_repair', False)
                and (stall_count >= 3
                     or layer_idx in {
                         self.DOMAIN_LAYER_L1,
                         self.DOMAIN_LAYER_L3,
                         self.DOMAIN_LAYER_L4,
                     })
            )

            t_mip0 = time.time() if use_mip else None
            repair_result = self._repair_removed_tasks(
                instance, partial, removed_tasks, current_sched, removed_ids,
                scale, effective_sub, force_mip=use_mip, repair_op=repair_op,
                dual_repair=dual_repair)
            if use_mip:
                mip_count += 1
                mip_secs = time.time() - t_mip0
                mip_times.append(mip_secs)
                mip_nsels.append(n_sel)

            if repair_result is None:
                self._record_destroy_operator(layer_idx, 'rejected')
                if self.ablation_flags.get('use_adaptive_weights', True):
                    self._apply_destroy_outcome(destroy_w, layer_idx, 'rejected')
                    if l3_sub_idx is not None:
                        self._update_layer_weights(l3_sub_w, l3_sub_idx, 'rejected')
                    if r_idx is not None:
                        self._update_layer_weights(repair_w, r_idx, 'rejected')
                T *= alpha
                stall_count += 1
                since_best += 1
                if (self.ablation_flags.get('use_adaptive_weights', True)
                        and iters % 100 == 0):
                    destroy_w = [max(0.2, w * 0.7 + 0.3) for w in destroy_w]
                    self._enforce_destroy_weight_floors(destroy_w)
                    l3_sub_w = [max(0.2, w * 0.7 + 0.3) for w in l3_sub_w]
                    repair_w = [max(0.2, w * 0.7 + 0.3) for w in repair_w]
                continue

            new_obj, new_sched = repair_result
            delta = new_obj - current_obj
            if self._sa_accept(delta, T):
                current_sched, current_obj = new_sched, new_obj
                accepted += 1
                outcome = 'accepted'
                if current_obj < best_obj - 0.01:
                    best_sched, best_obj = current_sched, current_obj
                    improved += 1
                    outcome = 'improved'
                    stall_count = 0
                    since_best = 0
                    if getattr(self, '_last_repair_knowledge_changed', False):
                        self._bump_kg_stat('knowledge_improvement_count')
                else:
                    stall_count += 1
                    since_best += 1
                self._record_destroy_operator(layer_idx, outcome, delta=delta)
                if self.ablation_flags.get('use_adaptive_weights', True):
                    self._apply_destroy_outcome(destroy_w, layer_idx, outcome)
                    if l3_sub_idx is not None:
                        self._update_layer_weights(l3_sub_w, l3_sub_idx, outcome)
                    if r_idx is not None:
                        self._update_layer_weights(repair_w, r_idx, outcome)
            else:
                stall_count += 1
                since_best += 1
                self._record_destroy_operator(layer_idx, 'rejected', delta=delta)
                if self.ablation_flags.get('use_adaptive_weights', True):
                    self._apply_destroy_outcome(destroy_w, layer_idx, 'rejected')
                    if l3_sub_idx is not None:
                        self._update_layer_weights(l3_sub_w, l3_sub_idx, 'rejected')
                    if r_idx is not None:
                        self._update_layer_weights(repair_w, r_idx, 'rejected')
            T *= alpha
            self._maybe_trace_best(best_obj)
            if self.ablation_flags.get('use_adaptive_weights', True) and iters % 100 == 0:
                destroy_w = [max(0.2, w * 0.7 + 0.3) for w in destroy_w]
                self._enforce_destroy_weight_floors(destroy_w)
                l3_sub_w = [max(0.2, w * 0.7 + 0.3) for w in l3_sub_w]
                repair_w = [max(0.2, w * 0.7 + 0.3) for w in repair_w]

        self._finalize_destroy_operator_weights(destroy_w)
        loop_elapsed = time.time() - loop_start
        log_stats = bool(scale.get('log_mip_stats', False)) if scale else False
        if (self.verbose or log_stats) and iters > 0:
            layer_info = ', '.join(
                f"{name}={destroy_w[i]:.1f}"
                for i, name in enumerate(self.DOMAIN_DESTROY_LAYERS))
            mip_suffix = ''
            if self._mip_enabled():
                if mip_times:
                    avg_mip = sum(mip_times) / len(mip_times)
                    avg_n = sum(mip_nsels) / len(mip_nsels)
                    mip_sum = sum(mip_times)
                    mip_suffix = (f", MIP={mip_count}次/"
                                  f"均耗{avg_mip:.2f}s(合计{mip_sum:.1f}s)/"
                                  f"均邻域{avg_n:.1f}")
                else:
                    mip_suffix = ", MIP=0次"
            print(f"  {label}: {iters}轮, {improved}次改进, obj={best_obj:.1f}"
                  f"{mip_suffix}, 耗时{loop_elapsed:.1f}s, 破坏权重=[{layer_info}]",
                  flush=True)
            # 累计到求解器，供批量日志汇总
            stats = getattr(self, '_last_mip_stats', None) or {
                'mip_count': 0, 'mip_sec_sum': 0.0, 'iters': 0}
            stats['mip_count'] = stats.get('mip_count', 0) + mip_count
            stats['mip_sec_sum'] = stats.get('mip_sec_sum', 0.0) + sum(mip_times)
            stats['iters'] = stats.get('iters', 0) + iters
            self._last_mip_stats = stats
        return best_sched, best_obj, iters, improved, mip_count, accepted

    def _domain_refinement_phase(self, instance, best_schedule, best_obj, refine_deadline,
                                 engine, T0, alpha, scale, stall_early_exit=None):
        """精修阶段：领域破坏 + 通用/知识修复；末尾 skill-aware 轻量精修吃满余量。"""
        refine_mip = scale.get('refine_mip_prob', 0.0)
        stall_mip = scale.get('stall_mip_threshold', 3)
        sub_time = scale.get('sub_time', 15)
        # 预留最多 ~8% 精修墙钟给 skill-aware（至少 1.5s，不额外加时）
        loop_deadline = refine_deadline
        if self.ablation_flags.get('use_skill_refine', True):
            now = time.time()
            span = max(0.0, refine_deadline - now)
            reserve = min(max(1.5, 0.08 * span), max(0.0, span - 2.0))
            loop_deadline = refine_deadline - reserve
        result = self._run_domain_search_loop(
            instance, best_schedule, best_obj, loop_deadline,
            engine, max(T0 * 0.3, 1.0), alpha,
            mip_repair_prob=refine_mip, scale=scale, sub_time=sub_time,
            stall_mip_threshold=stall_mip,
            stall_early_exit=stall_early_exit, label='领域精修')
        best_sched, best_obj, n_iters = result[0], result[1], result[2]
        if (self.ablation_flags.get('use_skill_refine', True)
                and time.time() < refine_deadline - 0.3):
            refined = self._skill_aware_local_refine(
                best_sched, instance, deadline=refine_deadline, max_passes=3)
            if refined is not None:
                r_obj, r_sched = refined
                if r_obj <= best_obj + 1e-9:
                    if self.verbose or (scale and scale.get('log_mip_stats')):
                        if r_obj < best_obj - 0.01:
                            print(f"  技能精修: {best_obj:.1f} -> {r_obj:.1f}",
                                  flush=True)
                    best_sched, best_obj = r_sched, r_obj
        return best_sched, best_obj, n_iters

    def _construct_diversified_initials(self, instance, scale, init_deadline,
                                        num_starts, solve_start):
        """多起点并行构造，避免单一 ALNS/贪心局部陷阱。

        产生最多 3 类候选：多起点贪心、短 ALNS、第二贪心种子群，取最优。
        """
        candidates = []
        n_greedy = max(3, min(num_starts, 8))
        g_sched, g_obj = generate_multistart_greedy_schedule(
            instance, num_starts=n_greedy)
        if g_sched is not None:
            candidates.append(('greedy', g_sched, g_obj))

        # 第二波不同种子贪心
        if time.time() + 2.0 < init_deadline:
            g2_sched, g2_obj = generate_multistart_greedy_schedule(
                instance, num_starts=max(2, n_greedy // 2))
            # 轻微扰动：改 seed 已在函数内用 random，再跑一次即可多样
            if g2_sched is not None:
                candidates.append(('greedy2', g2_sched, g2_obj))

        # 短 ALNS（若预算允许）
        short_alns = float(scale.get('multi_init_alns_sec', 0) or 0)
        if short_alns > 0 and time.time() + short_alns < init_deadline:
            try:
                seed = instance.get('instance_id', 0) * 17 + 7
                warm = candidates[0][1] if candidates else None
                alns = ALNSSolver(time_limit=short_alns, verbose=False, num_runs=1)
                sched, obj, _, _, _, _ = alns._solve_single(
                    instance, seed, warm_start_schedule=warm)
                if sched is not None:
                    candidates.append(('alns_short', sched, obj))
            except Exception:
                pass

        if not candidates:
            return generate_multistart_greedy_schedule(instance, num_starts=num_starts)
        best = min(candidates, key=lambda x: x[2])
        if self.verbose:
            detail = ', '.join(f"{n}={o:.1f}" for n, _, o in candidates)
            print(f"  多起点初解: [{detail}] → 选 {best[0]}={best[2]:.1f}")
        return best[1], best[2]

    def _construct_initial_solution(self, instance, scale, init_deadline,
                                    num_starts, num_tasks, num_ships, solve_start):
        """多起点贪心 / 滚动时域 / 多样化初解；优先使用共享初解（与基线对齐）。"""
        shared = instance.get('_shared_init_schedule')
        if shared is not None:
            utils = HeuristicScheduler()
            utils._load_team_config(instance)
            utils._load_resource_config(instance)
            obj = utils._compute_objective(shared, instance)
            if self.verbose:
                print(f"  使用共享初解 obj={obj:.1f}", flush=True)
            return [dict(e) for e in shared], obj

        if scale.get('diversified_init', True) and num_ships > 10:
            return self._construct_diversified_initials(
                instance, scale, init_deadline, num_starts, solve_start)

        tasks_per_team = num_tasks / max(instance.get('num_teams', 1), 1)
        if num_ships <= 10:
            rolling_ok = (not scale['prefer_greedy_init']
                          and num_tasks >= 8 and tasks_per_team >= 2.5)
        else:
            rolling_ok = num_tasks >= 15 and tasks_per_team >= 3.0
        use_rolling = rolling_ok and time.time() + 30 < init_deadline

        if use_rolling:
            if self.verbose:
                scale_tag = "小规模" if num_ships <= 10 else "大规模"
                print(f"  使用滚动时域初始解 [{scale_tag}] (任务={num_tasks}, "
                      f"任务/队={tasks_per_team:.1f}, "
                      f"window={max(10, min(30, num_tasks // 5))}, "
                      f"init_limit={init_deadline - solve_start:.0f}s)")
            init_schedule, init_obj = self._rolling_horizon_initial_solution(
                instance, deadline=init_deadline)
            if init_schedule is not None:
                return init_schedule, init_obj
            if self.verbose:
                print(f"  滚动时域失败/超时，回退到多起点贪心 (starts={num_starts})")
            return generate_multistart_greedy_schedule(instance, num_starts=num_starts)

        if self.verbose:
            print(f"  使用多起点贪心 (starts={num_starts})")
        return generate_multistart_greedy_schedule(instance, num_starts=num_starts)

    def _solve_scale_aware_hybrid(self, instance, scale, solve_start, deadline):
        """规模感知混合：任务数大时以 ALNS 为主引擎，MIP 作关键路径局部强化。

        注意：同等墙钟主表（equal_budget_tight_mip）禁止进入此旁路。
        本路径用于解质量优先辅实验；结束后用剩余墙钟做领域搜索，尽量吃满时限。
        """
        num_tasks = instance.get('num_tasks', len(instance.get('tasks_data', [])))
        log_stats = bool(scale.get('log_mip_stats', False))
        main_frac = float(scale.get('scale_aware_alns_frac', 0.75))
        refine_reserve = max(12.0, (deadline - solve_start) * (1.0 - main_frac))
        alns_deadline = deadline - refine_reserve
        alns_sec = max(8.0, alns_deadline - time.time())
        self._last_mip_stats = {'mip_count': 0, 'mip_sec_sum': 0.0, 'iters': 0}

        if self.verbose or log_stats:
            print(f"\n[KG-ALNS·规模感知] 任务={num_tasks}>阈值，"
                  f"ALNS主搜索≤{alns_sec:.0f}s + MIP强化≤{refine_reserve:.0f}s",
                  flush=True)

        best_sched, best_obj = generate_multistart_greedy_schedule(
            instance, num_starts=max(4, int(scale.get('num_starts', 6))))
        try:
            seed = instance.get('instance_id', 0) * 1000 + 91
            alns = ALNSSolver(time_limit=alns_sec, verbose=False, num_runs=1)
            sched, obj, _, _, _, _ = alns._solve_single(
                instance, seed, warm_start_schedule=best_sched)
            if sched is not None and obj < best_obj - 0.01:
                best_sched, best_obj = sched, obj
                if self.verbose or log_stats:
                    print(f"  ALNS主引擎: obj={best_obj:.1f}", flush=True)
        except Exception as e:
            if self.verbose or log_stats:
                print(f"  ALNS主引擎失败: {e}", flush=True)

        # MIP 局部强化：反复抽取关键路径 ≤max_nei 任务
        polish_rounds = int(scale.get('scale_aware_mip_rounds', 4))
        max_nei = int(scale.get('max_mip_nei', 15) or 15)
        no_gain = 0
        mip_count = 0
        mip_sec_sum = 0.0
        for pr in range(polish_rounds):
            if time.time() >= deadline - 1.0:
                break
            remaining = max(1.0, deadline - time.time())
            round_budget = remaining / max(1, polish_rounds - pr)
            utils = HeuristicScheduler()
            utils._load_team_config(instance)
            utils._load_resource_config(instance)
            crit_idx = utils._compute_critical_path(best_sched)
            crit_ids = {best_sched[i]['task_id'] for i in crit_idx}
            crit_ids = self._select_mip_core_task_ids(
                best_sched, crit_ids, instance, max_nei)
            if len(crit_ids) < 2:
                break
            fixed = self._fix_other_tasks(best_sched, crit_ids)
            prev = self.mip_sub_time_limit
            self.mip_sub_time_limit = min(
                float(scale.get('polish_sub_cap', 12)), round_budget)
            t0 = time.time()
            result = self._solve_mip_subproblem(
                instance, crit_ids, fixed,
                warm_start_schedule=best_sched, fix_space=True)
            self.mip_sub_time_limit = prev
            mip_count += 1
            mip_sec_sum += time.time() - t0
            if result is None:
                no_gain += 1
                if no_gain >= 2:
                    break
                continue
            new_sched, new_obj = result
            if new_obj < best_obj - 0.01:
                if self.verbose or log_stats:
                    print(f"  MIP强化#{pr + 1}: {best_obj:.1f} -> {new_obj:.1f}",
                          flush=True)
                best_sched, best_obj = new_sched, new_obj
                no_gain = 0
            else:
                no_gain += 1
                if no_gain >= 2:
                    break

        self._last_mip_stats = {
            'mip_count': mip_count, 'mip_sec_sum': mip_sec_sum, 'iters': 0}

        # 剩余墙钟：领域 LNS 吃满（避免 ALNS≈98s 后 MIP 早退导致总耗时不满）
        if time.time() < deadline - 1.5:
            engine = self._get_insertion_engine(instance)
            best_sched, best_obj, _, _, _, _ = self._run_domain_search_loop(
                instance, best_sched, best_obj, deadline - 0.5,
                engine, 30.0, 0.995,
                mip_repair_prob=scale.get('mip_repair_prob', 0.05),
                scale=scale, sub_time=scale.get('sub_time', 2),
                stall_mip_threshold=scale.get('stall_mip_threshold', 5),
                label='规模感知·余量LNS')

        if (self.ablation_flags.get('use_skill_refine', True)
                and time.time() < deadline - 0.5):
            refined = self._skill_aware_local_refine(
                best_sched, instance, deadline=deadline, max_passes=2)
            if refined is not None and refined[0] < best_obj - 0.01:
                best_obj, best_sched = refined
        if self.ablation_flags.get('use_team_swap', True) and time.time() < deadline - 0.5:
            swapped = self._team_swap_postprocess(best_sched, instance, deadline=deadline)
            utils = HeuristicScheduler()
            utils._load_team_config(instance)
            utils._load_resource_config(instance)
            swapped_obj = utils._compute_objective(swapped, instance)
            if swapped_obj < best_obj - 0.01:
                best_sched, best_obj = swapped, swapped_obj

        solve_time = time.time() - solve_start
        schedule_feasible = True
        if best_sched:
            rep_sched, rep_obj, schedule_feasible = report_objective_from_schedule(
                best_sched, instance, penalty_coeff=self.penalty_coeff)
            if rep_sched is not None:
                best_sched, best_obj = rep_sched, rep_obj

        st = getattr(self, '_last_mip_stats', None) or {}
        if self.verbose or log_stats:
            mip_tail = ''
            if self._mip_enabled():
                avg = (st['mip_sec_sum'] / st['mip_count']) if st.get('mip_count') else 0.0
                mip_tail = (f", MIP={st.get('mip_count', 0)}次/均耗{avg:.2f}s")
            print(f"  规模感知完成: obj={best_obj:.1f}, 耗时={solve_time:.1f}s, "
                  f"LNS={st.get('iters', 0)}轮{mip_tail}",
                  flush=True)

        return {
            'status': 'KG_ALNS',
            'objective': best_obj,
            'solve_time': solve_time,
            'search_time': solve_time,
            'postprocess_time': 0.0,
            'time_limit': scale['time_limit'],
            'feasible': schedule_feasible,
            'schedule': best_sched,
            'num_variables': 0,
            'num_constraints': 0,
            'num_binary': 0,
            'num_continuous': 0,
            'node_count': st.get('iters', 0),
            'iteration_count': 0,
            'accepted_count': 0,
            'mip_repair_count': st.get('mip_count', 0),
            'ablation_variant': self.ablation_variant,
            'ablation_flags': dict(self.ablation_flags),
            'best_bound': None,
            'mip_gap': None,
        }

    # ================================================================
    # 主求解入口
    # ================================================================

    def _plan_b_improve_only_intensify(self, best_sched, best_obj, instance, scale,
                                         deadline, since_best=0, solve_start=None):
        """方案B强化：领域优先；加深停滞/尾段才短 MIP；仅严格改进才返回。

        返回 (obj, sched) 或 None。
        """
        if best_sched is None or time.time() >= deadline - 0.8:
            return None
        best = (best_obj, [dict(e) for e in best_sched])
        improved = False
        domain_improved = False

        # 1) 领域关键路径连通邻域强化（无 Gurobi）
        if (scale.get('critical_path_intensify', True)
                and self.ablation_flags.get('use_domain_repair', True)):
            inten = self._intensify_critical_path_domain(best[1], instance, scale)
            if inten is not None and inten[0] < best[0] - 0.01:
                best = inten
                improved = True
                domain_improved = True

        # 2) MIP：深停滞/尾段触发；默认即使领域已改进也可再试（避免被领域「饿死」）
        deep_stall = int(scale.get('mip_deep_stall_threshold',
                                   max(8, int(scale.get('stall_mip_threshold', 4) or 4) * 2)))
        late_frac = float(scale.get('mip_late_wall_frac', 0.75) or 0.75)
        t_now = time.time()
        span = 1.0
        if solve_start is not None:
            span = max(1.0, deadline - float(solve_start))
            wall_progress = (t_now - float(solve_start)) / span
        else:
            wall_progress = 0.0
        is_late = wall_progress >= late_frac
        deep_or_late = since_best >= deep_stall or is_late
        # 深停滞/尾段：允许在领域成功后再打 MIP；浅停滞仍优先省预算给 ALNS
        allow_after_domain = bool(scale.get('mip_allow_after_domain', True)) and deep_or_late
        allow_mip = (
            self.ablation_flags.get('use_mip_repair', True)
            and scale.get('intensify_mip', True)
            and t_now < deadline - 1.5
            and deep_or_late
            and (not domain_improved or allow_after_domain)
            and t_now >= float(getattr(self, '_next_mip_allowed_at', 0.0) or 0.0)
            and self._mip_budget_remaining() >= 1.0
        )

        if allow_mip:
            max_nei = int(scale.get('max_mip_nei', 10) or 10)
            utils = HeuristicScheduler()
            utils._load_team_config(instance)
            utils._load_resource_config(instance)
            crit_idx = list(utils._compute_critical_path(best[1]))
            seed = {best[1][i]['task_id'] for i in crit_idx if i < len(best[1])}
            remove_ids = self._select_connected_high_score_ids(
                best[1], instance, max(2, min(max_nei, len(best[1]))),
                seed_candidates=seed if seed else None)
            if len(remove_ids) >= 2:
                # 中途短、尾段略长；均受预算帽约束
                if is_late:
                    want = float(scale.get('intensify_mip_late_sec',
                                           scale.get('iter_sub_cap', 3)) or 3)
                else:
                    want = float(scale.get('intensify_mip_sec',
                                           scale.get('iter_sub_cap', 3)) or 3)
                sub = min(want, self._mip_budget_remaining(),
                          max(1.0, deadline - t_now - 0.5))
                sub = max(1, int(sub))

                # 尾段且未用过：可尝试放开空间；否则 fix_space
                fix_space = True
                if (is_late and scale.get('intensify_unfix_space_late', True)
                        and not getattr(self, '_late_unfix_mip_used', False)):
                    fix_space = False
                    self._late_unfix_mip_used = True

                partial = [dict(e) for e in best[1] if e['task_id'] not in remove_ids]
                removed_tasks = [dict(e) for e in best[1] if e['task_id'] in remove_ids]
                # 临时覆盖时限与 fix_space，避免 equal_budget_tight 把尾段 MIP 压回 1–2s
                saved_fix = scale.get('mip_fix_space', True)
                saved_cap = scale.get('iter_sub_cap', 3)
                saved_floor = scale.get('mip_sub_time_floor', 2)
                scale = dict(scale)
                scale['mip_fix_space'] = fix_space
                scale['iter_sub_cap'] = sub
                scale['mip_sub_time_floor'] = min(saved_floor, sub)
                scale['sub_time'] = sub
                self._active_scale = scale

                t0 = time.time()
                repaired = self._normalize_repair_result(self._mip_repair(
                    instance, best[1], partial, removed_tasks, remove_ids, scale, sub))
                dt = time.time() - t0
                self._charge_mip_budget(dt)
                cooldown = float(scale.get('mip_cooldown_sec', 8) or 8)
                self._next_mip_allowed_at = time.time() + cooldown

                stats = getattr(self, '_last_mip_stats', None) or {
                    'mip_count': 0, 'mip_sec_sum': 0.0, 'iters': 0}
                stats['mip_count'] = stats.get('mip_count', 0) + 1
                stats['mip_sec_sum'] = stats.get('mip_sec_sum', 0.0) + dt
                stats['mip_improve'] = stats.get('mip_improve', 0)
                stats['mip_budget_left'] = self._mip_budget_remaining()
                self._last_mip_stats = stats

                if repaired is not None and repaired[0] < best[0] - 0.01:
                    best = repaired
                    improved = True
                    stats['mip_improve'] = stats.get('mip_improve', 0) + 1
                elif not fix_space:
                    # 放开空间无改进：同邻域再试固定空间（若预算仍够）
                    if self._mip_budget_remaining() >= 1.0 and time.time() < deadline - 1.0:
                        scale['mip_fix_space'] = True
                        self._active_scale = scale
                        sub2 = min(sub, int(self._mip_budget_remaining()),
                                   max(1, int(deadline - time.time())))
                        scale['iter_sub_cap'] = sub2
                        scale['sub_time'] = sub2
                        t1 = time.time()
                        repaired2 = self._normalize_repair_result(self._mip_repair(
                            instance, best[1], partial, removed_tasks, remove_ids,
                            scale, sub2))
                        dt2 = time.time() - t1
                        self._charge_mip_budget(dt2)
                        stats['mip_count'] += 1
                        stats['mip_sec_sum'] += dt2
                        if repaired2 is not None and repaired2[0] < best[0] - 0.01:
                            best = repaired2
                            improved = True
                            stats['mip_improve'] = stats.get('mip_improve', 0) + 1
                scale['mip_fix_space'] = saved_fix
                scale['iter_sub_cap'] = saved_cap
                scale['mip_sub_time_floor'] = saved_floor
                self._active_scale = scale

        if not improved:
            return None
        return best

    def _solve_plan_b(self, instance, scale, solve_start, deadline):
        """方案B主流程：ALNS 同构主搜索 + improve-only 领域/MIP 强化 + Polish/Swap。"""
        postprocess_reserve = float(scale.get('postprocess_reserve_sec', 5) or 5)
        search_deadline = deadline - postprocess_reserve
        instance_id = instance.get('instance_id', '?')
        num_tasks = instance.get('num_tasks', len(instance['tasks_data']))
        num_ships = instance.get('num_ships', 0)
        num_starts = int(scale.get('num_starts', 4))
        init_cap = float(scale.get('init_max_sec', 4))
        init_deadline = min(solve_start + init_cap, search_deadline - 2.0)
        log_stats = bool(scale.get('log_mip_stats', False))

        self._last_mip_stats = {
            'mip_count': 0, 'mip_sec_sum': 0.0, 'iters': 0,
            'mip_improve': 0, 'mip_budget_left': 0.0,
            'polish_improve': 0,
        }
        self._active_scale = scale
        self._neighborhood_cap = scale.get('neighborhood_cap', 12)
        ov = getattr(self, '_neighborhood_cap_override', None)
        if ov is not None:
            self._neighborhood_cap = int(ov)
        self._iter_sub_cap = scale.get('iter_sub_cap', 3)
        self._polish_sub_cap = scale.get('polish_sub_cap', 10)
        saved_sub = self.mip_sub_time_limit
        self.mip_sub_time_limit = scale.get('sub_time', 3)
        mip_budget = self._init_plan_b_mip_budget(scale, solve_start, search_deadline)
        self._mid_intensify_budget_left = float(
            scale.get('mid_intensify_budget_sec', 3) or 0)
        self._last_mip_stats['mip_budget_left'] = mip_budget

        if self.verbose or log_stats:
            if self._mip_enabled():
                print(f"\n[KG-ALNS] 算例 {instance_id}: {num_ships}船, {num_tasks}任务, "
                      f"墙钟={scale['time_limit']}s(ALNS主≤{search_deadline - solve_start:.0f}s+"
                      f"后处理{postprocess_reserve:.0f}s), "
                      f"nei≤{scale.get('neighborhood_cap')}, "
                      f"PolishNei≤{scale.get('polish_max_nei')}, "
                      f"中途MIP={'ON' if scale.get('intensify_mip') else 'OFF'}, "
                      f"领域强化预算={self._mid_intensify_budget_left:.0f}s, "
                      f"Polish≤{scale.get('polish_sub_cap')}s",
                      flush=True)
            else:
                print(f"\n[KG-ALNS] 算例 {instance_id}: {num_ships}船, {num_tasks}任务, "
                      f"墙钟={scale['time_limit']}s(主搜索≤{search_deadline - solve_start:.0f}s+"
                      f"后处理{postprocess_reserve:.0f}s), "
                      f"邻域≤{scale.get('neighborhood_cap')}",
                      flush=True)

        # 初始解（不用外部 ALNS 热启动；ALNS 即主引擎）
        incumbent_schedule, incumbent_obj = self._construct_initial_solution(
            instance, scale, init_deadline, num_starts, num_tasks, num_ships, solve_start)
        if self.verbose or log_stats:
            print(f"  初解 obj={incumbent_obj:.1f}", flush=True)

        stall_thr = int(scale.get('stall_mip_threshold', 4) or 4)

        def _intensify(best_sched, best_obj, since_best=0):
            return self._plan_b_improve_only_intensify(
                best_sched, best_obj, instance, scale, search_deadline,
                since_best=since_best, solve_start=solve_start)

        use_intensify = (
            self.ablation_flags.get('use_domain_repair', True)
            or (scale.get('intensify_mip', False)
                and self.ablation_flags.get('use_mip_repair', True)))
        run_id = int(instance.get('_run_id', 0) or 0)
        seed = instance.get('instance_id', 0) * 1000 + run_id * 17 + 42
        alns_budget = max(5.0, search_deadline - time.time())
        use_archive = bool(scale.get('polish_from_archive', False))
        alns = ALNSSolver(time_limit=alns_budget, verbose=False, num_runs=1)
        best_sched, best_obj, alns_time, iters, _, n_int = alns._solve_single(
            instance, seed,
            warm_start_schedule=incumbent_schedule,
            wall_deadline=search_deadline,
            intensify_fn=_intensify if use_intensify else None,
            stall_threshold=stall_thr,
            collect_archive=use_archive,
            archive_eta=float(scale.get('archive_eta', 0.10) or 0.10),
            archive_capacity=int(scale.get('archive_capacity', 5) or 5),
        )
        polish_archive = list(getattr(alns, 'last_archive', []) or [])
        self._last_mip_stats['iters'] = iters
        self._last_mip_stats['intensify_success'] = n_int
        self._last_mip_stats['archive_size'] = len(polish_archive)
        if self.verbose or log_stats:
            st = self._last_mip_stats
            mc = st.get('mip_count', 0)
            ms = st.get('mip_sec_sum', 0.0)
            avg = (ms / mc) if mc else 0.0
            arch_msg = (f", 档案={len(polish_archive)}" if use_archive else "")
            print(f"  ALNS主搜索: {iters}轮, obj={best_obj:.1f}, "
                  f"强化成功{n_int}次, MIP尝试={mc}次/改进{st.get('mip_improve', 0)}/"
                  f"均耗{avg:.2f}s/预算余{st.get('mip_budget_left', 0):.1f}s"
                  f"{arch_msg}, 耗时{alns_time:.1f}s",
                  flush=True)

        incumbent_schedule, incumbent_obj = best_sched, best_obj

        # 主搜结束仍无中途 MIP：强制补打 1 次（跳过领域，直接走精确修复）
        if (scale.get('force_mid_mip_once', True)
                and scale.get('intensify_mip', True)
                and self.ablation_flags.get('use_mip_repair', True)
                and int(self._last_mip_stats.get('mip_count', 0) or 0) == 0
                and self._mip_budget_remaining() >= 1.0
                and time.time() < search_deadline - 1.5):
            force_scale = dict(scale)
            force_scale['critical_path_intensify'] = False
            force_scale['mip_allow_after_domain'] = True
            forced = self._plan_b_improve_only_intensify(
                incumbent_schedule, incumbent_obj, instance, force_scale,
                search_deadline,
                since_best=max(int(scale.get('mip_deep_stall_threshold', 10) or 10), 99),
                solve_start=solve_start)
            if forced is not None and forced[0] < incumbent_obj - 0.01:
                incumbent_schedule, incumbent_obj = forced[1], forced[0]
                self._last_mip_stats['intensify_success'] = (
                    int(self._last_mip_stats.get('intensify_success', 0) or 0) + 1)
            if self.verbose or log_stats:
                print(f"  强制中途MIP: 尝试={self._last_mip_stats.get('mip_count', 0)}次, "
                      f"obj={incumbent_obj:.1f}, 预算余"
                      f"{self._last_mip_stats.get('mip_budget_left', 0):.1f}s",
                      flush=True)

        search_time = time.time() - solve_start
        postprocess_start = time.time()

        # Final Polish（尾段精确磨光；这是 MIP 贡献的主入口）
        if (scale.get('polish') and self.ablation_flags.get('use_polish', True)
                and time.time() < deadline - 0.5):
            remaining = max(0.0, deadline - time.time())
            polish_want = float(scale.get('polish_sub_cap', 7) or 7)
            # 给 Team Swap / 尾段 ALNS 留 ≥1s
            if remaining >= 2.5:
                pre_obj = incumbent_obj
                polished = self._final_polish(
                    incumbent_schedule, instance,
                    time_budget=min(remaining - 1.2, polish_want),
                    archive=polish_archive if use_archive else None)
                utils = HeuristicScheduler()
                utils._load_team_config(instance)
                utils._load_resource_config(instance)
                polished_obj = utils._compute_objective(polished, instance)
                if polished_obj < incumbent_obj - 0.01:
                    if self.verbose or log_stats:
                        n_tgt = self._last_mip_stats.get('archive_polish_targets', 1)
                        print(f"  Final Polish: {incumbent_obj:.1f} -> {polished_obj:.1f}"
                              f" (靶点={n_tgt})",
                              flush=True)
                    incumbent_schedule, incumbent_obj = polished, polished_obj
                    self._last_mip_stats['polish_improve'] = (
                        self._last_mip_stats.get('polish_improve', 0) + 1)
                elif self.verbose or log_stats:
                    print(f"  Final Polish: 无改进 (obj={pre_obj:.1f})", flush=True)

        # Skill-aware 精修 / Team Swap
        if (self.ablation_flags.get('use_skill_refine', True)
                and time.time() < deadline - 0.5):
            refined = self._skill_aware_local_refine(
                incumbent_schedule, instance, deadline=deadline, max_passes=2)
            if refined is not None and refined[0] < incumbent_obj - 0.01:
                incumbent_obj, incumbent_schedule = refined
        if (self.ablation_flags.get('use_team_swap', True)
                and time.time() < deadline - 0.5):
            swapped = self._team_swap_postprocess(
                incumbent_schedule, instance, deadline=deadline)
            utils = HeuristicScheduler()
            utils._load_team_config(instance)
            utils._load_resource_config(instance)
            swapped_obj = utils._compute_objective(swapped, instance)
            if swapped_obj < incumbent_obj - 0.01:
                incumbent_schedule, incumbent_obj = swapped, swapped_obj

        # 尾段：后处理提前结束则把剩余墙钟还给 ALNS（无中途 MIP，尽量追平基线迭代）
        if time.time() < deadline - 1.5:
            rem = deadline - time.time() - 0.3
            if rem > 2.0:
                alns2 = ALNSSolver(time_limit=rem, verbose=False, num_runs=1)
                s2, o2, _, it2, _, n2 = alns2._solve_single(
                    instance, seed + 7,
                    warm_start_schedule=incumbent_schedule,
                    wall_deadline=deadline - 0.2,
                    intensify_fn=_intensify if use_intensify else None,
                    stall_threshold=stall_thr,
                )
                self._last_mip_stats['iters'] = self._last_mip_stats.get('iters', 0) + it2
                if o2 < incumbent_obj - 0.01:
                    incumbent_schedule, incumbent_obj = s2, o2
                    if self.verbose or log_stats:
                        print(f"  尾段ALNS: obj={incumbent_obj:.1f} (+强化成功{n2})",
                              flush=True)

        postprocess_time = time.time() - postprocess_start
        solve_time = time.time() - solve_start
        self.mip_sub_time_limit = saved_sub

        schedule_feasible = True
        if incumbent_schedule:
            rep_sched, rep_obj, schedule_feasible = report_objective_from_schedule(
                incumbent_schedule, instance, penalty_coeff=self.penalty_coeff)
            if rep_sched is not None:
                incumbent_schedule, incumbent_obj = rep_sched, rep_obj

        st = self._last_mip_stats
        mc = int(st.get('mip_count', 0) or 0)
        if self.verbose or log_stats:
            mip_tail = ''
            if self._mip_enabled():
                ms = st.get('mip_sec_sum', 0.0)
                avg = (ms / mc) if mc else 0.0
                mip_tail = (f", MIP={mc}次(改进{st.get('mip_improve', 0)},"
                            f"均耗{avg:.2f}s)")
            print(f"  KG-ALNS完成: LNS={st.get('iters', 0)}轮{mip_tail}, "
                  f"obj={incumbent_obj:.1f}, "
                  f"耗时={solve_time:.1f}s(主{search_time:.1f}+后{postprocess_time:.1f})",
                  flush=True)
            if self._mip_enabled():
                print(f"    MIP明细: 中途={mc - int(st.get('polish_mip_count', 0) or 0)}次, "
                      f"Polish={int(st.get('polish_mip_count', 0) or 0)}次, "
                      f"改进={st.get('mip_improve', 0)}, "
                      f"Polish改进={st.get('polish_improve', 0)}",
                      flush=True)

        return {
            'status': 'KG_ALNS',
            'objective': incumbent_obj,
            'solve_time': solve_time,
            'search_time': search_time,
            'postprocess_time': postprocess_time,
            'time_limit': scale['time_limit'],
            'feasible': schedule_feasible,
            'schedule': incumbent_schedule,
            'num_variables': 0,
            'num_constraints': 0,
            'num_binary': 0,
            'num_continuous': 0,
            'node_count': st.get('iters', 0),
            'iteration_count': 0,
            'accepted_count': 0,
            'mip_repair_count': mc,
            'ablation_variant': self.ablation_variant,
            'ablation_flags': dict(self.ablation_flags),
            'destroy_profile': str(globals().get('DESTROY_PROFILE', 'baseline') or 'baseline'),
            'kg_plus_stats': dict(getattr(self, '_kg_plus_stats', {}) or {}),
            'convergence_trace': self._export_convergence_trace(),
            **({'destroy_operator_stats': self._export_destroy_operator_stats()}
               if getattr(self, '_collect_op_stats', False) else {}),
            'best_bound': None,
            'mip_gap': None,
        }

    def solve(self, instance):
        """KG-ALNS 求解主流程

        同等墙钟主表：方案B（plan_b_alns_main=True）—
        通用 ALNS 主搜索 + 领域/MIP 强化 + Final Polish。
        小规模 quick_mode 仍走领域混合路径（可含 MIP 修复）。
        """
        self._selection_counts = {}
        self._insertion_engine = None
        instance = dict(instance)
        instance['_penalty_coeff'] = self.penalty_coeff
        scale = self._apply_instance_scale(instance)
        self._active_scale = scale
        self._neighborhood_cap = scale['neighborhood_cap']
        ov = getattr(self, '_neighborhood_cap_override', None)
        if ov is not None:
            self._neighborhood_cap = int(ov)
        self._iter_sub_cap = scale['iter_sub_cap']
        self._polish_sub_cap = scale.get('polish_sub_cap', 120)
        self._domain_insert_limit = scale.get('domain_insert_limit', 12)
        self._destroy_ratio_mult = scale.get('destroy_ratio_mult', 1.0)
        self._mid_intensify_budget_left = float(
            scale.get('mid_intensify_budget_sec', 0) or 0)
        stall_early = scale.get('stall_early_exit')
        quick_mode = scale.get('quick_mode', False)

        solve_start = time.time()
        deadline = solve_start + scale['time_limit']
        self._record_convergence = bool(
            globals().get('ABLATION_RECORD_CONVERGENCE', False))
        if self._record_convergence:
            self._reset_convergence_trace(solve_start, scale['time_limit'])

        # 方案B：同等墙钟主表默认路径
        if scale.get('plan_b_alns_main', False) and not quick_mode:
            return self._solve_plan_b(instance, scale, solve_start, deadline)

        num_tasks = instance.get('num_tasks', len(instance.get('tasks_data', [])))
        thr = int(scale.get('scale_aware_threshold', 120) or 120)
        # 同等墙钟主表禁止旁路：scale_aware 吃不满 120s，且与梯度1主流程不一致
        if (scale.get('scale_aware_hybrid') and not quick_mode
                and not scale.get('equal_budget_tight_mip')
                and num_tasks > thr
                and self.ablation_flags.get('use_mip_repair', True)):
            return self._solve_scale_aware_hybrid(
                instance, scale, solve_start, deadline)

        postprocess_reserve = scale.get('postprocess_reserve_sec', 0)
        if scale.get('polish') and postprocess_reserve <= 0 and not quick_mode:
            postprocess_reserve = 15
        search_deadline = deadline - postprocess_reserve
        refine_ratio = scale.get('alns_refine_ratio', 0.0)
        bootstrap_ratio = scale.get('alns_bootstrap_ratio', 0.0)

        if quick_mode:
            refine_budget = 0.0
            bootstrap_budget = 0.0
            hybrid_budget = scale.get('hybrid_max_sec', min(20, scale['time_limit']))
            init_deadline = solve_start + min(8, scale['time_limit'] * 0.4)
            bootstrap_deadline = solve_start + bootstrap_budget
            hybrid_deadline = solve_start + hybrid_budget
        else:
            init_cap = scale.get('init_max_sec', min(45, scale['time_limit'] * 0.12))
            init_deadline = solve_start + init_cap
            refine_budget = bootstrap_budget = hybrid_budget = 0.0
            bootstrap_deadline = hybrid_deadline = search_deadline
        instance_id = instance.get('instance_id', '?')
        num_tasks = instance.get('num_tasks', len(instance['tasks_data']))
        num_ships = instance.get('num_ships', 0)

        num_starts = scale['num_starts']
        sub_time = scale['sub_time']
        max_alns_iters = scale['mip_alns_iters']
        mip_repair_prob = scale.get('mip_repair_prob', 0.1)
        saved_sub_time = self.mip_sub_time_limit
        self.mip_sub_time_limit = sub_time

        T0, alpha = 100.0, 0.995

        self._last_mip_stats = {'mip_count': 0, 'mip_sec_sum': 0.0, 'iters': 0}
        self._collect_op_stats = bool(
            globals().get('ABLATION_COLLECT_OPERATOR_STATS', False))
        self._init_kg_plus_stats()
        if self._collect_op_stats:
            self._init_destroy_operator_stats()
        if self.verbose or scale.get('log_mip_stats'):
            ab = self.ablation_variant
            if ab and ab != 'full':
                print(f"  [消融={ab}] {self.ablation_flags}", flush=True)
            if quick_mode:
                print(f"\n[KG-ALNS] 算例 {instance_id}: {num_ships}船, {num_tasks}任务, "
                      f"小规模快速模式, 时限≤{scale['time_limit']}s, "
                      f"迭代≤{scale.get('max_hybrid_iters', 25)}", flush=True)
            else:
                search_window = max(0.0, scale['time_limit'] - postprocess_reserve)
                hdr = (f"\n[KG-ALNS] 算例 {instance_id}: {num_ships}船, {num_tasks}任务, "
                       f"墙钟={scale['time_limit']}s(主搜索≤{search_window:.0f}s+"
                       f"后处理{postprocess_reserve:.0f}s), "
                       f"邻域≤{scale.get('neighborhood_cap', '?')}")
                if self._mip_enabled():
                    hdr += (f", MaxMIPNei={scale.get('max_mip_nei', '?')}, "
                            f"子时限floor/cap="
                            f"{scale.get('mip_sub_time_floor')}/{scale.get('iter_sub_cap')}, "
                            f"MIPprob={mip_repair_prob:.0%}, "
                            f"stallThr={scale.get('stall_mip_threshold')}, "
                            f"onlyStall={scale.get('mip_only_on_stall', False)}")
                print(hdr, flush=True)
        # 阶段一：初始解
        incumbent_schedule, incumbent_obj = self._construct_initial_solution(
            instance, scale, init_deadline, num_starts, num_tasks, num_ships, solve_start)
        incumbent_schedule, incumbent_obj = self._try_domain_warmstart(
            instance, scale, init_deadline, incumbent_schedule, incumbent_obj)
        if not quick_mode:
            incumbent_schedule, incumbent_obj = self._try_alns_warmstart(
                instance, scale, incumbent_schedule, incumbent_obj, search_deadline)

        if self.verbose or scale.get('log_mip_stats'):
            print(f"  热启动后 obj={incumbent_obj:.1f}", flush=True)
        self._kg_plus_stats['obj_at_init'] = float(incumbent_obj)
        self._maybe_trace_best(incumbent_obj)

        best_schedule, best_obj = incumbent_schedule, incumbent_obj
        engine = self._get_insertion_engine(instance)

        iter_count = 0
        improved_count = 0
        accepted_count = 0
        mip_repair_count = 0

        # init + 热启动完成后，按实际剩余主搜索窗口分配 bootstrap → hybrid → refine
        phase_now = time.time()
        if quick_mode:
            bootstrap_end = bootstrap_deadline
            hybrid_end = hybrid_deadline
        else:
            search_remaining = max(8.0, search_deadline - phase_now)
            refine_budget = refine_ratio * search_remaining
            bootstrap_budget = bootstrap_ratio * search_remaining
            hybrid_budget = max(0.0, search_remaining - refine_budget - bootstrap_budget)
            bootstrap_end = phase_now + bootstrap_budget
            hybrid_end = bootstrap_end + hybrid_budget
            if self.verbose or scale.get('log_mip_stats'):
                budget_line = (
                    f"  阶段预算(剩余{search_remaining:.0f}s): "
                    f"领域={bootstrap_budget:.0f}s, 混合={hybrid_budget:.0f}s, "
                    f"精修={refine_budget:.0f}s "
                    f"| warm={scale.get('alns_warm_sec', 0)}s, "
                    f"nei≤{scale.get('neighborhood_cap')}")
                if self._mip_enabled():
                    budget_line += (f", MIPprob={mip_repair_prob:.0%}, "
                                    f"stall={scale.get('stall_mip_threshold')}")
                print(budget_line, flush=True)
        # 阶段 2a：领域主搜索
        if bootstrap_budget > 0 and time.time() < bootstrap_end - 0.5:
            bootstrap_mip = scale.get('bootstrap_mip_prob', 0.0)
            best_schedule, best_obj, n_it, n_imp, n_mip, n_acc = self._run_domain_search_loop(
                instance, best_schedule, best_obj, bootstrap_end,
                engine, T0, alpha, mip_repair_prob=bootstrap_mip, scale=scale,
                sub_time=sub_time, stall_mip_threshold=scale.get('stall_mip_threshold', 3),
                stall_early_exit=stall_early, label='领域主搜索')
            iter_count += n_it
            improved_count += n_imp
            mip_repair_count += n_mip
            accepted_count += n_acc
            self._maybe_trace_best(best_obj)

        # 阶段 2b：混合强化
        if hybrid_budget > 0 and time.time() < hybrid_end - 0.5:
            loop_kwargs = {}
            if quick_mode:
                loop_kwargs = {
                    'max_iters': scale.get('max_hybrid_iters', 25),
                    'stall_early_exit': scale.get('stall_early_exit', 8),
                }
            elif stall_early:
                loop_kwargs = {'stall_early_exit': stall_early}
            stall_mip = scale.get('stall_mip_threshold', 3)
            best_schedule, best_obj, n_it, n_imp, n_mip, n_acc = self._run_domain_search_loop(
                instance, best_schedule, best_obj, hybrid_end,
                engine, max(T0 * 0.5, 1.0), alpha,
                mip_repair_prob=mip_repair_prob, scale=scale, sub_time=sub_time,
                stall_mip_threshold=stall_mip,
                label='快速混合' if quick_mode else '混合强化',
                **loop_kwargs)
            iter_count += n_it
            improved_count += n_imp
            mip_repair_count += n_mip
            accepted_count += n_acc
            self._maybe_trace_best(best_obj)

        incumbent_schedule, incumbent_obj = best_schedule, best_obj

        # 阶段 2c：领域精修
        if refine_budget > 0:
            best_schedule, best_obj, refine_iters = self._domain_refinement_phase(
                instance, incumbent_schedule, incumbent_obj, search_deadline, engine, T0, alpha,
                scale, stall_early_exit=stall_early)
            incumbent_schedule, incumbent_obj = best_schedule, best_obj
            iter_count += refine_iters
            self._maybe_trace_best(best_obj)

        # 主搜索结束（init + 热启动 + bootstrap/hybrid/refine），后处理纳入同一墙钟预算
        search_time = time.time() - solve_start
        postprocess_start = time.time()

        # 阶段三：Final Polish（可多轮，每轮重算关键路径）
        if (scale['polish'] and self.ablation_flags.get('use_polish', True)
                and time.time() < deadline - 0.5):
            polish_rounds = scale.get('polish_rounds', 1) if scale.get('multi_polish') else 1
            no_gain_streak = 0
            for pr in range(max(1, polish_rounds)):
                if time.time() >= deadline - 0.5:
                    break
                remaining = max(0.0, deadline - time.time())
                if remaining < 2.0:
                    break
                round_budget = (remaining / max(1, polish_rounds - pr)
                                if scale.get('multi_polish') else remaining)
                polished = self._final_polish(
                    incumbent_schedule, instance, time_budget=round_budget)
                utils = HeuristicScheduler()
                utils._load_team_config(instance)
                utils._load_resource_config(instance)
                polished_obj = utils._compute_objective(polished, instance)
                if polished_obj < incumbent_obj - 0.01:
                    if self.verbose or scale.get('log_mip_stats'):
                        print(f"  Final Polish#{pr + 1}: {incumbent_obj:.1f} -> {polished_obj:.1f} "
                              f"(改进{incumbent_obj - polished_obj:.1f}, 预算≤{round_budget:.0f}s)",
                              flush=True)
                    incumbent_schedule = polished
                    incumbent_obj = polished_obj
                    no_gain_streak = 0
                else:
                    no_gain_streak += 1
                    if scale.get('multi_polish') and no_gain_streak >= 2:
                        break

        # 阶段四：skill-aware 精修（终局 best refinement；含团队重匹配/交换）
        if (self.ablation_flags.get('use_skill_refine', True)
                and time.time() < deadline - 0.5):
            refined = self._skill_aware_local_refine(
                incumbent_schedule, instance, deadline=deadline, max_passes=3)
            if refined is not None:
                r_obj, r_sched = refined
                if r_obj <= incumbent_obj + 1e-9:
                    if (self.verbose or scale.get('log_mip_stats')) and r_obj < incumbent_obj - 0.01:
                        print(f"  Skill Refine: {incumbent_obj:.1f} -> {r_obj:.1f} "
                              f"(改进{incumbent_obj - r_obj:.1f})")
                    incumbent_schedule, incumbent_obj = r_sched, r_obj
        if (self.ablation_flags.get('use_team_swap', True)
                and time.time() < deadline - 0.5):
            swapped = self._team_swap_postprocess(
                incumbent_schedule, instance, deadline=deadline)
            utils = HeuristicScheduler()
            utils._load_team_config(instance)
            utils._load_resource_config(instance)
            swapped_obj = utils._compute_objective(swapped, instance)
            if swapped_obj < incumbent_obj - 0.01:
                if self.verbose:
                    print(f"  Team Swap: {incumbent_obj:.1f} -> {swapped_obj:.1f} "
                          f"(改进{incumbent_obj - swapped_obj:.1f})")
                incumbent_schedule = swapped
                incumbent_obj = swapped_obj

        # 后处理若提前结束，用剩余墙钟继续混合搜索（尾段 MIP 略增）
        if (not quick_mode and time.time() < deadline - 1.5
                and self.ablation_flags.get('use_mip_repair', True)):
            tail_deadline = deadline - 0.5
            stall_mip = scale.get('stall_mip_threshold', 3)
            tail_mip = min(0.38, scale.get('refine_mip_prob', mip_repair_prob * 2.5))
            best_schedule, best_obj, n_it, n_imp, n_mip, n_acc = self._run_domain_search_loop(
                instance, incumbent_schedule, incumbent_obj, tail_deadline,
                engine, max(T0 * 0.3, 1.0), alpha,
                mip_repair_prob=tail_mip, scale=scale, sub_time=sub_time,
                stall_mip_threshold=max(3, stall_mip - 1), label='尾段强化')
            if n_it > 0 and self.verbose:
                print(f"  尾段强化: +{n_it}轮, +{n_imp}次改进, MIP{n_mip}次, obj={best_obj:.1f}")
            incumbent_schedule, incumbent_obj = best_schedule, best_obj
            iter_count += n_it
            improved_count += n_imp
            mip_repair_count += n_mip
            accepted_count += n_acc

            accepted_count += n_acc
            self._maybe_trace_best(best_obj)

        postprocess_time = time.time() - postprocess_start
        solve_time = time.time() - solve_start
        self.mip_sub_time_limit = saved_sub_time

        schedule_feasible = True
        if incumbent_schedule:
            rep_sched, rep_obj, schedule_feasible = report_objective_from_schedule(
                incumbent_schedule, instance, penalty_coeff=self.penalty_coeff)
            if rep_sched is not None:
                incumbent_schedule, incumbent_obj = rep_sched, rep_obj

        if self._record_convergence:
            self._finalize_convergence_trace(incumbent_obj, scale['time_limit'])

        if self.verbose or scale.get('log_mip_stats'):
            feas_tag = '' if schedule_feasible else ' [可行性校验未通过]'
            mip_tail = ''
            if self._mip_enabled():
                st = getattr(self, '_last_mip_stats', None) or {}
                mc = st.get('mip_count', mip_repair_count)
                ms = st.get('mip_sec_sum', 0.0)
                avg = (ms / mc) if mc else 0.0
                mip_tail = (f", MIP修复{mc}次(均耗{avg:.2f}s/合计{ms:.1f}s)")
            print(f"  KG-ALNS完成: {iter_count}轮, {improved_count}次改进, "
                  f"{accepted_count}次接受{mip_tail}, "
                  f"最终obj={incumbent_obj:.1f}, "
                  f"耗时={solve_time:.1f}s(主搜索{search_time:.1f}+后处理{postprocess_time:.1f}, "
                  f"上限{scale['time_limit']:.0f}s){feas_tag}",
                  flush=True)
        self._kg_plus_stats['obj_final'] = float(incumbent_obj)
        if self.verbose or scale.get('log_mip_stats'):
            self._log_kg_plus_stats(force=True)

        return {
            'status': 'KG_ALNS',
            'objective': incumbent_obj,
            'solve_time': solve_time,
            'search_time': search_time,
            'postprocess_time': postprocess_time,
            'time_limit': scale['time_limit'],
            'feasible': schedule_feasible,
            'schedule': incumbent_schedule,
            'num_variables': 0,
            'num_constraints': 0,
            'num_binary': 0,
            'num_continuous': 0,
            'node_count': iter_count,
            'iteration_count': improved_count,
            'accepted_count': accepted_count,
            'mip_repair_count': mip_repair_count,
            'ablation_variant': self.ablation_variant,
            'ablation_flags': dict(self.ablation_flags),
            'destroy_profile': str(globals().get('DESTROY_PROFILE', 'baseline') or 'baseline'),
            'kg_plus_stats': dict(getattr(self, '_kg_plus_stats', {}) or {}),
            'convergence_trace': self._export_convergence_trace(),
            **({'destroy_operator_stats': self._export_destroy_operator_stats()}
               if getattr(self, '_collect_op_stats', False) else {}),
            'best_bound': None,
            'mip_gap': None,
        }





# ============================================================
# 遗传算法 (GA) 对比基线求解器
# ============================================================

class GeneticAlgorithmSolver:
    """对比基线：通用遗传算法（不含修船厂领域算子/后处理）

    两段式整数编码：团队分配 + 空间类型(0=泊位,1=干船坞)
    锦标赛选择、两点交叉、精英保留、可行性修复
    目标函数与 Gurobi / KG-ALNS 完全一致：
    total_flow + 10 × misallocation_count
    """

    INFEASIBLE_PENALTY = 1e9

    def __init__(self, time_limit=600, pop_size=80, max_generations=200,
                 elite_size=2, crossover_rate=0.8, mutation_rate=0.15,
                 tournament_size=3, stall_generations=30, num_runs=5, verbose=False):
        self.time_limit = time_limit
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.stall_generations = stall_generations
        self.num_runs = num_runs
        self.verbose = verbose
        self._utils = HeuristicScheduler()

    def _setup(self, instance):
        self._utils._load_team_config(instance)
        self._utils._load_resource_config(instance)
        self._arrival_times = instance['arrival_times']
        self._tasks_ordered = sorted(instance['tasks_data'], key=lambda t: t['task_id'])
        self._n = len(self._tasks_ordered)
        self._task_index = {t['task_id']: i for i, t in enumerate(self._tasks_ordered)}
        self._require_dock = [
            task_requires_dock_space(t) for t in self._tasks_ordered
        ]
        self._eligible = {}
        self._duration = {}
        for task in self._tasks_ordered:
            tid = task['task_id']
            teams = self._utils._get_eligible_teams_for_task(task)
            self._eligible[tid] = teams
            self._duration[tid] = {k: self._utils._get_task_duration(task, k) for k in teams}

    def _schedule_to_chromosome(self, schedule):
        team_genes = [0] * self._n
        space_genes = [0] * self._n
        for entry in schedule:
            idx = self._task_index[entry['task_id']]
            team_genes[idx] = entry['team']
            space_genes[idx] = 1 if entry['space_type'] == 'dock' else 0
        for i, req in enumerate(self._require_dock):
            if req:
                space_genes[i] = 1
        return team_genes, space_genes

    def _decode(self, team_genes, space_genes, instance):
        decode_order = sorted(
            self._tasks_ordered,
            key=lambda t: (self._arrival_times.get(t['ship_id'], 0.0),
                           t['ship_id'], t['task_id']))
        space_avail = {('berth', b): 0.0 for b in self._utils._berths}
        for d in self._utils._docks:
            space_avail[('dock', d)] = 0.0
        team_avail = {k: 0.0 for k in self._utils._teams}
        ship_prev = {}

        schedule = []
        for task in decode_order:
            idx = self._task_index[task['task_id']]
            tid = task['task_id']
            team_id = team_genes[idx]
            space_bit = 1 if self._require_dock[idx] else space_genes[idx]

            if team_id not in self._eligible[tid]:
                return None, self.INFEASIBLE_PENALTY

            if space_bit == 1 or self._require_dock[idx]:
                candidates = feasible_spaces_for_task(
                    task, [], self._utils._docks)
            else:
                candidates = feasible_spaces_for_task(
                    task, self._utils._berths, self._utils._docks)
            if not candidates:
                return None, self.INFEASIBLE_PENALTY

            arrival = self._arrival_times.get(task['ship_id'], 0.0)
            prev_end = ship_prev.get(task['ship_id'], arrival)
            dur = self._duration[tid][team_id]

            best_start, best_space = None, None
            for st, sid in candidates:
                sk = (st, sid)
                start = max(arrival, prev_end,
                            space_avail.get(sk, 0.0), team_avail.get(team_id, 0.0))
                if best_start is None or start < best_start:
                    best_start, best_space = start, (st, sid)

            end = best_start + dur
            schedule.append({
                'task_id': tid,
                'ship_id': task['ship_id'],
                'start': best_start,
                'completion': end,
                'duration': dur,
                'space': best_space[1],
                'space_type': best_space[0],
                'team': team_id,
                'skill_req': task.get('skill_req', {}),
            })
            space_avail[best_space] = end
            team_avail[team_id] = end
            ship_prev[task['ship_id']] = end

        obj = compute_objective_from_schedule(schedule, instance, self._utils._team_skills)
        return schedule, obj

    def _evaluate(self, team_genes, space_genes, instance):
        return self._decode(team_genes, space_genes, instance)

    def _random_individual(self, rng):
        team_genes, space_genes = [], []
        for i, task in enumerate(self._tasks_ordered):
            tid = task['task_id']
            team_genes.append(rng.choice(self._eligible[tid]))
            if self._require_dock[i]:
                space_genes.append(1)
            else:
                space_genes.append(rng.randint(0, 1))
        return team_genes, space_genes

    def _repair_feasibility(self, team_genes, space_genes, rng):
        for i, task in enumerate(self._tasks_ordered):
            tid = task['task_id']
            if team_genes[i] not in self._eligible[tid]:
                team_genes[i] = rng.choice(self._eligible[tid])
            if self._require_dock[i]:
                space_genes[i] = 1
        return team_genes, space_genes

    def _init_population(self, instance, rng):
        population = []
        for _ in range(70):
            ind = self._random_individual(rng)
            population.append(ind)
        for seed in range(10):
            sched = _greedy_construct_single(instance, random_seed=seed * 137 + 42)
            sched = self._utils._rebuild_schedule_times(sched, instance)
            population.append(self._schedule_to_chromosome(sched))
        return population

    def _tournament_select(self, population, fitness, rng):
        candidates = rng.sample(range(len(population)), min(self.tournament_size, len(population)))
        best_i = min(candidates, key=lambda i: fitness[i])
        return population[best_i]

    def _two_point_crossover(self, p1, p2, rng):
        t1, s1 = list(p1[0]), list(p1[1])
        t2, s2 = list(p2[0]), list(p2[1])
        if self._n < 2:
            return (t1, s1), (t2, s2)
        a, b = sorted(rng.sample(range(self._n), 2))
        for i in range(a, b + 1):
            t1[i], t2[i] = t2[i], t1[i]
            if not self._require_dock[i]:
                s1[i], s2[i] = s2[i], s1[i]
        return (t1, s1), (t2, s2)

    def _mutate(self, individual, rng):
        team_genes, space_genes = list(individual[0]), list(individual[1])
        for i in range(self._n):
            if rng.random() < self.mutation_rate:
                tid = self._tasks_ordered[i]['task_id']
                choices = [k for k in self._eligible[tid] if k != team_genes[i]]
                if choices:
                    team_genes[i] = rng.choice(choices)
            if not self._require_dock[i] and rng.random() < self.mutation_rate:
                space_genes[i] = 1 - space_genes[i]
        return team_genes, space_genes

    def _solve_single(self, instance, seed):
        rng = random.Random(seed)
        self._setup(instance)
        solve_start = time.time()
        deadline = solve_start + self.time_limit

        population = self._init_population(instance, rng)
        fitness = []
        schedules = []
        for ind in population:
            sched, obj = self._evaluate(ind[0], ind[1], instance)
            fitness.append(obj)
            schedules.append(sched)

        best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
        best_obj = fitness[best_idx]
        best_ind = population[best_idx]
        best_sched = schedules[best_idx]
        stall = 0
        convergence = []
        gen = 0
        _sci_tracer = None
        self.last_wall_trace = []
        self.last_best_solution_time = None
        if globals().get('SCI_RECORD_WALL_TRACE', False):
            _cps = globals().get('SCI_TRACE_CHECKPOINTS', SCI_TRACE_CHECKPOINTS)
            _sci_tracer = SciWallClockTracer(
                _cps, solve_start, time_limit=self.time_limit)
            _sci_tracer.update(best_obj, current_obj=best_obj, iteration=0)

        for gen in range(1, self.max_generations + 1):
            if time.time() >= deadline:
                break

            ranked = sorted(range(len(population)), key=lambda i: fitness[i])
            elites = [population[i] for i in ranked[:self.elite_size]]

            new_pop = elites[:]
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(population, fitness, rng)
                p2 = self._tournament_select(population, fitness, rng)
                if rng.random() < self.crossover_rate:
                    c1, c2 = self._two_point_crossover(p1, p2, rng)
                else:
                    c1, c2 = p1, p2
                c1 = self._repair_feasibility(*self._mutate(c1, rng), rng)
                c2 = self._repair_feasibility(*self._mutate(c2, rng), rng)
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            population = new_pop[:self.pop_size]
            fitness, schedules = [], []
            for ind in population:
                sched, obj = self._evaluate(ind[0], ind[1], instance)
                fitness.append(obj)
                schedules.append(sched)

            gen_best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
            if fitness[gen_best_idx] < best_obj - 1e-6:
                best_obj = fitness[gen_best_idx]
                best_ind = population[gen_best_idx]
                best_sched = schedules[gen_best_idx]
                stall = 0
            else:
                stall += 1

            if gen % 20 == 0:
                convergence.append({'generation': gen, 'objective': best_obj})
            if _sci_tracer is not None:
                _sci_tracer.update(
                    best_obj, current_obj=fitness[gen_best_idx], iteration=gen)

            if stall >= self.stall_generations:
                break

        solve_time = time.time() - solve_start
        if best_sched is not None:
            best_obj = compute_objective_from_schedule(
                best_sched, instance, self._utils._team_skills)
        if _sci_tracer is not None:
            self.last_wall_trace = _sci_tracer.finalize(
                best_obj, current_obj=best_obj, iteration=gen)
            self.last_best_solution_time = _sci_tracer.best_time
        return best_sched, best_obj, solve_time, gen, convergence

    def solve(self, instance):
        self._setup(instance)
        objs, times, all_conv = [], [], []
        bests = float('inf')
        best_sched = None
        for run in range(self.num_runs):
            seed = instance.get('instance_id', 0) * 1000 + run * 31 + 42
            sched, obj, t, gens, conv = self._solve_single(instance, seed)
            if sched is not None:
                sched, obj, _ = report_objective_from_schedule(
                    sched, instance, self._utils._team_skills)
            objs.append(obj)
            times.append(t)
            all_conv.append(conv)
            if sched is not None and obj < bests:
                bests = obj
                best_sched = sched

        return {
            'status': 'GA',
            'objective': float(np.mean(objs)),
            'objective_std': float(np.std(objs)) if len(objs) > 1 else 0.0,
            'objective_min': float(min(objs)) if objs else None,
            'objective_max': float(max(objs)) if objs else None,
            'objective_runs': objs,
            'solve_time': float(np.mean(times)),
            'solve_time_std': float(np.std(times)) if len(times) > 1 else 0.0,
            'feasible': best_sched is not None,
            'schedule': best_sched,
            'num_variables': 0,
            'num_constraints': 0,
            'num_binary': 0,
            'num_continuous': 0,
            'node_count': 0,
            'iteration_count': gens if self.num_runs == 1 else 0,
            'best_bound': None,
            'mip_gap': None,
            'convergence_history': all_conv[0] if all_conv else [],
            'num_runs': self.num_runs,
        }


def make_large_scale_ga_solver(time_limit, num_runs, equal_compute_budget=True):
    """大规模对比用 GA：同等预算模式下跑满 time_limit，仅由时限终止。"""
    kwargs = dict(time_limit=time_limit, num_runs=num_runs, verbose=False)
    if equal_compute_budget:
        kwargs['stall_generations'] = 10 ** 9
        kwargs['max_generations'] = 10 ** 6
    return GeneticAlgorithmSolver(**kwargs)


def run_kg_alns_multi_standalone(instance, time_limit, mip_sub_time_limit, num_runs,
                                 show_progress=False):
    """KG-ALNS 多次独立运行（模块级，可供 ProcessPool 子进程调用）。"""
    solver = MatheuristicSolver(
        time_limit=time_limit, mip_sub_time_limit=mip_sub_time_limit,
        max_iters=150, verbose=False)
    objs, times, search_times, post_times = [], [], [], []
    best_sched = None
    best_obj_val = float('inf')
    iid = instance.get('instance_id', '?')
    for run in range(num_runs):
        if show_progress:
            print(f"    Case {iid}: KG-ALNS 第 {run + 1}/{num_runs} 次...", flush=True)
        run_start = time.time()
        random.seed(instance.get('instance_id', 0) * 100 + run + 42)
        run_instance = dict(instance)
        run_instance['_run_id'] = run
        r = solver.solve(run_instance)
        sched = r['schedule']
        if sched is not None:
            sched, obj, _ = report_objective_from_schedule(sched, instance)
        else:
            obj = r['objective']
        objs.append(obj)
        times.append(r['solve_time'])
        search_times.append(r.get('search_time', r['solve_time']))
        post_times.append(r.get('postprocess_time', 0.0))
        if obj is not None and obj < best_obj_val:
            best_obj_val = obj
            if LARGE_SAVE_SCHEDULE_FILES:
                best_sched = sched
        del r, sched
        if show_progress:
            wall = time.time() - run_start
            print(f"      -> obj={obj:.1f}, 总耗时={wall:.1f}s "
                  f"(主搜索{search_times[-1]:.1f}+后处理{post_times[-1]:.1f})",
                  flush=True)
    if show_progress and objs:
        obj_std = float(np.std(objs)) if len(objs) > 1 else 0.0
        print(f"    → KG-ALNS {len(objs)}次: best={min(objs):.1f}, "
              f"mean={float(np.mean(objs)):.1f}±{obj_std:.1f}",
              flush=True)
    if LARGE_LOW_MEMORY_MODE:
        solver._insertion_engine = None
        gc.collect()
    return {
        'objective': float(min(objs)),
        'objective_mean': float(np.mean(objs)),
        'objective_std': float(np.std(objs)) if num_runs > 1 else 0.0,
        'objective_min': float(min(objs)),
        'objective_max': float(max(objs)),
        'objective_runs': objs,
        'solve_time': float(np.mean(times)),
        'solve_time_std': float(np.std(times)) if num_runs > 1 else 0.0,
        'search_time': float(np.mean(search_times)),
        'search_time_std': float(np.std(search_times)) if num_runs > 1 else 0.0,
        'postprocess_time': float(np.mean(post_times)),
        'postprocess_time_std': float(np.std(post_times)) if num_runs > 1 else 0.0,
        'schedule': best_sched,
        'feasible': True,
        'status': 'KG_ALNS',
    }


def _large_scale_kg_worker(args):
    """ProcessPool 子进程：KG-ALNS 多次独立运行。"""
    instance, time_limit, mip_sub_time_limit, num_runs = args
    return run_kg_alns_multi_standalone(
        instance, time_limit, mip_sub_time_limit, num_runs, show_progress=False)


def _large_scale_ga_worker(args):
    """ProcessPool 子进程：单例 GA 求解（与主进程算法相同，仅并行执行）。"""
    instance, time_limit, num_runs, equal_compute_budget = args
    return make_large_scale_ga_solver(
        time_limit, num_runs, equal_compute_budget).solve(instance)


def _large_scale_alns_worker(args):
    """ProcessPool 子进程：单例 ALNS 求解。"""
    instance, time_limit, num_runs = args
    return ALNSSolver(
        time_limit=time_limit, num_runs=num_runs, verbose=False).solve(instance)


def _release_solver_payload(result, low_memory=None):
    """低开销模式：剔除 schedule / 收敛曲线等大对象，仅保留 CSV 需要的标量。"""
    if low_memory is None:
        low_memory = LARGE_LOW_MEMORY_MODE
    if not low_memory or not isinstance(result, dict):
        return result
    result.pop('schedule', None)
    result.pop('convergence_history', None)
    return result


def build_shared_initial_schedule(instance, num_starts=6):
    """KG-ALNS 与通用 ALNS 共用的初解（多起点贪心），保证破坏算子对比归因干净。"""
    sched, obj = generate_multistart_greedy_schedule(
        instance, num_starts=max(3, int(num_starts)))
    return [dict(e) for e in sched], float(obj)


def _run_baselines_serial(instance, ga_time_limit, alns_time_limit, ga_num_runs, alns_num_runs,
                          compare_ga, compare_alns, equal_compute_budget=True):
    """串行运行 GA / ALNS（内存占用低，Windows 长跑更稳定）。"""
    skip = {'objective': None, 'objective_std': None, 'objective_runs': [],
            'solve_time': None, 'feasible': False, 'schedule': None, 'status': 'SKIP'}
    ga_result, alns_result = dict(skip), dict(skip)
    if compare_ga:
        try:
            # 基线 GA 始终跑满 ga_time_limit（不等预算仅指 MIP 可更长）
            ga_result = make_large_scale_ga_solver(
                ga_time_limit, ga_num_runs, equal_compute_budget=True).solve(instance)
        except Exception as e:
            print(f"  GA 异常: {e}", flush=True)
            ga_result = {**skip, 'status': 'ERROR'}
        if LARGE_LOW_MEMORY_MODE:
            _release_solver_payload(ga_result)
        gc.collect()
    if compare_alns:
        try:
            alns_result = ALNSSolver(
                time_limit=alns_time_limit, num_runs=alns_num_runs, verbose=False).solve(instance)
        except Exception as e:
            print(f"  ALNS 异常: {type(e).__name__}: {e}", flush=True)
            alns_result = {**skip, 'status': 'ERROR'}
        if LARGE_LOW_MEMORY_MODE:
            _release_solver_payload(alns_result)
        gc.collect()
    return ga_result, alns_result


def _run_large_scale_baselines(instance, ga_time_limit, alns_time_limit, ga_num_runs, alns_num_runs,
                               compare_ga, compare_alns, parallel_baselines,
                               equal_compute_budget=True):
    """运行 GA / ALNS 基线；两者均启用时可并行以缩短墙钟时间。

    返回 (ga_result, alns_result, parallel_ok)。
    parallel_ok=False 表示并行失败并已回退串行，调用方应关闭后续并行。

    Windows 下 ProcessPool 会 spawn 子进程并重新 import 整份脚本(+pandas/gurobi)，
    与 KG-ALNS 连续 Gurobi 求解叠加后易触发「页面文件太小」；失败时自动回退串行。
    """
    if not (compare_ga and compare_alns and parallel_baselines):
        ga, alns = _run_baselines_serial(
            instance, ga_time_limit, alns_time_limit, ga_num_runs, alns_num_runs,
            compare_ga, compare_alns, equal_compute_budget)
        return ga, alns, True

    gc.collect()
    ga_args = (instance, ga_time_limit, ga_num_runs, True)
    alns_args = (instance, alns_time_limit, alns_num_runs)
    try:
        with ProcessPoolExecutor(max_workers=2) as pool:
            ga_future = pool.submit(_large_scale_ga_worker, ga_args)
            alns_future = pool.submit(_large_scale_alns_worker, alns_args)
            ga_result = ga_future.result()
            alns_result = alns_future.result()
        if LARGE_LOW_MEMORY_MODE:
            _release_solver_payload(ga_result)
            _release_solver_payload(alns_result)
        return ga_result, alns_result, True
    except (BrokenExecutor, OSError, MemoryError) as e:
        err_hint = '页面文件/RAM 不足' if '页面文件' in str(e) else str(e)
        print(f"  [并行基线失败] {type(e).__name__}: {err_hint}", flush=True)
        print(f"  [回退] GA 与 ALNS 改串行求解（算法不变）...", flush=True)
        gc.collect()
        ga, alns = _run_baselines_serial(
            instance, ga_time_limit, alns_time_limit, ga_num_runs, alns_num_runs,
            compare_ga, compare_alns, equal_compute_budget)
        return ga, alns, False
    except Exception as e:
        print(f"  [并行基线失败] {type(e).__name__}: {e}", flush=True)
        print(f"  [回退] GA 与 ALNS 改串行求解...", flush=True)
        gc.collect()
        ga, alns = _run_baselines_serial(
            instance, ga_time_limit, alns_time_limit, ga_num_runs, alns_num_runs,
            compare_ga, compare_alns, equal_compute_budget)
        return ga, alns, False


def _run_large_scale_triple_parallel(instance, mip_time_limit, ga_time_limit, alns_time_limit,
                                     mip_sub_time_limit, num_runs, ga_num_runs, alns_num_runs,
                                     equal_compute_budget=True):
    """KG-ALNS ∥ GA ∥ ALNS 三进程并行（先共享初解，再同时开跑）。

    返回 (heu_result, ga_result, alns_result, parallel_ok)。
    失败时回退：主进程串行 KG，再走基线并行/串行逻辑由调用方处理。
    """
    skip = {'objective': None, 'objective_std': None, 'objective_runs': [],
            'solve_time': None, 'feasible': False, 'schedule': None, 'status': 'ERROR'}
    gc.collect()
    kg_args = (instance, mip_time_limit, mip_sub_time_limit, num_runs)
    ga_args = (instance, ga_time_limit, ga_num_runs, True)
    alns_args = (instance, alns_time_limit, alns_num_runs)
    try:
        with ProcessPoolExecutor(max_workers=3) as pool:
            kg_f = pool.submit(_large_scale_kg_worker, kg_args)
            ga_f = pool.submit(_large_scale_ga_worker, ga_args)
            alns_f = pool.submit(_large_scale_alns_worker, alns_args)
            heu_result = kg_f.result()
            ga_result = ga_f.result()
            alns_result = alns_f.result()
        if LARGE_LOW_MEMORY_MODE:
            _release_solver_payload(heu_result)
            _release_solver_payload(ga_result)
            _release_solver_payload(alns_result)
        return heu_result, ga_result, alns_result, True
    except (BrokenExecutor, OSError, MemoryError) as e:
        err_hint = '页面文件/RAM 不足' if '页面文件' in str(e) else str(e)
        print(f"  [三方并行失败] {type(e).__name__}: {err_hint}", flush=True)
        print("  [回退] KG 串行 → 再跑 GA/ALNS（算法不变）...", flush=True)
        gc.collect()
        try:
            heu_result = run_kg_alns_multi_standalone(
                instance, mip_time_limit, mip_sub_time_limit, num_runs,
                show_progress=True)
        except Exception as e2:
            print(f"  KG-ALNS 异常: {e2}", flush=True)
            heu_result = dict(skip)
        ga_result, alns_result, _ = _run_large_scale_baselines(
            instance, ga_time_limit, alns_time_limit, ga_num_runs, alns_num_runs,
            True, True, parallel_baselines=True, equal_compute_budget=equal_compute_budget)
        return heu_result, ga_result, alns_result, False
    except Exception as e:
        print(f"  [三方并行失败] {type(e).__name__}: {e}", flush=True)
        print("  [回退] KG 串行 → 再跑 GA/ALNS...", flush=True)
        gc.collect()
        try:
            heu_result = run_kg_alns_multi_standalone(
                instance, mip_time_limit, mip_sub_time_limit, num_runs,
                show_progress=True)
        except Exception as e2:
            print(f"  KG-ALNS 异常: {e2}", flush=True)
            heu_result = dict(skip)
        ga_result, alns_result, _ = _run_large_scale_baselines(
            instance, ga_time_limit, alns_time_limit, ga_num_runs, alns_num_runs,
            True, True, parallel_baselines=False, equal_compute_budget=equal_compute_budget)
        return heu_result, ga_result, alns_result, False


def estimate_large_batch_minutes(pending, time_limit, mip_num_runs, ga_num_runs, alns_num_runs,
                                 compare_ga=True, compare_alns=True,
                                 parallel_baselines=False,
                                 parallel_triple=False,
                                 typical_wall_factor=0.82):
    """估算阶段二批量墙钟时间（分钟）。

    worst: 每算法跑满 time_limit × runs（上界）
    typical: 考虑 KG-ALNS 早停与基线未跑满时限的典型占比
    """
    mip_sec = time_limit * mip_num_runs
    ga_sec = time_limit * ga_num_runs if compare_ga else 0
    alns_sec = time_limit * alns_num_runs if compare_alns else 0
    if parallel_triple and compare_ga and compare_alns:
        # KG∥GA∥ALNS：每例墙钟 ≈ max(三侧)
        per_case_worst = max(mip_sec, ga_sec, alns_sec)
        baseline_sec = max(ga_sec, alns_sec)
    elif compare_ga and compare_alns and parallel_baselines:
        baseline_sec = max(ga_sec, alns_sec)
        per_case_worst = mip_sec + baseline_sec
    else:
        baseline_sec = ga_sec + alns_sec
        per_case_worst = mip_sec + baseline_sec
    per_case_typical = per_case_worst * typical_wall_factor
    return {
        'pending': pending,
        'mip_budget_sec': mip_sec,
        'baseline_budget_sec': baseline_sec,
        'per_case_worst_sec': per_case_worst,
        'per_case_typical_sec': per_case_typical,
        'worst_min': pending * per_case_worst / 60.0,
        'typical_min': pending * per_case_typical / 60.0,
        'parallel_baselines': bool(parallel_baselines and compare_ga and compare_alns),
        'parallel_triple': bool(parallel_triple and compare_ga and compare_alns),
    }


class BatchExperimentGurobi:
    """批量实验管理器（Gurobi + 启发式对比）

    参数:
      output_dir     : 输出目录
      heuristic_type : 启发式求解器类型
                       - 'grasp' (默认): GRASP 随机化贪心 + 局部搜索（原版）
                       - 'constructive': 构造型启发式（设计文档 §2-4，先资源分配后时间排程）
                       - 'both': 同时运行两种启发式进行对比
    """

    def __init__(self, output_dir=None, heuristic_type='lNS'):
        desktop_path = os.path.normpath(os.path.expanduser("~/Desktop"))
        if output_dir is None:
            output_dir = os.path.join(desktop_path, "算法设计52")
        self.output_dir = os.path.normpath(output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.results = []
        self.instances = []
        self.experiment_configs = []
        self.detailed_schedules = []
        self.heuristic_type = heuristic_type  # 'grasp' | 'constructive' | 'both'

    def set_experiment_configs(self, configs):
        self.experiment_configs = configs

    def generate_instances(self, force_regen=False, quiet=False):
        if force_regen:
            pattern = os.path.join(self.output_dir, "instance_*")
            old_files = glob.glob(pattern)
            for f in old_files:
                os.remove(f)
            if old_files:
                print(f"\n[force_regen] 已删除 {len(old_files)} 个旧算例文件: {self.output_dir}")
            self.instances = []
        elif not self.instances and self.load_instances_from_disk(quiet=quiet):
            return self.instances

        instance_counter = len(self.instances) + 1
        start_idx = len(self.instances)
        if start_idx >= len(self.experiment_configs):
            return self.instances

        for config_idx, config in enumerate(self.experiment_configs[start_idx:], start=start_idx):
            group_name = config.get('group_name', f'Group_{config_idx + 1}')
            num_instances = config.get('num_instances', 1)

            if not quiet:
                print(f"\n生成配置组: {group_name}")
                print(f"  船舶数: {config.get('num_ships')}, 任务数范围: {config.get('tasks_per_ship_range')}")
                print(f"  生成算例数: {num_instances}")

            for i in range(num_instances):
                # 叠加 REPRODUCIBILITY_SEED，避免与旧版设计6 同号算例撞车
                seed = (instance_counter * 100 + i) ^ int(REPRODUCIBILITY_SEED)
                generator = RandomInstanceGenerator(seed=seed)
                instance = generator.generate_instance(instance_counter, config)
                instance['group_name'] = group_name
                instance['config_idx'] = config_idx
                self.instances.append(instance)
                if not quiet:
                    print(f"    算例 {instance_counter}: {instance['num_ships']}船, "
                          f"{instance['num_tasks']}任务, {instance['num_teams']}团队")
                instance_counter += 1

        return self.instances

    def _create_heuristic_solver(self):
        """根据 heuristic_type 创建启发式求解器

        MatheuristicSolver (KG-ALNS) 主算法：领域定制破坏主循环
        ALNSSolver / GeneticAlgorithmSolver：通用对比基线
        """
        if self.heuristic_type == 'constructive' or self.heuristic_type == 'lNS':
            return MatheuristicSolver(
                time_limit=120,
                mip_sub_time_limit=20,
                max_iters=25,
                verbose=False,
            ), 'KG-ALNS'
        elif self.heuristic_type == 'grasp':
            return HeuristicScheduler(), 'GRASP(greedy)'
        elif self.heuristic_type == 'both':
            return None, '双启发式对比'
        else:
            return MatheuristicSolver(
                time_limit=120, mip_sub_time_limit=20, max_iters=25, verbose=False,
            ), 'KG-ALNS(默认)'

    def _checkpoint_batch_results(self):
        """每完成一例即写入 batch_results.csv，中断后可续跑。"""
        if not self.results:
            return
        _safe_dataframe_to_csv(
            pd.DataFrame(self.results), f"{self.output_dir}/batch_results.csv")

    def run_batch(self, default_time_limit=300, skip_instance_ids=None,
                  existing_results=None):
        """批量求解所有算例 — Gurobi + 启发式对比（支持断点续跑）。"""
        skip_instance_ids = set(skip_instance_ids or [])
        if existing_results is not None:
            self.results = list(existing_results)
        heuristic_solver, heu_name = self._create_heuristic_solver()

        # 双启发式模式时，额外创建另一个求解器
        grasp_solver = HeuristicScheduler() if self.heuristic_type == 'both' else None

        print(f"\n{'=' * 145}")
        heu_label = 'KG-ALNS' if self.heuristic_type in ('constructive', 'lNS') else ('Heu(GRASP)' if self.heuristic_type == 'grasp' else 'Heuristic')
        print(f"启发式算法: {heu_name}")
        print(f"{'=' * 145}")
        print(f"{'Case':>6s} | {'N':>3s} | {'Tasks':>6s} | {'状态':>10s} | {'Gurobi':>10s} | "
              f"{heu_label:>10s} | {'Gap%':>8s} | {'胜负':>6s} | {'效率%':>8s} | {'加速%':>8s} | "
              f"{'Gurobi耗时':>10s} | {'Heu耗时':>10s}")
        print(f"{'=' * 145}")

        skipped = len(skip_instance_ids)
        pending = sum(1 for inst in self.instances if inst['instance_id'] not in skip_instance_ids)
        print(f"待跑 {pending} 例{f'，跳过已完成 {skipped} 例' if skipped else ''}")
        print(f"{'=' * 145}")

        for instance in self.instances:
            iid = instance['instance_id']
            if iid in skip_instance_ids:
                print(f"  >> 跳过 Case {iid}（已完成）", flush=True)
                continue

            config = instance.get('config', {})
            time_limit = config.get('time_limit', default_time_limit)
            mip_gap = config.get('mip_gap', 0.05)

            heuristic_solver.time_limit = time_limit

            try:
                # —— 启发式求解（先运行，为 Gurobi 提供热启动）——
                heu_result = heuristic_solver.solve(instance)

                # —— Gurobi 精确求解（热启动：注入启发式解加速收敛）——
                warm_start = heu_result['schedule'] if heu_result['feasible'] else None
                scheduler = ShipyardSchedulerGurobi(time_limit=time_limit, mip_gap=mip_gap)
                result = scheduler.solve(instance, log_dir=self.output_dir, warm_start_schedule=warm_start)
            except KeyboardInterrupt:
                self._checkpoint_batch_results()
                print(f"\n[中断] 已保存 {len(self.results)} 例 → "
                      f"{self.output_dir}/batch_results.csv", flush=True)
                raise
            except Exception as e:
                self._checkpoint_batch_results()
                print(f"\n[错误] Case {iid} 求解异常，已保存已完成 {len(self.results)} 例到 "
                      f"{self.output_dir}/batch_results.csv", flush=True)
                print(f"  修复代码后重新运行即可从 Case {iid} 继续（AUTO_RESUME=True）", flush=True)
                raise

            heu_obj = heu_result['objective']

            gap_to_optimal = None
            if result['objective'] is not None and result['objective'] > 0 and heu_obj is not None:
                gap_to_optimal = (heu_obj - result['objective']) / result['objective'] * 100
                if abs(gap_to_optimal) < 1e-5:
                    gap_to_optimal = 0.0
                elif result.get('status') == 'OPTIMAL':
                    gap_to_optimal = max(0.0, gap_to_optimal)

            heu_superiority = '-'
            if result['objective'] is not None and heu_obj is not None:
                if result.get('status') == 'OPTIMAL':
                    heu_superiority = ('TIE' if abs(heu_obj - result['objective']) < 0.01
                                       else '-')
                elif heu_obj < result['objective'] - 0.01:
                    heu_superiority = 'WIN'
                elif abs(heu_obj - result['objective']) < 0.01:
                    heu_superiority = 'TIE'

            # 新增三项指标
            obj_diff = None
            efficiency = None
            speedup_percent = None
            if result['objective'] is not None and heu_obj is not None:
                obj_diff = heu_obj - result['objective']
                if heu_obj > 0:
                    efficiency = result['objective'] / heu_obj * 100
            if result['solve_time'] is not None and heu_result['solve_time'] is not None and result['solve_time'] > 0:
                speedup_percent = (result['solve_time'] - heu_result['solve_time']) / result['solve_time'] * 100

            result_record = {
                'instance_id': instance['instance_id'],
                'group_name': instance.get('group_name', ''),
                'config_idx': instance.get('config_idx', -1),
                'num_ships': instance['num_ships'],
                'num_tasks': instance['num_tasks'],
                'num_berths': instance['num_berths'],
                'num_docks': instance['num_docks'],
                'num_teams': instance['num_teams'],
                'objective': result['objective'],
                'solve_time': result['solve_time'],
                'num_variables': result['num_variables'],
                'num_constraints': result['num_constraints'],
                'num_binary': result['num_binary'],
                'num_continuous': result['num_continuous'],
                'node_count': result['node_count'],
                'iteration_count': result['iteration_count'],
                'best_bound': result['best_bound'],
                'mip_gap': result['mip_gap'],
                'status': result['status'],
                'feasible': result['feasible'],
                'heuristic_obj': heu_obj,
                'heuristic_time': heu_result['solve_time'],
                'gap_to_optimal': gap_to_optimal,
                'heu_superiority': heu_superiority,
                'obj_diff': obj_diff,
                'efficiency': efficiency,
                'speedup_percent': speedup_percent,
            }

            self.results.append(result_record)

            if result['feasible'] and result['schedule']:
                self._save_schedule(instance, result['schedule'])
                self._save_case_result(instance, result['schedule'])
                self._save_team_config(instance)
                self.detailed_schedules.append({
                    'instance_id': instance['instance_id'],
                    'group_name': instance.get('group_name', ''),
                    'schedule': result['schedule'],
                    'arrival_times': instance['arrival_times']
                })

            # 保存启发式调度结果（用于对比甘特图）
            if heu_result['feasible'] and heu_result['schedule']:
                self._save_heuristic_schedule(instance, heu_result['schedule'])

            # —— 控制台输出 ————————————————————————————————
            obj_str = f"{result['objective']:.1f}" if result['objective'] else "N/A"
            heu_str = f"{heu_obj:.1f}" if heu_obj is not None else "N/A"
            gap_str = f"{gap_to_optimal:.2f}%" if gap_to_optimal is not None else "N/A"
            eff_str = f"{efficiency:.1f}%" if efficiency is not None else "N/A"
            spd_str = f"{speedup_percent:.1f}%" if speedup_percent is not None else "N/A"
            heu_time_display = f"!{heu_result['solve_time']:.4f}s!" if heu_result['solve_time'] < 0.1 else f" {heu_result['solve_time']:.4f}s "
            win_display = f">> {heu_superiority} <<" if heu_superiority == 'WIN' else heu_superiority
            print(f"  {instance['instance_id']:>4d} | {instance['num_ships']:>3d} | {instance['num_tasks']:>6d} | "
                  f"{result['status']:>10s} | {obj_str:>10s} | {heu_str:>10s} | "
                  f"{gap_str:>8s} | {win_display:>10s} | {eff_str:>8s} | {spd_str:>8s} | "
                  f"{result['solve_time']:>8.2f}s | {heu_time_display:>14s}")

            self._checkpoint_batch_results()

        print(f"{'=' * 120}")
        return self.results


    def _run_mip_lns_multi(self, solver, instance, num_runs=5, show_progress=True):
        """KG-ALNS 多次独立运行，返回 best/mean/标准差及分列墙钟"""
        utils = HeuristicScheduler()
        utils._load_team_config(instance)
        objs, times, search_times, post_times = [], [], [], []
        best_sched = None
        best_obj_val = float('inf')
        iid = instance.get('instance_id', '?')
        for run in range(num_runs):
            if show_progress:
                print(f"    Case {iid}: KG-ALNS 第 {run + 1}/{num_runs} 次...", flush=True)
            run_start = time.time()
            random.seed(instance.get('instance_id', 0) * 100 + run + 42)
            # 与基线 ALNS 一样按 run 分化种子；否则多次独立运行几乎同轨迹，mean/best 双输
            run_instance = dict(instance)
            run_instance['_run_id'] = run
            r = solver.solve(run_instance)
            sched = r['schedule']
            if sched is not None:
                sched, obj, _ = report_objective_from_schedule(sched, instance)
            else:
                obj = r['objective']
            objs.append(obj)
            times.append(r['solve_time'])
            search_times.append(r.get('search_time', r['solve_time']))
            post_times.append(r.get('postprocess_time', 0.0))
            if obj is not None and obj < best_obj_val:
                best_obj_val = obj
                if LARGE_SAVE_SCHEDULE_FILES:
                    best_sched = sched
            del r, sched
            if show_progress:
                wall = time.time() - run_start
                st = getattr(solver, '_last_mip_stats', None) or {}
                itc = st.get('iters', 0)
                detail = f"迭代={itc}轮"
                if solver._mip_enabled():
                    mc = st.get('mip_count', 0)
                    ms = st.get('mip_sec_sum', 0.0)
                    n_int = st.get('intensify_success', 0)
                    avg = (ms / mc) if mc else 0.0
                    detail = (f"ALNS={itc}轮, 强化成功={n_int}, "
                              f"MIP={mc}次/均耗{avg:.2f}s")
                print(f"      -> obj={obj:.1f}, 总耗时={wall:.1f}s "
                      f"(主搜索{search_times[-1]:.1f}+后处理{post_times[-1]:.1f})"
                      f" | {detail}",
                      flush=True)
        if show_progress and objs:
            obj_std = float(np.std(objs)) if len(objs) > 1 else 0.0
            print(f"    → KG-ALNS {len(objs)}次: best={min(objs):.1f}, "
                  f"mean={float(np.mean(objs)):.1f}±{obj_std:.1f}",
                  flush=True)
        if LARGE_LOW_MEMORY_MODE:
            solver._insertion_engine = None
            gc.collect()
        return {
            'objective': float(min(objs)),
            'objective_mean': float(np.mean(objs)),
            'objective_std': float(np.std(objs)) if num_runs > 1 else 0.0,
            'objective_min': float(min(objs)),
            'objective_max': float(max(objs)),
            'objective_runs': objs,
            'solve_time': float(np.mean(times)),
            'solve_time_std': float(np.std(times)) if num_runs > 1 else 0.0,
            'search_time': float(np.mean(search_times)),
            'search_time_std': float(np.std(search_times)) if num_runs > 1 else 0.0,
            'postprocess_time': float(np.mean(post_times)),
            'postprocess_time_std': float(np.std(post_times)) if num_runs > 1 else 0.0,
            'schedule': best_sched,
            'feasible': True,
            'status': 'KG_ALNS',
        }

    def _save_large_scale_csv(self, large_results, equal_compute_budget=True):
        """每完成一例即写入 CSV，防止长跑中断丢结果（支持与磁盘合并，可并行跑不同梯度）。"""
        if not large_results:
            return
        results_path = f"{self.output_dir}/large_scale_results.csv"
        comp_path = f"{self.output_dir}/large_scale_comparison.csv"
        new_df = pd.DataFrame(large_results)
        comp_cols = [
            'instance_id', 'num_ships', 'num_tasks', 'tier',
            'mip_lns_obj_best', 'mip_lns_obj_mean', 'mip_lns_obj_std',
            'mip_lns_time', 'mip_lns_search_time', 'mip_lns_postprocess_time',
            'ga_obj_best', 'ga_obj_mean', 'ga_obj_std', 'ga_time',
            'alns_obj_best', 'alns_obj_mean', 'alns_obj_std', 'alns_time',
            'gap_to_ga_pct', 'gap_to_alns_pct',
            'gap_to_ga_pct_mean', 'gap_to_alns_pct_mean',
            'gap_to_ga_pct_best', 'gap_to_alns_pct_best',
            't_stat_ga', 'p_value_ga', 't_stat_alns', 'p_value_alns',
            'wilcoxon_stat_ga', 'wilcoxon_p_ga', 'wilcoxon_stat_alns', 'wilcoxon_p_alns',
            'mip_lns_better_ga', 'mip_lns_better_alns',
            'mip_lns_better_ga_mean', 'mip_lns_better_alns_mean',
            'mip_lns_better_ga_best', 'mip_lns_better_alns_best',
            'obj_winner', 'obj_winner_best', 'quality_grade', 'quality_vs_alns',
            'mip_time_limit', 'ga_time_limit', 'alns_time_limit', 'equal_compute_budget',
        ]
        if not equal_compute_budget:
            comp_cols.insert(comp_cols.index('quality_grade'), 'speed_winner')
        comp_df = new_df[[c for c in comp_cols if c in new_df.columns]]

        for attempt in range(8):
            try:
                merged = merge_large_scale_results_df(new_df, results_path)
                _safe_dataframe_to_csv(merged, results_path)
                merged_comp = merge_large_scale_results_df(comp_df, comp_path)
                _safe_dataframe_to_csv(merged_comp, comp_path)
                return
            except PermissionError:
                time.sleep(0.5 * (attempt + 1))
        merged = merge_large_scale_results_df(new_df, results_path)
        _safe_dataframe_to_csv(merged, results_path)
        merged_comp = merge_large_scale_results_df(comp_df, comp_path)
        _safe_dataframe_to_csv(merged_comp, comp_path)

    def run_large_scale_batch(self, num_runs=3, ga_num_runs=None, alns_num_runs=5,
                              batch_time_limit=300,
                              mip_time_limit=None, ga_time_limit=None, alns_time_limit=None,
                              mip_sub_time_limit=45, compare_ga=True, compare_alns=True,
                              parallel_baselines=False, equal_compute_budget=True,
                              primary_metric='mean',
                              skip_instance_ids=None, existing_results=None,
                              only_tiers=None):
        """批量求解大规模算例 — KG-ALNS vs GA vs ALNS 三方对比

        batch_time_limit: 默认墙钟（三算法未单独指定时共用）
        mip_time_limit / ga_time_limit / alns_time_limit: 分算法墙钟上限（秒）
        num_runs: KG-ALNS 独立重复次数
        equal_compute_budget: True 时 GA 跑满时限，对比仅比解质量（不比速度）
        """
        ga_num_runs = ga_num_runs if ga_num_runs is not None else num_runs
        skip_instance_ids = skip_instance_ids or set()
        only_tiers = set(only_tiers) if only_tiers else None
        large_results = list(existing_results or [])

        tl = resolve_large_scale_time_limits(
            batch_time_limit=batch_time_limit,
            mip_time_limit=mip_time_limit,
            ga_time_limit=ga_time_limit,
            alns_time_limit=alns_time_limit,
        )
        mip_tl, ga_tl, alns_tl = tl['mip'], tl['ga'], tl['alns']
        if not tl['equal']:
            equal_compute_budget = False

        def _inst_tier(inst):
            return inst.get('config', {}).get('tier', '')

        def _runs_this_batch(inst):
            iid = inst['instance_id']
            if iid in skip_instance_ids:
                return False
            if only_tiers is not None and _inst_tier(inst) not in only_tiers:
                return False
            return True

        mip_solver = MatheuristicSolver(
            time_limit=mip_tl, mip_sub_time_limit=mip_sub_time_limit,
            max_iters=150, verbose=False)
        heu_name = 'KG-ALNS'

        algo_desc = [heu_name]
        if compare_ga:
            algo_desc.append('GA')
        if compare_alns:
            algo_desc.append('ALNS')
        skipped = len(skip_instance_ids)
        pending = sum(1 for inst in self.instances if _runs_this_batch(inst))
        tier_note = ''
        if only_tiers:
            tier_note = f", 梯度 {','.join(sorted(only_tiers))}"

        if tl['equal']:
            time_note = f"严格墙钟上限 {mip_tl}s/次(KG领域主搜·无Polish)"
        else:
            time_note = (f"墙钟 MIP={mip_tl}s GA={ga_tl}s ALNS={alns_tl}s "
                         f"(不等预算·辅实验)")

        print(f"\n{'=' * 195}")
        print(f"大规模算例批量求解（共 {len(self.instances)} 例, 待跑 {pending} 例"
              f"{f', 跳过 {skipped} 例' if skipped else ''}{tier_note}, 30-60 艘船）")
        use_parallel_triple = bool(
            globals().get('LARGE_PARALLEL_TRIPLE', False) and compare_ga and compare_alns)
        if use_parallel_triple:
            baseline_mode = 'KG∥GA∥ALNS 三方并行'
        elif compare_ga and compare_alns and parallel_baselines:
            baseline_mode = 'GA∥ALNS 并行'
        else:
            baseline_mode = 'GA→ALNS 串行'
        print(f"对比算法: {' | '.join(algo_desc)} | "
              f"KG-ALNS {num_runs} 次 | GA {ga_num_runs} 次 | ALNS {alns_num_runs} 次 | "
              f"{time_note} | 主口径={primary_metric} | "
              f"辅口径=best-of-runs | "
              f"调度: {baseline_mode} | "
              f"{'同等预算·仅比解质量' if equal_compute_budget else '不等预算·MIP可更长(辅实验)'} | "
              f"目标: total_flow + 10×misallocation_count"
              f"{' | 低内存模式' if LARGE_LOW_MEMORY_MODE else ''}")
        print(f"  胜负: 三算法各跑 {num_runs}/{ga_num_runs}/{alns_num_runs} 次，"
              f"主口径={primary_metric}（{'取最优单次' if primary_metric == 'best' else '取均值'}）",
              flush=True)
        print(f"{'=' * 195}")
        batch_wall_start = time.time()
        completed_case_times = []
        use_parallel_baselines = parallel_baselines and not use_parallel_triple
        if use_parallel_triple and sys.platform == 'win32':
            print("[提示] Windows 三方并行内存较高；失败将自动回退。"
                  "可设 LARGE_PARALLEL_TRIPLE=False。", flush=True)
        elif use_parallel_baselines and sys.platform == 'win32':
            print("[提示] Windows 下 GA∥ALNS 并行会 spawn 子进程并重载 pandas/Gurobi，"
                  "易触发「页面文件太小」。若报错请设 LARGE_PARALLEL_BASELINES=False。", flush=True)
        if compare_ga and compare_alns:
            mip_hdr = 'KG best' if primary_metric == 'best' else 'KG均值'
            hdr = (f"{'Case':>5s} | {'N':>3s} | {'Tasks':>5s} | {'梯度':>8s} | "
                   f"{mip_hdr:>10s} | {'KG±':>7s} | {'KG(s)':>7s} | "
                   f"{'GA':>10s} | {'GA±':>7s} | {'GA(s)':>7s} | "
                   f"{'ALNS':>10s} | {'ALN±':>7s} | {'ALN(s)':>7s} | "
                   f"{'胜负':>8s}")
            if primary_metric == 'mean':
                hdr += f" | {'胜(best)':>8s}"
            if not equal_compute_budget:
                hdr += f" | {'速度胜':>6s}"
            print(hdr)
            print(f"{'=' * 195}")

        for instance in self.instances:
            config = instance.get('config', {})
            iid = instance['instance_id']
            if only_tiers is not None and config.get('tier', '') not in only_tiers:
                continue
            if iid in skip_instance_ids:
                print(f"\n>> 跳过 Case {iid}（已完成）", flush=True)
                continue

            case_wall_start = time.time()
            print(f"\n>> 开始 Case {iid}: {instance['num_ships']}船, "
                  f"{instance['num_tasks']}任务 [{config.get('tier', '')}]", flush=True)

            # 共享初解：KG 与通用 ALNS 同一起点（GA 仍用自身种群初始化）
            try:
                shared_sched, shared_obj = build_shared_initial_schedule(
                    instance, num_starts=6)
                instance = dict(instance)
                instance['_shared_init_schedule'] = shared_sched
                print(f"    共享初解 obj={shared_obj:.1f}", flush=True)
            except Exception as e:
                print(f"    共享初解失败（将各自构造）: {e}", flush=True)

            if use_parallel_triple:
                print(f"    KG∥GA∥ALNS 三方并行求解 "
                      f"(各 {num_runs}/{ga_num_runs}/{alns_num_runs} 次 × 墙钟)...",
                      flush=True)
                try:
                    heu_result, ga_result, alns_result, parallel_ok = (
                        _run_large_scale_triple_parallel(
                            instance, mip_tl, ga_tl, alns_tl, mip_sub_time_limit,
                            num_runs, ga_num_runs, alns_num_runs, equal_compute_budget))
                except KeyboardInterrupt:
                    self._save_large_scale_csv(large_results, equal_compute_budget)
                    print(f"\n[中断] 已保存 {len(large_results)} 例 → "
                          f"{self.output_dir}/large_scale_results.csv", flush=True)
                    raise
                except Exception as e:
                    self._save_large_scale_csv(large_results, equal_compute_budget)
                    print(f"\n[错误] Case {iid} 三方并行异常，已保存 {len(large_results)} 例到 "
                          f"{self.output_dir}/large_scale_results.csv", flush=True)
                    print(f"  修复后设 resume_large 续跑即可（AUTO_RESUME=True）", flush=True)
                    raise
                if not parallel_ok:
                    use_parallel_triple = False
                    use_parallel_baselines = parallel_baselines
                    print("  [提示] 后续算例关闭三方并行（避免再次 OOM）", flush=True)
                if LARGE_LOW_MEMORY_MODE:
                    _release_solver_payload(heu_result)
                    _release_solver_payload(ga_result)
                    _release_solver_payload(alns_result)
                    gc.collect()
            else:
                try:
                    heu_result = self._run_mip_lns_multi(mip_solver, instance, num_runs)
                except Exception as e:
                    print(f"  KG-ALNS 异常 (Case {iid}): {e}", flush=True)
                    heu_result = {'objective': None, 'objective_std': None, 'objective_runs': [],
                                  'solve_time': None, 'feasible': False, 'schedule': None, 'status': 'ERROR'}

                gc.collect()  # KG-ALNS 释放 Gurobi 内存后再跑 GA/ALNS

                if compare_ga and compare_alns:
                    if use_parallel_baselines:
                        print(f"    GA ({ga_num_runs} 次) 与 ALNS ({alns_num_runs} 次) 并行求解...", flush=True)
                    else:
                        print(f"    GA ({ga_num_runs} 次) 与 ALNS ({alns_num_runs} 次) 串行求解...", flush=True)
                elif compare_ga:
                    print(f"    GA 求解中 ({ga_num_runs} 次)...", flush=True)
                elif compare_alns:
                    print(f"    ALNS 求解中 ({alns_num_runs} 次)...", flush=True)
                try:
                    ga_result, alns_result, parallel_ok = _run_large_scale_baselines(
                        instance, ga_tl, alns_tl, ga_num_runs, alns_num_runs,
                        compare_ga, compare_alns, use_parallel_baselines, equal_compute_budget)
                except KeyboardInterrupt:
                    self._save_large_scale_csv(large_results, equal_compute_budget)
                    print(f"\n[中断] 已保存 {len(large_results)} 例 → "
                          f"{self.output_dir}/large_scale_results.csv", flush=True)
                    raise
                except Exception as e:
                    self._save_large_scale_csv(large_results, equal_compute_budget)
                    print(f"\n[错误] Case {iid} 基线求解异常，已保存 {len(large_results)} 例到 "
                          f"{self.output_dir}/large_scale_results.csv", flush=True)
                    print(f"  修复代码后重新运行即可从 Case {iid} 继续（AUTO_RESUME=True）", flush=True)
                    raise
                if use_parallel_baselines and not parallel_ok:
                    use_parallel_baselines = False
                    print("  [提示] 后续算例将全程使用 GA→ALNS 串行（避免再次 OOM）", flush=True)
                if LARGE_LOW_MEMORY_MODE:
                    _release_solver_payload(ga_result)
                    _release_solver_payload(alns_result)
                    gc.collect()

            t_ga = None
            t_alns = None
            mip_runs = heu_result.get('objective_runs') or []
            ga_runs = ga_result.get('objective_runs') or []
            alns_runs = alns_result.get('objective_runs') or []
            if len(mip_runs) >= 2 and len(ga_runs) >= 2 and len(mip_runs) == len(ga_runs):
                t_ga = self._paired_t_test(mip_runs, ga_runs)
            if len(mip_runs) >= 2 and len(alns_runs) >= 2:
                n_pair = min(len(mip_runs), len(alns_runs))
                t_alns = self._paired_t_test(mip_runs[:n_pair], alns_runs[:n_pair])

            mip_best = heu_result.get('objective_min') or heu_result.get('objective')
            ga_best = ga_result.get('objective_min') or ga_result.get('objective')
            alns_best = alns_result.get('objective_min') or alns_result.get('objective')

            mip_better_ga = (mip_best is not None and ga_best is not None and mip_best < ga_best)
            mip_better_alns = (mip_best is not None and alns_best is not None and mip_best < alns_best)

            gap_to_ga = None
            if ga_best and mip_best and ga_best > 0:
                gap_to_ga = (mip_best - ga_best) / ga_best * 100

            gap_to_alns = None
            if alns_best and mip_best and alns_best > 0:
                gap_to_alns = (mip_best - alns_best) / alns_best * 100

            record = {
                'instance_id': instance['instance_id'],
                'group_name': instance.get('group_name', ''),
                'tier': config.get('tier', ''),
                'num_ships': instance['num_ships'],
                'num_tasks': instance['num_tasks'],
                'num_berths': instance['num_berths'],
                'num_docks': instance['num_docks'],
                'num_teams': instance['num_teams'],
                'mip_lns_obj_best': mip_best,
                'mip_lns_obj_mean': heu_result.get('objective_mean') or heu_result.get('objective'),
                'mip_lns_obj_std': heu_result.get('objective_std'),
                'mip_lns_obj_min': heu_result.get('objective_min'),
                'mip_lns_obj_max': heu_result.get('objective_max'),
                'mip_lns_time': heu_result.get('solve_time'),
                'mip_lns_search_time': heu_result.get('search_time'),
                'mip_lns_postprocess_time': heu_result.get('postprocess_time'),
                'mip_lns_feasible': heu_result.get('feasible', False),
                'heuristic_obj': mip_best,
                'heuristic_time': heu_result.get('solve_time'),
                'heuristic_search_time': heu_result.get('search_time'),
                'heuristic_postprocess_time': heu_result.get('postprocess_time'),
                'heuristic_feasible': heu_result.get('feasible', False),
                'ga_obj_best': ga_best,
                'ga_obj_mean': ga_result.get('objective'),
                'ga_obj_std': ga_result.get('objective_std'),
                'ga_obj_min': ga_result.get('objective_min'),
                'ga_obj_max': ga_result.get('objective_max'),
                'ga_time': ga_result.get('solve_time'),
                'ga_feasible': ga_result.get('feasible', False),
                'alns_obj_best': alns_best,
                'alns_obj_mean': alns_result.get('objective'),
                'alns_obj_std': alns_result.get('objective_std'),
                'alns_obj_min': alns_result.get('objective_min'),
                'alns_obj_max': alns_result.get('objective_max'),
                'alns_time': alns_result.get('solve_time'),
                'alns_feasible': alns_result.get('feasible', False),
                'gap_to_ga_pct': gap_to_ga,
                'gap_to_alns_pct': gap_to_alns,
                't_stat_ga': t_ga['t_stat'] if t_ga else None,
                'p_value_ga': t_ga['p_value'] if t_ga else None,
                't_stat_alns': t_alns['t_stat'] if t_alns else None,
                'p_value_alns': t_alns['p_value'] if t_alns else None,
                't_stat': t_alns['t_stat'] if t_alns else (t_ga['t_stat'] if t_ga else None),
                'p_value': t_alns['p_value'] if t_alns else (t_ga['p_value'] if t_ga else None),
                'mip_lns_better': mip_better_alns,
                'mip_lns_better_ga': mip_better_ga,
                'mip_lns_better_alns': mip_better_alns,
                'mip_time_limit': mip_tl,
                'ga_time_limit': ga_tl,
                'alns_time_limit': alns_tl,
                'equal_compute_budget': equal_compute_budget,
                'comparison_mode': ('equal_budget' if equal_compute_budget
                                    else 'quality_first'),
                'kg_alns_profile_version': globals().get(
                    'KG_ALNS_PROFILE_VERSION', ''),
                'destroy_profile': str(globals().get(
                    'DESTROY_PROFILE', 'baseline') or 'baseline'),
            }
            self._enrich_large_scale_record(record, heu_name,
                                            equal_compute_budget=equal_compute_budget,
                                            primary_metric=primary_metric)
            large_results.append(record)
            self._save_large_scale_csv(large_results, equal_compute_budget)

            if (LARGE_SAVE_SCHEDULE_FILES
                    and heu_result.get('feasible') and heu_result.get('schedule')):
                self._save_heuristic_schedule(instance, heu_result['schedule'])
                self._save_large_case_result(instance, heu_result['schedule'])
            if LARGE_LOW_MEMORY_MODE:
                heu_result.pop('schedule', None)
                gc.collect()

            mip_mean = heu_result.get('objective_mean') or mip_best
            ga_mean = ga_result.get('objective') or ga_best
            alns_mean = alns_result.get('objective') or alns_best
            if primary_metric == 'best':
                mip_disp, ga_disp, alns_disp = mip_best, ga_best, alns_best
            else:
                mip_disp, ga_disp, alns_disp = mip_mean, ga_mean, alns_mean
            mip_s = f"{mip_disp:.1f}" if mip_disp is not None else "N/A"
            mip_std = (f"±{heu_result['objective_std']:.1f}"
                       if heu_result.get('objective_std') is not None and num_runs > 1 else "-")
            mip_t = f"{heu_result['solve_time']:.1f}" if heu_result.get('solve_time') else "N/A"
            ga_s = f"{ga_disp:.1f}" if ga_disp is not None else "N/A"
            ga_std = (f"±{ga_result['objective_std']:.1f}"
                      if ga_result.get('objective_std') is not None and ga_num_runs > 1 else "-")
            ga_t = f"{ga_result['solve_time']:.1f}" if ga_result.get('solve_time') else "N/A"
            alns_s = f"{alns_disp:.1f}" if alns_disp is not None else "N/A"
            alns_std = (f"±{alns_result['objective_std']:.1f}"
                        if alns_result.get('objective_std') is not None and alns_num_runs > 1 else "-")
            alns_t = f"{alns_result['solve_time']:.1f}" if alns_result.get('solve_time') else "N/A"
            row_line = (f"  {instance['instance_id']:>4d} | {instance['num_ships']:>3d} | "
                  f"{instance['num_tasks']:>5d} | {config.get('tier', ''):>8s} | "
                  f"{mip_s:>10s} | {mip_std:>7s} | {mip_t:>7s} | "
                  f"{ga_s:>10s} | {ga_std:>7s} | {ga_t:>7s} | "
                  f"{alns_s:>10s} | {alns_std:>7s} | {alns_t:>7s} | "
                  f"{str(record.get('obj_winner', '-')):>8s}")
            if primary_metric == 'mean':
                row_line += f" | {str(record.get('obj_winner_best', '-')):>8s}"
            if not equal_compute_budget:
                row_line += f" | {str(record.get('speed_winner', '-')):>6s}"
            print(row_line)

            case_wall_sec = time.time() - case_wall_start
            completed_case_times.append(case_wall_sec)
            done_in_batch = len(completed_case_times)
            still_pending = pending - done_in_batch
            if still_pending > 0:
                avg_case_sec = sum(completed_case_times) / done_in_batch
                eta_min = still_pending * avg_case_sec / 60.0
                print(f"    [进度] 本例墙钟 {case_wall_sec:.0f}s | 已完成 {done_in_batch}/{pending} | "
                      f"滚动预计剩余 ≈ {eta_min:.0f} 分钟", flush=True)

        print(f"{'=' * 195}")
        if completed_case_times:
            total_batch_min = (time.time() - batch_wall_start) / 60.0
            print(f"本批实际墙钟: {total_batch_min:.1f} 分钟 "
                  f"(平均每例 {sum(completed_case_times) / len(completed_case_times):.0f}s)")
        self._print_large_scale_summary(large_results)
        self._print_large_scale_comparison(large_results, heu_name,
                                           equal_compute_budget=equal_compute_budget,
                                           primary_metric=primary_metric)

        self._save_large_scale_csv(large_results, equal_compute_budget)
        return large_results

    def _save_large_case_result(self, instance, schedule):
        """保存大规模算例的单例结果（精简版，仅保存船舶级汇总）"""
        ship_completions = {}
        for row in schedule:
            ship_id = row['ship_id']
            ship_completions.setdefault(ship_id, []).append(row)

        ship_rows = []
        for ship_id, tasks in ship_completions.items():
            arrival_time = instance['arrival_times'].get(ship_id, 0)
            completion_time = max(task['completion'] for task in tasks)
            first_task_start = min(task['start'] for task in tasks)
            ship_wait_time = first_task_start - arrival_time
            teams_used = sorted(set(task['team'] for task in tasks if task['team'] is not None))
            resources_used = sorted(set(
                f"Berth{int(task['space'])}" if task['space_type'] == 'berth' else f"Dock{int(task['space'])}"
                for task in tasks
            ))
            ship_rows.append({
                'instance_id': instance['instance_id'],
                'group_name': instance.get('group_name', ''),
                'ship_id': ship_id,
                'num_tasks': len(tasks),
                'arrival_time': arrival_time,
                'departure_time': completion_time,
                'Wait_Time': round(ship_wait_time, 2),
                'total_flow_time': round(completion_time - arrival_time, 2),
                'teams_used': ','.join(f"T{t}" for t in teams_used),
                'resources_used': ','.join(resources_used)
            })

        ship_df = pd.DataFrame(ship_rows)
        filename = f"{self.output_dir}/Large_Case_{instance['instance_id']:03d}_ship_summary.csv"
        ship_df.to_csv(filename, index=False, encoding='utf-8-sig')


    def _print_large_scale_summary(self, large_results):
        """按梯度打印 KG-ALNS vs GA vs ALNS 统计汇总"""
        print(f"\n{'=' * 140}")
        print("大规模算例 — KG-ALNS vs GA vs ALNS 梯度统计 (run-mean 主口径)")
        print(f"{'=' * 140}")
        df = pd.DataFrame(large_results)
        tiers_order = LARGE_SCALE_TIER_LABELS
        print(f"{'梯度':>8s} | {'数量':>4s} | {'KG-ALNS':>10s} | {'GA':>10s} | {'ALNS':>10s} | "
              f"{'KG(s)':>8s} | {'GA(s)':>8s} | {'ALN(s)':>8s} | "
              f"{'胜GA':>7s} | {'胜ALN':>8s} | {'目标胜':>6s}")
        print('-' * 140)

        def _tier_avg(tdf, col):
            if col not in tdf.columns:
                return None
            s = tdf[col].dropna()
            return s.mean() if len(s) > 0 else None

        for tier in tiers_order:
            tdf = df[df['tier'] == tier]
            if len(tdf) == 0:
                continue
            mip = (_tier_avg(tdf, 'mip_lns_obj_mean') or _tier_avg(tdf, 'mip_lns_obj_best')
                   or _tier_avg(tdf, 'heuristic_obj'))
            ga = _tier_avg(tdf, 'ga_obj_mean') or _tier_avg(tdf, 'ga_obj_best')
            alns = _tier_avg(tdf, 'alns_obj_mean') or _tier_avg(tdf, 'alns_obj_best')
            ga_s = f"{ga:.1f}" if ga is not None and not pd.isna(ga) else "N/A"
            alns_s = f"{alns:.1f}" if alns is not None and not pd.isna(alns) else "N/A"
            wins_ga = int(tdf['mip_lns_better_ga'].sum()) if 'mip_lns_better_ga' in tdf.columns else 0
            wins_alns = int(tdf['mip_lns_better_alns'].sum()) if 'mip_lns_better_alns' in tdf.columns else 0
            obj_wins = int((tdf['obj_winner'] == 'KG-ALNS').sum()) if 'obj_winner' in tdf.columns else 0
            print(f"  {tier:>6s} | {len(tdf):>4d} | {mip:>10.1f} | {ga_s:>10s} | {alns_s:>10s} | "
                  f"{tdf['mip_lns_time'].mean():>8.2f} | "
                  f"{tdf['ga_time'].mean():>8.2f} | "
                  f"{tdf['alns_time'].mean():>8.2f} | "
                  f"{wins_ga:>7d} | {wins_alns:>8d} | {obj_wins:>6d}")

        print('-' * 140)
        mip_all = (_tier_avg(df, 'mip_lns_obj_mean') or _tier_avg(df, 'mip_lns_obj_best')
                   or _tier_avg(df, 'heuristic_obj'))
        ga_all = _tier_avg(df, 'ga_obj_mean') or _tier_avg(df, 'ga_obj_best')
        alns_all = _tier_avg(df, 'alns_obj_mean') or _tier_avg(df, 'alns_obj_best')
        ga_str = f"{ga_all:.1f}" if ga_all is not None and not pd.isna(ga_all) else "N/A"
        alns_str = f"{alns_all:.1f}" if alns_all is not None and not pd.isna(alns_all) else "N/A"
        wins_ga_all = int(df['mip_lns_better_ga'].sum()) if 'mip_lns_better_ga' in df.columns else 0
        wins_alns_all = int(df['mip_lns_better_alns'].sum()) if 'mip_lns_better_alns' in df.columns else 0
        obj_wins_all = int((df['obj_winner'] == 'KG-ALNS').sum()) if 'obj_winner' in df.columns else 0
        print(f"  合计 {len(df)} 例 | KG-ALNS={mip_all:.1f} | GA={ga_str} | ALNS={alns_str} | "
              f"胜GA {wins_ga_all}/{len(df)} | 胜ALNS {wins_alns_all}/{len(df)} | "
              f"三方目标胜 {obj_wins_all}/{len(df)}")

    @staticmethod
    def _pick_winner(values, names, lower_better=True, tol=0.01):
        valid = [(v, n) for v, n in zip(values, names)
                 if v is not None and not (isinstance(v, float) and pd.isna(v))]
        if not valid:
            return '-'
        if lower_better:
            best_v = min(v for v, _ in valid)
            winners = [n for v, n in valid if abs(v - best_v) < tol]
        else:
            best_v = max(v for v, _ in valid)
            winners = [n for v, n in valid if abs(v - best_v) < tol]
        return winners[0] if len(winners) == 1 else 'TIE'

    @staticmethod
    def _paired_t_test(sample_a, sample_b):
        a = np.array(sample_a, dtype=float)
        b = np.array(sample_b, dtype=float)
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        if len(a) < 2:
            return None
        d = a - b
        mean_d = d.mean()
        std_d = d.std(ddof=1)
        if std_d < 1e-12:
            return {'t_stat': 0.0, 'p_value': 1.0, 'mean_diff': mean_d, 'n': len(a)}
        t_stat = mean_d / (std_d / np.sqrt(len(d)))
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
        return {'t_stat': t_stat, 'p_value': p_value, 'mean_diff': mean_d, 'n': len(a)}

    @staticmethod
    def _paired_wilcoxon_test(sample_a, sample_b):
        """跨算例配对 Wilcoxon 符号秩检验（基于均值，SCI 推荐非参数检验）。"""
        a = np.array(sample_a, dtype=float)
        b = np.array(sample_b, dtype=float)
        mask = ~(np.isnan(a) | np.isnan(b))
        d = (a[mask] - b[mask])
        d = d[d != 0]
        if len(d) < 5:
            return None
        try:
            from scipy.stats import wilcoxon
            stat, p = wilcoxon(d, alternative='two-sided', zero_method='wilcox')
            return {
                'statistic': float(stat), 'p_value': float(p),
                'mean_diff': float(d.mean()), 'n': len(d),
            }
        except ImportError:
            pass
        except Exception:
            return None
        # scipy 不可用：符号检验近似
        n_pos = int((d > 0).sum())
        n_neg = int((d < 0).sum())
        n_eff = n_pos + n_neg
        if n_eff < 5:
            return None
        k = min(n_pos, n_neg)
        # 二项检验双侧 p 值（n_eff 次试验，成功概率 0.5）
        p_one = sum(
            math.comb(n_eff, i) * (0.5 ** n_eff)
            for i in range(k + 1)
        )
        p_value = min(1.0, 2 * p_one)
        return {
            'statistic': float(k), 'p_value': float(p_value),
            'mean_diff': float(d.mean()), 'n': n_eff,
            'method': 'sign_test_fallback',
        }

    @staticmethod
    def _gap_pct(mip_obj, baseline_obj):
        if mip_obj is None or baseline_obj is None:
            return None
        try:
            if pd.isna(mip_obj) or pd.isna(baseline_obj) or baseline_obj <= 0:
                return None
        except TypeError:
            if baseline_obj <= 0:
                return None
        return (mip_obj - baseline_obj) / baseline_obj * 100

    @staticmethod
    def _enrich_large_scale_record(record, heu_name='KG-ALNS', equal_compute_budget=True,
                                   primary_metric='mean'):
        mip_best = record.get('mip_lns_obj_best') or record.get('heuristic_obj')
        ga_best = record.get('ga_obj_best')
        alns_best = record.get('alns_obj_best')
        mip_mean = record.get('mip_lns_obj_mean') or mip_best
        ga_mean = record.get('ga_obj_mean') or ga_best
        alns_mean = record.get('alns_obj_mean') or alns_best

        record['gap_to_ga_pct_best'] = record.get('gap_to_ga_pct')
        record['gap_to_alns_pct_best'] = record.get('gap_to_alns_pct')
        record['mip_lns_better_ga_best'] = record.get('mip_lns_better_ga')
        record['mip_lns_better_alns_best'] = record.get('mip_lns_better_alns')

        gap_ga_mean = BatchExperimentGurobi._gap_pct(mip_mean, ga_mean)
        gap_alns_mean = BatchExperimentGurobi._gap_pct(mip_mean, alns_mean)
        record['gap_to_ga_pct_mean'] = gap_ga_mean
        record['gap_to_alns_pct_mean'] = gap_alns_mean
        record['mip_lns_better_ga_mean'] = (
            mip_mean is not None and ga_mean is not None and mip_mean < ga_mean)
        record['mip_lns_better_alns_mean'] = (
            mip_mean is not None and alns_mean is not None and mip_mean < alns_mean)

        if primary_metric == 'mean':
            mip_obj, ga_obj, alns_obj = mip_mean, ga_mean, alns_mean
            record['gap_to_ga_pct'] = gap_ga_mean
            record['gap_to_alns_pct'] = gap_alns_mean
            record['mip_lns_better_ga'] = record['mip_lns_better_ga_mean']
            record['mip_lns_better_alns'] = record['mip_lns_better_alns_mean']
            record['heuristic_obj'] = mip_mean
        else:
            mip_obj, ga_obj, alns_obj = mip_best, ga_best, alns_best

        record['obj_winner_best'] = BatchExperimentGurobi._pick_winner(
            [mip_best, ga_best, alns_best], [heu_name, 'GA', 'ALNS'])

        mip_time = record.get('mip_lns_time') or record.get('heuristic_time')
        ga_time = record.get('ga_time')
        alns_time = record.get('alns_time')
        algo_names = [heu_name, 'GA', 'ALNS']

        record['obj_winner'] = BatchExperimentGurobi._pick_winner(
            [mip_obj, ga_obj, alns_obj], algo_names)
        gap_ga = record.get('gap_to_ga_pct')
        gap_alns = record.get('gap_to_alns_pct')
        if equal_compute_budget:
            record.pop('speed_winner', None)
            record.pop('speedup_vs_ga', None)
            record.pop('speedup_vs_alns', None)
        else:
            record['speed_winner'] = BatchExperimentGurobi._pick_winner(
                [mip_time, ga_time, alns_time], algo_names)
            if mip_time and ga_time and mip_time > 0:
                record['speedup_vs_ga'] = ga_time / mip_time
            if mip_time and alns_time and mip_time > 0:
                record['speedup_vs_alns'] = alns_time / mip_time

        if gap_ga is not None:
            if gap_ga <= 0:
                record['quality_grade'] = '优'
            elif gap_ga <= 5:
                record['quality_grade'] = '良'
            else:
                record['quality_grade'] = '一般'
        else:
            record['quality_grade'] = '-'

        if gap_alns is not None:
            if gap_alns <= 0:
                record['quality_vs_alns'] = '优'
            elif gap_alns <= 5:
                record['quality_vs_alns'] = '良'
            else:
                record['quality_vs_alns'] = '一般'
        else:
            record['quality_vs_alns'] = '-'

        record['mip_lns_better'] = record.get('mip_lns_better_alns')

    def _print_large_scale_comparison(self, large_results, heu_name='KG-ALNS',
                                      equal_compute_budget=True, primary_metric='mean'):
        """打印 KG-ALNS vs GA vs ALNS 综合对比表（主口径 mean，辅口径 best）"""
        if not large_results:
            return
        df = pd.DataFrame(large_results)
        n = len(df)
        algo_names = [heu_name, 'GA', 'ALNS']

        def _avg(col):
            if col not in df.columns:
                return None
            s = df[col].dropna()
            return s.mean() if len(s) > 0 else None

        def _fmt(v, d=1):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return 'N/A'
            return f"{v:.{d}f}"

        mip_mean = _avg('mip_lns_obj_mean') or _avg('heuristic_obj')
        ga_mean = _avg('ga_obj_mean')
        alns_mean = _avg('alns_obj_mean')
        mip_best = _avg('mip_lns_obj_best') or mip_mean
        ga_best = _avg('ga_obj_best') or ga_mean
        alns_best = _avg('alns_obj_best') or alns_mean
        mip_std = _avg('mip_lns_obj_std')
        ga_std = _avg('ga_obj_std')
        alns_std = _avg('alns_obj_std')
        mip_time = _avg('mip_lns_time') or _avg('heuristic_time')
        mip_search_time = _avg('mip_lns_search_time') or _avg('heuristic_search_time')
        mip_post_time = _avg('mip_lns_postprocess_time') or _avg('heuristic_postprocess_time')
        ga_time = _avg('ga_time')
        alns_time = _avg('alns_time')
        avg_gap_ga = _avg('gap_to_ga_pct_mean') or _avg('gap_to_ga_pct')
        avg_gap_alns = _avg('gap_to_alns_pct_mean') or _avg('gap_to_alns_pct')
        avg_gap_ga_best = _avg('gap_to_ga_pct_best')
        avg_gap_alns_best = _avg('gap_to_alns_pct_best')

        obj_wins = {name: 0 for name in algo_names}
        obj_wins['TIE'] = 0
        obj_wins_best = {name: 0 for name in algo_names}
        obj_wins_best['TIE'] = 0
        spd_wins = {name: 0 for name in algo_names}
        spd_wins['TIE'] = 0
        for _, row in df.iterrows():
            w = row.get('obj_winner', '-')
            if w in obj_wins:
                obj_wins[w] += 1
            wb = row.get('obj_winner_best', '-')
            if wb in obj_wins_best:
                obj_wins_best[wb] += 1
            if not equal_compute_budget:
                w = row.get('speed_winner', '-')
                if w in spd_wins:
                    spd_wins[w] += 1

        mip_mean_col = 'mip_lns_obj_mean' if 'mip_lns_obj_mean' in df.columns else 'heuristic_obj'
        ga_mean_col = 'ga_obj_mean' if 'ga_obj_mean' in df.columns else 'ga_obj_best'
        alns_mean_col = 'alns_obj_mean' if 'alns_obj_mean' in df.columns else 'alns_obj_best'
        mip_best_col = 'mip_lns_obj_best' if 'mip_lns_obj_best' in df.columns else mip_mean_col
        ga_best_col = 'ga_obj_best' if 'ga_obj_best' in df.columns else ga_mean_col
        alns_best_col = 'alns_obj_best' if 'alns_obj_best' in df.columns else alns_mean_col

        t_ga = self._paired_t_test(df[mip_mean_col].tolist(), df[ga_mean_col].tolist())
        t_alns = self._paired_t_test(df[mip_mean_col].tolist(), df[alns_mean_col].tolist())
        w_ga = self._paired_wilcoxon_test(df[mip_mean_col].tolist(), df[ga_mean_col].tolist())
        w_alns = self._paired_wilcoxon_test(df[mip_mean_col].tolist(), df[alns_mean_col].tolist())
        t_ga_best = self._paired_t_test(df[mip_best_col].tolist(), df[ga_best_col].tolist())
        t_alns_best = self._paired_t_test(df[mip_best_col].tolist(), df[alns_best_col].tolist())

        print(f"\n{'=' * 130}")
        print(f"大规模算例 — {heu_name} vs GA vs ALNS 综合对比表 "
              f"(主口径={primary_metric}, 辅口径=best-of-runs)")
        print(f"{'=' * 130}")
        budget_note = ("同等计算预算：三算法墙钟上限相同（KG-ALNS=领域破坏主循环·无MIP/Polish），仅比较解质量；"
                       if equal_compute_budget else "")
        print(f"说明: {budget_note}各算法独立重复运行（KG×{LARGE_MIP_NUM_RUNS}, "
              f"GA×{LARGE_GA_NUM_RUNS}, ALNS×{LARGE_ALNS_NUM_RUNS}），不同 seed；"
              "主结果报告 run-mean±std，辅结果报告 best-of-runs；"
              f"KG-ALNS 墙钟分列：主搜索 + 后处理，合计 ≤ {LARGE_BATCH_TIME_LIMIT}s；"
              "跨算例统计采用配对 t 检验与 Wilcoxon 符号秩检验。")
        print(f"{'=' * 130}")

        print(f"\n【总体对比 — 主口径 run-mean】  算例数: {n}")
        hdr = f"{'指标':<22s} | {heu_name:>12s} | {'GA':>12s} | {'ALNS':>12s} | {'优胜方':>8s}"
        print(hdr)
        print('-' * len(hdr))
        print(f"{'平均目标值(mean)':<22s} | {_fmt(mip_mean):>12s} | {_fmt(ga_mean):>12s} | "
              f"{_fmt(alns_mean):>12s} | "
              f"{self._pick_winner([mip_mean, ga_mean, alns_mean], algo_names):>8s}")
        print(f"{'run内标准差(均值)':<22s} | {_fmt(mip_std, 2):>12s} | {_fmt(ga_std, 2):>12s} | "
              f"{_fmt(alns_std, 2):>12s} | {'越小越稳':>8s}")
        print(f"{'平均目标值(best)':<22s} | {_fmt(mip_best):>12s} | {_fmt(ga_best):>12s} | "
              f"{_fmt(alns_best):>12s} | "
              f"{self._pick_winner([mip_best, ga_best, alns_best], algo_names):>8s}")
        time_note = '同等墙钟验证' if equal_compute_budget else '越快越优'
        print(f"{'平均总墙钟(s)':<22s} | {_fmt(mip_time, 2):>12s} | {_fmt(ga_time, 2):>12s} | "
              f"{_fmt(alns_time, 2):>12s} | {time_note:>8s}")
        if mip_search_time is not None or mip_post_time is not None:
            print(f"{'  └主搜索(s)':<22s} | {_fmt(mip_search_time, 2):>12s} | {'—':>12s} | {'—':>12s} | "
                  f"{'KG':>8s}")
            print(f"{'  └后处理(s)':<22s} | {_fmt(mip_post_time, 2):>12s} | {'—':>12s} | {'—':>12s} | "
                  f"{'后处理':>8s}")
        print(f"{'目标胜(mean)':<22s} | {obj_wins[heu_name]:>12d} | {obj_wins['GA']:>12d} | "
              f"{obj_wins['ALNS']:>12d} | "
              f"{self._pick_winner([obj_wins[heu_name], obj_wins['GA'], obj_wins['ALNS']], algo_names, lower_better=False):>8s}")
        print(f"{'目标胜(best)':<22s} | {obj_wins_best[heu_name]:>12d} | {obj_wins_best['GA']:>12d} | "
              f"{obj_wins_best['ALNS']:>12d} | "
              f"{self._pick_winner([obj_wins_best[heu_name], obj_wins_best['GA'], obj_wins_best['ALNS']], algo_names, lower_better=False):>8s}")
        if not equal_compute_budget:
            print(f"{'求解更快次数':<22s} | {spd_wins[heu_name]:>12d} | {spd_wins['GA']:>12d} | "
                  f"{spd_wins['ALNS']:>12d} | "
                  f"{self._pick_winner([spd_wins[heu_name], spd_wins['GA'], spd_wins['ALNS']], algo_names, lower_better=False):>8s}")
        print(f"{'平均Gap to GA(mean%)':<22s} | {_fmt(avg_gap_ga, 2):>12s} | {'—':>12s} | {'—':>12s} | "
              f"{'负=KG优':>8s}")
        print(f"{'平均Gap to ALNS(mean%)':<22s} | {_fmt(avg_gap_alns, 2):>12s} | {'—':>12s} | {'—':>12s} | "
              f"{'负=KG优':>8s}")
        print(f"{'平均Gap to GA(best%)':<22s} | {_fmt(avg_gap_ga_best, 2):>12s} | {'—':>12s} | {'—':>12s} | "
              f"{'辅':>8s}")
        print(f"{'平均Gap to ALNS(best%)':<22s} | {_fmt(avg_gap_alns_best, 2):>12s} | {'—':>12s} | {'—':>12s} | "
              f"{'辅':>8s}")

        sig_ga = int((df['p_value_ga'] < 0.05).sum()) if 'p_value_ga' in df.columns else 0
        sig_alns = int((df['p_value_alns'] < 0.05).sum()) if 'p_value_alns' in df.columns else 0
        wins_ga_mean = int(df['mip_lns_better_ga_mean'].sum()) if 'mip_lns_better_ga_mean' in df.columns else (
            int(df['mip_lns_better_ga'].sum()) if 'mip_lns_better_ga' in df.columns else 0)
        wins_alns_mean = int(df['mip_lns_better_alns_mean'].sum()) if 'mip_lns_better_alns_mean' in df.columns else (
            int(df['mip_lns_better_alns'].sum()) if 'mip_lns_better_alns' in df.columns else 0)
        wins_ga_best = int(df['mip_lns_better_ga_best'].sum()) if 'mip_lns_better_ga_best' in df.columns else 0
        wins_alns_best = int(df['mip_lns_better_alns_best'].sum()) if 'mip_lns_better_alns_best' in df.columns else 0

        if t_ga:
            sig_g = "显著" if t_ga['p_value'] < 0.05 else "不显著"
            better = heu_name if t_ga['mean_diff'] < 0 else 'GA'
            print(f"\n【跨算例配对t检验 vs GA】(run-mean, n={t_ga['n']})")
            print(f"  t={t_ga['t_stat']:.3f}, p={t_ga['p_value']:.4f} ({sig_g}), "
                  f"均值差={t_ga['mean_diff']:.2f} → {better} 更优")
        if w_ga:
            sig_w = "显著" if w_ga['p_value'] < 0.05 else "不显著"
            method = w_ga.get('method', 'wilcoxon')
            print(f"【跨算例Wilcoxon vs GA】(run-mean, n={w_ga['n']}, {method})")
            print(f"  stat={w_ga['statistic']:.3f}, p={w_ga['p_value']:.4f} ({sig_w}), "
                  f"中位差≈{w_ga['mean_diff']:.2f}")
        print(f"  逐算例run内显著(p<0.05): {sig_ga}/{n} 例 | "
              f"KG mean<GA: {wins_ga_mean}/{n} | KG best<GA: {wins_ga_best}/{n}")

        if t_alns:
            sig_a = "显著" if t_alns['p_value'] < 0.05 else "不显著"
            better_a = heu_name if t_alns['mean_diff'] < 0 else 'ALNS'
            print(f"\n【跨算例配对t检验 vs ALNS】(run-mean, n={t_alns['n']})")
            print(f"  t={t_alns['t_stat']:.3f}, p={t_alns['p_value']:.4f} ({sig_a}), "
                  f"均值差={t_alns['mean_diff']:.2f} → {better_a} 更优")
        if w_alns:
            sig_w = "显著" if w_alns['p_value'] < 0.05 else "不显著"
            method = w_alns.get('method', 'wilcoxon')
            print(f"【跨算例Wilcoxon vs ALNS】(run-mean, n={w_alns['n']}, {method})")
            print(f"  stat={w_alns['statistic']:.3f}, p={w_alns['p_value']:.4f} ({sig_w}), "
                  f"中位差≈{w_alns['mean_diff']:.2f}")
        print(f"  逐算例run内显著(p<0.05): {sig_alns}/{n} 例 | "
              f"KG mean<ALNS: {wins_alns_mean}/{n} | KG best<ALNS: {wins_alns_best}/{n}")

        if t_ga_best:
            print(f"\n【辅：跨算例配对t检验 vs GA】(best-of-runs, n={t_ga_best['n']}) "
                  f"t={t_ga_best['t_stat']:.3f}, p={t_ga_best['p_value']:.4f}")
        if t_alns_best:
            print(f"【辅：跨算例配对t检验 vs ALNS】(best-of-runs, n={t_alns_best['n']}) "
                  f"t={t_alns_best['t_stat']:.3f}, p={t_alns_best['p_value']:.4f}")

        q_ex = int((df['quality_grade'] == '优').sum()) if 'quality_grade' in df.columns else 0
        q_good = int((df['quality_grade'] == '良').sum()) if 'quality_grade' in df.columns else 0
        q_fair = int((df['quality_grade'] == '一般').sum()) if 'quality_grade' in df.columns else 0
        qa_ex = int((df['quality_vs_alns'] == '优').sum()) if 'quality_vs_alns' in df.columns else 0
        qa_good = int((df['quality_vs_alns'] == '良').sum()) if 'quality_vs_alns' in df.columns else 0
        qa_fair = int((df['quality_vs_alns'] == '一般').sum()) if 'quality_vs_alns' in df.columns else 0
        print(f"  解质量(vs GA): 优/良/一般 = {q_ex}/{q_good}/{q_fair}")
        print(f"  解质量(vs ALNS): 优/良/一般 = {qa_ex}/{qa_good}/{qa_fair}")

        print(f"\n【逐例对比】")
        if equal_compute_budget:
            print(f"{'Case':>5s} | {'Tasks':>5s} | {'梯度':>8s} | {heu_name:>9s} | {'GA':>9s} | {'ALNS':>9s} | "
                  f"{'目标胜':>6s} | {'KG(s)':>7s} | {'GA(s)':>7s} | {'ALN(s)':>7s} | "
                  f"{'GapGA%':>7s} | {'GapALN%':>8s} | {'p(ALN)':>7s} | {'质量ALN':>7s}")
        else:
            print(f"{'Case':>5s} | {'Tasks':>5s} | {'梯度':>8s} | {heu_name:>9s} | {'GA':>9s} | {'ALNS':>9s} | "
                  f"{'目标胜':>6s} | {'KG(s)':>7s} | {'GA(s)':>7s} | {'ALN(s)':>7s} | "
                  f"{'速度胜':>6s} | {'GapGA%':>7s} | {'GapALN%':>8s} | {'p(ALN)':>7s} | {'质量ALN':>7s}")
        print('-' * 130)
        for _, row in df.sort_values('instance_id').iterrows():
            gap_ga = row.get('gap_to_ga_pct')
            gap_alns = row.get('gap_to_alns_pct')
            gap_ga_s = f"{gap_ga:.2f}" if gap_ga is not None and not pd.isna(gap_ga) else '-'
            gap_alns_s = f"{gap_alns:.2f}" if gap_alns is not None and not pd.isna(gap_alns) else '-'
            pv = row.get('p_value_alns')
            pv_s = f"{pv:.3f}" if pv is not None and not pd.isna(pv) else '-'
            base = (f"  {int(row['instance_id']):>4d} | {int(row['num_tasks']):>5d} | "
                    f"{str(row.get('tier', '')):>8s} | "
                    f"{_fmt(row.get('mip_lns_obj_mean') or row.get('mip_lns_obj_best')):>9s} | "
                    f"{_fmt(row.get('ga_obj_mean') or row.get('ga_obj_best')):>9s} | "
                    f"{_fmt(row.get('alns_obj_mean') or row.get('alns_obj_best')):>9s} | "
                    f"{str(row.get('obj_winner', '-')):>6s} | "
                    f"{_fmt(row.get('mip_lns_time'), 2):>7s} | {_fmt(row.get('ga_time'), 2):>7s} | "
                    f"{_fmt(row.get('alns_time'), 2):>7s} | ")
            if equal_compute_budget:
                print(f"{base}{gap_ga_s:>7s} | {gap_alns_s:>8s} | {pv_s:>7s} | "
                      f"{str(row.get('quality_vs_alns', '-')):>7s}")
            else:
                print(f"{base}{str(row.get('speed_winner', '-')):>6s} | {gap_ga_s:>7s} | "
                      f"{gap_alns_s:>8s} | {pv_s:>7s} | {str(row.get('quality_vs_alns', '-')):>7s}")
        print(f"{'=' * 130}")

        stats_row = {
            'n_instances': n,
            'equal_compute_budget': equal_compute_budget,
            'primary_metric': primary_metric,
            'mip_num_runs': LARGE_MIP_NUM_RUNS,
            'ga_num_runs': LARGE_GA_NUM_RUNS,
            'alns_num_runs': LARGE_ALNS_NUM_RUNS,
            'avg_mip_lns_obj_mean': mip_mean,
            'avg_ga_obj_mean': ga_mean,
            'avg_alns_obj_mean': alns_mean,
            'avg_mip_lns_obj_best': mip_best,
            'avg_ga_obj_best': ga_best,
            'avg_alns_obj_best': alns_best,
            'avg_mip_lns_std': mip_std,
            'avg_ga_std': ga_std,
            'avg_alns_std': alns_std,
            'avg_mip_lns_time': mip_time,
            'avg_mip_lns_search_time': mip_search_time,
            'avg_mip_lns_postprocess_time': mip_post_time,
            'avg_ga_time': ga_time,
            'avg_alns_time': alns_time,
            'avg_gap_to_ga_pct_mean': avg_gap_ga,
            'avg_gap_to_alns_pct_mean': avg_gap_alns,
            'avg_gap_to_ga_pct_best': avg_gap_ga_best,
            'avg_gap_to_alns_pct_best': avg_gap_alns_best,
            'mip_lns_win_obj_mean': obj_wins[heu_name],
            'ga_win_obj_mean': obj_wins['GA'],
            'alns_win_obj_mean': obj_wins['ALNS'],
            'mip_lns_win_obj_best': obj_wins_best[heu_name],
            'ga_win_obj_best': obj_wins_best['GA'],
            'alns_win_obj_best': obj_wins_best['ALNS'],
            'mip_lns_better_ga_mean': wins_ga_mean,
            'mip_lns_better_alns_mean': wins_alns_mean,
            'mip_lns_better_ga_best': wins_ga_best,
            'mip_lns_better_alns_best': wins_alns_best,
            't_stat_vs_ga_mean': t_ga['t_stat'] if t_ga else None,
            'p_value_vs_ga_mean': t_ga['p_value'] if t_ga else None,
            't_stat_vs_alns_mean': t_alns['t_stat'] if t_alns else None,
            'p_value_vs_alns_mean': t_alns['p_value'] if t_alns else None,
            'wilcoxon_stat_vs_ga': w_ga['statistic'] if w_ga else None,
            'wilcoxon_p_vs_ga': w_ga['p_value'] if w_ga else None,
            'wilcoxon_stat_vs_alns': w_alns['statistic'] if w_alns else None,
            'wilcoxon_p_vs_alns': w_alns['p_value'] if w_alns else None,
            't_stat_vs_ga_best': t_ga_best['t_stat'] if t_ga_best else None,
            'p_value_vs_ga_best': t_ga_best['p_value'] if t_ga_best else None,
            't_stat_vs_alns_best': t_alns_best['t_stat'] if t_alns_best else None,
            'p_value_vs_alns_best': t_alns_best['p_value'] if t_alns_best else None,
        }
        if not equal_compute_budget:
            stats_row.update({
                'mip_lns_win_speed': spd_wins[heu_name],
                'ga_win_speed': spd_wins['GA'],
                'alns_win_speed': spd_wins['ALNS'],
            })
        pd.DataFrame([stats_row]).to_csv(
            f"{self.output_dir}/large_scale_ttest_summary.csv", index=False, encoding='utf-8-sig')

    def _save_large_scale_excel(self, large_results):
        """保存大规模算例 Excel 汇总"""
        if not OPENPYXL_AVAILABLE:
            return

        from openpyxl import Workbook
        from openpyxl.utils.dataframe import dataframe_to_rows

        df = pd.DataFrame(large_results)
        wb = Workbook()
        ws = wb.active
        ws.title = "大规模实验结果"

        export_cols = {
            'instance_id': '算例编号',
            'tier': '梯度',
            'num_ships': '船舶数',
            'num_tasks': '任务数',
            'num_berths': '泊位数',
            'num_docks': '干船坞数',
            'num_teams': '团队数',
            'heuristic_obj': '目标值(KG-ALNS)',
            'heuristic_time': '耗时(KG-ALNS_秒)',
            'mip_lns_obj_best': 'KG-ALNS目标最优',
            'mip_lns_obj_mean': 'KG-ALNS目标均值',
            'mip_lns_obj_std': 'KG-ALNS目标标准差',
            'mip_lns_time': 'KG-ALNS耗时(秒)',
            'ga_obj_best': 'GA目标最优',
            'ga_obj_mean': 'GA目标均值',
            'ga_obj_std': 'GA目标标准差',
            'ga_time': 'GA耗时(秒)',
            'gap_to_ga_pct': 'Gap_to_GA(%)',
            'alns_obj_best': 'ALNS目标最优',
            'alns_obj_mean': 'ALNS目标均值',
            'alns_obj_std': 'ALNS目标标准差',
            'alns_time': 'ALNS耗时(秒)',
            'gap_to_alns_pct': 'Gap_to_ALNS(%)',
            'mip_lns_better_ga': 'KG-ALNS优于GA',
            'mip_lns_better_alns': 'KG-ALNS优于ALNS',
            'quality_vs_alns': '解质量(vs ALNS)',
            't_stat_ga': 't统计量(vs GA)',
            'p_value_ga': 'p值(vs GA)',
            't_stat_alns': 't统计量(vs ALNS)',
            'p_value_alns': 'p值(vs ALNS)',
            'mip_lns_better': 'KG-ALNS更优(vs ALNS)',
            'obj_winner': '目标值优胜(三方)',
            'quality_grade': '解质量(vs GA)',
        }
        if 'speed_winner' in df.columns:
            export_cols['speed_winner'] = '速度优胜(三方)'
        export_df = df[[c for c in export_cols if c in df.columns]].rename(columns=export_cols)

        for col_idx, col_name in enumerate(export_df.columns, 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.value = col_name
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            cell.alignment = Alignment(horizontal="center", vertical="center")

        for row_idx, row in enumerate(dataframe_to_rows(export_df, index=False, header=False), 2):
            for col_idx, value in enumerate(row, 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.value = value
                cell.alignment = Alignment(horizontal="center", vertical="center")

        for col_idx, col_name in enumerate(export_df.columns, 1):
            max_length = max(len(str(col_name)), max(
                [len(str(val)) for val in export_df.iloc[:, col_idx - 1]]))
            col_letter = chr(64 + col_idx) if col_idx <= 26 else None
            if col_letter:
                ws.column_dimensions[col_letter].width = min(max_length + 2, 28)

        excel_file = f"{self.output_dir}/大规模算例汇总.xlsx"
        try:
            wb.save(excel_file)
            print(f"大规模 Excel 汇总已保存: {excel_file}")
        except PermissionError:
            alt_file = f"{self.output_dir}/大规模算例汇总_{datetime.now().strftime('%H%M%S')}.xlsx"
            wb.save(alt_file)
            print(f"原文件被占用，已保存至: {alt_file}")

    def _save_schedule(self, instance, schedule):
        df = pd.DataFrame(schedule)
        filename = f"{self.output_dir}/schedule_instance_{instance['instance_id']}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')

    def _save_heuristic_schedule(self, instance, schedule):
        df = pd.DataFrame(schedule)
        filename = f"{self.output_dir}/heuristic_schedule_instance_{instance['instance_id']}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')

    def _save_case_result(self, instance, schedule):
        ship_completions = {}
        for row in schedule:
            ship_id = row['ship_id']
            ship_completions.setdefault(ship_id, []).append(row)

        task_rows = []
        ship_rows = []
        team_config = instance.get('team_config')

        for ship_id, tasks in ship_completions.items():
            arrival_time = instance['arrival_times'].get(ship_id, 0)
            completion_time = max(task['completion'] for task in tasks)
            first_task_start = min(task['start'] for task in tasks)
            ship_wait_time = first_task_start - arrival_time
            total_wait = sum(task['start'] - arrival_time for task in tasks)
            task_count = len(tasks)
            teams_used = sorted(set(task['team'] for task in tasks if task['team'] is not None))
            resources_used = sorted(set(
                f"Berth{int(task['space'])}" if task['space_type'] == 'berth' else f"Dock{int(task['space'])}"
                for task in tasks
            ))

            for task in tasks:
                resource_name = (
                    f"Berth{int(task['space'])}" if task['space_type'] == 'berth' else f"Dock{int(task['space'])}"
                )
                wait_time = task['start'] - arrival_time

                team_skills_detail = ""
                if task['team'] is not None and team_config:
                    team_id = task['team']
                    ts = team_config.get('team_skills', {}).get(team_id, {})
                    if ts:
                        skills_list = [f"{skill}({level})" for skill, level in sorted(ts.items())]
                        team_skills_detail = f"T{team_id}: {', '.join(skills_list)}"

                task_rows.append({
                    'instance_id': instance['instance_id'],
                    'group_name': instance.get('group_name', ''),
                    'ship_id': ship_id,
                    'ship_arrival': arrival_time,
                    'ship_completion': completion_time,
                    'task_id': task['task_id'],
                    'task_start': task['start'],
                    'task_completion': task['completion'],
                    'task_duration': task['duration'],
                    'wait_time': wait_time,
                    'assigned_team': task['team'],
                    'Assigned_Team_Skills': team_skills_detail,
                    'resource': resource_name,
                    'skill_req': task['skill_req']
                })

            ship_rows.append({
                'instance_id': instance['instance_id'],
                'group_name': instance.get('group_name', ''),
                'ship_id': ship_id,
                'num_tasks': task_count,
                'arrival_time': arrival_time,
                'departure_time': completion_time,
                'Wait_Time': round(ship_wait_time, 2),
                'total_wait_time': total_wait,
                'teams_used': ','.join(f"T{t}" for t in teams_used),
                'resources_used': ','.join(resources_used)
            })

        task_df = pd.DataFrame(task_rows)
        task_df.to_csv(f"{self.output_dir}/Case_{instance['instance_id']:02d}_task_details.csv",
                       index=False, encoding='utf-8-sig')

        ship_df = pd.DataFrame(ship_rows)
        ship_df.to_csv(f"{self.output_dir}/Case_{instance['instance_id']:02d}_ship_summary.csv",
                       index=False, encoding='utf-8-sig')

    def _save_team_config(self, instance):
        team_config = instance.get('team_config')
        if not team_config:
            return

        rows = []
        for team_id, skills_dict in team_config['team_skills'].items():
            for skill, level in skills_dict.items():
                proc_time = team_config['processing_time'].get(team_id, {}).get(skill, None)
                rows.append({
                    'instance_id': instance['instance_id'],
                    'group_name': instance.get('group_name', ''),
                    'team_id': team_id,
                    'skill': skill,
                    'level': level,
                    'processing_time': proc_time
                })

        df = pd.DataFrame(rows)
        df.to_csv(f"{self.output_dir}/Case_{instance['instance_id']:02d}_team_config.csv",
                  index=False, encoding='utf-8-sig')

    def save_summary_table(self):
        """保存汇总表：船舶数、变量数、约束数、求解耗时、最终Gap值"""
        if not self.results:
            print("没有结果数据，无法生成汇总表")
            return

        summary_rows = []
        for r in self.results:
            summary_rows.append({
                '算例编号': r['instance_id'],
                '组别': r['group_name'],
                '船舶数': r['num_ships'],
                '任务数': r['num_tasks'],
                '团队数': r['num_teams'],
                '泊位数': r['num_berths'],
                '干船坞数': r['num_docks'],
                '变量数': r['num_variables'],
                '约束数': r['num_constraints'],
                '二进制变量': r['num_binary'],
                '连续变量': r['num_continuous'],
                'Gurobi耗时(秒)': round(r['solve_time'], 2),
                '目标值(Gurobi)': round(r['objective'], 2) if r['objective'] else None,
                '目标值(Heuristic)': round(r['heuristic_obj'], 2) if r['heuristic_obj'] else None,
                'Heuristic耗时(秒)': round(r['heuristic_time'], 6) if r['heuristic_time'] is not None else None,
                '加速比': round(r['solve_time'] / r['heuristic_time'], 1) if r['heuristic_time'] and r['heuristic_time'] > 0 else None,
                'Gap_to_Optimal(%)': round(r['gap_to_optimal'], 4) if r['gap_to_optimal'] is not None else None,
                'Obj_Diff': round(r['obj_diff'], 2) if r['obj_diff'] is not None else None,
                'Efficiency(%)': round(r['efficiency'], 2) if r['efficiency'] is not None else None,
                'Speedup_Percent(%)': round(r['speedup_percent'], 2) if r['speedup_percent'] is not None else None,
                '胜负': r.get('heu_superiority', '-'),
                'BestBound': round(r['best_bound'], 2) if r['best_bound'] else None,
                '最终Gap': round(r['mip_gap'], 6) if r['mip_gap'] is not None else None,
                '求解状态': r['status'],
                '是否可行': '是' if r['feasible'] else '否',
            })

        summary_df = pd.DataFrame(summary_rows)
        _safe_dataframe_to_csv(summary_df, f"{self.output_dir}/summary_table.csv")
        print(f"\n汇总表已保存: {self.output_dir}/summary_table.csv")

        # 打印汇总表
        print("\n" + "=" * 160)
        print("实验结果汇总表（Gurobi vs 启发式 — 严格限时 120s）")
        print("=" * 160)
        header_cols = ['算例编号', '组别', '船舶数', '任务数', '目标值(Gurobi)', '目标值(Heuristic)', 'Gap_to_Optimal(%)', 'Obj_Diff', 'Efficiency(%)', 'Speedup_Percent(%)', '胜负', 'Gurobi耗时(秒)', 'Heuristic耗时(秒)', '加速比', '求解状态']
        print(summary_df[header_cols].to_string(index=False))
        print("=" * 160)

        return summary_df

    def save_benchmark_csv(self):
        """导出 comparison_15_cases.csv — Gurobi vs 启发式对比"""
        if not self.results:
            print("没有结果数据，无法导出 benchmark CSV")
            return

        rows = []
        for r in self.results:
            rows.append({
                'N_ships': r['num_ships'],
                'N_tasks': r['num_tasks'],
                'N_berths': r['num_berths'],
                'N_docks': r['num_docks'],
                'N_teams': r['num_teams'],
                'num_variables': r['num_variables'],
                'num_binary': r['num_binary'],
                'num_continuous': r['num_continuous'],
                'num_constraints': r['num_constraints'],
                'solve_time_Gurobi_s': round(r['solve_time'], 2),
                'Objective_Gurobi': round(r['objective'], 2) if r['objective'] else None,
                'Objective_Heuristic': round(r['heuristic_obj'], 2) if r['heuristic_obj'] else None,
                'Obj_Diff': round(r['obj_diff'], 2) if r['obj_diff'] is not None else None,
                'Efficiency_pct': round(r['efficiency'], 2) if r['efficiency'] is not None else None,
                'Speedup_Percent': round(r['speedup_percent'], 2) if r['speedup_percent'] is not None else None,
                'Heuristic_Time_s': round(r['heuristic_time'], 6) if r['heuristic_time'] is not None else None,
                'Gap_to_Optimal_pct': round(r['gap_to_optimal'], 4) if r['gap_to_optimal'] is not None else None,
                'Heu_Superiority': r.get('heu_superiority', '-'),
                'Time_Speedup_x': round(r['solve_time'] / r['heuristic_time'], 1) if r['heuristic_time'] and r['heuristic_time'] > 0 else None,
                'best_bound': round(r['best_bound'], 2) if r['best_bound'] else None,
                'MIPGap': round(r['mip_gap'], 6) if r['mip_gap'] is not None else None,
                'status': r['status'],
                'feasible': r['feasible'],
            })

        df = pd.DataFrame(rows)
        path = f"{self.output_dir}/comparison_15_cases.csv"
        _safe_dataframe_to_csv(df, path)
        print(f"\nBenchmark CSV 已保存: {path}")
        return df

    def generate_excel_report(self):
        """生成格式化的Excel汇总报告"""
        if not self.results:
            print("没有结果数据，无法生成Excel报告")
            return

        if not OPENPYXL_AVAILABLE:
            print("警告: openpyxl 未安装，无法生成 Excel 文件。请运行: pip install openpyxl")
            return

        from openpyxl import Workbook
        from openpyxl.utils.dataframe import dataframe_to_rows

        # 汇总表
        summary_rows = []
        for r in self.results:
            summary_rows.append({
                'Case ID': r['instance_id'],
                'Case Name': r['group_name'],
                '船舶数': r['num_ships'],
                '任务数': r['num_tasks'],
                '团队数': r['num_teams'],
                '泊位数': r['num_berths'],
                '干船坞数': r['num_docks'],
                '目标值(Gurobi)': round(r['objective'], 2) if r['objective'] else None,
                '目标值(Heuristic)': round(r['heuristic_obj'], 2) if r['heuristic_obj'] else None,
                'Obj_Diff': round(r['obj_diff'], 2) if r['obj_diff'] is not None else None,
                'Efficiency(%)': round(r['efficiency'], 2) if r['efficiency'] is not None else None,
                'Speedup_Percent(%)': round(r['speedup_percent'], 2) if r['speedup_percent'] is not None else None,
                'Gap_to_Optimal(%)': round(r['gap_to_optimal'], 4) if r['gap_to_optimal'] is not None else None,
                '胜负': r.get('heu_superiority', '-'),
                '求解时间_Gurobi(秒)': round(r['solve_time'], 2),
                '求解时间_Heuristic(秒)': round(r['heuristic_time'], 6) if r['heuristic_time'] is not None else None,
                '加速比': round(r['solve_time'] / r['heuristic_time'], 1) if r['heuristic_time'] and r['heuristic_time'] > 0 else None,
                '变量数': r['num_variables'],
                '约束数': r['num_constraints'],
                '二进制变量': r['num_binary'],
                '连续变量': r['num_continuous'],
                '节点数': r['node_count'],
                '迭代数': r['iteration_count'],
                'BestBound': round(r['best_bound'], 2) if r['best_bound'] else None,
                '最终Gap': round(r['mip_gap'], 6) if r['mip_gap'] is not None else None,
                '求解状态': r['status'],
                '可行': '是' if r['feasible'] else '否',
            })

        export_df = pd.DataFrame(summary_rows)

        wb = Workbook()
        ws = wb.active
        ws.title = "实验结果汇总"

        for col_idx, col_name in enumerate(export_df.columns, 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.value = col_name
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            cell.alignment = Alignment(horizontal="center", vertical="center")

        for row_idx, row in enumerate(dataframe_to_rows(export_df, index=False, header=False), 2):
            for col_idx, value in enumerate(row, 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.value = value
                cell.alignment = Alignment(horizontal="center", vertical="center")
                col_name = export_df.columns[col_idx - 1]
                if col_name in ['目标值(Gurobi)', '目标值(Heuristic)', 'Obj_Diff', 'Efficiency(%)', 'Speedup_Percent(%)', '求解时间_Gurobi(秒)', '求解时间_Heuristic(秒)'] and isinstance(value, (int, float)):
                    cell.number_format = '0.00'

        for col_idx, col_name in enumerate(export_df.columns, 1):
            max_length = max(len(str(col_name)), max([len(str(val)) for val in export_df.iloc[:, col_idx - 1]]))
            col_letter = chr(64 + col_idx) if col_idx <= 26 else None
            if col_letter:
                ws.column_dimensions[col_letter].width = min(max_length + 2, 28)

        # 任务明细
        if self.detailed_schedules:
            task_rows = []
            ship_rows_agg = []
            for item in self.detailed_schedules:
                instance_id = item['instance_id']
                group_name = item['group_name']
                arrival_times = item['arrival_times']
                schedule = item['schedule']

                instance_obj = next((inst for inst in self.instances if inst['instance_id'] == instance_id), None)
                team_config = instance_obj.get('team_config') if instance_obj else None

                ship_tasks_agg = {}
                for task in schedule:
                    ship_id = task['ship_id']
                    ship_tasks_agg.setdefault(ship_id, []).append(task)
                for ship_id, tasks in ship_tasks_agg.items():
                    arrival_time = arrival_times.get(ship_id, 0)
                    comp_time = max(task['completion'] for task in tasks)
                    first_start = min(task['start'] for task in tasks)
                    ship_wait = round(first_start - arrival_time, 2)
                    total_wait = sum(task['start'] - arrival_time for task in tasks)
                    teams_used = sorted(set(task['team'] for task in tasks if task['team'] is not None))
                    resources_used = sorted(set(
                        f"Berth{int(task['space'])}" if task['space_type'] == 'berth' else f"Dock{int(task['space'])}"
                        for task in tasks
                    ))
                    ship_rows_agg.append({
                        'instance_id': instance_id,
                        'group_name': group_name,
                        'ship_id': ship_id,
                        'num_tasks': len(tasks),
                        'arrival_time': arrival_time,
                        'departure_time': comp_time,
                        'Wait_Time': ship_wait,
                        'total_wait_time': round(total_wait, 2),
                        'teams_used': ','.join(f"T{t}" for t in teams_used),
                        'resources_used': ','.join(resources_used)
                    })
                    for task in tasks:
                        resource_name = (
                            f"Berth{int(task['space'])}" if task['space_type'] == 'berth' else f"Dock{int(task['space'])}"
                        )
                        team_skills_detail = ""
                        if task['team'] is not None and team_config:
                            ts = team_config.get('team_skills', {}).get(task['team'], {})
                            if ts:
                                skills_list = [f"{skill}({level})" for skill, level in sorted(ts.items())]
                                team_skills_detail = f"T{task['team']}: {', '.join(skills_list)}"
                        task_rows.append({
                            'instance_id': instance_id,
                            'group_name': group_name,
                            'ship_id': task['ship_id'],
                            'ship_arrival': arrival_time,
                            'ship_completion': comp_time,
                            'task_id': task['task_id'],
                            'task_start': task['start'],
                            'task_completion': task['completion'],
                            'task_duration': task['duration'],
                            'wait_time': round(task['start'] - arrival_time, 2),
                            'assigned_team': task['team'],
                            'Assigned_Team_Skills': team_skills_detail,
                            'resource': resource_name,
                            'skill_req': json.dumps(task['skill_req'], ensure_ascii=False)
                        })

            if task_rows:
                task_df = pd.DataFrame(task_rows)
                task_ws = wb.create_sheet(title="任务明细")
                for col_idx, col_name in enumerate(task_df.columns, 1):
                    cell = task_ws.cell(row=1, column=col_idx)
                    cell.value = col_name
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                for row_idx, row in enumerate(dataframe_to_rows(task_df, index=False, header=False), 2):
                    for col_idx, value in enumerate(row, 1):
                        cell = task_ws.cell(row=row_idx, column=col_idx)
                        cell.value = value
                        cell.alignment = Alignment(horizontal="center", vertical="center")
                        if task_df.columns[col_idx - 1] in ['task_start', 'task_completion', 'task_duration', 'wait_time'] and isinstance(value, (int, float)):
                            cell.number_format = '0.00'
                for col_idx, col_name in enumerate(task_df.columns, 1):
                    max_length = max(len(str(col_name)), max([len(str(val)) for val in task_df.iloc[:, col_idx - 1]]))
                    col_letter = chr(64 + col_idx) if col_idx <= 26 else None
                    if col_letter:
                        task_ws.column_dimensions[col_letter].width = min(max_length + 2, 30)

            if ship_rows_agg:
                ship_df = pd.DataFrame(ship_rows_agg)
                ship_ws = wb.create_sheet(title="船舶汇总")
                for col_idx, col_name in enumerate(ship_df.columns, 1):
                    cell = ship_ws.cell(row=1, column=col_idx)
                    cell.value = col_name
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                for row_idx, row in enumerate(dataframe_to_rows(ship_df, index=False, header=False), 2):
                    for col_idx, value in enumerate(row, 1):
                        cell = ship_ws.cell(row=row_idx, column=col_idx)
                        cell.value = value
                        cell.alignment = Alignment(horizontal="center", vertical="center")
                        if ship_df.columns[col_idx - 1] in ['arrival_time', 'departure_time', 'Wait_Time', 'total_wait_time'] and isinstance(value, (int, float)):
                            cell.number_format = '0.00'
                for col_idx, col_name in enumerate(ship_df.columns, 1):
                    max_length = max(len(str(col_name)), max([len(str(val)) for val in ship_df.iloc[:, col_idx - 1]]))
                    col_letter = chr(64 + col_idx) if col_idx <= 26 else None
                    if col_letter:
                        ship_ws.column_dimensions[col_letter].width = min(max_length + 2, 30)

        # 团队配置表
        team_rows = []
        for instance in self.instances:
            team_config = instance.get('team_config')
            if not team_config:
                continue
            for team_id, skills_dict in team_config['team_skills'].items():
                for skill, level in skills_dict.items():
                    proc_time = team_config['processing_time'].get(team_id, {}).get(skill, None)
                    team_rows.append({
                        'instance_id': instance['instance_id'],
                        'group_name': instance.get('group_name', ''),
                        'team_id': team_id,
                        'skill': skill,
                        'level': level,
                        'processing_time': proc_time
                    })
        if team_rows:
            team_df = pd.DataFrame(team_rows)
            team_ws = wb.create_sheet(title="团队配置")
            for col_idx, col_name in enumerate(team_df.columns, 1):
                cell = team_ws.cell(row=1, column=col_idx)
                cell.value = col_name
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
                cell.alignment = Alignment(horizontal="center", vertical="center")
            for row_idx, row in enumerate(dataframe_to_rows(team_df, index=False, header=False), 2):
                for col_idx, value in enumerate(row, 1):
                    cell = team_ws.cell(row=row_idx, column=col_idx)
                    cell.value = value
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    if team_df.columns[col_idx - 1] == 'processing_time' and isinstance(value, (int, float)):
                        cell.number_format = '0.00'
            for col_idx, col_name in enumerate(team_df.columns, 1):
                max_length = max(len(str(col_name)), max([len(str(val)) for val in team_df.iloc[:, col_idx - 1]]))
                col_letter = chr(64 + col_idx) if col_idx <= 26 else None
                if col_letter:
                    team_ws.column_dimensions[col_letter].width = min(max_length + 2, 30)

        excel_file = f"{self.output_dir}/算法对比汇总.xlsx"
        try:
            wb.save(excel_file)
            print(f"Excel汇总表已保存: {excel_file}")
        except PermissionError:
            alt_file = f"{self.output_dir}/算法对比汇总_{datetime.now().strftime('%H%M%S')}.xlsx"
            wb.save(alt_file)
            print(f"原文件被占用，Excel汇总表已保存至: {alt_file}")
        return excel_file

    def save_summary(self):
        """保存汇总结果和统计分析"""
        # 算例摘要
        summary_rows = []
        for instance in self.instances:
            summary_rows.append({
                'instance_id': instance['instance_id'],
                'group_name': instance.get('group_name', ''),
                'num_ships': instance['num_ships'],
                'num_tasks': instance['num_tasks'],
                'num_teams': instance['num_teams'],
                'num_berths': instance['num_berths'],
                'num_docks': instance['num_docks']
            })
        _safe_dataframe_to_csv(
            pd.DataFrame(summary_rows), f"{self.output_dir}/Summary_Table.csv")

        df = pd.DataFrame(self.results)
        _safe_dataframe_to_csv(df, f"{self.output_dir}/batch_results.csv")

        # 分组统计
        group_stats = []
        for group_name in df['group_name'].unique():
            group_df = df[df['group_name'] == group_name]
            feasible_df = group_df[group_df['feasible'] == True]

            if len(feasible_df) > 0:
                stats = {
                    'group_name': group_name,
                    'num_instances': len(group_df),
                    'num_feasible': len(feasible_df),
                    'feasibility_rate': len(feasible_df) / len(group_df) * 100,
                    'avg_objective': feasible_df['objective'].mean(),
                    'std_objective': feasible_df['objective'].std(),
                    'min_objective': feasible_df['objective'].min(),
                    'max_objective': feasible_df['objective'].max(),
                    'avg_solve_time': feasible_df['solve_time'].mean(),
                    'avg_variables': feasible_df['num_variables'].mean(),
                    'avg_constraints': feasible_df['num_constraints'].mean(),
                    'avg_gap': feasible_df['mip_gap'].mean() if 'mip_gap' in feasible_df else None,
                }
                group_stats.append(stats)

        if group_stats:
            stats_df = pd.DataFrame(group_stats)
            _safe_dataframe_to_csv(stats_df, f"{self.output_dir}/group_statistics.csv")
            print("\n" + "=" * 100)
            print("分组统计汇总")
            print("=" * 100)
            print(stats_df.to_string(index=False))

        return df, group_stats if group_stats else None

    def save_instances(self):
        """保存算例配置"""
        for instance in self.instances:
            tasks_df = pd.DataFrame(instance['tasks_data'])
            tasks_df.to_csv(f"{self.output_dir}/instance_{instance['instance_id']}_tasks.csv",
                            index=False, encoding='utf-8-sig')

            arrival_df = pd.DataFrame([instance['arrival_times']])
            arrival_df.to_csv(f"{self.output_dir}/instance_{instance['instance_id']}_arrivals.csv",
                              index=False, encoding='utf-8-sig')

            if 'due_dates' in instance:
                due_df = pd.DataFrame([instance['due_dates']])
                due_df.to_csv(f"{self.output_dir}/instance_{instance['instance_id']}_due_dates.csv",
                              index=False, encoding='utf-8-sig')

            metadata = {
                'instance_id': instance['instance_id'],
                'group_name': instance.get('group_name', ''),
                'config': instance.get('config', {}),
                'num_ships': instance['num_ships'],
                'num_tasks': instance['num_tasks'],
                'num_teams': instance['num_teams'],
                'num_berths': instance.get('num_berths'),
                'num_docks': instance.get('num_docks')
            }
            with open(f"{self.output_dir}/instance_{instance['instance_id']}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            team_config = instance.get('team_config')
            if team_config:
                with open(f"{self.output_dir}/instance_{instance['instance_id']}_teams.json", 'w') as f:
                    json.dump(team_config, f, indent=2)

    def load_instances_from_disk(self, quiet=False):
        """从已保存的 CSV/JSON 加载算例（消融/敏感性续跑时避免重复随机生成）。"""
        import ast
        meta_files = sorted(
            glob.glob(os.path.join(self.output_dir, 'instance_*_metadata.json')),
            key=lambda p: int(re.search(r'instance_(\d+)_metadata', os.path.basename(p)).group(1)),
        )
        if not meta_files:
            return False
        if self.experiment_configs and len(meta_files) != len(self.experiment_configs):
            if not quiet:
                print(f"  [load] 磁盘算例 {len(meta_files)} 个 != 配置 {len(self.experiment_configs)} 个，将重新生成")
            return False

        instances = []
        for mf in meta_files:
            with open(mf, encoding='utf-8') as f:
                metadata = json.load(f)
            iid = int(metadata['instance_id'])
            tasks_path = os.path.join(self.output_dir, f'instance_{iid}_tasks.csv')
            if not os.path.isfile(tasks_path):
                if not quiet:
                    print(f"  [load] 缺少 instance_{iid}_tasks.csv，将重新生成")
                return False

            tasks_df = pd.read_csv(tasks_path, encoding='utf-8-sig')
            tasks_data = []
            for _, row in tasks_df.iterrows():
                skill_raw = row['skill_req']
                if isinstance(skill_raw, str):
                    skill_req = ast.literal_eval(skill_raw)
                else:
                    skill_req = skill_raw
                tasks_data.append({
                    'task_id': int(row['task_id']),
                    'ship_id': int(row['ship_id']),
                    'skill_req': skill_req,
                    'require_dock': bool(row['require_dock']),
                    'min_qual': int(row['min_qual']),
                    'complexity': float(row['complexity']),
                })

            arrival_path = os.path.join(self.output_dir, f'instance_{iid}_arrivals.csv')
            arrival_df = pd.read_csv(arrival_path, encoding='utf-8-sig')
            ship_ids = [int(c) for c in arrival_df.columns]
            arrival_times = {sid: float(arrival_df[str(sid)].iloc[0]) for sid in ship_ids}

            due_dates = {}
            due_path = os.path.join(self.output_dir, f'instance_{iid}_due_dates.csv')
            if os.path.isfile(due_path):
                due_df = pd.read_csv(due_path, encoding='utf-8-sig')
                due_dates = {int(c): float(due_df[str(c)].iloc[0]) for c in due_df.columns}

            team_config = None
            teams_path = os.path.join(self.output_dir, f'instance_{iid}_teams.json')
            if os.path.isfile(teams_path):
                with open(teams_path, encoding='utf-8') as f:
                    team_config = normalize_team_config_keys(json.load(f))

            config = metadata.get('config', {})
            num_berths = metadata.get('num_berths') or config.get('num_berths', 2)
            num_docks = metadata.get('num_docks') or config.get('num_docks', 1)
            instance = {
                'instance_id': iid,
                'group_name': metadata.get('group_name', ''),
                'config': config,
                'config_idx': iid - 1,
                'num_ships': int(metadata['num_ships']),
                'num_tasks': int(metadata['num_tasks']),
                'num_berths': int(num_berths),
                'num_docks': int(num_docks),
                'num_teams': int(metadata['num_teams']),
                'berths': list(range(1, int(num_berths) + 1)),
                'docks': list(range(1, int(num_docks) + 1)),
                'arrival_times': arrival_times,
                'due_dates': due_dates,
                'tasks_data': tasks_data,
                'team_config': team_config,
            }
            instances.append(instance)

        self.instances = instances
        if not quiet:
            print(f"  [load] 从磁盘加载 {len(instances)} 个算例: {self.output_dir}")
        return True

# 算法设计8 专用种子（与算法设计6/7 的 20250608、20250601 不同，保证算例全集不同）
REPRODUCIBILITY_SEED = 20260719
SMALL_SCALE_CONFIG_SEED = 20260711   # 阶段一算例配置固定种子（续跑时复用同一批算例）

# 大规模算例梯度（对标上海洋山港四期：年约 2500 艘 ≈ 周 50 艘，调度窗口取 30–60 艘）
LARGE_SCALE_SHIP_RANGE = (30, 60)
LARGE_SCALE_TIER_SPECS = [
    (30, 40, 20),   # 梯度1 Light
    (41, 50, 20),   # 梯度2 Medium
    (51, 60, 20),   # 梯度3 Heavy
]
LARGE_SCALE_TIER_LABELS = [f'{lo}-{hi}' for lo, hi, _ in LARGE_SCALE_TIER_SPECS]
LARGE_SCALE_TIER_NAMES = ('Light', 'Medium', 'Heavy')  # 梯度1/2/3 简称


def generate_large_scale_configs(num_instances=60, seed=REPRODUCIBILITY_SEED):
    """生成大规模算例配置（30-60艘船，3梯度 × 20组 = 60组）

    梯度划分（论文 §5 大规模实验设计）：
      梯度1 (Light):  30-40艘 × 20组 — 接近周调度下限
      梯度2 (Medium): 41-50艘 × 20组 — 对标洋山港四期周船次（≈50）
      梯度3 (Heavy):  51-60艘 × 20组 — 高峰周/缓冲上限，仍可控于 KG-ALNS 优势区间
    """
    random.seed(seed)
    np.random.seed(seed)
    configs = []

    for tier_min, tier_max, count in LARGE_SCALE_TIER_SPECS:
        for i in range(count):
            n_ships = random.randint(tier_min, tier_max)
            n_berths = min(8, max(2, n_ships // 5 + 2))
            n_docks = min(4, max(1, n_ships // 10 + 1))
            n_teams = min(12, max(3, n_ships // 3 + 2))

            if n_ships <= 40:
                tasks_per_ship = (2, 3)
            elif n_ships <= 50:
                tasks_per_ship = (2, 3)
            else:
                tasks_per_ship = (2, 4)

            cfg = {
                'num_ships': n_ships,
                'tasks_per_ship_range': tasks_per_ship,
                'arrival_max': n_ships * 3,
                'num_berths': n_berths,
                'num_docks': n_docks,
                'force_team_count': n_teams,
                'num_instances': 1,
                'time_limit': 600,
                'mip_gap': 0.0,
                'group_name': f'Tier_{tier_min}-{tier_max}',
                'tier': f'{tier_min}-{tier_max}',
                'config_type': 'baseline',
            }
            configs.append(cfg)

    return configs


def generate_ablation_xl_configs(seed=None):
    """生成消融专用超大梯度配置（默认 71-80 × 5），不改动阶段二 30-60 配置。

    资源上限略高于阶段二（泊位/船坞/班组），避免 70+ 船在硬顶 8/4/12 下过度不可行，
    同时仍显著紧于船队规模，便于拉开消融差距。
    """
    seed = int(seed if seed is not None else globals().get(
        'ABLATION_XL_CONFIG_SEED', 20260821))
    random.seed(seed)
    np.random.seed(seed)
    configs = []
    specs = globals().get('ABLATION_XL_TIER_SPECS', [(71, 80, 5)])
    for tier_min, tier_max, count in specs:
        for _ in range(count):
            n_ships = random.randint(tier_min, tier_max)
            n_berths = min(12, max(2, n_ships // 5 + 2))
            n_docks = min(6, max(1, n_ships // 10 + 1))
            n_teams = min(16, max(3, n_ships // 3 + 2))
            configs.append({
                'num_ships': n_ships,
                'tasks_per_ship_range': (2, 4),
                'arrival_max': n_ships * 3,
                'num_berths': n_berths,
                'num_docks': n_docks,
                'force_team_count': n_teams,
                'num_instances': 1,
                'time_limit': 600,
                'mip_gap': 0.0,
                'group_name': f'Tier_{tier_min}-{tier_max}',
                'tier': f'{tier_min}-{tier_max}',
                'config_type': 'ablation_xl',
            })
    return configs


def load_or_create_ablation_xl_instances(output_root):
    """加载或生成消融 XL 算例（独立目录；ID 从 ABLATION_XL_ID_OFFSET+1 起）。"""
    xl_subdir = globals().get('ABLATION_XL_POOL_SUBDIR', 'ablation_xl_pool')
    xl_dir = os.path.join(output_root, xl_subdir)
    os.makedirs(xl_dir, exist_ok=True)
    offset = int(globals().get('ABLATION_XL_ID_OFFSET', 100) or 100)
    configs = load_large_configs(xl_dir)
    if not configs:
        configs = generate_ablation_xl_configs()
        save_large_configs(configs, xl_dir)
        print(f"[消融XL] 已生成 {len(configs)} 组配置 → {xl_dir}/large_configs.json",
              flush=True)

    exp = BatchExperimentGurobi(output_dir=xl_dir, heuristic_type='lNS')
    exp.set_experiment_configs(configs)
    if exp.load_instances_from_disk(quiet=True):
        # 校验 ID 偏移，避免与阶段二 1–60 冲突
        bad = [int(i['instance_id']) for i in exp.instances
               if int(i['instance_id']) <= offset]
        if bad:
            print(f"[消融XL] 检测到旧 ID {bad}，将按偏移 {offset}+ 重新生成",
                  flush=True)
            for f in glob.glob(os.path.join(xl_dir, 'instance_*')):
                try:
                    os.remove(f)
                except OSError:
                    pass
            exp.instances = []
        else:
            print(f"[消融XL] 从磁盘加载 {len(exp.instances)} 例 "
                  f"(ids={[int(i['instance_id']) for i in exp.instances]})",
                  flush=True)
            for inst in exp.instances:
                tc = inst.get('team_config')
                if tc is not None:
                    inst['team_config'] = normalize_team_config_keys(tc)
            return list(exp.instances)

    # 无可用磁盘算例：按偏移 ID 生成并落盘（不触碰 large_scale/）
    for f in glob.glob(os.path.join(xl_dir, 'instance_*')):
        try:
            os.remove(f)
        except OSError:
            pass
    instances = []
    for i, config in enumerate(configs):
        iid = offset + 1 + i
        seed = (iid * 100 + 17) ^ int(REPRODUCIBILITY_SEED) ^ int(
            globals().get('ABLATION_XL_CONFIG_SEED', 20260821))
        generator = RandomInstanceGenerator(seed=seed)
        instance = generator.generate_instance(iid, config)
        instance['group_name'] = config.get(
            'group_name', f'Tier_{config.get("tier", "xl")}')
        instance['config_idx'] = i
        instances.append(instance)
        print(f"  [消融XL] 算例 {iid}: {instance['num_ships']}船, "
              f"{instance['num_tasks']}任务, {instance['num_teams']}团队 "
              f"(tier={config.get('tier')})", flush=True)
    exp.instances = instances
    exp.save_instances()
    save_large_configs(configs, xl_dir)
    print(f"[消融XL] 已保存 {len(instances)} 例 → {xl_dir}/", flush=True)
    return list(instances)


OUTPUT_ROOT = os.path.join(os.path.expanduser("~/Desktop"), "算法设计52")

# ---------------------------------------------------------------------------
# 分阶段交互确认（每阶段完成后输入 yes 才继续，保证同进程数据连贯）
# ---------------------------------------------------------------------------
PROMPT_BETWEEN_STAGES = False      # 无人值守：关闭阶段间 yes
PROMPT_YES = 'yes'                 # 确认关键字（小写）

# ---- 断点续跑（当前已启用）----
# True: 从零重跑（清空 OUTPUT_ROOT、重新生成算例、不读断点 CSV）
# False + AUTO_RESUME: 跳过 CSV 中已完成算例，从未完成处继续
FRESH_START = False                # 断点续跑：不清空、不重生成已有结果
AUTO_RESUME = True                 # 自动跳过 small/large CSV 中已完成算例
SMALL_SCALE_NUM_CASES = 15         # 阶段一每轨算例数（基准/增强各 15）

# ---------------------------------------------------------------------------
# Windows 防黑屏安全档（内存/页面文件不足、一运行就黑屏时务必 True）
# True → 阶段一(小规模) + 阶段二(大规模)；默认不开消融/敏感性
# ---------------------------------------------------------------------------
WINDOWS_SAFE_RUN = True
# KG∥GA∥ALNS 三方并行 + 300s×3 → 算法设计52（KG-ALNS+ / V1_kg_endgame）
_RERUN_ARCHIVE_POLISH = False

# ---------------------------------------------------------------------------
# 内存优化（Windows ≤16GB / 页面文件不足时建议 True）
# 不改变算法逻辑与 CSV 主指标；仅减少驻留大对象与 I/O 开销
# ---------------------------------------------------------------------------
LARGE_LOW_MEMORY_MODE = True
LARGE_SAVE_SCHEDULE_FILES = not LARGE_LOW_MEMORY_MODE   # False=不写每例调度/ship_summary CSV
GUROBI_SUB_THREADS = 2 if LARGE_LOW_MEMORY_MODE else 0  # 2=质量/内存折中，0=Gurobi 默认

# ---------------------------------------------------------------------------
# 实验运行开关
# ---------------------------------------------------------------------------
# 'pilot_destroy'— 轻量：15 例 × 3 destroy profile（baseline/knowledge_high/stable），不跑全量 60
# 'pilot_test'   — 旧：15例×60s 筛 V1/V2/V3 → 9例×300s 验证
# 'full'         — 阶段一 + 阶段二 + 消融 + 敏感性
# 'interactive'  — 同 full，但默认分阶段 yes 确认
# 'large_only'   — 跳过阶段一，只跑大规模
# 'resume_large' — 从 large_scale_results.csv 断点续跑
# 'ablation'     — 仅跑消融（SCI 主消融 → ablation_sci_main_51_60_x5/）
# 'ablation_regression' — Full 回归测试 Case 2/25/43，对照 5.3 基准
# 'sensitivity'  — 仅跑旧算法参数敏感性（compact_9x3）
# 'sensitivity_lambda' — 仅跑惩罚系数 λ 敏感性（新目录，不覆盖旧结果）
# 'ablation_and_lambda' — 消融后接 λ 敏感性（新目录，不覆盖旧结果）
#
# ★ 改这里切换单阶段实验（WINDOWS_SAFE_RUN 不再覆盖此值）：
EXPERIMENT_MODE = 'ablation'          # 当前=knowledge_trajectory（收敛轨迹+触发率）
# EXPERIMENT_MODE = 'sensitivity_lambda'  # λ 敏感性（已完成可注释）
# EXPERIMENT_MODE = 'ablation_regression' # Full 回归 Case 2/25/43
# ABLATION_STUDY: 'main'|'secondary'|'knowledge_trajectory'|'stress'|'operators'
if WINDOWS_SAFE_RUN:
    # 防黑屏：不跑阶段一/全管道；仅执行上方 EXPERIMENT_MODE 指定的单阶段
    RUN_SMALL_SCALE = False
    RUN_ABLATION_AFTER_LARGE = False
    RUN_SENSITIVITY_AFTER_ABLATION = False
else:
    if EXPERIMENT_MODE in ('interactive', 'full', 'resume_large', 'large_only'):
        pass  # 保持用户设定的 EXPERIMENT_MODE
    elif EXPERIMENT_MODE == 'ablation':
        pass
    else:
        EXPERIMENT_MODE = 'interactive' if PROMPT_BETWEEN_STAGES else 'full'
    RUN_SMALL_SCALE = True
    RUN_ABLATION_AFTER_LARGE = True
    RUN_SENSITIVITY_AFTER_ABLATION = False

if globals().get('_RERUN_ARCHIVE_POLISH', False):
    EXPERIMENT_MODE = 'resume_large'
    RUN_SMALL_SCALE = False
    RUN_ABLATION_AFTER_LARGE = False
    RUN_SENSITIVITY_AFTER_ABLATION = False
    PROMPT_BETWEEN_STAGES = False

CLEAR_OUTPUT_ON_START = FRESH_START  # 启动时清空输出根目录
# False：full 与消融变体同代码/同种子/同 300s 重跑（SCI 公平基准）；True：从 §5.3 导入（口径不一致，勿用于正文）
ABLATION_IMPORT_FULL_FROM_LARGE = False

# ---------------------------------------------------------------------------
# 消融实验分表（SCI 双表结构）
#   'main'      — 主消融：4 配置 × 全部 20 个 Tier 51–60 × 5 runs（已完成，勿为显著性重跑）
#   'secondary' — Endgame 拆分：w/o Elite / w/o Skill（复用主消融 Full；独立 Holm）
#   'stress'    — 附加压力测试：71–80（独立成表，不与主显著性混合）
#   'operators' — L1/L3/L4 知识算子消融
# ---------------------------------------------------------------------------
# ★ Knowledge 定位调整（勿再冲终局显著性）：
#   - 可保留 KG-ALNS / 知识框架；Elite=主性能驱动；Knowledge=领域嵌入设计
#   - 已有：主消融 + Endgame拆分 + 收敛轨迹；机制分析：H_i 相关已可跑
#   - 目标分解(flow/misc)需新跑才有列；交互可选 knowledge_interaction（+100次）
# ABLATION_STUDY = 'knowledge_interaction'  # 可选：仅补 w/o K+E，约 1.5–2h
ABLATION_STUDY = 'knowledge_trajectory'   # 已完成；改 interaction 前请确认老师是否要求

# 收敛轨迹检查点（秒）；统计单位仍为 20 实例配对，非 100 runs
CONVERGENCE_CHECKPOINT_SEC = (30, 60, 120, 180, 300)
ABLATION_RECORD_CONVERGENCE = False   # knowledge_trajectory 模式下自动 True

# ★ SCI 正式主消融（不改 LARGE_SCALE / 敏感性）
#   全部 20 个 Tier 51–60 × 4 主要配置 × 5 runs × 300s
#   主要假设：知识引导搜索 / 自适应学习 / 终局强化（组合关闭，减少 Holm 负担）
ABLATION_MAIN_VARIANTS = [
    'full',
    'wo_knowledge_guided',  # 关评价+知识破坏+α=0；保留同一 repair 骨架
    'wo_adaptive',          # 关自适应；算子池不变，固定选择概率
    'wo_endgame',           # 关 elite + skill + team_swap；预算仍跑满主搜索
]
ABLATION_MAIN_SUBDIR = 'ablation_sci_main_51_60_x5'
ABLATION_MAIN_SAMPLE_PER_TIER = 20      # 取满该梯度全部 20 例
ABLATION_MAIN_TIER_FILTER = ['51-60']
ABLATION_MAIN_NUM_RUNS = 5
ABLATION_SAMPLE_SEED = 20250717         # 仅抽样用；主消融取满梯度时不起筛除作用
ABLATION_FIXED_INSTANCE_IDS = None      # 主消融：不挑例，用该梯度全部实例
# 配对种子：seed_ir = SEED0 + 1000*instance_id + run（各配置共用）
ABLATION_PAIR_SEED0 = 20260822
# 无自适应时的固定 destroy 选择概率（实验前锁定；顺序 L1,L2,L3,L4,diversity）
ABLATION_FIXED_DESTROY_WEIGHTS = (0.40, 0.15, 0.15, 0.30, 0.00)

# 次级消融：拆分 Endgame（最推荐补跑；复用主消融 Full，勿与三项主检验混做 Holm）
#   w/o Elite — 仅关 elite restart
#   w/o Skill — 关 skill refine（+ team_swap；Full 默认本就关 swap）
ABLATION_SECONDARY_VARIANTS = [
    'full',                 # 从 ablation_sci_main_51_60_x5 导入，不重跑
    'wo_elite_restart',
    'wo_skill_refine',      # flags 内同时关 team_swap
]
ABLATION_SECONDARY_SUBDIR = 'ablation_endgame_split_51_60_x5'
ABLATION_SECONDARY_SAMPLE_PER_TIER = 20
ABLATION_SECONDARY_TIER_FILTER = ['51-60']
ABLATION_SECONDARY_NUM_RUNS = 5
# 次级实验：从已完成的 SCI 主消融导入 Full 基准（同实例、同 5-run 均值）
ABLATION_IMPORT_FULL_FROM_SUBDIR = 'ablation_sci_main_51_60_x5'

# Knowledge 收敛轨迹：Full vs w/o Knowledge（同实例同种子；记录 30–300s 检查点 + 知识计数器）
ABLATION_KNOWLEDGE_TRAJECTORY_VARIANTS = [
    'full',
    'wo_knowledge_guided',
]
ABLATION_KNOWLEDGE_TRAJECTORY_SUBDIR = 'ablation_knowledge_trajectory_51_60_x5'
ABLATION_KNOWLEDGE_TRAJECTORY_SAMPLE_PER_TIER = 20
ABLATION_KNOWLEDGE_TRAJECTORY_TIER_FILTER = ['51-60']
ABLATION_KNOWLEDGE_TRAJECTORY_NUM_RUNS = 5

# Knowledge×Elite 交互：仅新增 w/o K+E（100 次）；其余从主/次级消融导入
ABLATION_KNOWLEDGE_INTERACTION_VARIANTS = [
    'full',                     # import ablation_sci_main_51_60_x5
    'wo_knowledge_guided',      # import trajectory 或主消融
    'wo_elite_restart',         # import ablation_endgame_split_51_60_x5
    'wo_knowledge_no_elite',    # 新增跑
]
ABLATION_KNOWLEDGE_INTERACTION_SUBDIR = 'ablation_knowledge_elite_interaction_51_60_x5'
ABLATION_KNOWLEDGE_INTERACTION_SAMPLE_PER_TIER = 20
ABLATION_KNOWLEDGE_INTERACTION_TIER_FILTER = ['51-60']
ABLATION_KNOWLEDGE_INTERACTION_NUM_RUNS = 5
# 导入已有 Full / w/o K / w/o Elite（不重跑）
ABLATION_IMPORT_VARIANTS_FROM = {
    'full': 'ablation_sci_main_51_60_x5',
    'wo_knowledge_guided': 'ablation_sci_main_51_60_x5',
    'wo_elite_restart': 'ablation_endgame_split_51_60_x5',
}

# 旧「子模块打包」列表（若需全开 K/D/R，可临时改 ABLATION_SECONDARY_VARIANTS）
ABLATION_MODULE_SECONDARY_VARIANTS = [
    'full',
    'wo_knowledge',
    'wo_destroy',
    'wo_repair_alpha0',
    'wo_elite_restart',
    'wo_skill_refine',
]
# ---- 71–80 附加压力测试（独立目录；不进入主 Friedman/Holm）----
ABLATION_STRESS_SUBDIR = 'ablation_stress_71_80_x5'
ABLATION_STRESS_SAMPLE_PER_TIER = 5     # 现有 XL 池 5 例；扩到 10 后再宣称规模依赖
ABLATION_STRESS_TIER_FILTER = ['71-80']
ABLATION_STRESS_NUM_RUNS = 5
ABLATION_STRESS_VARIANTS = list(ABLATION_MAIN_VARIANTS)

# ---- 消融专用 XL 算例池（独立目录，不写入 large_scale/）----
ABLATION_XL_POOL_SUBDIR = 'ablation_xl_pool'
ABLATION_XL_ID_OFFSET = 100             # 实例 ID = 101, 102, … 避免与 1–60 冲突
ABLATION_XL_CONFIG_SEED = 20260821     # 仅用于生成 71-80 配置/实例
ABLATION_XL_TIER_SPECS = [
    (71, 80, 5),   # 压力测试池；可日后扩到 10
]
ABLATION_XL_TIER_LABELS = [f'{lo}-{hi}' for lo, hi, _ in ABLATION_XL_TIER_SPECS]

# 试点：每梯度 2 例；验证通过后设 ABLATION_PILOT=False
ABLATION_PILOT = False
ABLATION_PILOT_SAMPLE_PER_TIER = 2
ABLATION_PILOT_SUBDIR = 'ablation_sci_main_51_60_x5_pilot'

# Full 回归测试：三梯度代表例（与 5.3 对照，改消融前建议先跑）
ABLATION_REGRESSION_INSTANCE_IDS = [2, 25, 43]
ABLATION_REGRESSION_TOLERANCE = 0.01

# 算子诊断：默认关闭，避免公共求解热路径额外计数（5.3 Full 不受影响）
ABLATION_COLLECT_OPERATOR_STATS = False

# 消融并行：扁平池（A 方案）跨变体×算例×run；OOM 时将 FLAT_WORKERS 降至 4 或 2
ABLATION_FLAT_PARALLEL = True
ABLATION_FLAT_WORKERS = 6
ABLATION_PARALLEL_RUNS = False
ABLATION_PARALLEL_WORKERS = 3
ABLATION_PARALLEL_INSTANCES = False
ABLATION_INSTANCE_WORKERS = 2

# 表2：知识算子级消融 — 突出修船领域知识贡献
ABLATION_OPERATOR_VARIANTS = [
    'full',
    'wo_l1',
    'wo_l3',
    'wo_l4',
]
ABLATION_OPERATOR_SUBDIR = 'ablation_operators'
ABLATION_OPERATOR_SAMPLE_PER_TIER = 10
ABLATION_OPERATOR_TIER_FILTER = ['51-60']
ABLATION_OPERATOR_NUM_RUNS = 3

# 运行时解析（由 ABLATION_STUDY / ABLATION_PILOT 切换）
if ABLATION_STUDY == 'operators':
    ABLATION_VARIANTS_TO_RUN = list(ABLATION_OPERATOR_VARIANTS)
    ABLATION_OUTPUT_SUBDIR = ABLATION_OPERATOR_SUBDIR
    ABLATION_SAMPLE_PER_TIER = ABLATION_OPERATOR_SAMPLE_PER_TIER
    ABLATION_TIER_FILTER = list(ABLATION_OPERATOR_TIER_FILTER)
    ABLATION_NUM_RUNS = ABLATION_OPERATOR_NUM_RUNS
elif ABLATION_STUDY == 'stress':
    ABLATION_VARIANTS_TO_RUN = list(ABLATION_STRESS_VARIANTS)
    ABLATION_OUTPUT_SUBDIR = ABLATION_STRESS_SUBDIR
    ABLATION_SAMPLE_PER_TIER = ABLATION_STRESS_SAMPLE_PER_TIER
    ABLATION_TIER_FILTER = list(ABLATION_STRESS_TIER_FILTER)
    ABLATION_NUM_RUNS = ABLATION_STRESS_NUM_RUNS
elif ABLATION_STUDY == 'secondary':
    ABLATION_VARIANTS_TO_RUN = list(ABLATION_SECONDARY_VARIANTS)
    ABLATION_OUTPUT_SUBDIR = ABLATION_SECONDARY_SUBDIR
    ABLATION_SAMPLE_PER_TIER = ABLATION_SECONDARY_SAMPLE_PER_TIER
    ABLATION_TIER_FILTER = list(ABLATION_SECONDARY_TIER_FILTER)
    ABLATION_NUM_RUNS = ABLATION_SECONDARY_NUM_RUNS
elif ABLATION_STUDY == 'knowledge_trajectory':
    ABLATION_VARIANTS_TO_RUN = list(ABLATION_KNOWLEDGE_TRAJECTORY_VARIANTS)
    ABLATION_OUTPUT_SUBDIR = ABLATION_KNOWLEDGE_TRAJECTORY_SUBDIR
    ABLATION_SAMPLE_PER_TIER = ABLATION_KNOWLEDGE_TRAJECTORY_SAMPLE_PER_TIER
    ABLATION_TIER_FILTER = list(ABLATION_KNOWLEDGE_TRAJECTORY_TIER_FILTER)
    ABLATION_NUM_RUNS = ABLATION_KNOWLEDGE_TRAJECTORY_NUM_RUNS
    ABLATION_RECORD_CONVERGENCE = True
elif ABLATION_STUDY == 'knowledge_interaction':
    ABLATION_VARIANTS_TO_RUN = list(ABLATION_KNOWLEDGE_INTERACTION_VARIANTS)
    ABLATION_OUTPUT_SUBDIR = ABLATION_KNOWLEDGE_INTERACTION_SUBDIR
    ABLATION_SAMPLE_PER_TIER = ABLATION_KNOWLEDGE_INTERACTION_SAMPLE_PER_TIER
    ABLATION_TIER_FILTER = list(ABLATION_KNOWLEDGE_INTERACTION_TIER_FILTER)
    ABLATION_NUM_RUNS = ABLATION_KNOWLEDGE_INTERACTION_NUM_RUNS
else:
    ABLATION_VARIANTS_TO_RUN = list(ABLATION_MAIN_VARIANTS)
    ABLATION_NUM_RUNS = ABLATION_MAIN_NUM_RUNS
    ABLATION_TIER_FILTER = ABLATION_MAIN_TIER_FILTER
    if ABLATION_PILOT:
        ABLATION_OUTPUT_SUBDIR = ABLATION_PILOT_SUBDIR
        ABLATION_SAMPLE_PER_TIER = ABLATION_PILOT_SAMPLE_PER_TIER
    else:
        ABLATION_OUTPUT_SUBDIR = ABLATION_MAIN_SUBDIR
        ABLATION_SAMPLE_PER_TIER = ABLATION_MAIN_SAMPLE_PER_TIER


def ablation_paired_seed(instance_id, run_id):
    """各配置共用的配对种子：seed0 + 1000*i + r。"""
    seed0 = int(globals().get('ABLATION_PAIR_SEED0', 20260822) or 20260822)
    return int(seed0) + 1000 * int(instance_id) + int(run_id)
SENSITIVITY_PER_TIER = 3
# 小论文精简池：每梯度 3 例共 9（linspace）；覆盖 Light/Medium/Heavy
SENSITIVITY_INSTANCE_IDS = [
    1, 10, 20, 21, 30, 40, 41, 50, 60,
]
SENSITIVITY_BASE_TIME_LIMIT = 300
SENSITIVITY_BASE_MIP_SUB = 3
# 旧算法参数敏感性（已完成，勿覆盖）
SENSITIVITY_PROTOCOL = 'sensitivity_compact_9x3'
SENSITIVITY_NUM_RUNS = 3
SENSITIVITY_RESUME = True
SENSITIVITY_STUDIES_TO_RUN = [
    'S1_knowledge_weights',
    'S2_neighborhood_cap',
    'S3_repair_alpha',
]
SENSITIVITY_TIME_CLIP_FACTOR = 1.10  # time_valid：solve_time ≤ 1.10×时限

# ---- 新增：模型参数 λ 敏感性（独立协议目录，不碰 compact_9x3）----
SENSITIVITY_LAMBDA_PROTOCOL = 'sensitivity_lambda'
SENSITIVITY_LAMBDA_STUDIES = ['S4_lambda']
SENSITIVITY_LAMBDA_NUM_RUNS = 3
SENSITIVITY_LAMBDA_RESUME = True
# 敏感性并行（扁平任务池：level×算例×run；Windows 建议 ≤3，OOM 时改 1 或 2）
SENSITIVITY_PARALLEL = True
SENSITIVITY_PARALLEL_WORKERS = 3

# ---------------------------------------------------------------------------
# Destroy 自适应初始权重 profile（只改 destroy 选择先验，不改 repair）
# 层序 = DOMAIN_DESTROY_LAYERS:
#   [critical_task, random, skill_complexity, resource_conflict, diversity]
# Profile A baseline     : 等权
# Profile B knowledge_high: random0.30 / resource0.35 / critical0.25 / skill0.10
# Profile C stable       : random0.45 / resource0.30 / critical0.15 / skill0.10
# ---------------------------------------------------------------------------
DESTROY_PROFILE = 'baseline'  # 'baseline' | 'knowledge_high' | 'stable'
DESTROY_PROFILE_INIT_WEIGHTS = {
    'baseline': [1.0, 1.0, 1.0, 1.0, 1.0],
    #                L1crit  L2rand  L3skill L4res   diversity
    'knowledge_high': [0.25, 0.30, 0.10, 0.35, 0.0],
    'stable': [0.15, 0.45, 0.10, 0.30, 0.0],
}
DESTROY_PROFILE_LABELS = {
    'baseline': 'A_baseline',
    'knowledge_high': 'B_knowledge_high',
    'stable': 'C_stable',
}

# ---------------------------------------------------------------------------
# 试点测试（pilot_destroy）：15 代表性大规模实例 × 三 profile
# ---------------------------------------------------------------------------
PILOT_DESTROY_SEED = 20260803
PILOT_DESTROY_NUM_PER_TIER = 5
PILOT_DESTROY_TIME_LIMIT = 300
PILOT_DESTROY_NUM_RUNS = 3
PILOT_DESTROY_PROFILES = ('baseline', 'knowledge_high', 'stable')

# 旧 pilot_test 参数（保留兼容）
PILOT_TUNE_SEED = 20260803
PILOT_SCREEN_NUM_PER_TIER = 5
PILOT_SCREEN_TIME_LIMIT = 60
PILOT_SCREEN_NUM_RUNS = 3
PILOT_VALIDATE_PER_TIER = 3
PILOT_VALIDATE_TIME_LIMIT = 300
PILOT_VALIDATE_NUM_RUNS = 3
PILOT_VALIDATE_MIN_WINS = 7
PILOT_SCREEN_BUDGET_OVERRIDES = {
    'init_max_sec': 6,
    'num_starts': 6,
    'postprocess_reserve_sec': 3,
    'mid_intensify_budget_sec': 8,
    'domain_intensify_max_sec': 0.8,
}

# ---------------------------------------------------------------------------
# 大规模 KG-ALNS 配置（领域破坏主循环 · 同等墙钟 300s 充分求解）
# LARGE_MIP_ALNS_PROFILE:
#   'quality' — 论文三方对比：跑满预算，目标胜 GA/ALNS
#   'speed'   — 调试/快速批跑：早停，牺牲质量换墙钟
# ---------------------------------------------------------------------------
LARGE_MIP_NUM_RUNS = 3          # 三算法重复次数须一致；主口径 run-mean
LARGE_GA_NUM_RUNS = 3
LARGE_ALNS_NUM_RUNS = 3
LARGE_NUM_RUNS = LARGE_MIP_NUM_RUNS  # 向后兼容别名
LARGE_COMPARE_PRIMARY = 'mean'  # SCI 稳定性：run-mean 主口径
LARGE_BATCH_TIME_LIMIT = 300      # 充分求解：三算法同等墙钟 300s
LARGE_BASELINE_TIME_LIMIT = 300   # GA / ALNS 墙钟上限
LARGE_MIP_QUALITY_TIME_LIMIT = 600    # 不等预算辅表用（本主表关闭）
LARGE_ALNS_TIME_LIMIT = LARGE_BASELINE_TIME_LIMIT
LARGE_GA_TIME_LIMIT = LARGE_BASELINE_TIME_LIMIT
LARGE_COMPARE_GA = True
LARGE_COMPARE_ALNS = True
# True: 每例 KG∥GA∥ALNS 三进程同时跑（约省 2/3 墙钟；内存压力最大）
# False + LARGE_PARALLEL_BASELINES=True: 先 KG，再 GA∥ALNS
# 全 False: KG → GA → ALNS 全串行（最省内存）
# Windows 可能因 spawn+重载触发「页面文件太小」→ 自动回退
LARGE_PARALLEL_TRIPLE = True
LARGE_PARALLEL_BASELINES = True   # 三方并行失败时回退用；也可单独开 GA∥ALNS
LARGE_EQUAL_COMPUTE_BUDGET = True    # 定稿主表：三算法同等墙钟
# 同等墙钟：三算法均 300s
LARGE_MIP_TIME_LIMIT = (
    LARGE_BATCH_TIME_LIMIT if LARGE_EQUAL_COMPUTE_BUDGET
    else LARGE_MIP_QUALITY_TIME_LIMIT)
LARGE_MIP_SUB_TIME_LIMIT = 22 if not LARGE_EQUAL_COMPUTE_BUDGET else 3
LARGE_QUALITY_FIRST_MODE = False     # True 时 KG-ALNS 可长于基线（辅表）
MIP_ALNS_TIER1_USE_EXTENDED = False  # True: 长时限专用参数档
LARGE_TIER_APPLY_DOMINANCE_PROFILE = False  # run_large_tier*.py 定稿时 True
LARGE_STRICT_WALL_CLOCK = True       # True: KG-ALNS 全流程严格 ≤ batch_time_limit
LARGE_TYPICAL_WALL_FACTOR = 0.92     # 300s 充分求解，接近跑满
LARGE_MANUAL_SKIP_IDS = set()

# 阶段二按 3 个梯度分批求解（30-40 / 41-50 / 51-60 各 20 例），配合 PROMPT 每批需 yes 确认
LARGE_SPLIT_BY_TIER = True

MIP_ALNS_USE_TUNED_PROFILE = True
LARGE_MIP_ALNS_PROFILE = 'quality'   # 'quality' | 'speed'
# 写入结果 CSV，区分算法版本（勿与旧 V1 / 算法设计51 结果混目录）
# V1_kg_endgame = V1_min_weight_protect + 弱知识 repair(ΔC-α·Ki)
#               + skill-aware 终局精修 + elite restart
KG_ALNS_PROFILE_VERSION = 'V1_kg_endgame'

# --- 质量档（默认，论文三方对比）---
MIP_ALNS_LARGE_QUALITY = {
    # 主搜索预算在 init/热启动后按剩余墙钟分配；后处理不足时由尾段强化吃满 120s
    'init_max_sec': 6,
    'num_starts': 10,
    'alns_init_sec': 3,
    'alns_warm_sec': 18,
    'postprocess_reserve_sec': 8,
    'alns_bootstrap_ratio': 0.10,
    'alns_refine_ratio': 0.15,
    'bootstrap_mip_prob': 0.10,
    'refine_mip_prob': 0.25,
    'mip_repair_prob': 0.26,
    'stall_mip_threshold': 2,
    'mip_min_time_left': 4,
    'neighborhood_cap': 18,
    'domain_insert_limit': 22,
    'destroy_ratio_mult': 1.0,
    'use_full_sub_time': True,
    'iter_sub_cap': 18,
    'mip_sub_time_floor': 10,
    'sub_mip_gap': 0.005,
    'sub_mip_gap_large': 0.012,
    'polish': True,
    'polish_sub_cap': 20,
    'stall_early_exit': None,
}

# 解质量优先（MIP 长时限）：充分发挥 MIP 精确修复 + 多轮 Polish
MIP_ALNS_QUALITY_FIRST = {
    'init_max_sec': 6,
    'num_starts': 12,
    'alns_init_sec': 0,
    'alns_warm_sec': 0,
    'postprocess_reserve_sec': 25,
    'alns_bootstrap_ratio': 0.0,
    'alns_refine_ratio': 0.18,
    'bootstrap_mip_prob': 0.0,
    'refine_mip_prob': 0.48,
    'mip_repair_prob': 0.22,
    'mip_late_boost': 2.0,
    'stall_mip_threshold': 2,
    'mip_stall_force_after': 2,
    'always_compare_mip_on_stall': True,
    'mip_repair_n_min': 2,
    'mip_repair_n_max': 18,
    'mip_min_time_left': 4,
    'neighborhood_cap': 24,
    'domain_insert_limit': 28,
    'destroy_ratio_mult': 1.05,
    'use_full_sub_time': True,
    'iter_sub_cap': 22,
    'mip_sub_time_floor': 12,
    'sub_mip_gap': 0.002,
    'sub_mip_gap_large': 0.006,
    'polish': True,
    'polish_sub_cap': 35,
    'multi_polish': True,
    'polish_rounds': 5,
    'stall_early_exit': None,
    'max_mip_nei': 22,
    'mip_fix_space': True,
    'scale_aware_hybrid': False,
    'diversified_init': True,
    'multi_init_alns_sec': 4,
}

# 600s 解质量优先：长时限下加大 hybrid 迭代 + 更强 MIP 精修/Polish
MIP_ALNS_QUALITY_FIRST_LONG = {
    'postprocess_reserve_sec': 40,
    'alns_bootstrap_ratio': 0.06,
    'alns_refine_ratio': 0.22,
    'refine_mip_prob': 0.55,
    'mip_repair_prob': 0.16,
    'mip_late_boost': 2.4,
    'mip_repair_n_max': 22,
    'iter_sub_cap': 30,
    'mip_sub_time_floor': 16,
    'neighborhood_cap': 26,
    'domain_insert_limit': 30,
    'polish_sub_cap': 50,
    'polish_rounds': 6,
}

# 同等墙钟主表参数档（领域破坏 + 通用修复骨架；知识作 repair tie-break / skill refine）
# 墙钟秒数由 LARGE_BATCH_TIME_LIMIT=300 注入；勿开 MIP repair
KG_ALNS_EQUAL_BUDGET_120 = {
    'plan_b_alns_main': False,     # 领域破坏主循环（非方案B）
    'init_max_sec': 8,             # 300s 档：略增初解预算
    'num_starts': 8,
    'alns_warm_sec': 0,
    'postprocess_reserve_sec': 0,
    'alns_bootstrap_ratio': 0.10,
    'alns_refine_ratio': 0.10,
    'bootstrap_mip_prob': 0.0,
    'refine_mip_prob': 0.0,
    'mip_repair_prob': 0.0,
    'mip_late_boost': 1.0,
    'mip_only_on_stall': True,
    'mip_stall_force_after': 999,
    'always_compare_mip_on_stall': False,

    'intensify_mip': False,
    'mip_budget_cap_sec': 0,
    'mip_budget_frac': 0.0,
    'mip_deep_stall_threshold': 999,
    'mip_late_wall_frac': 0.99,
    'mip_allow_after_domain': False,
    'force_mid_mip_once': False,
    'intensify_mip_sec': 3,
    'intensify_mip_late_sec': 5,
    'intensify_unfix_space_late': False,
    'mip_cooldown_sec': 12,
    'mip_repair_n_min': 2,
    'mip_repair_n_max': 8,

    'critical_path_intensify': False,
    'critical_intensify_rounds': 0,
    'stall_mip_threshold': 6,
    'mid_intensify_budget_sec': 0,
    'domain_intensify_max_sec': 0.0,

    'mip_min_time_left': 4,
    'neighborhood_cap': 14,
    'domain_insert_limit': 18,
    'destroy_ratio_mult': 1.0,
    'use_full_sub_time': False,
    'equal_budget_tight_mip': False,
    'lock_neighborhood': True,
    'sub_time': 3,
    'iter_sub_cap': 3,
    'mip_sub_time_floor': 2,
    'sub_mip_gap': 0.02,
    'sub_mip_gap_large': 0.05,

    'polish': False,
    'polish_sub_cap': 0,
    'polish_max_nei': 8,
    'polish_unfix_space': False,
    'multi_polish': False,
    'polish_rounds': 1,
    'polish_from_archive': False,

    'stall_early_exit': None,
    'max_mip_nei': 8,
    'mip_fix_space': True,
    'diversified_init': True,
    'multi_init_alns_sec': 0,
    'log_mip_stats': True,
    'scale_aware_hybrid': False,
    'scale_aware_threshold': 9999,
    'scale_aware_alns_frac': 0.80,
    'scale_aware_mip_rounds': 3,
}

# V2：领域修复 + 动态邻域 + 有限预算关键路径强化（仍无 MIP，纯 KG-ALNS）
KG_ALNS_EQUAL_BUDGET_300 = {
    'plan_b_alns_main': False,

    # 初始解
    'init_max_sec': 12,
    'num_starts': 12,
    'diversified_init': True,
    'alns_warm_sec': 0,
    'multi_init_alns_sec': 0,

    # 搜索阶段
    'alns_bootstrap_ratio': 0.08,
    'alns_refine_ratio': 0.15,
    'postprocess_reserve_sec': 12,
    'bootstrap_mip_prob': 0.0,
    'refine_mip_prob': 0.0,

    # 邻域（lock=False 时由 _finalize_instance_scale 按任务数动态覆盖）
    'neighborhood_cap': 22,
    'domain_insert_limit': 30,
    'destroy_ratio_mult': 1.10,
    'lock_neighborhood': False,

    # 关键路径强化（停滞触发，300s 内约 ≤25s）
    'critical_path_intensify': True,
    'critical_intensify_rounds': 3,
    'domain_intensify_max_sec': 1.5,
    'mid_intensify_budget_sec': 25,
    'stall_mip_threshold': 6,

    # 暂不使用 MIP（保持纯 KG-ALNS，非 Matheuristic）
    'mip_repair_prob': 0.0,
    'mip_late_boost': 1.0,
    'mip_only_on_stall': True,
    'mip_stall_force_after': 999,
    'always_compare_mip_on_stall': False,
    'intensify_mip': False,
    'mip_budget_cap_sec': 0,
    'mip_budget_frac': 0.0,
    'mip_deep_stall_threshold': 999,
    'mip_late_wall_frac': 0.99,
    'mip_allow_after_domain': False,
    'force_mid_mip_once': False,
    'intensify_mip_sec': 3,
    'intensify_mip_late_sec': 5,
    'intensify_unfix_space_late': False,
    'mip_cooldown_sec': 12,
    'mip_repair_n_min': 2,
    'mip_repair_n_max': 8,
    'mip_min_time_left': 4,

    'use_full_sub_time': False,
    'equal_budget_tight_mip': False,
    'sub_time': 3,
    'iter_sub_cap': 3,
    'mip_sub_time_floor': 2,
    'sub_mip_gap': 0.02,
    'sub_mip_gap_large': 0.05,

    # 后处理
    'polish': False,
    'polish_sub_cap': 0,
    'polish_max_nei': 8,
    'polish_unfix_space': False,
    'multi_polish': False,
    'polish_rounds': 1,
    'polish_from_archive': False,
    'stall_early_exit': None,

    'max_mip_nei': 8,
    'mip_fix_space': True,
    'log_mip_stats': True,
    'scale_aware_hybrid': False,
    'scale_aware_threshold': 9999,
    'scale_aware_alns_frac': 0.80,
    'scale_aware_mip_rounds': 3,
}

# 兼容旧名：主表 = 原始 KG-ALNS（领域破坏 + 通用修复；轻量 destroy 权重实验）
MIP_ALNS_EQUAL_BUDGET_120 = KG_ALNS_EQUAL_BUDGET_120

# 兼容旧名：与主表完全相同（不再分档）
MIP_ALNS_EQUAL_BUDGET_120_HEAVY = dict(MIP_ALNS_EQUAL_BUDGET_120)

# 兼容旧名：同等墙钟档 = EQUAL_BUDGET
MIP_ALNS_TIER1_DOMINANCE = dict(MIP_ALNS_EQUAL_BUDGET_120)
MIP_ALNS_TIER1_EXTENDED = dict(MIP_ALNS_EQUAL_BUDGET_120)

# 试点筛选三配置（预算表 + 消融开关；勿与正式 60 例混目录）
# V1=原始；V2=仅开领域修复；V3=领域修复+动态邻域+关键路径
def _pilot_kg_profile_specs():
    v1_flags = {
        'use_mip_repair': False,
        'use_l1_destroy': True,
        'use_l3_destroy': True,
        'use_l4_destroy': True,
        'use_adaptive_weights': True,
        'use_domain_repair': False,
        'use_knowledge_eval': True,
        'use_knowledge_repair': True,
        'use_polish': False,
        'use_team_swap': False,
        'use_skill_refine': True,
        'use_elite_restart': True,
    }
    v2_flags = dict(v1_flags)
    v2_flags.update({'use_domain_repair': True, 'use_team_swap': True})
    v3_flags = dict(v2_flags)
    return {
        'V1': {
            'label': '原始KG-ALNS',
            'budget': dict(KG_ALNS_EQUAL_BUDGET_120),
            'flags': v1_flags,
        },
        'V2': {
            'label': '领域修复',
            'budget': dict(KG_ALNS_EQUAL_BUDGET_120),
            'flags': v2_flags,
        },
        'V3': {
            'label': '领域修复+动态邻域+关键路径',
            'budget': dict(KG_ALNS_EQUAL_BUDGET_300),
            'flags': v3_flags,
        },
    }


MIP_ALNS_LARGE_QUALITY_HEAVY = {
    **MIP_ALNS_LARGE_QUALITY,
    'init_max_sec': 15,
    'num_starts': 10,
    'alns_warm_sec': 28,
    'mip_repair_prob': 0.22,
    'refine_mip_prob': 0.22,
    'neighborhood_cap': 16,
    'iter_sub_cap': 14,
    'polish_sub_cap': 28,
}

# --- 速度档（调试批跑，Case1 约 31s/次，质量通常劣于 ALNS）---
MIP_ALNS_LARGE_TUNED = {
    'init_max_sec': 5,
    'num_starts': 5,
    'alns_bootstrap_ratio': 0.0,
    'alns_refine_ratio': 0.04,
    'alns_init_sec': 0,
    'alns_warm_sec': 0,
    'bootstrap_mip_prob': 0.0,
    'refine_mip_prob': 0.0,
    'mip_repair_prob': 0.24,
    'stall_mip_threshold': 2,
    'mip_min_time_left': 10,
    'neighborhood_cap': 9,
    'domain_insert_limit': 15,
    'destroy_ratio_mult': 1.2,
    'use_full_sub_time': False,
    'iter_sub_cap': 6,
    'mip_sub_time_floor': 6,
    'sub_mip_gap': 0.012,
    'sub_mip_gap_large': 0.028,
    'polish': False,
    'polish_sub_cap': 0,
    'stall_early_exit': 100,
}

MIP_ALNS_LARGE_TUNED_HEAVY = {
    **MIP_ALNS_LARGE_TUNED,
    'init_max_sec': 7,
    'num_starts': 5,
    'mip_repair_prob': 0.28,
    'neighborhood_cap': 11,
    'iter_sub_cap': 7,
    'sub_mip_gap_large': 0.032,
    'stall_early_exit': 120,
}

# ---------------------------------------------------------------------------
# 消融实验变体（第五章 §5.4；围绕知识引导创新模块）
# 命名：KG-ALNS / KG-ALNS^{-K,-D,-R,-A,-E}
# ---------------------------------------------------------------------------
ABLATION_VARIANTS = {
    'full': {
        'label': 'KG-ALNS (完整)',
        'paper_name': 'KG-ALNS',
        'flags': {},
        'modules': {
            'knowledge_eval': True,
            'knowledge_destroy': True,
            'knowledge_repair': True,
            'adaptive': True,
            'elite_restart': True,
            'skill_refine': True,
            'endgame': True,
            'destroy_l1': True,
            'destroy_l3': True,
            'destroy_l4': True,
        },
    },
    'wo_knowledge': {
        # 单因素补充：仅关闭知识评价 Ki；保留弱知识 repair 骨架
        'label': 'KG-ALNS without knowledge evaluation',
        'paper_name': 'w/o K',
        'flags': {
            'use_knowledge_eval': False,
        },
        'modules': {
            'knowledge_eval': False,
            'knowledge_destroy': True,
            'knowledge_repair': True,
            'adaptive': True,
            'elite_restart': True,
            'skill_refine': True,
            'endgame': True,
        },
    },
    'wo_knowledge_guided': {
        # SCI 主消融 / 轨迹实验：知识置零但保留同一 repair 骨架与算子槽位
        'label': 'KG-ALNS without knowledge-guided search',
        'paper_name': 'w/o Knowledge',
        'flags': {
            'use_knowledge_eval': False,
            'use_knowledge_repair': True,       # 必须保留同一修复函数
            'knowledge_repair_alpha': 0.0,      # 仅置零知识贡献
            'use_knowledge_priority': False,    # L1/L3/L4 仍被选中，但用随机破坏替代知识优先级
        },
        'modules': {
            'knowledge_eval': False,
            'knowledge_destroy': False,
            'knowledge_repair': True,   # 骨架保留；贡献为 0
            'adaptive': True,
            'elite_restart': True,
            'skill_refine': True,
            'endgame': True,
            'destroy_l1': True,
            'destroy_l3': True,
            'destroy_l4': True,
        },
    },
    'wo_knowledge_no_elite': {
        # Knowledge×Elite 交互：关 Knowledge + 关 Elite（其余与 w/o Knowledge 对齐）
        'label': 'KG-ALNS without knowledge and elite restart',
        'paper_name': 'w/o K+E',
        'flags': {
            'use_knowledge_eval': False,
            'use_knowledge_repair': True,
            'knowledge_repair_alpha': 0.0,
            'use_knowledge_priority': False,
            'use_elite_restart': False,
            'use_skill_refine': True,
            'use_team_swap': False,
        },
        'modules': {
            'knowledge_eval': False,
            'knowledge_destroy': False,
            'knowledge_repair': True,
            'adaptive': True,
            'elite_restart': False,
            'skill_refine': True,
            'endgame': False,
            'destroy_l1': True,
            'destroy_l3': True,
            'destroy_l4': True,
        },
    },
    'wo_destroy': {
        'label': 'KG-ALNS without knowledge destroy',
        'paper_name': 'w/o D',
        'flags': {
            'use_l1_destroy': False,
            'use_l3_destroy': False,
            'use_l4_destroy': False,
            # 仅保留 L2 random + diversity
        },
        'modules': {
            'knowledge_eval': True,
            'knowledge_destroy': False,
            'knowledge_repair': True,
            'adaptive': True,
            'elite_restart': True,
            'skill_refine': True,
            'endgame': True,
        },
    },
    'wo_repair': {
        # 旧变体：直接关 use_knowledge_repair（会切换修复路径；主消融勿用）
        'label': 'KG-ALNS without knowledge repair',
        'paper_name': 'w/o R',
        'flags': {
            'use_knowledge_repair': False,
        },
        'modules': {
            'knowledge_eval': True,
            'knowledge_destroy': True,
            'knowledge_repair': False,
            'adaptive': True,
            'elite_restart': True,
            'skill_refine': True,
            'endgame': True,
        },
    },
    'wo_repair_alpha0': {
        # 干净对照：α=0 + 保留 repair 骨架（推荐替代旧 wo_repair）
        'label': 'KG-ALNS with knowledge repair alpha=0',
        'paper_name': 'w/o R(α=0)',
        'flags': {
            'use_knowledge_repair': True,
            'knowledge_repair_alpha': 0.0,
        },
        'modules': {
            'knowledge_eval': True,
            'knowledge_destroy': True,
            'knowledge_repair': True,
            'adaptive': True,
            'elite_restart': True,
            'skill_refine': True,
            'endgame': True,
        },
    },
    'wo_adaptive': {
        'label': 'KG-ALNS without adaptive weights',
        'paper_name': 'w/o Adaptive',
        'flags': {
            'use_adaptive_weights': False,
        },
        'modules': {
            'knowledge_eval': True,
            'knowledge_destroy': True,
            'knowledge_repair': True,
            'adaptive': False,
            'elite_restart': True,
            'skill_refine': True,
            'endgame': True,
        },
    },
    'wo_elite_restart': {
        'label': 'KG-ALNS without elite restart',
        'paper_name': 'w/o Elite',
        'flags': {
            'use_elite_restart': False,
        },
        'modules': {
            'knowledge_eval': True,
            'knowledge_destroy': True,
            'knowledge_repair': True,
            'adaptive': True,
            'elite_restart': False,
            'skill_refine': True,
            'endgame': True,
        },
    },
    'wo_skill_refine': {
        # 次级 Endgame 拆分：关 skill refine；team_swap 一并关闭（与 Full 对齐，Full 默认本就 False）
        'label': 'KG-ALNS without skill refine',
        'paper_name': 'w/o Skill',
        'flags': {
            'use_skill_refine': False,
            'use_team_swap': False,
        },
        'modules': {
            'knowledge_eval': True,
            'knowledge_destroy': True,
            'knowledge_repair': True,
            'adaptive': True,
            'elite_restart': True,
            'skill_refine': False,
            'endgame': True,
        },
    },
    'wo_endgame': {
        # SCI 主消融：关闭终局强化；墙钟仍 300s，剩余时间留给主搜索
        'label': 'KG-ALNS without endgame reinforcement',
        'paper_name': 'w/o Endgame',
        'flags': {
            'use_elite_restart': False,
            'use_skill_refine': False,
            'use_team_swap': False,
        },
        'modules': {
            'knowledge_eval': True,
            'knowledge_destroy': True,
            'knowledge_repair': True,
            'adaptive': True,
            'elite_restart': False,
            'skill_refine': False,
            'endgame': False,
            'destroy_l1': True,
            'destroy_l3': True,
            'destroy_l4': True,
        },
    },
    # ---- 表2：知识算子级消融（L1 / L3 / L4 分别关闭）----
    'wo_l1': {
        'label': 'KG-ALNS^{-L1} 无关键路径破坏',
        'paper_name': 'KG-ALNS^{-L1}',
        'flags': {'use_l1_destroy': False},
        'modules': {
            'knowledge_eval': True,
            'knowledge_destroy': True,
            'knowledge_repair': True,
            'adaptive': True,
            'elite_restart': True,
            'skill_refine': True,
            'endgame': True,
            'destroy_l1': False,
            'destroy_l3': True,
            'destroy_l4': True,
        },
    },
    'wo_l3': {
        'label': 'KG-ALNS^{-L3} 无技能-船坞链破坏',
        'paper_name': 'KG-ALNS^{-L3}',
        'flags': {'use_l3_destroy': False},
        'modules': {
            'knowledge_eval': True,
            'knowledge_destroy': True,
            'knowledge_repair': True,
            'adaptive': True,
            'elite_restart': True,
            'skill_refine': True,
            'endgame': True,
            'destroy_l1': True,
            'destroy_l3': False,
            'destroy_l4': True,
        },
    },
    'wo_l4': {
        'label': 'KG-ALNS^{-L4} 无瓶颈资源破坏',
        'paper_name': 'KG-ALNS^{-L4}',
        'flags': {'use_l4_destroy': False},
        'modules': {
            'knowledge_eval': True,
            'knowledge_destroy': True,
            'knowledge_repair': True,
            'adaptive': True,
            'elite_restart': True,
            'skill_refine': True,
            'endgame': True,
            'destroy_l1': True,
            'destroy_l3': True,
            'destroy_l4': False,
        },
    },
    # ---- 兼容旧变体键（不进默认运行列表）----
    'knowledge_off': {
        'label': '仅通用扰动 (L2+diversity)',
        'paper_name': 'knowledge_off',
        'flags': {
            'use_l1_destroy': False,
            'use_l3_destroy': False,
            'use_l4_destroy': False,
            'use_domain_repair': False,
            'use_knowledge_eval': False,
            'use_knowledge_repair': False,
            'use_skill_refine': False,
            'use_team_swap': False,
            'use_elite_restart': False,
        },
        'modules': {
            'knowledge_eval': False,
            'knowledge_destroy': False,
            'knowledge_repair': False,
            'adaptive': True,
            'elite_restart': False,
            'skill_refine': False,
            'endgame': False,
        },
    },
    'wo_knowledge_repair': {
        'label': 'w/o 弱知识repair',
        'paper_name': 'wo_knowledge_repair',
        'flags': {'use_knowledge_repair': False},
        'modules': {
            'knowledge_eval': True,
            'knowledge_destroy': True,
            'knowledge_repair': False,
            'adaptive': True,
            'elite_restart': True,
            'skill_refine': True,
            'endgame': True,
        },
    },
    'v1_min_weight_only': {
        'label': '仅V1权重保护(关三模块)',
        'paper_name': 'v1_min_weight_only',
        'flags': {
            'use_knowledge_eval': False,
            'use_knowledge_repair': False,
            'use_skill_refine': False,
            'use_elite_restart': False,
        },
        'modules': {
            'knowledge_eval': False,
            'knowledge_destroy': True,
            'knowledge_repair': False,
            'adaptive': True,
            'elite_restart': False,
            'skill_refine': False,
            'endgame': False,
        },
    },
}


def load_completed_ablation_ids(output_dir, variant_key, rows=None):
    """读取某消融变体已成功完成的算例 ID（objective 有效）"""
    if rows is not None:
        done = set()
        for r in rows:
            if r.get('variant') != variant_key:
                continue
            obj = r.get('objective')
            if obj is None or (isinstance(obj, float) and pd.isna(obj)):
                continue
            done.add(int(r['instance_id']))
        return done
    csv_path = os.path.join(output_dir, 'ablation_results.csv')
    if not os.path.isfile(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        if df.empty or 'variant' not in df.columns:
            return set()
        sub = df[df['variant'] == variant_key]
        done = set()
        for _, r in sub.iterrows():
            obj = r.get('objective')
            if obj is None or (isinstance(obj, float) and pd.isna(obj)):
                continue
            done.add(int(r['instance_id']))
        return done
    except Exception:
        return set()


def filter_ablation_rows_to_instances(rows, instances):
    """丢弃与当前算例池 num_ships/num_tasks 不一致的旧记录（算例规模变更后）。"""
    inst_map = {int(i['instance_id']): i for i in instances}
    valid, dropped = [], 0
    for r in rows:
        iid = int(r['instance_id'])
        inst = inst_map.get(iid)
        if inst is None:
            dropped += 1
            continue
        if (int(r.get('num_ships', -1)) != int(inst['num_ships'])
                or int(r.get('num_tasks', -1)) != int(inst['num_tasks'])):
            dropped += 1
            continue
        valid.append(r)
    return valid, dropped


def import_full_baseline_from_large_scale(large_dir, instances=None):
    """从阶段二 large_scale_results.csv 导入 KG-ALNS(完整) 作为消融 V0 基准"""
    csv_path = os.path.join(large_dir, 'large_scale_results.csv')
    if not os.path.isfile(csv_path):
        return []
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception:
        return []
    if df.empty or 'instance_id' not in df.columns:
        return []
    inst_map = {int(i['instance_id']): i for i in (instances or [])}
    spec = ABLATION_VARIANTS['full']
    rows = []
    skipped = 0
    for _, r in df.iterrows():
        iid = int(r['instance_id'])
        if inst_map:
            inst = inst_map.get(iid)
            if inst is None:
                skipped += 1
                continue
            if (int(r.get('num_ships', -1)) != int(inst['num_ships'])
                    or int(r.get('num_tasks', -1)) != int(inst['num_tasks'])):
                skipped += 1
                continue
            tier = inst.get('config', {}).get('tier', r.get('tier', ''))
        else:
            tier = r.get('tier', '')
        obj = r.get('mip_lns_obj_mean')
        if obj is None or (isinstance(obj, float) and pd.isna(obj)):
            obj = r.get('mip_lns_obj_best')
        if obj is None or (isinstance(obj, float) and pd.isna(obj)):
            obj = r.get('heuristic_obj')
        if obj is None or (isinstance(obj, float) and pd.isna(obj)):
            continue
        rows.append({
            'variant': 'full',
            'variant_label': spec['label'],
            'paper_name': spec.get('paper_name', 'KG-ALNS'),
            'instance_id': iid,
            'num_ships': inst_map[iid]['num_ships'] if inst_map else r.get('num_ships'),
            'num_tasks': inst_map[iid]['num_tasks'] if inst_map else r.get('num_tasks'),
            'tier': tier,
            'objective': float(obj),
            'solve_time': r.get('mip_lns_time') or r.get('heuristic_time'),
            'mip_repair_count': r.get('mip_repair_count'),
            'gap_vs_full_pct': 0.0,
            'source': 'large_scale_comparison',
        })
    if skipped:
        print(f"[消融] 跳过 {skipped} 条 large_scale 导入（与当前算例规模不一致）")
    return rows


def import_full_baseline_from_ablation_dir(ablation_src_dir, instances=None):
    """从已完成的消融目录导入 full 行（次级 Endgame 拆分复用 SCI 主消融 Full）。

    要求源目录 ablation_results.csv 中 variant=='full'，且与当前实例 ID/规模一致。
    """
    csv_path = os.path.join(ablation_src_dir, 'ablation_results.csv')
    if not os.path.isfile(csv_path):
        print(f"[消融] 未找到可导入的 Full：{csv_path}", flush=True)
        return []
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception as e:
        print(f"[消融] 读取 Full 源失败: {e}", flush=True)
        return []
    if df.empty or 'variant' not in df.columns:
        return []
    full_df = df[df['variant'].astype(str) == 'full'].copy()
    if full_df.empty:
        return []
    inst_map = {int(i['instance_id']): i for i in (instances or [])}
    spec = ABLATION_VARIANTS['full']
    rows = []
    skipped = 0
    for _, r in full_df.iterrows():
        iid = int(r['instance_id'])
        obj = r.get('objective')
        if obj is None or (isinstance(obj, float) and pd.isna(obj)):
            skipped += 1
            continue
        if inst_map:
            inst = inst_map.get(iid)
            if inst is None:
                skipped += 1
                continue
            if ('num_ships' in r and not pd.isna(r.get('num_ships'))
                    and int(r['num_ships']) != int(inst['num_ships'])):
                skipped += 1
                continue
            tier = inst.get('config', {}).get('tier', r.get('tier', ''))
            n_ships = inst['num_ships']
            n_tasks = inst['num_tasks']
        else:
            tier = r.get('tier', '')
            n_ships = r.get('num_ships')
            n_tasks = r.get('num_tasks')
        row = {
            'variant': 'full',
            'variant_label': spec['label'],
            'paper_name': spec.get('paper_name', 'KG-ALNS'),
            'instance_id': iid,
            'num_ships': n_ships,
            'num_tasks': n_tasks,
            'tier': tier,
            'objective': float(obj),
            'objective_std': r.get('objective_std'),
            'objective_best': r.get('objective_best'),
            'objective_runs': r.get('objective_runs'),
            'seed_runs': r.get('seed_runs'),
            'feasible_runs': r.get('feasible_runs'),
            'all_runs_feasible': r.get('all_runs_feasible'),
            'n_feasible_runs': r.get('n_feasible_runs'),
            'solve_time': r.get('solve_time'),
            'mip_repair_count': r.get('mip_repair_count'),
            'gap_vs_full_pct': 0.0,
            'source': f'imported_full:{os.path.basename(ablation_src_dir.rstrip(os.sep))}',
            'eval_protocol': r.get('eval_protocol', 'report_objective_from_schedule'),
        }
        rows.append(row)
    print(f"[消融] 从 {ablation_src_dir} 导入 full {len(rows)} 例"
          f"（跳过 {skipped}）", flush=True)
    return rows


def import_ablation_variant_from_dir(var_key, ablation_src_dir, instances=None):
    """从已完成消融目录导入指定变体行（交互实验复用主/次级结果）。"""
    if var_key not in ABLATION_VARIANTS:
        return []
    csv_path = os.path.join(ablation_src_dir, 'ablation_results.csv')
    if not os.path.isfile(csv_path):
        print(f"[消融] 未找到可导入变体 {var_key}: {csv_path}", flush=True)
        return []
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception as e:
        print(f"[消融] 读取 {var_key} 源失败: {e}", flush=True)
        return []
    if df.empty or 'variant' not in df.columns:
        return []
    sub = df[df['variant'].astype(str) == str(var_key)].copy()
    if sub.empty:
        return []
    inst_map = {int(i['instance_id']): i for i in (instances or [])}
    spec = ABLATION_VARIANTS[var_key]
    rows = []
    skipped = 0
    for _, r in sub.iterrows():
        iid = int(r['instance_id'])
        obj = r.get('objective')
        if obj is None or (isinstance(obj, float) and pd.isna(obj)):
            skipped += 1
            continue
        if inst_map:
            inst = inst_map.get(iid)
            if inst is None:
                skipped += 1
                continue
            if ('num_ships' in r and not pd.isna(r.get('num_ships'))
                    and int(r['num_ships']) != int(inst['num_ships'])):
                skipped += 1
                continue
            tier = inst.get('config', {}).get('tier', r.get('tier', ''))
            n_ships, n_tasks = inst['num_ships'], inst['num_tasks']
        else:
            tier = r.get('tier', '')
            n_ships, n_tasks = r.get('num_ships'), r.get('num_tasks')
        gap = float(r.get('gap_vs_full_pct', 0) or 0)
        if var_key == 'full':
            gap = 0.0
        row = {
            'variant': var_key,
            'variant_label': spec['label'],
            'paper_name': spec.get('paper_name', var_key),
            'instance_id': iid,
            'num_ships': n_ships,
            'num_tasks': n_tasks,
            'tier': tier,
            'objective': float(obj),
            'objective_std': r.get('objective_std'),
            'objective_best': r.get('objective_best'),
            'objective_runs': r.get('objective_runs'),
            'seed_runs': r.get('seed_runs'),
            'feasible_runs': r.get('feasible_runs'),
            'all_runs_feasible': r.get('all_runs_feasible'),
            'n_feasible_runs': r.get('n_feasible_runs'),
            'solve_time': r.get('solve_time'),
            'mip_repair_count': r.get('mip_repair_count'),
            'total_flow': r.get('total_flow'),
            'misallocation_count': r.get('misallocation_count'),
            'misallocation_penalty': r.get('misallocation_penalty'),
            'dock_misuse_count': r.get('dock_misuse_count'),
            'gap_vs_full_pct': gap,
            'source': f'imported:{var_key}:{os.path.basename(ablation_src_dir.rstrip(os.sep))}',
            'eval_protocol': r.get('eval_protocol', 'report_objective_from_schedule'),
        }
        rows.append(row)
    print(f"[消融] 从 {ablation_src_dir} 导入 {var_key} {len(rows)} 例"
          f"（跳过 {skipped}）", flush=True)
    return rows


def _parse_ablation_json_field(val, default=None):
    """CSV 续跑后 JSON 字段可能是 str / dict / NaN。"""
    if default is None:
        default = {}
    if val is None:
        return default
    try:
        if isinstance(val, float) and pd.isna(val):
            return default
    except Exception:
        pass
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() == 'none':
            return default
        try:
            return json.loads(s)
        except Exception:
            return default
    return default


def _ablation_paired_wilcoxon(sample_a, sample_b):
    """配对 Wilcoxon（最小化目标：a vs b 的差 a-b）。"""
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    d = a[mask] - b[mask]
    d = d[np.abs(d) > 1e-12]
    if len(d) < 5:
        return None
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(d, alternative='two-sided', zero_method='wilcox')
        return {
            'statistic': float(stat), 'p_value': float(p),
            'mean_diff': float(d.mean()), 'n': int(len(d)), 'method': 'wilcoxon',
        }
    except Exception:
        pass
    n_pos = int((d > 0).sum())
    n_neg = int((d < 0).sum())
    n_eff = n_pos + n_neg
    if n_eff < 5:
        return None
    k = min(n_pos, n_neg)
    p_one = sum(math.comb(n_eff, i) * (0.5 ** n_eff) for i in range(k + 1))
    return {
        'statistic': float(k), 'p_value': float(min(1.0, 2 * p_one)),
        'mean_diff': float(d.mean()), 'n': int(n_eff), 'method': 'sign_test',
    }


def _holm_adjust(p_values):
    """Holm–Bonferroni 校正；返回与输入同序的校正 p。"""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adj = [1.0] * m
    running = 0.0
    for rank, (idx, p) in enumerate(indexed):
        corr = (m - rank) * float(p)
        running = max(running, corr)
        adj[idx] = min(1.0, running)
    return adj


def _ablation_friedman_test(matrix):
    """Friedman 检验。matrix: shape (n_instances, n_variants)，目标越小越好。"""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        return None
    n, k = arr.shape
    if n < 3 or k < 3:
        return None
    # 逐行排名（并列取平均秩）
    ranks = np.zeros_like(arr)
    for i in range(n):
        row = arr[i]
        order = np.argsort(row)
        sorted_vals = row[order]
        raw_ranks = np.empty(k, dtype=float)
        j = 0
        while j < k:
            t = j
            while t + 1 < k and abs(sorted_vals[t + 1] - sorted_vals[j]) < 1e-12:
                t += 1
            avg_r = 0.5 * (j + 1 + t + 1)
            for u in range(j, t + 1):
                raw_ranks[order[u]] = avg_r
            j = t + 1
        ranks[i] = raw_ranks
    rj = ranks.sum(axis=0)
    chi2 = (12.0 / (n * k * (k + 1.0))) * float(np.sum(rj ** 2)) - 3.0 * n * (k + 1.0)
    df = k - 1
    try:
        from scipy.stats import chi2 as chi2_dist
        p = float(chi2_dist.sf(chi2, df))
    except Exception:
        # 粗略：大 df 用正态近似不可靠；无 scipy 时仅返回统计量
        p = None
    return {
        'statistic': float(chi2), 'df': int(df), 'p_value': p, 'n': int(n), 'k': int(k),
        'mean_ranks': {i: float(rj[i] / n) for i in range(k)},
    }


def build_ablation_summary(df, variants):
    """生成消融汇总（总体 + 分梯度 + 配对 PD / 胜平负）。"""
    summary_rows = []
    tier_rows = []
    # 梯度顺序：阶段二标签 → 消融 XL 标签 → 数据中出现的其余 tier
    tier_order = list(LARGE_SCALE_TIER_LABELS)
    for t in globals().get('ABLATION_XL_TIER_LABELS', ()) or ():
        if t not in tier_order:
            tier_order.append(t)
    if df is not None and not df.empty and 'tier' in df.columns:
        for t in sorted(df['tier'].dropna().astype(str).unique()):
            if t and t not in tier_order:
                tier_order.append(t)
    full_df = df[df['variant'] == 'full'].copy() if not df.empty else pd.DataFrame()
    full_map = {}
    if not full_df.empty:
        for _, r in full_df.iterrows():
            if r.get('objective') is None or (
                    isinstance(r.get('objective'), float) and pd.isna(r.get('objective'))):
                continue
            full_map[int(r['instance_id'])] = float(r['objective'])

    for var_key in variants:
        if var_key not in ABLATION_VARIANTS:
            continue
        sub = df[df['variant'] == var_key]
        if sub.empty:
            continue
        var_avg = sub['objective'].mean()
        mods = ABLATION_VARIANTS[var_key].get('modules') or {}

        # 逐实例配对 PD / 胜平负（相对 full；最小化）
        pds, n_worse, n_tie, n_better = [], 0, 0, 0
        for _, r in sub.iterrows():
            obj = r.get('objective')
            if obj is None or (isinstance(obj, float) and pd.isna(obj)):
                continue
            base = full_map.get(int(r['instance_id']))
            if base is None or base <= 0:
                continue
            pd_i = (float(obj) - base) / base * 100.0
            pds.append(pd_i)
            if pd_i > 0.01:
                n_worse += 1
            elif pd_i < -0.01:
                n_better += 1
            else:
                n_tie += 1

        n_feas_ok = None
        if 'all_runs_feasible' in sub.columns:
            try:
                n_feas_ok = int(sub['all_runs_feasible'].astype(bool).sum())
            except Exception:
                n_feas_ok = None

        summary_rows.append({
            'variant': var_key,
            'label': ABLATION_VARIANTS[var_key]['label'],
            'paper_name': ABLATION_VARIANTS[var_key].get(
                'paper_name', ABLATION_VARIANTS[var_key]['label']),
            'knowledge_eval': 'Y' if mods.get('knowledge_eval') else 'N',
            'knowledge_destroy': 'Y' if mods.get('knowledge_destroy') else 'N',
            'knowledge_repair': 'Y' if mods.get('knowledge_repair') else 'N',
            'adaptive': 'Y' if mods.get('adaptive') else 'N',
            'elite_restart': ('-' if 'elite_restart' not in mods
                              else ('Y' if mods['elite_restart'] else 'N')),
            'skill_refine': ('-' if 'skill_refine' not in mods
                             else ('Y' if mods['skill_refine'] else 'N')),
            'endgame': 'Y' if mods.get('endgame') else 'N',
            'destroy_l1': ('-' if 'destroy_l1' not in mods
                           else ('Y' if mods['destroy_l1'] else 'N')),
            'destroy_l3': ('-' if 'destroy_l3' not in mods
                           else ('Y' if mods['destroy_l3'] else 'N')),
            'destroy_l4': ('-' if 'destroy_l4' not in mods
                           else ('Y' if mods['destroy_l4'] else 'N')),
            'n_cases': len(sub),
            'n_all_runs_feasible': n_feas_ok,
            'avg_objective': var_avg,
            'avg_time_s': sub['solve_time'].mean(),
            'avg_mip_repairs': sub['mip_repair_count'].mean()
            if 'mip_repair_count' in sub.columns else None,
            'mean_pd_vs_full_pct': float(np.mean(pds)) if pds else None,
            'median_pd_vs_full_pct': float(np.median(pds)) if pds else None,
            'avg_gap_vs_full_pct': float(np.mean(pds)) if pds else None,
            'n_worse_than_full': n_worse,
            'n_tie_vs_full': n_tie,
            'n_better_than_full': n_better,
            'wtl_vs_full': f'{n_worse}-{n_tie}-{n_better}',
        })
        for tier in tier_order:
            tsub = sub[sub['tier'] == tier]
            if tsub.empty:
                continue
            t_pds = []
            for _, r in tsub.iterrows():
                obj = r.get('objective')
                if obj is None or (isinstance(obj, float) and pd.isna(obj)):
                    continue
                base = full_map.get(int(r['instance_id']))
                if base is None or base <= 0:
                    continue
                t_pds.append((float(obj) - base) / base * 100.0)
            t_avg = tsub['objective'].mean()
            tier_rows.append({
                'variant': var_key,
                'label': ABLATION_VARIANTS[var_key]['label'],
                'paper_name': ABLATION_VARIANTS[var_key].get(
                    'paper_name', ABLATION_VARIANTS[var_key]['label']),
                'tier': tier,
                'n_cases': len(tsub),
                'avg_objective': t_avg,
                'mean_pd_vs_full_pct': float(np.mean(t_pds)) if t_pds else None,
                'avg_gap_vs_full_pct': float(np.mean(t_pds)) if t_pds else None,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(tier_rows)


def _rank_biserial_from_diffs(diffs):
    """配对秩二列相关系数（Kerby 2014）：(R+ - R-) / (R+ + R-)。

    diffs = variant - full；最小化问题中 diffs>0 表示变体更差。
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    d = d[np.abs(d) > 1e-12]
    n = len(d)
    if n < 1:
        return None
    try:
        from scipy.stats import rankdata
        ranks = rankdata(np.abs(d))
    except Exception:
        order = np.argsort(np.abs(d))
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1, dtype=float)
    r_pos = float(ranks[d > 0].sum())
    r_neg = float(ranks[d < 0].sum())
    denom = r_pos + r_neg
    if denom <= 0:
        return 0.0
    return (r_pos - r_neg) / denom


def _bootstrap_mean_ci(values, n_boot=5000, alpha=0.05, seed=42):
    """实例级指标均值的百分位 bootstrap 95% CI（默认）。"""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return None, None
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = float(np.mean(rng.choice(x, size=n, replace=True)))
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _pd_tail_stats(pds):
    """PD 分布尾部：>25% 例数、P90、max（用于说明 Endgame 重尾）。"""
    x = np.asarray(pds, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            'n_pd_gt_25': 0, 'p90_pd_pct': None, 'max_pd_pct': None,
            'ci95_mean_pd_lo': None, 'ci95_mean_pd_hi': None,
        }
    lo, hi = _bootstrap_mean_ci(x)
    return {
        'n_pd_gt_25': int((x > 25.0).sum()),
        'p90_pd_pct': float(np.percentile(x, 90)),
        'max_pd_pct': float(np.max(x)),
        'ci95_mean_pd_lo': lo,
        'ci95_mean_pd_hi': hi,
    }


def build_ablation_stats_tests(df, variants):
    """Friedman（全变体）+ 各变体 vs full 的 Wilcoxon + Holm + 效应量。

    统计单位：实例级目标（若有多 run，结果表中 objective 应为 run-mean）。
    次级 Endgame 拆分仅含 Elite/Skill 时，Holm 只校正这两项比较，勿与主消融三项混算。
    """
    if df is None or df.empty or 'full' not in set(variants):
        return pd.DataFrame(), {}
    work = df.dropna(subset=['objective']).copy()
    if work.empty:
        return pd.DataFrame(), {}
    need = [v for v in variants if v in ABLATION_VARIANTS]
    pivot = work.pivot_table(
        index='instance_id', columns='variant', values='objective', aggfunc='first')
    present = [v for v in need if v in pivot.columns]
    if 'full' not in present or len(present) < 2:
        return pd.DataFrame(), {}
    complete = pivot[present].dropna()
    friedman_info = {}
    if len(complete) >= 3 and len(present) >= 3:
        fr = _ablation_friedman_test(complete[present].values)
        if fr:
            friedman_info = {
                'friedman_stat': fr['statistic'],
                'friedman_df': fr['df'],
                'friedman_p': fr['p_value'],
                'friedman_n': fr['n'],
                'friedman_k': fr['k'],
                'mean_ranks': {
                    present[i]: fr['mean_ranks'][i] for i in range(len(present))
                },
            }

    rows = []
    raw_p = []
    full_vals = complete['full'].tolist() if 'full' in complete.columns else []
    full_arr = complete['full'].values.astype(float) if 'full' in complete.columns else None

    for var_key in present:
        paper = ABLATION_VARIANTS.get(var_key, {}).get('paper_name', var_key)
        if var_key == 'full':
            rows.append({
                'variant': 'full',
                'paper_name': ABLATION_VARIANTS['full'].get('paper_name', 'KG-ALNS'),
                'n_paired': len(complete),
                'mean_objective': float(np.mean(full_vals)) if full_vals else None,
                'mean_pd_pct': 0.0,
                'median_pd_pct': 0.0,
                'iqr_pd_pct': 0.0,
                'n_pd_gt_25': 0,
                'p90_pd_pct': 0.0,
                'max_pd_pct': 0.0,
                'ci95_mean_pd_lo': 0.0,
                'ci95_mean_pd_hi': 0.0,
                'full_win_tie_loss': f'{len(complete)}-0-0',
                'mean_diff_vs_full': 0.0,
                'wilcoxon_stat': None,
                'wilcoxon_p': None,
                'holm_p': None,
                'effect_size_rbc': 0.0,
                'method': None,
                'mean_rank': (friedman_info.get('mean_ranks') or {}).get('full'),
            })
            raw_p.append(None)
            continue

        var_arr = complete[var_key].values.astype(float)
        diffs = var_arr - full_arr
        with np.errstate(divide='ignore', invalid='ignore'):
            pds = np.where(np.abs(full_arr) > 1e-12,
                           diffs / full_arr * 100.0, np.nan)
        pds = pds[np.isfinite(pds)]
        n_worse = int((diffs > 1e-9).sum())   # variant worse
        n_better = int((diffs < -1e-9).sum())
        n_tie = int(len(diffs) - n_worse - n_better)
        # Full 胜 = variant 更差
        wtl = f'{n_worse}-{n_tie}-{n_better}'

        w = _ablation_paired_wilcoxon(complete[var_key].tolist(), full_vals)
        p = w['p_value'] if w else None
        raw_p.append(p)
        rbc = _rank_biserial_from_diffs(diffs)
        q1 = float(np.percentile(pds, 25)) if len(pds) else None
        q3 = float(np.percentile(pds, 75)) if len(pds) else None
        tail = _pd_tail_stats(pds)
        rows.append({
            'variant': var_key,
            'paper_name': paper,
            'n_paired': (w['n'] if w else len(complete)),
            'mean_objective': float(np.mean(var_arr)),
            'mean_pd_pct': float(np.mean(pds)) if len(pds) else None,
            'median_pd_pct': float(np.median(pds)) if len(pds) else None,
            'iqr_pd_pct': (q3 - q1) if (q1 is not None and q3 is not None) else None,
            'n_pd_gt_25': tail['n_pd_gt_25'],
            'p90_pd_pct': tail['p90_pd_pct'],
            'max_pd_pct': tail['max_pd_pct'],
            'ci95_mean_pd_lo': tail['ci95_mean_pd_lo'],
            'ci95_mean_pd_hi': tail['ci95_mean_pd_hi'],
            'full_win_tie_loss': wtl,
            'mean_diff_vs_full': (w['mean_diff'] if w else float(np.mean(diffs))),
            'wilcoxon_stat': (w['statistic'] if w else None),
            'wilcoxon_p': p,
            'holm_p': None,
            'effect_size_rbc': rbc,
            'method': (w.get('method') if w else None),
            'mean_rank': (friedman_info.get('mean_ranks') or {}).get(var_key),
        })

    adj_idx = [i for i, p in enumerate(raw_p) if p is not None]
    if adj_idx:
        adj_vals = _holm_adjust([raw_p[i] for i in adj_idx])
        for j, i in enumerate(adj_idx):
            rows[i]['holm_p'] = adj_vals[j]

    return pd.DataFrame(rows), friedman_info


def build_ablation_sci_main_table(stats_df, friedman_info=None):
    """论文主表：Mean/Median PD、IQR、WTL、Holm p、r_rb、95% CI、尾部统计。"""
    if stats_df is None or stats_df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in stats_df.iterrows():
        lo, hi = r.get('ci95_mean_pd_lo'), r.get('ci95_mean_pd_hi')
        if lo is not None and hi is not None and not (
                isinstance(lo, float) and (np.isnan(lo) or np.isnan(hi))):
            ci_str = f'[{lo:.3f}, {hi:.3f}]'
        else:
            ci_str = ''
        rows.append({
            '配置': r.get('paper_name', r.get('variant')),
            '变体键': r.get('variant'),
            'Mean PD (%)': r.get('mean_pd_pct'),
            'Median PD (%)': r.get('median_pd_pct'),
            'IQR': r.get('iqr_pd_pct'),
            'Full胜–平–负': r.get('full_win_tie_loss'),
            'Wilcoxon p': r.get('wilcoxon_p'),
            'Holm p': r.get('holm_p'),
            'Effect size (r_rb)': r.get('effect_size_rbc'),
            '95% CI (Mean PD)': ci_str,
            'n(PD>25%)': r.get('n_pd_gt_25'),
            'P90 PD (%)': r.get('p90_pd_pct'),
            'Max PD (%)': r.get('max_pd_pct'),
            'n': r.get('n_paired'),
            'Friedman mean rank': r.get('mean_rank'),
        })
    out = pd.DataFrame(rows)
    if friedman_info:
        out.attrs['friedman_p'] = friedman_info.get('friedman_p')
        out.attrs['friedman_stat'] = friedman_info.get('friedman_stat')
        out.attrs['friedman_n'] = friedman_info.get('friedman_n')
    return out


def build_ablation_instance_pd_long(df, variants=None):
    """实例级 PD 长表（散点/配对图用）：instance × variant → PD%。"""
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.dropna(subset=['objective']).copy()
    if 'variant' not in work.columns:
        return pd.DataFrame()
    variants = variants or sorted(work['variant'].dropna().unique().tolist())
    pivot = work.pivot_table(
        index='instance_id', columns='variant', values='objective', aggfunc='first')
    if 'full' not in pivot.columns:
        return pd.DataFrame()
    rows = []
    for iid, row in pivot.iterrows():
        base = row.get('full')
        if base is None or (isinstance(base, float) and (np.isnan(base) or base <= 0)):
            continue
        base = float(base)
        for v in variants:
            if v == 'full' or v not in pivot.columns:
                continue
            obj = row.get(v)
            if obj is None or (isinstance(obj, float) and np.isnan(obj)):
                continue
            pd_pct = (float(obj) - base) / base * 100.0
            rows.append({
                'instance_id': int(iid),
                'variant': v,
                'paper_name': ABLATION_VARIANTS.get(v, {}).get('paper_name', v),
                'full_obj': base,
                'variant_obj': float(obj),
                'pd_pct': pd_pct,
            })
    return pd.DataFrame(rows)


def _parse_convergence_trace_field(val):
    """解析 convergence_trace（dict / JSON str）。"""
    raw = _parse_ablation_json_field(val, {})
    out = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        try:
            t = int(k)
            fv = float(v)
            if math.isfinite(fv):
                out[t] = fv
        except (TypeError, ValueError):
            continue
    return out


def _normalized_convergence_auc(trace, f_ref, checkpoints=None, time_limit=300.0):
    """归一化收敛曲线下面积 AUC（越小越快）。

    AUC = (1/T) ∫ [(f(t)-f_ref)/f_ref] dt，梯形积分；检查点间线性插值。
    """
    checkpoints = list(checkpoints or globals().get(
        'CONVERGENCE_CHECKPOINT_SEC', (30, 60, 120, 180, 300)))
    T = float(time_limit)
    if f_ref is None or not math.isfinite(float(f_ref)) or float(f_ref) <= 0:
        return None
    f_ref = float(f_ref)
    if not trace:
        return None
    # 时间点：0 + 检查点（≤T）
    times = [0.0] + [float(t) for t in checkpoints if float(t) <= T]
    if times[-1] < T:
        times.append(T)
    # 每个时刻的目标：t=0 用首个检查点或 trace 最小值
    vals = []
    sorted_cps = sorted(trace.keys())
    first_obj = trace[sorted_cps[0]] if sorted_cps else None
    if first_obj is None:
        return None
    running = float(first_obj)
    for t in times:
        if t <= 0:
            vals.append(running)
            continue
        # best-so-far at time t
        cur = running
        for cp in sorted_cps:
            if cp <= t:
                cur = min(cur, float(trace[cp]))
        running = cur
        vals.append(running)
    area = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if dt <= 0:
            continue
        y0 = (vals[i - 1] - f_ref) / f_ref
        y1 = (vals[i] - f_ref) / f_ref
        area += 0.5 * (y0 + y1) * dt
    return area / T if T > 0 else None


def build_knowledge_activation_summary(df):
    """汇总 Knowledge 触发率 / 成功率（实例×变体）。"""
    if df is None or df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        stats = _parse_ablation_json_field(r.get('kg_plus_stats'), {})
        cand = float(stats.get('knowledge_candidate_count', 0) or 0)
        changed = float(stats.get('knowledge_changed_choice_count', 0) or 0)
        improved = float(stats.get('knowledge_improvement_count', 0) or 0)
        destroy = float(stats.get('knowledge_destroy_selected_count', 0) or 0)
        rows.append({
            'variant': r.get('variant'),
            'paper_name': r.get('paper_name', r.get('variant')),
            'instance_id': r.get('instance_id'),
            'knowledge_candidate_count': cand,
            'knowledge_changed_choice_count': changed,
            'knowledge_improvement_count': improved,
            'knowledge_destroy_selected_count': destroy,
            'activation_rate': (changed / cand) if cand > 0 else None,
            'success_rate': (improved / changed) if changed > 0 else None,
        })
    return pd.DataFrame(rows)


def build_knowledge_convergence_auc_tests(df, variants=None, wo_variant='wo_knowledge_guided'):
    """Full vs w/o Knowledge 的归一化 AUC 配对检验（主要假设，无需 Holm）。"""
    if df is None or df.empty:
        return pd.DataFrame(), {}
    variants = variants or ['full', wo_variant]
    if 'full' not in variants or wo_variant not in variants:
        return pd.DataFrame(), {}
    work = df.copy()
    cps = list(globals().get('CONVERGENCE_CHECKPOINT_SEC', (30, 60, 120, 180, 300)))
    T = float(cps[-1]) if cps else 300.0

    # 每实例 f_ref = 两配置全部检查点/终局目标中的全局最优
    ref_by_inst = {}
    trace_by = {}  # (iid, variant) -> trace dict (instance-mean)
    for _, r in work.iterrows():
        iid = int(r['instance_id'])
        var = str(r['variant'])
        trace = _parse_convergence_trace_field(r.get('convergence_trace'))
        if not trace and r.get('objective') is not None:
            obj = float(r['objective'])
            trace = {int(T): obj}
        if trace:
            trace_by[(iid, var)] = trace
        objs = [float(v) for v in trace.values()]
        if r.get('objective') is not None:
            objs.append(float(r['objective']))
        if objs:
            ref_by_inst[iid] = min(ref_by_inst.get(iid, objs[0]), min(objs))

    auc_full, auc_wo, paired_ids = [], [], []
    inst_ids = sorted({
        iid for (iid, v) in trace_by.keys()
        if v in ('full', wo_variant)})
    checkpoint_rows = []
    for iid in inst_ids:
        f_ref = ref_by_inst.get(iid)
        tf = trace_by.get((iid, 'full'))
        tw = trace_by.get((iid, wo_variant))
        if f_ref is None or tf is None or tw is None:
            continue
        af = _normalized_convergence_auc(tf, f_ref, cps, T)
        aw = _normalized_convergence_auc(tw, f_ref, cps, T)
        if af is None or aw is None:
            continue
        auc_full.append(af)
        auc_wo.append(aw)
        paired_ids.append(iid)
        for t in cps:
            checkpoint_rows.append({
                'instance_id': iid,
                'checkpoint_s': int(t),
                'full_obj': tf.get(int(t)),
                'wo_k_obj': tw.get(int(t)),
            })

    if len(auc_full) < 5:
        return pd.DataFrame(), {'n_paired': len(auc_full)}

    diffs = np.asarray(auc_wo, dtype=float) - np.asarray(auc_full, dtype=float)
    w = _ablation_paired_wilcoxon(auc_wo, auc_full)
    rbc = _rank_biserial_from_diffs(diffs)
    lo, hi = _bootstrap_mean_ci(diffs)
    n_worse = int((diffs > 1e-12).sum())   # wo worse (higher AUC)
    n_better = int((diffs < -1e-12).sum())
    n_tie = int(len(diffs) - n_worse - n_better)

    summary = pd.DataFrame([{
        'metric': 'normalized_AUC',
        'n_paired': len(diffs),
        'mean_auc_full': float(np.mean(auc_full)),
        'mean_auc_wo_k': float(np.mean(auc_wo)),
        'median_auc_full': float(np.median(auc_full)),
        'median_auc_wo_k': float(np.median(auc_wo)),
        'mean_auc_diff_wo_minus_full': float(np.mean(diffs)),
        'median_auc_diff_wo_minus_full': float(np.median(diffs)),
        'full_better_tie_worse': f'{n_better}-{n_tie}-{n_worse}',
        'wilcoxon_p': (w['p_value'] if w else None),
        'effect_size_rbc': rbc,
        'ci95_mean_diff_lo': lo,
        'ci95_mean_diff_hi': hi,
        'interpretation': (
            'Full faster (AUC lower)' if (w and w['p_value'] < 0.05 and np.median(diffs) > 0)
            else 'No significant AUC advantage'),
    }])
    meta = {
        'n_paired': len(diffs),
        'checkpoint_table': pd.DataFrame(checkpoint_rows),
        'per_instance_auc': pd.DataFrame({
            'instance_id': paired_ids,
            'auc_full': auc_full,
            'auc_wo_k': auc_wo,
            'auc_diff_wo_minus_full': diffs,
        }),
    }
    return summary, meta


def write_knowledge_trajectory_postprocess(ablation_dir):
    """Knowledge 轨迹后处理：AUC 检验、检查点表、触发率汇总。"""
    csv_path = os.path.join(ablation_dir, 'ablation_results.csv')
    if not os.path.isfile(csv_path):
        print(f"[Knowledge轨迹] 未找到 {csv_path}", flush=True)
        return pd.DataFrame(), {}
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception as e:
        print(f"[Knowledge轨迹] 读取失败: {e}", flush=True)
        return pd.DataFrame(), {}
    variants = sorted(df['variant'].dropna().unique().tolist()) if 'variant' in df.columns else []
    auc_df, meta = build_knowledge_convergence_auc_tests(df, variants=variants)
    if not auc_df.empty:
        _safe_dataframe_to_csv(
            auc_df, os.path.join(ablation_dir, 'knowledge_auc_tests.csv'))
        _safe_dataframe_to_csv(
            auc_df, os.path.join(ablation_dir, 'Knowledge_AUC检验.csv'))
    cp = meta.get('checkpoint_table')
    if cp is not None and not cp.empty:
        _safe_dataframe_to_csv(
            cp, os.path.join(ablation_dir, 'knowledge_convergence_checkpoints.csv'))
    per_auc = meta.get('per_instance_auc')
    if per_auc is not None and not per_auc.empty:
        _safe_dataframe_to_csv(
            per_auc, os.path.join(ablation_dir, 'knowledge_auc_per_instance.csv'))
    act = build_knowledge_activation_summary(df)
    if not act.empty:
        _safe_dataframe_to_csv(
            act, os.path.join(ablation_dir, 'knowledge_activation_summary.csv'))
        _safe_dataframe_to_csv(
            act, os.path.join(ablation_dir, 'Knowledge触发率汇总.csv'))
    if not auc_df.empty:
        row = auc_df.iloc[0]
        print(f"[Knowledge轨迹] AUC Wilcoxon p={row.get('wilcoxon_p')} "
              f"median_diff={row.get('median_auc_diff_wo_minus_full')} "
              f"WTL={row.get('full_better_tie_worse')} "
              f"n={row.get('n_paired')}", flush=True)
    return auc_df, meta


def _spearman_with_p(x, y):
    """Spearman 相关；无 scipy 时用秩相关近似。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return None, None, int(len(x))
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y)
        return float(r), float(p), int(len(x))
    except Exception:
        rx = pd.Series(x).rank().values
        ry = pd.Series(y).rank().values
        r = float(np.corrcoef(rx, ry)[0, 1])
        return r, None, int(len(x))


def build_knowledge_complexity_correlation(df, instances_by_id,
                                         wo_variant='wo_knowledge_guided'):
    """H_i 与 Knowledge PD 的 Spearman 相关（20 实例，非事后筛选）。"""
    if df is None or df.empty or 'full' not in set(df['variant']):
        return pd.DataFrame(), pd.DataFrame()
    pivot = df.pivot_table(
        index='instance_id', columns='variant', values='objective', aggfunc='first')
    if 'full' not in pivot.columns or wo_variant not in pivot.columns:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    for iid, row in pivot.iterrows():
        base = row.get('full')
        var = row.get(wo_variant)
        if base is None or var is None or base <= 0:
            continue
        inst = instances_by_id.get(int(iid))
        if inst is None:
            continue
        h = compute_instance_knowledge_intensity(inst)
        if h is None:
            continue
        pd_k = (float(var) - float(base)) / float(base) * 100.0
        rows.append({
            'instance_id': int(iid),
            'knowledge_intensity_H': h,
            'pd_knowledge_pct': pd_k,
            'full_obj': float(base),
            'wo_k_obj': float(var),
        })
    detail = pd.DataFrame(rows)
    if len(detail) < 5:
        return pd.DataFrame(), detail
    r, p, n = _spearman_with_p(
        detail['knowledge_intensity_H'].values,
        detail['pd_knowledge_pct'].values)
    summary = pd.DataFrame([{
        'metric': 'Spearman(H_i, PD^K)',
        'n_instances': n,
        'spearman_r': r,
        'spearman_p': p,
            'interpretation': (
            'Higher complexity → larger Knowledge PD (Full better)'
            if (r is not None and r > 0.3 and p is not None and p < 0.05)
            else 'No uniform complexity–benefit relationship'),
    }])
    return summary, detail


def build_knowledge_objective_component_tests(df, wo_variant='wo_knowledge_guided'):
    """流动时间 / 技能惩罚 配对 Wilcoxon + Holm（需 CSV 含分解列）。"""
    need = {'total_flow', 'misallocation_count'}
    if df is None or df.empty or not need.issubset(set(df.columns)):
        return pd.DataFrame(), 'missing_decomposition_columns'
    pivot_flow = df.pivot_table(
        index='instance_id', columns='variant', values='total_flow', aggfunc='first')
    pivot_misc = df.pivot_table(
        index='instance_id', columns='variant', values='misallocation_count', aggfunc='first')
    if wo_variant not in pivot_flow.columns or 'full' not in pivot_flow.columns:
        return pd.DataFrame(), 'missing_variants'
    rows = []
    raw_p = []
    tests = [
        ('total_flow', pivot_flow, 'Total flow time'),
        ('misallocation_count', pivot_misc, 'Misallocation count'),
    ]
    if 'misallocation_penalty' in df.columns:
        pivot_pen = df.pivot_table(
            index='instance_id', columns='variant',
            values='misallocation_penalty', aggfunc='first')
        tests.append(('misallocation_penalty', pivot_pen, 'Misallocation penalty'))
    if 'dock_misuse_count' in df.columns:
        pivot_dock = df.pivot_table(
            index='instance_id', columns='variant',
            values='dock_misuse_count', aggfunc='first')
        tests.append(('dock_misuse_count', pivot_dock, 'Dock misuse count'))

    for key, piv, label in tests:
        if wo_variant not in piv.columns or 'full' not in piv.columns:
            continue
        complete = piv[['full', wo_variant]].dropna()
        if len(complete) < 5:
            continue
        wo_vals = complete[wo_variant].astype(float).tolist()
        full_vals = complete['full'].astype(float).tolist()
        diffs = np.asarray(wo_vals) - np.asarray(full_vals)
        w = _ablation_paired_wilcoxon(wo_vals, full_vals)
        p = w['p_value'] if w else None
        raw_p.append(p)
        n_worse = int((diffs > 1e-9).sum())
        n_better = int((diffs < -1e-9).sum())
        n_tie = int(len(diffs) - n_worse - n_better)
        lo, hi = _bootstrap_mean_ci(diffs)
        rows.append({
            'component': key,
            'label': label,
            'n_paired': len(complete),
            'mean_full': float(np.mean(full_vals)),
            'mean_wo_k': float(np.mean(wo_vals)),
            'mean_diff_wo_minus_full': float(np.mean(diffs)),
            'median_diff_wo_minus_full': float(np.median(diffs)),
            'full_better_tie_worse': f'{n_better}-{n_tie}-{n_worse}',
            'wilcoxon_p': p,
            'holm_p': None,
            'effect_size_rbc': _rank_biserial_from_diffs(diffs),
            'ci95_mean_diff_lo': lo,
            'ci95_mean_diff_hi': hi,
        })
    if not rows:
        return pd.DataFrame(), 'no_complete_pairs'
    adj_idx = [i for i, p in enumerate(raw_p) if p is not None]
    if adj_idx:
        adj_vals = _holm_adjust([raw_p[i] for i in adj_idx])
        for j, i in enumerate(adj_idx):
            rows[i]['holm_p'] = adj_vals[j]
    return pd.DataFrame(rows), None


def build_knowledge_elite_interaction(df):
    """Knowledge×Elite 交互 I_i = (f_{-K,-E}-f_{K,-E}) - (f_{-K,E}-f_{K,E})。"""
    need = {'full', 'wo_knowledge_guided', 'wo_elite_restart', 'wo_knowledge_no_elite'}
    if df is None or df.empty:
        return pd.DataFrame(), {}
    present = set(df['variant'].dropna().unique())
    if not need.issubset(present):
        return pd.DataFrame(), {'missing': sorted(need - present)}
    pivot = df.pivot_table(
        index='instance_id', columns='variant', values='objective', aggfunc='first')
    complete = pivot[list(need)].dropna()
    if len(complete) < 5:
        return pd.DataFrame(), {'n_paired': len(complete)}
    interactions = []
    for iid, row in complete.iterrows():
        f_ke = float(row['full'])
        f_kE = float(row['wo_knowledge_guided'])
        fKe = float(row['wo_elite_restart'])
        fKE = float(row['wo_knowledge_no_elite'])
        I = (fKE - fKe) - (f_kE - f_ke)
        interactions.append({'instance_id': int(iid), 'interaction_I': I})
    idf = pd.DataFrame(interactions)
    diffs = idf['interaction_I'].values.astype(float)
    w = _ablation_paired_wilcoxon(diffs.tolist(), [0.0] * len(diffs))
    rbc = _rank_biserial_from_diffs(diffs)
    lo, hi = _bootstrap_mean_ci(diffs)
    summary = pd.DataFrame([{
        'metric': 'Knowledge×Elite interaction I',
        'n_paired': len(diffs),
        'mean_I': float(np.mean(diffs)),
        'median_I': float(np.median(diffs)),
        'wilcoxon_p': (w['p_value'] if w else None),
        'effect_size_rbc': rbc,
        'ci95_mean_I_lo': lo,
        'ci95_mean_I_hi': hi,
        'interpretation': (
            'Knowledge more helpful without Elite'
            if (w and w['p_value'] < 0.05 and np.median(diffs) > 0)
            else 'No significant interaction'),
    }])
    return summary, {'per_instance': idf}


def _load_instances_for_ablation_dir(ablation_dir, output_root=OUTPUT_ROOT):
    """从 ablation_instance_ids.json + large_scale 加载算例 dict。"""
    manifest = os.path.join(ablation_dir, 'ablation_instance_ids.json')
    if not os.path.isfile(manifest):
        return {}
    try:
        with open(manifest, encoding='utf-8') as f:
            raw = json.load(f)
    except Exception:
        return {}
    if isinstance(raw, dict):
        ids = raw.get('instance_ids') or []
    elif isinstance(raw, list):
        ids = raw
    else:
        return {}
    ids = [int(x) for x in ids]
    large_dir = os.path.join(output_root, 'large_scale')
    if not os.path.isfile(os.path.join(large_dir, 'large_configs.json')):
        for alt in ('有效数据', 'valid_data'):
            alt_dir = os.path.join(output_root, alt, 'large_scale')
            if os.path.isfile(os.path.join(alt_dir, 'large_configs.json')):
                large_dir = alt_dir
                break
    configs = load_large_configs(large_dir) or []
    exp = BatchExperimentGurobi(output_dir=large_dir, heuristic_type='lNS')
    if configs:
        exp.set_experiment_configs(configs)
    exp.generate_instances(force_regen=False, quiet=True)
    by_id = {int(i['instance_id']): i for i in exp.instances}
    for inst in by_id.values():
        tc = inst.get('team_config')
        if tc is not None:
            inst['team_config'] = normalize_team_config_keys(tc)
    return {iid: by_id[iid] for iid in ids if iid in by_id}


def write_knowledge_mechanism_analysis(ablation_dir, output_root=OUTPUT_ROOT):
    """Knowledge 机制分析：目标分解检验 + H_i 相关 +（若有）交互效应。"""
    csv_path = os.path.join(ablation_dir, 'ablation_results.csv')
    if not os.path.isfile(csv_path):
        print(f"[Knowledge机制] 未找到 {csv_path}", flush=True)
        return {}
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception as e:
        print(f"[Knowledge机制] 读取失败: {e}", flush=True)
        return {}
    out = {}
    comp_df, comp_note = build_knowledge_objective_component_tests(df)
    if not comp_df.empty:
        _safe_dataframe_to_csv(
            comp_df, os.path.join(ablation_dir, 'knowledge_objective_components.csv'))
        _safe_dataframe_to_csv(
            comp_df, os.path.join(ablation_dir, 'Knowledge目标分解检验.csv'))
        out['components'] = comp_df
        print("[Knowledge机制] 目标分解检验已写出", flush=True)
    else:
        print(f"[Knowledge机制] 目标分解跳过: {comp_note}"
              f"（需重跑并保存 total_flow/misallocation 列，或调用 decomposition replay）",
              flush=True)

    inst_map = _load_instances_for_ablation_dir(ablation_dir, output_root)
    corr_df, corr_detail = build_knowledge_complexity_correlation(df, inst_map)
    if not corr_df.empty:
        _safe_dataframe_to_csv(
            corr_df, os.path.join(ablation_dir, 'knowledge_complexity_correlation.csv'))
        _safe_dataframe_to_csv(
            corr_detail, os.path.join(ablation_dir, 'knowledge_complexity_detail.csv'))
        out['complexity'] = corr_df
        row = corr_df.iloc[0]
        print(f"[Knowledge机制] Spearman(H, PD^K): r={row.get('spearman_r')} "
              f"p={row.get('spearman_p')} n={row.get('n_instances')}", flush=True)
    if not corr_detail.empty:
        _safe_dataframe_to_csv(
            corr_detail, os.path.join(ablation_dir, 'Knowledge复杂度明细.csv'))

    inter_df, inter_meta = build_knowledge_elite_interaction(df)
    if not inter_df.empty:
        _safe_dataframe_to_csv(
            inter_df, os.path.join(ablation_dir, 'knowledge_elite_interaction.csv'))
        out['interaction'] = inter_df
        row = inter_df.iloc[0]
        print(f"[Knowledge机制] Knowledge×Elite I: median={row.get('median_I')} "
              f"p={row.get('wilcoxon_p')}", flush=True)
    elif inter_meta.get('missing'):
        print(f"[Knowledge机制] 交互分析待补变体: {inter_meta['missing']}", flush=True)
    return out


def build_ablation_operator_diag_df(df):
    """展开 destroy_operator_stats → 长表（variant×instance×operator）。"""
    if df is None or df.empty or 'destroy_operator_stats' not in df.columns:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        stats = _parse_ablation_json_field(r.get('destroy_operator_stats'), {})
        if not stats:
            continue
        for op_name, op in stats.items():
            if not isinstance(op, dict):
                continue
            rows.append({
                'variant': r.get('variant'),
                'paper_name': r.get('paper_name'),
                'instance_id': r.get('instance_id'),
                'tier': r.get('tier'),
                'operator': op_name,
                'selected': op.get('selected'),
                'accepted': op.get('accepted'),
                'improved': op.get('improved'),
                'new_best': op.get('new_best'),
                'avg_improve': op.get('avg_improve'),
                'final_weight': op.get('final_weight'),
            })
    return pd.DataFrame(rows)


def build_ablation_readable_tables(detail_df, summary_df, tier_df, variants=None):
    """生成便于阅读/论文粘贴的消融表（中文列名 + 透视）。

    返回 dict:
      main_overall  — 总体主表（变体 / 平均目标 / 相对full劣化% / 胜平负）
      main_by_tier  — 分梯度宽表（行=变体，列=各梯度 Mean PD%）
      pivot_obj     — 实例×变体 目标值
      pivot_pd      — 实例×变体 相对 full 劣化%
      case_detail   — 逐例可读明细（含船数、梯度、full 与各变体）
    """
    variants = variants or (
        list(summary_df['variant']) if summary_df is not None and not summary_df.empty
        else list(ABLATION_VARIANTS.keys()))

    main_overall = pd.DataFrame()
    if summary_df is not None and not summary_df.empty:
        rows = []
        for _, r in summary_df.iterrows():
            rows.append({
                '变体键': r.get('variant'),
                '论文名': r.get('paper_name', r.get('label', '')),
                '算例数': r.get('n_cases'),
                '平均目标': r.get('avg_objective'),
                '相对完整版劣化%(均值)': r.get('mean_pd_vs_full_pct',
                                         r.get('avg_gap_vs_full_pct')),
                '相对完整版劣化%(中位)': r.get('median_pd_vs_full_pct'),
                '胜平负(劣-平-优)': r.get('wtl_vs_full'),
                '平均耗时(s)': r.get('avg_time_s'),
            })
        main_overall = pd.DataFrame(rows)

    main_by_tier = pd.DataFrame()
    if tier_df is not None and not tier_df.empty and 'tier' in tier_df.columns:
        # 宽表：variant × tier → mean PD%
        work = tier_df.copy()
        work['paper_name'] = work.apply(
            lambda r: r.get('paper_name') or ABLATION_VARIANTS.get(
                r.get('variant'), {}).get('paper_name', r.get('variant')),
            axis=1)
        piv = work.pivot_table(
            index=['variant', 'paper_name'],
            columns='tier',
            values='mean_pd_vs_full_pct',
            aggfunc='first')
        # 列顺序：已知梯度优先
        tier_order = list(LARGE_SCALE_TIER_LABELS) + list(
            globals().get('ABLATION_XL_TIER_LABELS', ()) or ())
        cols = [c for c in tier_order if c in piv.columns]
        cols += [c for c in piv.columns if c not in cols]
        piv = piv.reindex(columns=cols)
        piv = piv.reset_index()
        rename = {'variant': '变体键', 'paper_name': '论文名'}
        for c in cols:
            rename[c] = f'{c}_劣化%'
        main_by_tier = piv.rename(columns=rename)

    pivot_obj = pd.DataFrame()
    pivot_pd = pd.DataFrame()
    case_detail = pd.DataFrame()
    if detail_df is not None and not detail_df.empty and 'instance_id' in detail_df.columns:
        work = detail_df.dropna(subset=['objective']).copy()
        if not work.empty:
            # 目标透视
            pivot_obj = work.pivot_table(
                index='instance_id', columns='variant',
                values='objective', aggfunc='first')
            present = [v for v in variants if v in pivot_obj.columns]
            extras = [c for c in pivot_obj.columns if c not in present]
            pivot_obj = pivot_obj.reindex(columns=present + extras)
            pivot_obj = pivot_obj.reset_index().rename(
                columns={'instance_id': '算例ID'})

            # 劣化%：相对同例 full
            full_map = {}
            full_sub = work[work['variant'] == 'full']
            for _, r in full_sub.iterrows():
                full_map[int(r['instance_id'])] = float(r['objective'])
            pd_rows = []
            for _, r in work.iterrows():
                iid = int(r['instance_id'])
                base = full_map.get(iid)
                obj = float(r['objective'])
                pd_pct = ((obj - base) / base * 100.0
                          if base is not None and base > 0 else None)
                pd_rows.append({
                    'instance_id': iid,
                    'variant': r['variant'],
                    'pd_pct': pd_pct,
                })
            pdf = pd.DataFrame(pd_rows)
            if not pdf.empty:
                pivot_pd = pdf.pivot_table(
                    index='instance_id', columns='variant',
                    values='pd_pct', aggfunc='first')
                present2 = [v for v in variants if v in pivot_pd.columns]
                extras2 = [c for c in pivot_pd.columns if c not in present2]
                pivot_pd = pivot_pd.reindex(columns=present2 + extras2)
                pivot_pd = pivot_pd.reset_index().rename(
                    columns={'instance_id': '算例ID'})

            # 逐例可读：船数/梯度 + full + 各变体目标与劣化
            meta = {}
            for _, r in work.iterrows():
                iid = int(r['instance_id'])
                if iid not in meta:
                    meta[iid] = {
                        '算例ID': iid,
                        '梯度': r.get('tier', ''),
                        '船舶数': r.get('num_ships'),
                        '任务数': r.get('num_tasks'),
                    }
            case_rows = []
            for iid in sorted(meta.keys()):
                row = dict(meta[iid])
                row['full目标'] = full_map.get(iid)
                for v in variants:
                    if v == 'full':
                        continue
                    sub = work[(work['instance_id'].astype(int) == iid)
                               & (work['variant'] == v)]
                    if sub.empty:
                        row[f'{v}_目标'] = None
                        row[f'{v}_劣化%'] = None
                    else:
                        obj = float(sub.iloc[0]['objective'])
                        base = full_map.get(iid)
                        row[f'{v}_目标'] = obj
                        row[f'{v}_劣化%'] = (
                            (obj - base) / base * 100.0
                            if base is not None and base > 0 else None)
                case_rows.append(row)
            case_detail = pd.DataFrame(case_rows)

    return {
        'main_overall': main_overall,
        'main_by_tier': main_by_tier,
        'pivot_obj': pivot_obj,
        'pivot_pd': pivot_pd,
        'case_detail': case_detail,
    }


def write_ablation_outputs(all_rows, ablation_dir, all_variant_keys=None):
    """写入/更新消融明细、汇总 CSV、Excel（含可读主表/透视表）。"""
    all_variant_keys = all_variant_keys or list(ABLATION_VARIANTS.keys())
    os.makedirs(ablation_dir, exist_ok=True)
    results_path = os.path.join(ablation_dir, 'ablation_results.csv')
    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    if not df.empty:
        results_path = _safe_dataframe_to_csv(df, results_path)
    summary_df, tier_df = build_ablation_summary(df, all_variant_keys)
    summary_path = os.path.join(ablation_dir, 'ablation_summary.csv')
    tier_path = os.path.join(ablation_dir, 'ablation_summary_by_tier.csv')
    summary_path = _safe_dataframe_to_csv(summary_df, summary_path)
    tier_path = _safe_dataframe_to_csv(tier_df, tier_path)

    tables = build_ablation_readable_tables(
        df, summary_df, tier_df, variants=all_variant_keys)
    table_paths = {}
    name_map = {
        'main_overall': '消融主表_总体.csv',
        'main_by_tier': '消融主表_分梯度.csv',
        'pivot_obj': '消融透视_目标.csv',
        'pivot_pd': '消融透视_劣化pct.csv',
        'case_detail': '消融逐例对照表.csv',
    }
    for key, fname in name_map.items():
        tdf = tables.get(key)
        if tdf is not None and not tdf.empty:
            table_paths[key] = _safe_dataframe_to_csv(
                tdf, os.path.join(ablation_dir, fname))

    excel_path = os.path.join(ablation_dir, 'ablation_report.xlsx')
    if not df.empty:
        save_ablation_excel_report(
            df, summary_df, tier_df, excel_path, readable_tables=tables)
    return df, summary_df, tier_df, results_path, summary_path, tier_path


def write_ablation_stats_postprocess(ablation_dir, variants=None):
    """独立后处理：读 ablation_results.csv，写 Friedman/Wilcoxon/Holm 检验 CSV。

    用法（消融跑完后）::
        write_ablation_stats_postprocess('~/Desktop/算法设计52/ablation_hard_51_80_x3')
    """
    csv_path = os.path.join(ablation_dir, 'ablation_results.csv')
    if not os.path.isfile(csv_path):
        print(f"[消融后处理] 未找到 {csv_path}", flush=True)
        return pd.DataFrame(), {}
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception as e:
        print(f"[消融后处理] 读取失败: {e}", flush=True)
        return pd.DataFrame(), {}
    if variants is None:
        variants = sorted(df['variant'].dropna().unique().tolist()) if 'variant' in df.columns else []
    stats_df, friedman_info = build_ablation_stats_tests(df, variants)
    if not stats_df.empty:
        _safe_dataframe_to_csv(
            stats_df, os.path.join(ablation_dir, 'ablation_stats_tests.csv'))
        sci_df = build_ablation_sci_main_table(stats_df, friedman_info)
        if not sci_df.empty:
            _safe_dataframe_to_csv(
                sci_df, os.path.join(ablation_dir, '消融SCI主表.csv'))
            _safe_dataframe_to_csv(
                sci_df, os.path.join(ablation_dir, 'ablation_sci_main_table.csv'))
    if friedman_info:
        with open(os.path.join(ablation_dir, 'ablation_friedman.json'),
                  'w', encoding='utf-8') as f:
            json.dump(friedman_info, f, indent=2, ensure_ascii=False)
        print(f"[消融后处理] Friedman χ²={friedman_info.get('friedman_stat')} "
              f"p={friedman_info.get('friedman_p')} "
              f"n={friedman_info.get('friedman_n')}", flush=True)
    pd_long = build_ablation_instance_pd_long(df, variants)
    if not pd_long.empty:
        _safe_dataframe_to_csv(
            pd_long, os.path.join(ablation_dir, 'ablation_instance_pd_long.csv'))
        _safe_dataframe_to_csv(
            pd_long, os.path.join(ablation_dir, '消融实例PD长表.csv'))
    if 'destroy_operator_stats' in df.columns:
        op_df = build_ablation_operator_diag_df(df)
        if not op_df.empty:
            _safe_dataframe_to_csv(
                op_df, os.path.join(ablation_dir, 'ablation_operator_diag.csv'))
    # 同步刷新可读表（跑完后/中断后都可单独调用）
    summary_df, tier_df = build_ablation_summary(df, variants)
    tables = build_ablation_readable_tables(df, summary_df, tier_df, variants=variants)
    for key, fname in {
        'main_overall': '消融主表_总体.csv',
        'main_by_tier': '消融主表_分梯度.csv',
        'pivot_obj': '消融透视_目标.csv',
        'pivot_pd': '消融透视_劣化pct.csv',
        'case_detail': '消融逐例对照表.csv',
    }.items():
        tdf = tables.get(key)
        if tdf is not None and not tdf.empty:
            _safe_dataframe_to_csv(tdf, os.path.join(ablation_dir, fname))
    excel_path = os.path.join(ablation_dir, 'ablation_report.xlsx')
    if not df.empty:
        save_ablation_excel_report(
            df, summary_df, tier_df, excel_path, readable_tables=tables)
    print(f"[消融后处理] 统计检验 → {ablation_dir}/ablation_stats_tests.csv", flush=True)
    print(f"[消融后处理] 可读表/Excel 已刷新 → {ablation_dir}", flush=True)
    if globals().get('ABLATION_STUDY') == 'knowledge_trajectory' or (
            'convergence_trace' in df.columns):
        try:
            write_knowledge_trajectory_postprocess(ablation_dir)
        except Exception as e:
            print(f"[消融后处理] Knowledge轨迹分析失败: {e}", flush=True)
    try:
        write_knowledge_mechanism_analysis(ablation_dir)
    except Exception as e:
        print(f"[消融后处理] Knowledge机制分析失败: {e}", flush=True)
    return stats_df, friedman_info


def run_ablation_full_regression_test(output_root=OUTPUT_ROOT, batch_time_limit=None,
                                      instance_ids=None, tolerance=None,
                                      baseline_csv=None, num_runs=1):
    """Full 算法回归：固定代表例 × 300s，与阶段二 large_scale 基准对比。

    默认 Case 2 / 25 / 43（三梯度各 1 例）。|Δobj| ≤ tolerance 视为 Full 行为未变。
    结果写入 {output_root}/ablation_regression/regression_full_check.csv
    """
    batch_time_limit = batch_time_limit or LARGE_BATCH_TIME_LIMIT
    tolerance = tolerance if tolerance is not None else ABLATION_REGRESSION_TOLERANCE
    instance_ids = list(instance_ids or ABLATION_REGRESSION_INSTANCE_IDS)
    out_dir = os.path.join(output_root, 'ablation_regression')
    os.makedirs(out_dir, exist_ok=True)

    large_dir = os.path.join(output_root, 'large_scale')
    if not os.path.isfile(os.path.join(large_dir, 'large_configs.json')):
        for alt_name in ('有效数据', 'valid_data'):
            alt = os.path.join(output_root, alt_name, 'large_scale')
            if os.path.isfile(os.path.join(alt, 'large_configs.json')):
                large_dir = alt
                break

    configs = load_large_configs(large_dir)
    if not configs:
        configs = generate_large_scale_configs(60, seed=REPRODUCIBILITY_SEED)
    exp = BatchExperimentGurobi(output_dir=large_dir, heuristic_type='lNS')
    exp.set_experiment_configs(configs)
    exp.generate_instances(force_regen=False, quiet=True)
    id_set = {int(x) for x in instance_ids}
    work = [i for i in exp.instances if int(i['instance_id']) in id_set]
    work.sort(key=lambda x: int(x['instance_id']))
    if len(work) < len(id_set):
        print(f"[回归] 警告：仅找到 {len(work)}/{len(id_set)} 个算例", flush=True)

    baseline_map = {}
    baseline_csv = baseline_csv or os.path.join(large_dir, 'large_scale_results.csv')
    if os.path.isfile(baseline_csv):
        try:
            bdf = pd.read_csv(baseline_csv, encoding='utf-8-sig')
            for _, r in bdf.iterrows():
                iid = int(r.get('instance_id', -1))
                obj = r.get('mip_lns_obj_mean')
                if obj is None or (isinstance(obj, float) and pd.isna(obj)):
                    obj = r.get('mip_lns_obj_best')
                if obj is None or (isinstance(obj, float) and pd.isna(obj)):
                    obj = r.get('heuristic_obj')
                if obj is not None and not (isinstance(obj, float) and pd.isna(obj)):
                    baseline_map[iid] = float(obj)
        except Exception as e:
            print(f"[回归] 读取基准 CSV 失败: {e}", flush=True)

    solver = MatheuristicSolver(
        time_limit=batch_time_limit,
        mip_sub_time_limit=LARGE_MIP_SUB_TIME_LIMIT,
        max_iters=150,
        verbose=True,
        ablation_flags={},
        ablation_variant='full',
    )
    rows = []
    print(f"\n[Full回归] {len(work)} 例 × {batch_time_limit}s | 容差={tolerance}", flush=True)
    for inst in work:
        iid = int(inst['instance_id'])
        tier = (inst.get('config') or {}).get('tier', '')
        seed_base = iid * 100 + 42
        result = _ablation_solve_one(
            solver, inst, seed_base=seed_base, num_runs=num_runs, executor=None,
            parallel_workers=1)
        obj = result.get('objective')
        base = baseline_map.get(iid)
        delta = (float(obj) - base) if (obj is not None and base is not None) else None
        ok = (delta is not None and abs(delta) <= tolerance)
        rows.append({
            'instance_id': iid,
            'tier': tier,
            'objective_full_now': obj,
            'objective_baseline_53': base,
            'delta': delta,
            'abs_delta': abs(delta) if delta is not None else None,
            'within_tolerance': ok,
            'tolerance': tolerance,
            'seed_base': seed_base,
            'num_runs': num_runs,
            'baseline_csv': baseline_csv,
        })
        tag = 'OK' if ok else ('N/A' if delta is None else 'DIFF')
        print(f"  Case {iid} [{tier}]: now={obj} base={base} Δ={delta} → {tag}", flush=True)

    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, 'regression_full_check.csv')
    _safe_dataframe_to_csv(df, out_path)
    n_ok = int(df['within_tolerance'].sum()) if 'within_tolerance' in df.columns else 0
    n_cmp = int(df['delta'].notna().sum()) if 'delta' in df.columns else 0
    print(f"\n[Full回归] {n_ok}/{n_cmp} 例在容差内 → {out_path}", flush=True)
    if n_cmp and n_ok < n_cmp:
        print("[Full回归] 存在明显差异：请检查是否改动了 Full 默认开关/算子逻辑；"
              "必要时重跑 5.3 KG-ALNS 60 例", flush=True)
    return rows, df


def save_ablation_excel_report(detail_df, summary_df, tier_df, output_path,
                              readable_tables=None):
    """保存消融实验 Excel（主表/分梯度/透视/明细）。"""
    if detail_df.empty:
        return None
    if not OPENPYXL_AVAILABLE:
        detail_df.to_csv(output_path.replace('.xlsx', '_detail.csv'),
                         index=False, encoding='utf-8-sig')
        summary_df.to_csv(output_path.replace('.xlsx', '_summary.csv'),
                          index=False, encoding='utf-8-sig')
        print(f"  openpyxl 未安装，消融结果已存 CSV")
        return output_path.replace('.xlsx', '_detail.csv')

    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows

    readable_tables = readable_tables or build_ablation_readable_tables(
        detail_df, summary_df, tier_df)

    wb = Workbook()
    # Sheet1: 总体主表（优先）
    ws_main = wb.active
    ws_main.title = '主表_总体'
    main_overall = readable_tables.get('main_overall')
    if main_overall is not None and not main_overall.empty:
        for r in dataframe_to_rows(main_overall, index=False, header=True):
            ws_main.append(r)
    else:
        for r in dataframe_to_rows(summary_df, index=False, header=True):
            ws_main.append(r)

    ws_tier_main = wb.create_sheet('主表_分梯度')
    main_by_tier = readable_tables.get('main_by_tier')
    if main_by_tier is not None and not main_by_tier.empty:
        for r in dataframe_to_rows(main_by_tier, index=False, header=True):
            ws_tier_main.append(r)
    else:
        for r in dataframe_to_rows(tier_df, index=False, header=True):
            ws_tier_main.append(r)

    ws_case = wb.create_sheet('逐例对照')
    case_detail = readable_tables.get('case_detail')
    if case_detail is not None and not case_detail.empty:
        for r in dataframe_to_rows(case_detail, index=False, header=True):
            ws_case.append(r)

    ws_piv_obj = wb.create_sheet('透视_目标')
    pivot_obj = readable_tables.get('pivot_obj')
    if pivot_obj is not None and not pivot_obj.empty:
        for r in dataframe_to_rows(pivot_obj, index=False, header=True):
            ws_piv_obj.append(r)

    ws_piv_pd = wb.create_sheet('透视_劣化pct')
    pivot_pd = readable_tables.get('pivot_pd')
    if pivot_pd is not None and not pivot_pd.empty:
        for r in dataframe_to_rows(pivot_pd, index=False, header=True):
            ws_piv_pd.append(r)

    ws_sum = wb.create_sheet('汇总_原始')
    for r in dataframe_to_rows(summary_df, index=False, header=True):
        ws_sum.append(r)
    ws_tier = wb.create_sheet('分梯度_原始')
    for r in dataframe_to_rows(tier_df, index=False, header=True):
        ws_tier.append(r)
    ws_det = wb.create_sheet('明细')
    for r in dataframe_to_rows(detail_df, index=False, header=True):
        ws_det.append(r)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    try:
        wb.save(output_path)
    except PermissionError:
        output_path = output_path.replace(
            '.xlsx', f"_{datetime.now().strftime('%H%M%S')}.xlsx")
        wb.save(output_path)
    print(f"  消融 Excel: {output_path}")
    return output_path

def sample_instances_per_tier(instances, per_tier=5, seed=None, tier_filter=None,
                              fixed_ids=None):
    """按梯度分层抽样（用于消融，避免 60 例全量）。

    tier_filter: 如 ['51-60'] 时仅保留困难梯度；None=全部梯度。
    fixed_ids: 非空则按给定 instance_id 列表取算例（正式实验冻结用）。
    seed: 分层随机种子；默认 ABLATION_SAMPLE_SEED。
    """
    if seed is None:
        seed = int(globals().get('ABLATION_SAMPLE_SEED', 20250717) or 20250717)
    if fixed_ids is None:
        fixed_ids = globals().get('ABLATION_FIXED_INSTANCE_IDS', None)
    if fixed_ids:
        want = {int(x) for x in fixed_ids}
        chosen = [i for i in instances if int(i.get('instance_id', -1)) in want]
        chosen.sort(key=lambda x: int(x.get('instance_id', 0)))
        print(f"  [抽样] 使用固定实例 ID ({len(chosen)}/{len(want)}): "
              f"{[int(i['instance_id']) for i in chosen]}", flush=True)
        return chosen
    allow = set(tier_filter) if tier_filter else None
    if not per_tier or per_tier <= 0:
        pool_all = list(instances)
        if allow:
            pool_all = [
                i for i in pool_all
                if (i.get('config') or {}).get('tier', '') in allow
            ]
        return pool_all
    by_tier = {}
    for inst in instances:
        tier = (inst.get('config') or {}).get('tier', 'unknown')
        if allow and tier not in allow:
            continue
        by_tier.setdefault(tier, []).append(inst)
    if not by_tier:
        print(f"  [抽样] tier_filter={tier_filter} 无匹配算例", flush=True)
        return []
    rng = random.Random(seed)
    sampled = []
    for tier in sorted(by_tier.keys()):
        pool = list(by_tier[tier])
        pool.sort(key=lambda x: int(x.get('instance_id', 0)))
        k = min(int(per_tier), len(pool))
        if k >= len(pool):
            chosen = pool
        else:
            chosen = rng.sample(pool, k)
            chosen.sort(key=lambda x: int(x.get('instance_id', 0)))
        sampled.extend(chosen)
        print(f"  [抽样] 梯度 {tier}: {k}/{len(pool)} 例 → "
              f"ids={[int(i['instance_id']) for i in chosen]}", flush=True)
    return sampled


def _save_ablation_instance_manifest(ablation_dir, work_instances, per_tier, seed):
    """冻结本次消融算例 ID，便于复现与正式实验对齐。"""
    ids = [int(i['instance_id']) for i in work_instances]
    by_tier = {}
    for inst in work_instances:
        tier = (inst.get('config') or {}).get('tier', '')
        by_tier.setdefault(tier, []).append(int(inst['instance_id']))
    for t in by_tier:
        by_tier[t] = sorted(by_tier[t])
    payload = {
        'sample_seed': int(seed) if seed is not None else None,
        'sample_per_tier': per_tier,
        'pilot': bool(globals().get('ABLATION_PILOT', False)),
        'fixed_instance_ids': globals().get('ABLATION_FIXED_INSTANCE_IDS', None),
        'instance_ids': ids,
        'instance_ids_by_tier': by_tier,
        'n_instances': len(ids),
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    path = os.path.join(ablation_dir, 'ablation_instance_ids.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[消融] 已冻结算例清单 → {path}", flush=True)
    return path


def _ablation_prepare_run_instance(instance, seed):
    """规范化算例并构造共享初解（串行/并行 run 共用）。"""
    random.seed(seed)
    try:
        np.random.seed(int(seed) % (2 ** 31 - 1))
    except Exception:
        pass
    run_inst = dict(instance)
    tc = run_inst.get('team_config')
    if tc is not None:
        run_inst['team_config'] = normalize_team_config_keys(tc)
    try:
        shared_sched, _ = build_shared_initial_schedule(run_inst, num_starts=6)
        run_inst['_shared_init_schedule'] = shared_sched
    except Exception:
        pass
    return run_inst


def _ablation_single_run_worker(args):
    """ProcessPool 子进程：消融单次独立运行（须模块级可 pickle）。"""
    (instance, time_limit, mip_sub_time_limit, ablation_flags, ablation_variant,
     penalty_coeff, seed, run_id) = args
    # 并行子进程降低 Gurobi 线程，避免 3 进程 × 多线程挤爆 CPU/内存
    global GUROBI_SUB_THREADS
    try:
        GUROBI_SUB_THREADS = 1
    except Exception:
        pass
    try:
        run_inst = _ablation_prepare_run_instance(instance, seed)
        solver = MatheuristicSolver(
            time_limit=time_limit,
            mip_sub_time_limit=mip_sub_time_limit,
            max_iters=150,
            verbose=False,
            ablation_flags=dict(ablation_flags or {}),
            ablation_variant=ablation_variant,
            penalty_coeff=penalty_coeff,
        )
        result = solver.solve(run_inst)
        sched = result.get('schedule')
        feasible = None
        if sched is not None:
            sched, obj, feasible = report_objective_from_schedule(
                sched, run_inst, penalty_coeff=solver.penalty_coeff)
            decomp = (decompose_schedule_objective(
                sched, run_inst, penalty_coeff=solver.penalty_coeff)
                if feasible else {})
        else:
            obj = result.get('objective')
            feasible = result.get('feasible')
            decomp = {}
        out = {
            'run_id': int(run_id),
            'seed': int(seed),
            'objective': obj,
            'solve_time': result.get('solve_time'),
            'mip_repair_count': result.get('mip_repair_count'),
            'feasible': bool(feasible) if feasible is not None else False,
            'kg_plus_stats': dict(result.get('kg_plus_stats') or {}),
            'convergence_trace': dict(result.get('convergence_trace') or {}),
            'total_flow': decomp.get('total_flow'),
            'misallocation_count': decomp.get('misallocation_count'),
            'misallocation_penalty': decomp.get('misallocation_penalty'),
            'dock_misuse_count': decomp.get('dock_misuse_count'),
            # 低内存：不回传调度表，主进程只汇总标量
            'schedule': None if LARGE_LOW_MEMORY_MODE else sched,
        }
        try:
            solver._insertion_engine = None
        except Exception:
            pass
        del solver, result, run_inst
        gc.collect()
        return out
    except Exception as e:
        gc.collect()
        return {
            'run_id': int(run_id),
            'seed': int(seed),
            'objective': None,
            'solve_time': None,
            'mip_repair_count': 0,
            'feasible': False,
            'schedule': None,
            'error': f'{type(e).__name__}: {e}',
        }


def _ablation_aggregate_run_results(run_results, num_runs):
    """将多次 run 结果聚合成与串行版相同的字典。

    均值/标准差仅纳入可行且目标有效的 run；不可行单独记入 feasible_runs。
    标准差使用样本标准差 (ddof=1)。
    """
    n = max(1, int(num_runs))
    ordered = sorted(run_results, key=lambda r: int(r.get('run_id', 0)))
    objs = [r.get('objective') for r in ordered]
    seeds = [r.get('seed') for r in ordered]
    feas_flags = []
    for r in ordered:
        f = r.get('feasible')
        feas_flags.append(bool(f) if f is not None else False)
    times = [r.get('solve_time') for r in ordered]
    mip_counts = [r.get('mip_repair_count') for r in ordered]
    best_sched = None
    best_obj = None
    for r in ordered:
        obj = r.get('objective')
        sched = r.get('schedule')
        if obj is None or (isinstance(obj, float) and pd.isna(obj)):
            continue
        if not r.get('feasible', True):
            continue
        if best_obj is None or obj < best_obj:
            best_obj = obj
            if sched is not None:
                best_sched = sched

    valid = []
    for r, o in zip(ordered, objs):
        if o is None or (isinstance(o, float) and pd.isna(o)):
            continue
        if not r.get('feasible', False):
            continue
        valid.append(float(o))

    all_runs_feasible = bool(feas_flags) and all(feas_flags)

    if not valid:
        empty_decomp = _aggregate_objective_decomposition(ordered)
        return {
            'objective': None,
            'objective_mean': None,
            'objective_std': None,
            'objective_best': None,
            'objective_runs': objs,
            'seed_runs': seeds,
            'feasible_runs': feas_flags,
            'all_runs_feasible': all_runs_feasible,
            'n_feasible_runs': 0,
            'solve_time': None,
            'mip_repair_count': 0,
            'num_runs': n,
            'feasible': all_runs_feasible,
            'schedule': best_sched,
            'kg_plus_stats': _aggregate_kg_plus_stats(ordered),
            'convergence_trace': _aggregate_convergence_traces(ordered),
            **empty_decomp,
        }
    std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    return {
        'objective': float(np.mean(valid)),
        'objective_mean': float(np.mean(valid)),
        'objective_std': std,
        'objective_best': float(min(valid)),
        'objective_runs': objs,
        'seed_runs': seeds,
        'feasible_runs': feas_flags,
        'all_runs_feasible': all_runs_feasible,
        'n_feasible_runs': len(valid),
        'solve_time': float(np.mean([t for t in times if t is not None] or [0.0])),
        'mip_repair_count': float(np.mean([
            m for m in mip_counts if m is not None] or [0.0])),
        'num_runs': n,
        'feasible': all_runs_feasible,
        'schedule': best_sched,
        'kg_plus_stats': _aggregate_kg_plus_stats(ordered),
        'convergence_trace': _aggregate_convergence_traces(ordered),
        **_aggregate_objective_decomposition(ordered),
    }


def _aggregate_objective_decomposition(run_results):
    """可行 run 上 total_flow / misallocation / dock_misuse 的均值。"""
    keys = (
        'total_flow', 'misallocation_count',
        'misallocation_penalty', 'dock_misuse_count',
    )
    buckets = {k: [] for k in keys}
    for r in run_results:
        if not r.get('feasible', False):
            continue
        for k in keys:
            v = r.get(k)
            if v is None or (isinstance(v, float) and (pd.isna(v) or not math.isfinite(v))):
                continue
            buckets[k].append(float(v))
    return {
        k: (float(np.mean(vals)) if vals else None)
        for k, vals in buckets.items()
    }


def _aggregate_kg_plus_stats(run_results):
    """多次 run 的 kg_plus_stats 求和（计数器）。"""
    keys = (
        'knowledge_guide_count', 'knowledge_candidate_count',
        'knowledge_changed_choice_count', 'knowledge_improvement_count',
        'knowledge_destroy_selected_count', 'skill_refine_improve',
        'elite_restart_count',
    )
    out = {k: 0 for k in keys}
    for r in run_results:
        stats = r.get('kg_plus_stats') or {}
        if isinstance(stats, str):
            stats = _parse_ablation_json_field(stats, {})
        for k in keys:
            try:
                out[k] += int(stats.get(k, 0) or 0)
            except (TypeError, ValueError):
                pass
    return out


def _aggregate_convergence_traces(run_results):
    """多次 run 在各检查点上的可行目标均值。"""
    cps = list(globals().get('CONVERGENCE_CHECKPOINT_SEC', (30, 60, 120, 180, 300)))
    buckets = {int(t): [] for t in cps}
    for r in run_results:
        if not r.get('feasible', False):
            continue
        trace = r.get('convergence_trace') or {}
        if isinstance(trace, str):
            trace = _parse_ablation_json_field(trace, {})
        parsed = _parse_convergence_trace_field(trace)
        for t in cps:
            v = parsed.get(int(t))
            if v is not None and math.isfinite(float(v)):
                buckets[int(t)].append(float(v))
    out = {}
    for t, vals in buckets.items():
        if vals:
            out[int(t)] = float(np.mean(vals))
    return out


def _ablation_instance_worker(args):
    """ProcessPool：整例求解（内部 runs 串行，用于 num_runs==1 的算例并行）。"""
    (instance, time_limit, mip_sub_time_limit, ablation_flags, ablation_variant,
     penalty_coeff, seed_base, num_runs) = args
    try:
        solver = MatheuristicSolver(
            time_limit=time_limit,
            mip_sub_time_limit=mip_sub_time_limit,
            max_iters=150,
            verbose=False,
            ablation_flags=dict(ablation_flags or {}),
            ablation_variant=ablation_variant,
            penalty_coeff=penalty_coeff,
        )
        # 子进程内强制串行 run，避免嵌套进程池
        solved = _ablation_solve_one(
            solver, instance, seed_base=seed_base, num_runs=num_runs,
            executor=None, parallel_workers=1)
        solved = dict(solved)
        solved['instance_id'] = int(instance['instance_id'])
        solved['num_ships'] = instance.get('num_ships')
        solved['num_tasks'] = instance.get('num_tasks')
        solved['tier'] = (instance.get('config') or {}).get('tier', '')
        solved.pop('schedule', None)
        try:
            solver._insertion_engine = None
        except Exception:
            pass
        del solver
        gc.collect()
        return solved
    except Exception as e:
        gc.collect()
        return {
            'instance_id': int(instance.get('instance_id', -1)),
            'num_ships': instance.get('num_ships'),
            'num_tasks': instance.get('num_tasks'),
            'tier': (instance.get('config') or {}).get('tier', ''),
            'objective': None,
            'objective_std': None,
            'objective_best': None,
            'objective_runs': None,
            'solve_time': None,
            'mip_repair_count': 0,
            'num_runs': num_runs,
            'error': f'{type(e).__name__}: {e}',
        }


def _ablation_build_result_row(var_key, spec, instance, result, full_baselines):
    """组装消融 CSV 行（含逐 run 目标/种子/可行性）。"""
    iid = int(instance['instance_id'])
    tier = (instance.get('config') or {}).get('tier', '')
    obj = result.get('objective')
    baseline = full_baselines.get(iid)
    gap_pct = None
    if baseline is not None and obj is not None and baseline > 0:
        gap_pct = (obj - baseline) / baseline * 100
    mods = spec.get('modules') or {}
    return {
        'variant': var_key,
        'variant_label': spec['label'],
        'paper_name': spec.get('paper_name', spec['label']),
        'knowledge_eval': 'Y' if mods.get('knowledge_eval') else 'N',
        'knowledge_destroy': 'Y' if mods.get('knowledge_destroy') else 'N',
        'knowledge_repair': 'Y' if mods.get('knowledge_repair') else 'N',
        'adaptive': 'Y' if mods.get('adaptive') else 'N',
        'elite_restart': ('-' if 'elite_restart' not in mods
                          else ('Y' if mods['elite_restart'] else 'N')),
        'skill_refine': ('-' if 'skill_refine' not in mods
                         else ('Y' if mods['skill_refine'] else 'N')),
        'endgame': 'Y' if mods.get('endgame') else 'N',
        'destroy_l1': ('-' if 'destroy_l1' not in mods
                       else ('Y' if mods['destroy_l1'] else 'N')),
        'destroy_l3': ('-' if 'destroy_l3' not in mods
                       else ('Y' if mods['destroy_l3'] else 'N')),
        'destroy_l4': ('-' if 'destroy_l4' not in mods
                       else ('Y' if mods['destroy_l4'] else 'N')),
        'instance_id': iid,
        'num_ships': instance['num_ships'],
        'num_tasks': instance['num_tasks'],
        'tier': tier,
        'objective': obj,
        'objective_std': result.get('objective_std'),
        'objective_best': result.get('objective_best'),
        'objective_runs': json.dumps(result.get('objective_runs')),
        'seed_runs': json.dumps(result.get('seed_runs')),
        'feasible_runs': json.dumps(result.get('feasible_runs')),
        'all_runs_feasible': result.get('all_runs_feasible'),
        'n_feasible_runs': result.get('n_feasible_runs'),
        'feasible': result.get('feasible'),
        'num_runs': result.get('num_runs', 1),
        'solve_time': result.get('solve_time'),
        'mip_repair_count': result.get('mip_repair_count'),
        'gap_vs_full_pct': gap_pct,
        'source': 'ablation_run',
        'eval_protocol': 'report_objective_from_schedule',
        'total_flow': result.get('total_flow'),
        'misallocation_count': result.get('misallocation_count'),
        'misallocation_penalty': result.get('misallocation_penalty'),
        'dock_misuse_count': result.get('dock_misuse_count'),
        'kg_plus_stats': json.dumps(
            result.get('kg_plus_stats') or {}, ensure_ascii=False),
        'convergence_trace': json.dumps(
            result.get('convergence_trace') or {}, ensure_ascii=False),
    }


def _ablation_solve_one(solver, instance, seed_base, num_runs=1, executor=None,
                        parallel_workers=None):
    """消融单例求解；num_runs>1 时返回 mean 目标（SCI 稳定性）。

    公平协议：每个 run 使用配对种子、共享初解逻辑，并重新创建求解器，
    避免权重/统计量残留。
    """
    n = max(1, int(num_runs))
    workers = parallel_workers
    if workers is None:
        workers = int(globals().get('ABLATION_PARALLEL_WORKERS', 1) or 1)
    use_parallel = (
        bool(globals().get('ABLATION_PARALLEL_RUNS', False))
        and n > 1
        and int(workers) > 1
    )

    flags = dict(getattr(solver, 'ablation_flags', {}) or {})
    variant = getattr(solver, 'ablation_variant', 'full')
    penalty = getattr(solver, 'penalty_coeff', MISALLOCATION_PENALTY_COEFF)
    time_limit = getattr(solver, 'time_limit', LARGE_BATCH_TIME_LIMIT)
    mip_sub = getattr(solver, 'mip_sub_time_limit', LARGE_MIP_SUB_TIME_LIMIT)

    def _seed_for_run(run):
        iid = int(instance.get('instance_id', 0) or 0)
        if callable(globals().get('ablation_paired_seed')):
            return ablation_paired_seed(iid, run)
        return int(seed_base) + int(run)

    if not use_parallel:
        run_results = []
        best_sched = None
        best_obj = None
        for run in range(n):
            seed = _seed_for_run(run)
            run_inst = _ablation_prepare_run_instance(instance, seed)
            # 每个 run 新建求解器，避免自适应权重/统计残留
            run_solver = MatheuristicSolver(
                time_limit=time_limit,
                mip_sub_time_limit=mip_sub,
                max_iters=150,
                verbose=False,
                ablation_flags=dict(flags),
                ablation_variant=variant,
                penalty_coeff=penalty,
            )
            result = run_solver.solve(run_inst)
            sched = result.get('schedule')
            feasible = None
            if sched is not None:
                sched, obj, feasible = report_objective_from_schedule(
                    sched, run_inst, penalty_coeff=run_solver.penalty_coeff)
                decomp = (decompose_schedule_objective(
                    sched, run_inst, penalty_coeff=run_solver.penalty_coeff)
                    if feasible else {})
            else:
                obj = result.get('objective')
                feasible = result.get('feasible')
                decomp = {}
            feasible = bool(feasible) if feasible is not None else False
            if (obj is not None and not (isinstance(obj, float) and pd.isna(obj))
                    and feasible):
                if best_obj is None or obj < best_obj:
                    best_obj = obj
                    best_sched = sched
            run_results.append({
                'run_id': run,
                'seed': seed,
                'objective': obj,
                'solve_time': result.get('solve_time'),
                'mip_repair_count': result.get('mip_repair_count'),
                'feasible': feasible,
                'kg_plus_stats': dict(result.get('kg_plus_stats') or {}),
                'convergence_trace': dict(result.get('convergence_trace') or {}),
                'total_flow': decomp.get('total_flow'),
                'misallocation_count': decomp.get('misallocation_count'),
                'misallocation_penalty': decomp.get('misallocation_penalty'),
                'dock_misuse_count': decomp.get('dock_misuse_count'),
                'schedule': None,
            })
            try:
                run_solver._insertion_engine = None
            except Exception:
                pass
            del run_solver, result, run_inst
            gc.collect()
        out = _ablation_aggregate_run_results(run_results, n)
        out['schedule'] = best_sched
        return out

    job_args = [
        (instance, time_limit, mip_sub, flags, variant, penalty,
         _seed_for_run(run), run)
        for run in range(n)
    ]
    max_workers = min(int(workers), n)
    own_pool = executor is None
    pool = executor
    try:
        if own_pool:
            pool = ProcessPoolExecutor(max_workers=max_workers)
        futures = [pool.submit(_ablation_single_run_worker, a) for a in job_args]
        run_results = [fut.result() for fut in futures]
        for rr in run_results:
            if rr.get('error'):
                print(f"      [run并行] run={rr.get('run_id')} {rr['error']}",
                      flush=True)
    finally:
        if own_pool and pool is not None:
            try:
                pool.shutdown(wait=True, cancel_futures=False)
            except TypeError:
                pool.shutdown(wait=True)
    return _ablation_aggregate_run_results(run_results, n)


def _ablation_flat_job_worker(args):
    """扁平任务池：单次 (variant×instance×run)，返回带 variant/instance_id 的 run 结果。"""
    (var_key, flags, instance, time_limit, mip_sub, penalty, seed, run_id) = args
    out = _ablation_single_run_worker(
        (instance, time_limit, mip_sub, flags, var_key, penalty, seed, run_id))
    out = dict(out)
    out['variant'] = var_key
    out['instance_id'] = int(instance.get('instance_id', -1))
    return out


def run_ablation_study(output_root=OUTPUT_ROOT, batch_time_limit=None,
                       mip_sub_time_limit=None, variants=None, resume=True,
                       large_exp=None, import_full_from_large_scale=False,
                       output_subdir=None, tier_filter=None, sample_per_tier=None,
                       num_runs=None):
    """批量消融实验：full 与各变体 × 抽样大规模算例

    公平协议（与大规模报告口径一致）：
      - 复用 large_scale/ 同一批算例
      - 每个变体（含 full）同墙钟、同种子规则
      - 最终目标统一经 report_objective_from_schedule
      - 默认不从 large_scale_results 导入 full

    output_subdir: 'ablation_final_x3' / 'ablation_final_x3_pilot' / 'ablation_operators'
    tier_filter: 如 ['51-60'] 仅难例
    num_runs: >1 时取可行 run 的 mean；ABLATION_FLAT_PARALLEL 时扁平并行
    """
    batch_time_limit = batch_time_limit or LARGE_BATCH_TIME_LIMIT
    mip_sub_time_limit = mip_sub_time_limit or LARGE_MIP_SUB_TIME_LIMIT
    variants = variants or ABLATION_VARIANTS_TO_RUN or list(ABLATION_VARIANTS.keys())
    output_subdir = output_subdir or globals().get('ABLATION_OUTPUT_SUBDIR', 'ablation')
    tier_filter = tier_filter if tier_filter is not None else globals().get(
        'ABLATION_TIER_FILTER', None)
    per_tier = sample_per_tier if sample_per_tier is not None else globals().get(
        'ABLATION_SAMPLE_PER_TIER', None)
    num_runs = int(num_runs if num_runs is not None else globals().get(
        'ABLATION_NUM_RUNS', 1) or 1)
    sample_seed = int(globals().get('ABLATION_SAMPLE_SEED', 20250717) or 20250717)

    flat_parallel = bool(globals().get('ABLATION_FLAT_PARALLEL', False))
    flat_workers = max(1, int(globals().get('ABLATION_FLAT_WORKERS', 3) or 3))
    parallel_runs = (
        (not flat_parallel)
        and bool(globals().get('ABLATION_PARALLEL_RUNS', False))
        and num_runs > 1
    )
    run_workers = max(1, int(globals().get('ABLATION_PARALLEL_WORKERS', 1) or 1))
    parallel_instances = (
        (not flat_parallel)
        and bool(globals().get('ABLATION_PARALLEL_INSTANCES', False))
        and num_runs <= 1
        and not parallel_runs
    )
    inst_workers = max(1, int(globals().get('ABLATION_INSTANCE_WORKERS', 1) or 1))
    if parallel_runs:
        run_workers = min(run_workers, num_runs)
    if parallel_instances:
        inst_workers = max(1, inst_workers)
    if flat_parallel:
        flat_workers = max(1, flat_workers)

    ablation_dir = os.path.join(output_root, output_subdir)
    large_dir = os.path.join(output_root, 'large_scale')
    # 若根目录无 large_scale，回退到「有效数据/large_scale」（整理后的正式数据）
    if not os.path.isfile(os.path.join(large_dir, 'large_configs.json')):
        for alt_name in ('有效数据', 'valid_data'):
            alt = os.path.join(output_root, alt_name, 'large_scale')
            if os.path.isfile(os.path.join(alt, 'large_configs.json')):
                print(f"[消融] 使用算例目录: {alt}", flush=True)
                large_dir = alt
                break
    os.makedirs(ablation_dir, exist_ok=True)
    os.makedirs(large_dir, exist_ok=True)
    print(f"\n[消融] 启动 — 输出目录: {ablation_dir}", flush=True)
    print(f"[消融] study={globals().get('ABLATION_STUDY', '?')} | "
          f"pilot={globals().get('ABLATION_PILOT', False)} | "
          f"subdir={output_subdir} | tiers={tier_filter or 'ALL'} | "
          f"runs×{num_runs}", flush=True)
    if flat_parallel:
        print(f"[消融] 并行: 扁平池 (variant×instance×run) ×{flat_workers} 进程",
              flush=True)
    elif parallel_runs:
        print(f"[消融] 并行: 同例 multi-run ×{run_workers} 进程", flush=True)
    elif parallel_instances:
        print(f"[消融] 并行: 同变体多算例 ×{inst_workers} 进程", flush=True)
    else:
        print("[消融] 并行: 关闭（串行）", flush=True)
    print(f"[消融] 公平协议: report_objective | full同跑="
          f"{not import_full_from_large_scale} | 时限={batch_time_limit}s",
          flush=True)

    if large_exp is not None and large_exp.instances:
        exp = large_exp
        configs = large_exp.experiment_configs
        print(f"\n[消融] 复用阶段二算例实例 ({len(exp.instances)} 例)")
    else:
        saved_configs = load_large_configs(large_dir)
        if saved_configs:
            configs = saved_configs
            print(f"\n[消融] 从 large_scale/large_configs.json 加载 {len(configs)} 组算例配置")
        else:
            random.seed(REPRODUCIBILITY_SEED)
            np.random.seed(REPRODUCIBILITY_SEED)
            configs = generate_large_scale_configs(60, seed=REPRODUCIBILITY_SEED)
            save_large_configs(configs, large_dir)
        exp = BatchExperimentGurobi(output_dir=large_dir, heuristic_type='lNS')
        exp.set_experiment_configs(configs)
        print("\n[消融] 加载算例实例...", flush=True)
        exp.generate_instances(force_regen=False, quiet=True)
        if not exp.instances:
            exp.generate_instances(force_regen=False, quiet=False)
        if not glob.glob(os.path.join(large_dir, 'instance_*_tasks.csv')):
            exp.save_instances()
            save_large_configs(configs, large_dir)

    # 防御：内存/磁盘算例一律规范化 team_config 键为 int
    for inst in exp.instances:
        tc = inst.get('team_config')
        if tc is not None:
            inst['team_config'] = normalize_team_config_keys(tc)

    # 本地池：勿改写 large_exp.instances（管道后续阶段可能复用）
    work_instances = list(exp.instances)

    # 消融 XL（如 71-80）：独立池合并进来；不写 large_scale，不影响阶段二/敏感性
    xl_labels = set(globals().get('ABLATION_XL_TIER_LABELS', ()) or ())
    need_xl = bool(xl_labels) and (
        tier_filter is None
        or any(t in xl_labels for t in (tier_filter or []))
    )
    if need_xl:
        xl_insts = load_or_create_ablation_xl_instances(output_root)
        if xl_insts:
            existing_ids = {int(i['instance_id']) for i in work_instances}
            added = 0
            for inst in xl_insts:
                iid = int(inst['instance_id'])
                if iid in existing_ids:
                    continue
                work_instances.append(inst)
                existing_ids.add(iid)
                added += 1
            print(f"[消融] 已合并 XL 算例 {added} 例 "
                  f"(tiers={sorted(xl_labels)})，池大小={len(work_instances)}",
                  flush=True)

    fixed_ids = globals().get('ABLATION_FIXED_INSTANCE_IDS', None)
    if per_tier or tier_filter or fixed_ids:
        print(f"[消融] 按梯度抽样：每梯度 ≤{per_tier or 'ALL'} | filter={tier_filter} | "
              f"seed={sample_seed}", flush=True)
        work_instances = sample_instances_per_tier(
            work_instances,
            per_tier=per_tier if per_tier else 9999,
            seed=sample_seed,
            tier_filter=tier_filter)
        print(f"[消融] 抽样后共 {len(work_instances)} 例", flush=True)

    if not work_instances:
        print("[消融] 无算例，退出")
        return [], pd.DataFrame()

    _save_ablation_instance_manifest(
        ablation_dir, work_instances, per_tier, sample_seed)

    results_path = os.path.join(ablation_dir, 'ablation_results.csv')
    existing_rows = []
    if resume and os.path.isfile(results_path):
        try:
            existing_rows = pd.read_csv(results_path, encoding='utf-8-sig').to_dict('records')
            print(f"[消融] 续跑：已有 {len(existing_rows)} 条记录")
        except Exception:
            existing_rows = []

    all_rows, dropped = filter_ablation_rows_to_instances(existing_rows, work_instances)
    if dropped:
        print(f"[消融] 已丢弃 {dropped} 条与当前算例不一致的旧记录（算例规模已变更）")

    all_variant_keys = list(variants) if variants else list(
        ABLATION_VARIANTS_TO_RUN or ABLATION_VARIANTS.keys())
    for k in list(variants or ABLATION_VARIANTS_TO_RUN or []):
        if k not in all_variant_keys:
            all_variant_keys.append(k)
    if all_rows or dropped:
        write_ablation_outputs(all_rows, ablation_dir, all_variant_keys)
        if dropped:
            print(f"[消融] 已重写 ablation_results.csv（保留 {len(all_rows)} 条有效记录）",
                  flush=True)

    existing_full_ids = {int(r['instance_id']) for r in all_rows
                         if r.get('variant') == 'full' and r.get('objective') is not None}

    # 次级 Endgame 拆分：从 SCI 主消融目录导入 Full（同实例、同 5-run 均值），不重跑 Full
    import_full_subdir = globals().get('ABLATION_IMPORT_FULL_FROM_SUBDIR', None)
    if (globals().get('ABLATION_STUDY') == 'secondary'
            and import_full_subdir
            and not import_full_from_large_scale):
        src_dir = import_full_subdir
        if not os.path.isabs(src_dir):
            src_dir = os.path.join(output_root, src_dir)
        imported_full = import_full_baseline_from_ablation_dir(src_dir, work_instances)
        for row in imported_full:
            iid = int(row['instance_id'])
            if iid not in existing_full_ids:
                all_rows.append(row)
                existing_full_ids.add(iid)
        if imported_full:
            write_ablation_outputs(all_rows, ablation_dir, all_variant_keys)

    # Knowledge×Elite 交互：从主/次级消融导入已有变体，仅跑 w/o K+E
    import_var_map = globals().get('ABLATION_IMPORT_VARIANTS_FROM', None)
    if (globals().get('ABLATION_STUDY') == 'knowledge_interaction'
            and import_var_map
            and not import_full_from_large_scale):
        existing_by_var = {}
        for r in all_rows:
            if r.get('objective') is None:
                continue
            vk = str(r.get('variant', ''))
            existing_by_var.setdefault(vk, set()).add(int(r['instance_id']))
        any_imported = False
        for var_key, subdir in import_var_map.items():
            if var_key not in (variants or []):
                continue
            src_dir = subdir
            if not os.path.isabs(src_dir):
                src_dir = os.path.join(output_root, src_dir)
            imported = import_ablation_variant_from_dir(
                var_key, src_dir, work_instances)
            have = existing_by_var.get(var_key, set())
            for row in imported:
                iid = int(row['instance_id'])
                if iid not in have:
                    all_rows.append(row)
                    have.add(iid)
                    any_imported = True
            existing_by_var[var_key] = have
            if var_key == 'full':
                existing_full_ids.update(have)
        if any_imported:
            write_ablation_outputs(all_rows, ablation_dir, all_variant_keys)

    if import_full_from_large_scale:
        print("[消融][警告] import_full_from_large_scale=True："
              "full 为大规模×3均值，与变体口径可能不一致", flush=True)
        imported_full = import_full_baseline_from_large_scale(large_dir, work_instances)
        if imported_full:
            print(f"[消融] 从 large_scale_results.csv 导入 full 基准 {len(imported_full)} 例")
            for row in imported_full:
                iid = int(row['instance_id'])
                if iid not in existing_full_ids:
                    all_rows.append(row)
                    existing_full_ids.add(iid)

    full_baselines = {
        int(r['instance_id']): float(r['objective'])
        for r in all_rows
        if r.get('variant') == 'full' and r.get('objective') is not None
    }

    variants_to_run = list(variants)
    # 已有完整 Full 基准（导入或续跑）→ 跳过重跑 Full
    if len(full_baselines) >= len(work_instances) and 'full' in variants_to_run:
        variants_to_run = [v for v in variants_to_run if v != 'full']
        print(f"[消融] full 已具备 {len(full_baselines)} 例基准，跳过 Full 重跑"
              f"（待跑变体={variants_to_run}）", flush=True)
    # 交互实验：已导入的变体也跳过
    if globals().get('ABLATION_STUDY') == 'knowledge_interaction':
        skip = []
        for vk in list(variants_to_run):
            n_have = sum(
                1 for r in all_rows
                if r.get('variant') == vk and r.get('objective') is not None)
            if n_have >= len(work_instances):
                skip.append(vk)
        if skip:
            variants_to_run = [v for v in variants_to_run if v not in skip]
            print(f"[消融] 已导入变体跳过重跑: {skip} → 待跑={variants_to_run}",
                  flush=True)

    if flat_parallel:
        wall_factor = 1.0 / flat_workers
    elif parallel_runs:
        wall_factor = 1.0 / run_workers
    elif parallel_instances:
        wall_factor = 1.0 / inst_workers
    else:
        wall_factor = 1.0
    est_per_case = batch_time_limit * num_runs / 60.0 * wall_factor
    est_total = len(variants_to_run) * len(work_instances) * est_per_case
    print(f"\n[消融] 待跑变体 {len(variants_to_run)} 个 × 算例 {len(work_instances)} 个 "
          f"× {num_runs} runs, 时限={batch_time_limit}s/run, "
          f"预计墙钟≈{est_total:.0f} 分钟"
          + (f"（并行加速约×{1.0 / wall_factor:.1f}）" if wall_factor < 1 else ""))
    print(f"[消融] 输出目录: {ablation_dir}")

    def _commit_ablation_instance_result(var_key, instance, result):
        nonlocal results_path
        iid = int(instance['instance_id'])
        obj = result.get('objective')
        if var_key == 'full' and obj is not None:
            full_baselines[iid] = float(obj)
        spec = ABLATION_VARIANTS[var_key]
        row = _ablation_build_result_row(
            var_key, spec, instance, result, full_baselines)
        all_rows.append(row)
        results_path = _safe_dataframe_to_csv(pd.DataFrame(all_rows), results_path)
        if obj is not None and num_runs > 1:
            print(f"  [{var_key}] Case {iid} → mean={obj:.2f} "
                  f"±{result.get('objective_std') or 0:.2f} "
                  f"(best={result.get('objective_best')}, "
                  f"feas={result.get('n_feasible_runs')}/{num_runs})",
                  flush=True)
        elif obj is not None:
            print(f"  [{var_key}] Case {iid} → obj={obj:.2f}", flush=True)

    run_pool = None
    try:
        if flat_parallel and flat_workers >= 1:
            if sys.platform == 'win32':
                print(f"[提示] Windows 扁平并行={flat_workers}；"
                      f"OOM 时设 ABLATION_FLAT_WORKERS=1 或 2", flush=True)
            # 收集所有待跑 (variant, instance, run)
            job_args = []
            inst_by_key = {}
            for var_key in variants_to_run:
                if var_key not in ABLATION_VARIANTS:
                    print(f"  跳过未知变体: {var_key}")
                    continue
                spec = ABLATION_VARIANTS[var_key]
                done_ids = (
                    load_completed_ablation_ids(ablation_dir, var_key, all_rows)
                    if resume else set())
                pending = [
                    inst for inst in work_instances
                    if int(inst['instance_id']) not in done_ids]
                print(f"\n[消融变体] {var_key}: {spec['label']} "
                      f"(待跑 {len(pending)}/{len(work_instances)})", flush=True)
                for inst in pending:
                    iid = int(inst['instance_id'])
                    inst_by_key[(var_key, iid)] = inst
                    for run in range(num_runs):
                        seed = ablation_paired_seed(iid, run)
                        job_args.append((
                            var_key, dict(spec['flags']), inst,
                            batch_time_limit, mip_sub_time_limit,
                            MISALLOCATION_PENALTY_COEFF, seed, run,
                        ))
            print(f"\n[消融] 扁平任务数={len(job_args)} "
                  f"(variant×instance×run)", flush=True)
            if not job_args:
                print("[消融] 全部变体/算例已完成，跳过求解", flush=True)
            else:
                from collections import defaultdict
                buckets = defaultdict(list)
                written = set()
                try:
                    with ProcessPoolExecutor(max_workers=flat_workers) as pool:
                        futs = {
                            pool.submit(_ablation_flat_job_worker, a): a
                            for a in job_args
                        }
                        for fut in as_completed(futs):
                            a = futs[fut]
                            var_key = a[0]
                            iid = int(a[2]['instance_id'])
                            try:
                                rr = fut.result()
                            except Exception as e:
                                rr = {
                                    'run_id': a[7], 'seed': a[6],
                                    'objective': None, 'feasible': False,
                                    'solve_time': None, 'mip_repair_count': 0,
                                    'destroy_operator_stats': {},
                                    'error': f'{type(e).__name__}: {e}',
                                    'variant': var_key, 'instance_id': iid,
                                }
                            if rr.get('error'):
                                print(f"    [{var_key}] Case {iid} run={rr.get('run_id')} "
                                      f"{rr['error']}", flush=True)
                            key = (var_key, iid)
                            buckets[key].append(rr)
                            if len(buckets[key]) >= num_runs and key not in written:
                                result = _ablation_aggregate_run_results(
                                    buckets[key], num_runs)
                                _commit_ablation_instance_result(
                                    var_key, inst_by_key[key], result)
                                written.add(key)
                                # 每完成一例刷新汇总
                                if len(written) % max(1, len(work_instances) // 3) == 0:
                                    write_ablation_outputs(
                                        all_rows, ablation_dir, all_variant_keys)
                except BrokenExecutor as e:
                    print(f"[扁平并行失败] {e} → 回退串行路径", flush=True)
                    flat_parallel = False
                except Exception as e:
                    print(f"[扁平并行失败] {type(e).__name__}: {e} → 回退串行路径",
                          flush=True)
                    flat_parallel = False
                write_ablation_outputs(all_rows, ablation_dir, all_variant_keys)

        if not flat_parallel:
            if parallel_runs and run_workers > 1:
                if sys.platform == 'win32':
                    print(f"[提示] Windows 消融 run 并行={run_workers}；"
                          f"若页面文件/OOM 请设 ABLATION_PARALLEL_WORKERS=1 或 2",
                          flush=True)
                run_pool = ProcessPoolExecutor(max_workers=run_workers)

            for var_key in variants_to_run:
                if var_key not in ABLATION_VARIANTS:
                    print(f"  跳过未知变体: {var_key}")
                    continue
                spec = ABLATION_VARIANTS[var_key]
                done_ids = (
                    load_completed_ablation_ids(ablation_dir, var_key, all_rows)
                    if resume else set())
                pending = [
                    inst for inst in work_instances
                    if int(inst['instance_id']) not in done_ids]
                print(f"\n{'=' * 72}\n[消融变体] {var_key}: {spec['label']} "
                      f"(待跑 {len(pending)}/{len(work_instances)})\n{'=' * 72}")

                if not pending:
                    print(f"  [{var_key}] 已全部完成，跳过求解")
                    write_ablation_outputs(all_rows, ablation_dir, all_variant_keys)
                    continue

                solver = MatheuristicSolver(
                    time_limit=batch_time_limit,
                    mip_sub_time_limit=mip_sub_time_limit,
                    max_iters=150,
                    verbose=False,
                    ablation_flags=spec['flags'],
                    ablation_variant=var_key,
                )

                ran_instance_parallel = False
                if parallel_instances and inst_workers > 1 and len(pending) > 1:
                    print(f"  [{var_key}] 算例并行 ×{min(inst_workers, len(pending))}",
                          flush=True)
                    job_args = [
                        (inst, batch_time_limit, mip_sub_time_limit, spec['flags'],
                         var_key, solver.penalty_coeff,
                         int(inst['instance_id']) * 100 + 42, num_runs)
                        for inst in pending
                    ]
                    inst_by_id = {int(i['instance_id']): i for i in pending}
                    try:
                        with ProcessPoolExecutor(
                                max_workers=min(inst_workers, len(pending))) as ipool:
                            futs = {
                                ipool.submit(_ablation_instance_worker, a):
                                a[0]['instance_id']
                                for a in job_args
                            }
                            for fut in futs:
                                iid = int(futs[fut])
                                try:
                                    result = fut.result()
                                except Exception as e:
                                    print(f"    Case {iid} 并行异常: {e}", flush=True)
                                    result = {
                                        'objective': None, 'solve_time': None,
                                        'mip_repair_count': 0, 'num_runs': num_runs,
                                        'objective_std': None, 'objective_best': None,
                                    }
                                if result.get('error'):
                                    print(f"    Case {iid} {result['error']}", flush=True)
                                _commit_ablation_instance_result(
                                    var_key, inst_by_id[iid], result)
                        ran_instance_parallel = True
                    except BrokenExecutor as e:
                        print(f"  [算例并行失败] {e} → 回退串行", flush=True)
                        parallel_instances = False
                    except Exception as e:
                        print(f"  [算例并行失败] {type(e).__name__}: {e} → 回退串行",
                              flush=True)
                        parallel_instances = False

                if not ran_instance_parallel:
                    for instance in pending:
                        if resume:
                            done_now = load_completed_ablation_ids(
                                ablation_dir, var_key, all_rows)
                            if int(instance['instance_id']) in done_now:
                                continue
                        iid = instance['instance_id']
                        tier = instance.get('config', {}).get('tier', '')
                        print(f"  [{var_key}] Case {iid}: {instance['num_ships']}船, "
                              f"{instance['num_tasks']}任务 [{tier}] ×{num_runs}"
                              + (f" (run∥{run_workers})" if parallel_runs else ""),
                              flush=True)
                        try:
                            solved = _ablation_solve_one(
                                solver, instance,
                                seed_base=ablation_paired_seed(iid, 0),
                                num_runs=num_runs, executor=run_pool,
                                parallel_workers=(
                                    run_workers if parallel_runs else 1))
                            result = solved
                        except BrokenExecutor as e:
                            print(f"    [run并行池损坏] {e} → 本例改串行", flush=True)
                            if run_pool is not None:
                                try:
                                    run_pool.shutdown(wait=False, cancel_futures=True)
                                except TypeError:
                                    try:
                                        run_pool.shutdown(wait=False)
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                                run_pool = None
                            parallel_runs = False
                            try:
                                result = _ablation_solve_one(
                                    solver, instance,
                                    seed_base=ablation_paired_seed(iid, 0),
                                    num_runs=num_runs, executor=None,
                                    parallel_workers=1)
                            except Exception as e2:
                                print(f"    异常: {e2}", flush=True)
                                result = {
                                    'objective': None, 'solve_time': None,
                                    'mip_repair_count': 0, 'num_runs': num_runs,
                                    'objective_std': None, 'objective_best': None,
                                    'objective_runs': None,
                                }
                        except Exception as e:
                            print(f"    异常: {e}", flush=True)
                            result = {
                                'objective': None, 'solve_time': None,
                                'mip_repair_count': 0, 'num_runs': num_runs,
                                'objective_std': None, 'objective_best': None,
                                'objective_runs': None,
                            }
                        _commit_ablation_instance_result(var_key, instance, result)

                write_ablation_outputs(all_rows, ablation_dir, all_variant_keys)
                print(f"  [{var_key}] 变体完成，已更新 ablation_summary.csv",
                      flush=True)
                gc.collect()
    finally:
        if run_pool is not None:
            try:
                run_pool.shutdown(wait=True, cancel_futures=False)
            except TypeError:
                try:
                    run_pool.shutdown(wait=True)
                except Exception:
                    pass
            except Exception:
                pass
        # 统一回填配对 Gap%（保证 full 后跑完的变体也有正确相对劣化）
        full_map = {
            int(r['instance_id']): float(r['objective'])
            for r in all_rows
            if r.get('variant') == 'full'
            and r.get('objective') is not None
            and not (isinstance(r.get('objective'), float) and pd.isna(r.get('objective')))
        }
        for r in all_rows:
            if r.get('variant') == 'full':
                r['gap_vs_full_pct'] = 0.0
                continue
            base = full_map.get(int(r['instance_id']))
            obj = r.get('objective')
            if base and obj is not None and base > 0 and not (
                    isinstance(obj, float) and pd.isna(obj)):
                r['gap_vs_full_pct'] = (float(obj) - base) / base * 100
        df, summary_df, tier_df, results_path, summary_path, tier_path = (
            write_ablation_outputs(all_rows, ablation_dir, all_variant_keys))

    report = {
        '阶段': '消融实验',
        'study': globals().get('ABLATION_STUDY', 'main'),
        'pilot': bool(globals().get('ABLATION_PILOT', False)),
        'output_subdir': output_subdir,
        'tier_filter': tier_filter,
        'sample_per_tier': per_tier,
        'sample_seed': sample_seed,
        'instance_ids': [int(i['instance_id']) for i in work_instances],
        'num_runs': num_runs,
        'flat_parallel': bool(globals().get('ABLATION_FLAT_PARALLEL', False)),
        'flat_workers': flat_workers,
        'parallel_runs': parallel_runs,
        'parallel_workers': run_workers,
        'std_ddof': 1,
        'aggregate_feasible_only': True,
        '实验时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        '算例来源': large_dir,
        'import_full_from_large_scale': import_full_from_large_scale,
        'eval_protocol': 'report_objective_from_schedule',
        'team_config_keys': 'int',
        '变体': {k: ABLATION_VARIANTS[k]['label'] for k in all_variant_keys
                 if k in ABLATION_VARIANTS},
        'batch_time_limit': batch_time_limit,
    }
    with open(os.path.join(ablation_dir, 'ablation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[消融完成] 明细: {results_path}")
    print(f"           汇总: {summary_path}")
    print(f"           分梯度: {tier_path}")
    print(f"           Excel: {os.path.join(ablation_dir, 'ablation_report.xlsx')}")
    print(f"           SCI主表: {ablation_dir}/消融SCI主表.csv")
    print(f"  统计检验（后处理）: write_ablation_stats_postprocess('{ablation_dir}')")
    try:
        write_ablation_stats_postprocess(ablation_dir, variants=all_variant_keys)
    except Exception as e:
        print(f"  [警告] 自动统计后处理失败: {e}", flush=True)
    if globals().get('ABLATION_STUDY') == 'knowledge_trajectory':
        try:
            write_knowledge_trajectory_postprocess(ablation_dir)
        except Exception as e:
            print(f"  [警告] Knowledge轨迹后处理失败: {e}", flush=True)
    tables = build_ablation_readable_tables(
        df, summary_df, tier_df, variants=all_variant_keys)
    main_overall = tables.get('main_overall')
    main_by_tier = tables.get('main_by_tier')
    if main_overall is not None and not main_overall.empty:
        print("\n  【主表·总体】相对完整版劣化%（越大=关掉该模块后越差）")
        print(f"  {'变体':>18s} | {'平均目标':>10s} | {'劣化%':>8s} | {'胜平负':>10s} | {'耗时':>7s}")
        print(f"  {'-' * 62}")
        for _, r in main_overall.iterrows():
            gap = r.get('相对完整版劣化%(均值)')
            gap_s = f"{float(gap):.2f}" if gap is not None and pd.notna(gap) else '-'
            obj = r.get('平均目标')
            obj_s = f"{float(obj):.1f}" if obj is not None and pd.notna(obj) else '-'
            t = r.get('平均耗时(s)')
            t_s = f"{float(t):.1f}" if t is not None and pd.notna(t) else '-'
            print(f"  {str(r.get('论文名', r.get('变体键'))):>18s} | {obj_s:>10s} | "
                  f"{gap_s:>8s} | {str(r.get('胜平负(劣-平-优)', '-')):>10s} | {t_s:>7s}")
    if main_by_tier is not None and not main_by_tier.empty:
        print("\n  【主表·分梯度】各梯度相对 full 的 Mean PD%")
        # 动态打印梯度列
        tier_cols = [c for c in main_by_tier.columns if str(c).endswith('_劣化%')]
        hdr = f"  {'变体':>18s}"
        for c in tier_cols:
            hdr += f" | {str(c).replace('_劣化%', ''):>10s}"
        print(hdr)
        print(f"  {'-' * (20 + 13 * max(1, len(tier_cols)))}")
        for _, r in main_by_tier.iterrows():
            line = f"  {str(r.get('论文名', r.get('变体键'))):>18s}"
            for c in tier_cols:
                v = r.get(c)
                line += f" | {(f'{float(v):.2f}' if v is not None and pd.notna(v) else '-'):>10s}"
            print(line)
    elif summary_df is not None and not summary_df.empty:
        print("\n  变体汇总（平均目标值 / 相对完整版 Gap%）")
        print(f"  {'变体':>16s} | {'平均目标':>10s} | {'Gap%':>8s} | {'耗时(s)':>8s}")
        print(f"  {'-' * 52}")
        for _, r in summary_df.iterrows():
            gap_s = f"{r['avg_gap_vs_full_pct']:.2f}" if pd.notna(r['avg_gap_vs_full_pct']) else '-'
            print(f"  {r['variant']:>16s} | {r['avg_objective']:>10.1f} | {gap_s:>8s} | "
                  f"{r['avg_time_s']:>8.1f}")
    else:
        print("  [消融] 汇总为空：尚无有效结果行")
    return all_rows, summary_df


# ---------------------------------------------------------------------------
# 敏感性分析（第五章 §5.5）— 知识权重 / 破坏比例 / 修复系数 α
# 旧 S1–S6 规格保留兼容；默认仅跑 SENSITIVITY_STUDIES_TO_RUN
# ---------------------------------------------------------------------------
S6_PHASE_BUDGETS = {
    'bias_init': {'label': '偏初解(35/45/20)', 'bootstrap': 0.35, 'refine': 0.20},
    'default': {'label': '默认(20/50/30)', 'bootstrap': 0.20, 'refine': 0.30},
    'bias_refine': {'label': '偏精修(10/40/50)', 'bootstrap': 0.10, 'refine': 0.50},
}

# 知识权重水平：字符串 "α1,α2,α3"（α1+α2+α3=1）
KNOWLEDGE_WEIGHT_LEVELS = {
    'A_path': '0.6,0.2,0.2',       # 偏路径/关键性 P
    'B_resource': '0.2,0.6,0.2',   # 偏资源竞争 R
    'C_skill': '0.2,0.2,0.6',     # 偏技能复杂 S
    'D_balanced': '0.333,0.333,0.334',  # 均衡（默认）
}

SENSITIVITY_SPECS = [
  # ---- 第五章主敏感性（默认启用）----
  {
    'study': 'S1_knowledge_weights',
    'group_label': 'S1 知识权重 (α1,α2,α3)',
    'sub_param': 'knowledge_weights',
    'param_label': 'α1P+α2R+α3S',
    'baseline': '0.333,0.333,0.334',
    'levels': [
        '0.6,0.2,0.2',
        '0.2,0.6,0.2',
        '0.2,0.2,0.6',
        '0.333,0.333,0.334',
    ],
    'modifies_instance': False,
    'solver_key': 'knowledge_weights',
  },
  {
    # 主算法真正起控的是邻域上限 q_max（默认 14）；低/基准/高 三档即可
    'study': 'S2_neighborhood_cap',
    'group_label': 'S2 邻域规模上限',
    'sub_param': 'neighborhood_cap',
    'param_label': 'q_max',
    'baseline': 14,
    'levels': [8, 14, 20],
    'modifies_instance': False,
    'solver_key': 'neighborhood_cap',
  },
  {
    # 低(无知识贡献)–基准–中–高；不必为画平滑曲线设 6 档
    'study': 'S3_repair_alpha',
    'group_label': 'S3 修复知识系数 α',
    'sub_param': 'knowledge_repair_alpha',
    'param_label': 'Score=ΔC-α·Ki 中的 α',
    'baseline': 0.05,
    'levels': [0.0, 0.05, 0.5, 1.0],
    'modifies_instance': False,
    'solver_key': 'knowledge_repair_alpha',
  },
  # ---- 旧规格（兼容；默认不跑）----
  {
    'study': 'S2_destroy_ratio',
    'group_label': 'S2 破坏比例 d（旧；固定比例旁路）',
    'sub_param': 'destroy_ratio',
    'param_label': 'destroy_ratio d',
    'baseline': 0.30,
    'levels': [0.10, 0.20, 0.30, 0.40, 0.50],
    'modifies_instance': False,
    'solver_key': 'destroy_ratio',
  },
  {
    'study': 'S1_resource',
    'group_label': 'S1 资源紧张度',
    'sub_param': 'berth_ratio',
    'param_label': '泊位比 N_B/N',
    'baseline': 0.25,
    'levels': [0.15, 0.20, 0.25, 0.30],
    'modifies_instance': True,
  },
  {
    'study': 'S1_resource',
    'group_label': 'S1 资源紧张度',
    'sub_param': 'team_ratio',
    'param_label': '团队比 N_K/N',
    'baseline': 0.33,
    'levels': [0.25, 0.33, 0.50],
    'modifies_instance': True,
  },
  {
    'study': 'S2_lambda',
    'group_label': 'S2 惩罚系数 λ（旧键名；请用 S4_lambda）',
    'sub_param': 'penalty_coeff',
    'param_label': 'λ',
    'baseline': 10,
    'levels': [0, 5, 10, 20, 50],
    'modifies_instance': False,
    'solver_key': 'penalty_coeff',
  },
  {
    # 投稿补充：模型参数 λ（技能错配惩罚）；独立 protocol=sensitivity_lambda
    'study': 'S4_lambda',
    'group_label': 'S4 技能错配惩罚系数 λ',
    'sub_param': 'penalty_coeff',
    'param_label': 'λ (misallocation penalty)',
    'baseline': 10,
    'levels': [0, 5, 10, 20, 50],
    'modifies_instance': False,
    'solver_key': 'penalty_coeff',
  },
  {
    'study': 'S3_arrival',
    'group_label': 'S3 到港强度',
    'sub_param': 'arrival_factor',
    'param_label': 'arrival_max = k·N',
    'baseline': 3,
    'levels': [2, 3, 5],
    'modifies_instance': True,
  },
  {
    'study': 'S4_time_limit',
    'group_label': 'S4 总求解时限',
    'sub_param': 'time_limit',
    'param_label': 'T (s)',
    'baseline': 120,
    'levels': [60, 120, 180],
    'modifies_instance': False,
    'solver_key': 'time_limit',
  },
  {
    'study': 'S5_mip_sub',
    'group_label': 'S5 MIP子问题时限',
    'sub_param': 'mip_sub_time_limit',
    'param_label': 'MIP子时限 (s)',
    'baseline': 25,
    'levels': [10, 25, 40],
    'modifies_instance': False,
    'solver_key': 'mip_sub_time_limit',
  },
  {
    'study': 'S6_phase',
    'group_label': 'S6 三阶段时间预算',
    'sub_param': 'phase_mode',
    'param_label': '预算方案',
    'baseline': 'default',
    'levels': ['bias_init', 'default', 'bias_refine'],
    'modifies_instance': False,
    'solver_key': 'phase_mode',
  },
]


def select_sensitivity_instances(instances, per_tier=SENSITIVITY_PER_TIER,
                                   instance_ids=None):
    """从大规模算例中按梯度抽样（默认每梯度 SENSITIVITY_PER_TIER 例）。

    若 SENSITIVITY_INSTANCE_IDS / instance_ids 非空，则严格按固定 ID 选取（SCI 可复现）。
    """
    ids = instance_ids if instance_ids is not None else SENSITIVITY_INSTANCE_IDS
    if ids:
        id_set = set(int(x) for x in ids)
        selected = [inst for inst in instances if int(inst['instance_id']) in id_set]
        selected.sort(key=lambda x: int(x['instance_id']))
        missing = sorted(id_set - {int(i['instance_id']) for i in selected})
        if missing:
            print(f"  [敏感性][警告] 固定算例缺失: {missing}", flush=True)
        return selected

    tiers = LARGE_SCALE_TIER_LABELS
    selected = []
    for tier in tiers:
        tier_insts = sorted(
            [i for i in instances if i.get('config', {}).get('tier') == tier],
            key=lambda x: x['instance_id'],
        )
        if not tier_insts:
            continue
        if len(tier_insts) <= per_tier:
            selected.extend(tier_insts)
        else:
            idxs = np.linspace(0, len(tier_insts) - 1, per_tier, dtype=int)
            selected.extend([tier_insts[i] for i in idxs])
    return selected


def filter_sensitivity_rows_to_instances(rows, selected_ids):
    """丢弃不在当前敏感性算例集中的旧行（防止混样污染汇总）。"""
    if not selected_ids:
        return list(rows), 0
    allow = set(int(x) for x in selected_ids)
    valid, dropped = [], 0
    for r in rows:
        try:
            iid = int(r['instance_id'])
        except (TypeError, ValueError, KeyError):
            dropped += 1
            continue
        if iid not in allow:
            dropped += 1
            continue
        valid.append(r)
    return valid, dropped


def sanitize_sensitivity_solve_time(solve_time, time_limit):
    """墙钟异常值钳制（休眠/挂起会导致 solve_time 远超 time_limit）。"""
    if solve_time is None or (isinstance(solve_time, float) and pd.isna(solve_time)):
        return solve_time
    try:
        t = float(solve_time)
        lim = float(time_limit) if time_limit is not None else None
    except (TypeError, ValueError):
        return solve_time
    factor = float(globals().get('SENSITIVITY_TIME_CLIP_FACTOR', 1.25) or 1.25)
    if lim is not None and lim > 0 and t > lim * factor:
        return float(lim)
    return t


def build_sensitivity_instance(base_instance, spec, level_value):
    """按敏感性水平修改算例（保持相同随机种子，仅改目标参数）"""
    if not spec.get('modifies_instance'):
        inst = copy.deepcopy(base_instance)
        # λ 敏感性：实例与求解器必须使用同一惩罚系数
        if spec.get('solver_key') == 'penalty_coeff':
            inst['_penalty_coeff'] = float(level_value)
        else:
            inst['_penalty_coeff'] = MISALLOCATION_PENALTY_COEFF
        return inst

    config = copy.deepcopy(base_instance.get('config', {}))
    instance_id = base_instance['instance_id']
    n_ships = base_instance['num_ships']
    sub = spec['sub_param']

    if sub == 'berth_ratio':
        config['num_berths'] = min(8, max(2, round(n_ships * level_value)))
    elif sub == 'team_ratio':
        config['force_team_count'] = min(12, max(3, int(n_ships * level_value) + 2))
    elif sub == 'arrival_factor':
        config['arrival_max'] = int(n_ships * level_value)

    seed = instance_id * 100
    generator = RandomInstanceGenerator(seed=seed)
    inst = generator.generate_instance(instance_id, config)
    inst['group_name'] = base_instance.get('group_name', '')
    inst['config_idx'] = base_instance.get('config_idx', 0)
    if spec.get('solver_key') == 'penalty_coeff':
        inst['_penalty_coeff'] = float(level_value)
    else:
        inst['_penalty_coeff'] = MISALLOCATION_PENALTY_COEFF
    return inst


def create_sensitivity_solver(spec, level_value, batch_time_limit, mip_sub_time_limit):
    """按敏感性规格创建 KG-ALNS 求解器（完整版 flags；每次 run 重新创建）。"""
    time_limit = batch_time_limit
    mip_sub = mip_sub_time_limit
    penalty_coeff = MISALLOCATION_PENALTY_COEFF
    phase_override = None
    knowledge_weights = None
    solver_key = spec.get('solver_key')

    if solver_key == 'penalty_coeff':
        penalty_coeff = level_value
    elif solver_key == 'time_limit':
        time_limit = level_value
    elif solver_key == 'mip_sub_time_limit':
        mip_sub = level_value
    elif solver_key == 'phase_mode':
        phase_override = S6_PHASE_BUDGETS[level_value]
    elif solver_key == 'knowledge_weights':
        if isinstance(level_value, str):
            knowledge_weights = tuple(
                float(x.strip()) for x in str(level_value).split(','))
        else:
            knowledge_weights = tuple(float(x) for x in level_value)

    solver = MatheuristicSolver(
        time_limit=time_limit,
        mip_sub_time_limit=mip_sub,
        max_iters=150,
        verbose=False,
        penalty_coeff=penalty_coeff,
        phase_ratios_override=phase_override,
        knowledge_weights=knowledge_weights,
        ablation_variant='full',
    )
    if solver_key == 'mip_sub_time_limit':
        solver._sub_time_override = mip_sub
    if solver_key == 'destroy_ratio':
        # 旧旁路：严格固定比例（不进默认 SENSITIVITY_STUDIES_TO_RUN）
        solver._destroy_ratio_fixed = float(level_value)
    if solver_key == 'neighborhood_cap':
        # 主敏感性 S2：覆盖 scale 中的 q_max（主算法真实控制参数）
        solver._neighborhood_cap_override = int(level_value)
    if solver_key == 'knowledge_repair_alpha':
        # α=0 仍保持同一 knowledge-repair 骨架；关闭骨架留给消融 wo_repair
        solver.KNOWLEDGE_REPAIR_ALPHA = float(level_value)
        solver.ablation_flags['use_knowledge_repair'] = True
    return solver


def load_completed_sensitivity_keys(output_dir):
    """读取已完成键 (study, sub_param, level, instance_id, run_id, seed)。"""
    csv_path = os.path.join(output_dir, 'sensitivity_results.csv')
    if not os.path.isfile(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        if df.empty:
            return set()
        keys = set()
        for _, r in df.iterrows():
            if r.get('objective') is None or pd.isna(r.get('objective')):
                continue
            if str(r.get('feasible', True)).lower() in ('false', '0', 'nan'):
                continue
            if str(r.get('time_valid', True)).lower() in ('false', '0', 'nan'):
                continue
            keys.add((
                str(r.get('study', '')),
                str(r.get('sub_param', '')),
                str(r.get('level_value')),
                int(r['instance_id']),
                int(r.get('run_id', 0)),
                int(r.get('seed', -1)),
            ))
        return keys
    except Exception:
        return set()


def _sensitivity_level_match(series, level):
    """数值/字符串水平匹配。"""
    out = []
    for v in series:
        if str(v) == str(level):
            out.append(True)
            continue
        try:
            out.append(abs(float(v) - float(level)) < 1e-12)
        except Exception:
            out.append(False)
    return pd.Series(out, index=series.index)


def build_sensitivity_summary(df, expected_n=None, raise_on_incomplete=False):
    """SCI 汇总：先对 instance×level 取 run-mean，再算配对 RPD%。

    仅保留 feasible & time_valid & objective 非空。
    主指标：mean_rpd_pct / median_rpd_pct / sd_rpd_pct。
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    valid = df.copy()
    if 'feasible' in valid.columns:
        valid = valid[valid['feasible'].astype(str).str.lower().isin(
            ['true', '1', 'yes']) | (valid['feasible'] == True)]
    if 'time_valid' in valid.columns:
        valid = valid[valid['time_valid'].astype(str).str.lower().isin(
            ['true', '1', 'yes']) | (valid['time_valid'] == True)]
    valid = valid[valid['objective'].notna()]
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()

    if expected_n is not None and raise_on_incomplete:
        counts = valid.groupby(
            ['study', 'sub_param', 'level_value'], dropna=False).size()
        if not counts.empty and not bool((counts == expected_n).all()):
            raise ValueError(
                f"敏感性实验不完整，期望每水平 {expected_n} 条：\n{counts}")

    # 每实例每水平：多 run 取 mean
    case_level = (
        valid.groupby(
            ['study', 'sub_param', 'level_value', 'instance_id', 'tier'],
            as_index=False,
        )['objective']
        .mean()
        .rename(columns={'objective': 'mean_objective'})
    )

    summary_rows = []
    tier_rows = []
    specs_by_key = {(s['study'], s['sub_param']): s for s in SENSITIVITY_SPECS}

    for (study, sub_param), spec in specs_by_key.items():
        sub = case_level[
            (case_level['study'] == study) & (case_level['sub_param'] == sub_param)
        ].copy()
        if sub.empty:
            continue
        baseline_val = spec['baseline']
        baseline = sub[_sensitivity_level_match(sub['level_value'], baseline_val)][
            ['instance_id', 'mean_objective']
        ].rename(columns={'mean_objective': 'baseline_objective'})
        if baseline.empty:
            continue
        paired = sub.merge(baseline, on='instance_id', how='inner')
        paired = paired[paired['baseline_objective'] > 0].copy()
        if paired.empty:
            continue
        paired['rpd_pct'] = (
            100.0 * (paired['mean_objective'] - paired['baseline_objective'])
            / paired['baseline_objective']
        )

        for level in spec['levels']:
            pl = paired[_sensitivity_level_match(paired['level_value'], level)]
            if pl.empty:
                continue
            # 辅助：原始明细上的时间/误配（valid 行）
            raw = valid[
                (valid['study'] == study) & (valid['sub_param'] == sub_param)
                & _sensitivity_level_match(valid['level_value'], level)
            ]
            summary_rows.append({
                'study': study,
                'group_label': spec['group_label'],
                'sub_param': sub_param,
                'param_label': spec['param_label'],
                'level_value': level,
                'level_label': str(level),
                'n_instances': int(pl['instance_id'].nunique()),
                'n_runs_rows': int(len(raw)),
                'avg_objective': float(pl['mean_objective'].mean()),
                'median_objective': float(pl['mean_objective'].median()),
                'mean_rpd_pct': float(pl['rpd_pct'].mean()),
                'sd_rpd_pct': float(pl['rpd_pct'].std(ddof=1)) if len(pl) > 1 else 0.0,
                'median_rpd_pct': float(pl['rpd_pct'].median()),
                'avg_gap_vs_baseline_pct': float(pl['rpd_pct'].mean()),
                'median_gap_vs_baseline_pct': float(pl['rpd_pct'].median()),
                'avg_misallocation': (
                    float(raw['misallocation_count'].mean())
                    if 'misallocation_count' in raw.columns and not raw.empty else None),
                'avg_time_s': (
                    float(raw['solve_time'].mean())
                    if 'solve_time' in raw.columns and not raw.empty else None),
                'is_baseline': (
                    str(level) == str(baseline_val)
                    or (abs(float(level) - float(baseline_val)) < 1e-12
                        if str(level).replace('.', '', 1).replace('-', '', 1).isdigit()
                        and str(baseline_val).replace('.', '', 1).replace('-', '', 1).isdigit()
                        else False)),
            })

        for tier in LARGE_SCALE_TIER_LABELS:
            pt = paired[paired['tier'] == tier]
            if pt.empty:
                continue
            for level in spec['levels']:
                pl = pt[_sensitivity_level_match(pt['level_value'], level)]
                if pl.empty:
                    continue
                tier_rows.append({
                    'study': study,
                    'sub_param': sub_param,
                    'tier': tier,
                    'level_value': level,
                    'n_instances': int(pl['instance_id'].nunique()),
                    'avg_objective': float(pl['mean_objective'].mean()),
                    'mean_rpd_pct': float(pl['rpd_pct'].mean()),
                    'median_rpd_pct': float(pl['rpd_pct'].median()),
                    'avg_gap_vs_baseline_pct': float(pl['rpd_pct'].mean()),
                    'median_gap_vs_baseline_pct': float(pl['rpd_pct'].median()),
                })

    return pd.DataFrame(summary_rows), pd.DataFrame(tier_rows)


def write_sensitivity_outputs(all_rows, sens_dir, selected_ids, all_specs,
                              batch_time_limit, mip_sub_time_limit, large_dir):
    """写入/更新敏感性明细、汇总 CSV、JSON、Excel。"""
    os.makedirs(sens_dir, exist_ok=True)
    results_path = os.path.join(sens_dir, 'sensitivity_results.csv')
    rows, dropped = filter_sensitivity_rows_to_instances(all_rows, selected_ids)
    if dropped:
        print(f"[敏感性] 汇总前丢弃 {dropped} 条非规范算例记录", flush=True)
    # 钳制异常 solve_time
    for r in rows:
        r['solve_time'] = sanitize_sensitivity_solve_time(
            r.get('solve_time'), r.get('time_limit_used', batch_time_limit))
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not df.empty:
        results_path = _safe_dataframe_to_csv(df, results_path)
    expected_n = (
        len(selected_ids) * int(globals().get('SENSITIVITY_NUM_RUNS', 1) or 1)
        if selected_ids else None)
    summary_df, tier_df = build_sensitivity_summary(
        df, expected_n=expected_n, raise_on_incomplete=False)
    summary_path = os.path.join(sens_dir, 'sensitivity_summary.csv')
    tier_path = os.path.join(sens_dir, 'sensitivity_summary_by_tier.csv')
    summary_path = _safe_dataframe_to_csv(summary_df, summary_path)
    tier_path = _safe_dataframe_to_csv(tier_df, tier_path)
    excel_path = os.path.join(sens_dir, 'sensitivity_report.xlsx')
    if not df.empty:
        save_sensitivity_excel_report(df, summary_df, tier_df, excel_path)
    report = {
        '阶段': '敏感性分析',
        'protocol': globals().get('SENSITIVITY_PROTOCOL', 'sensitivity_compact_9x3'),
        'num_runs': globals().get('SENSITIVITY_NUM_RUNS', 1),
        '实验时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        '算例来源': large_dir,
        '抽样算例': selected_ids,
        'batch_time_limit': batch_time_limit,
        'mip_sub_time_limit': mip_sub_time_limit,
        '规格': [{'study': s['study'], 'sub_param': s['sub_param'],
                  'levels': s['levels']} for s in all_specs],
    }
    with open(os.path.join(sens_dir, 'sensitivity_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return df, summary_df, tier_df, results_path, summary_path, tier_path


def save_sensitivity_excel_report(detail_df, summary_df, tier_df, output_path):
    """保存敏感性分析 Excel（汇总 + 分梯度 + 明细）"""
    if detail_df.empty:
        return None
    if not OPENPYXL_AVAILABLE:
        base = output_path.replace('.xlsx', '')
        detail_df.to_csv(f"{base}_detail.csv", index=False, encoding='utf-8-sig')
        summary_df.to_csv(f"{base}_summary.csv", index=False, encoding='utf-8-sig')
        print("  openpyxl 未安装，敏感性结果已存 CSV")
        return f"{base}_detail.csv"

    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows

    wb = Workbook()
    ws_sum = wb.active
    ws_sum.title = '敏感性汇总'
    for r in dataframe_to_rows(summary_df, index=False, header=True):
        ws_sum.append(r)
    ws_tier = wb.create_sheet('分梯度汇总')
    for r in dataframe_to_rows(tier_df, index=False, header=True):
        ws_tier.append(r)
    ws_det = wb.create_sheet('明细')
    for r in dataframe_to_rows(detail_df, index=False, header=True):
        ws_det.append(r)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        wb.save(output_path)
    except PermissionError:
        output_path = output_path.replace('.xlsx',
                                          f"_{datetime.now().strftime('%H%M%S')}.xlsx")
        wb.save(output_path)
    print(f"  敏感性 Excel: {output_path}")
    return output_path


def _is_parallel_resource_failure(exc):
    """页面文件/RAM/进程池损坏：应立刻停并行、改串行续跑。"""
    if isinstance(exc, (BrokenExecutor, MemoryError)):
        return True
    if isinstance(exc, OSError):
        return True
    msg = str(exc).lower()
    needles = (
        '页面文件', 'paging file', 'not enough memory', 'not enough storage',
        'memoryerror', 'winerror 1455', ' 1455', 'brokenprocesspool',
        'broken executor', 'cannot allocate', 'oom',
    )
    return any(n in msg for n in needles)


def _sensitivity_unfinished_jobs(pending_list, done_keys, selected_ids, sens_dir):
    """并行中断后收集尚未成功写入的任务。"""
    leftover = []
    try:
        done_now = load_completed_sensitivity_keys(sens_dir)
        allow = set(int(x) for x in selected_ids)
        done_now = {k for k in done_now if int(k[3]) in allow}
    except Exception:
        done_now = set()
    combined = set(done_keys) | done_now
    for a in pending_list:
        spec, lv, inst, run_id, seed = a[:5]
        iid = int(inst['instance_id'])
        key = (spec['study'], spec['sub_param'], str(lv), iid, run_id, seed)
        if key not in combined:
            leftover.append(a)
    return leftover


def _sensitivity_execute_one_run(spec, level, base_inst, run_id, seed,
                                 batch_time_limit, mip_sub_time_limit, protocol):
    """单次敏感性求解（串行/并行共用），返回 (row, key, ok, warn)。"""
    study = spec['study']
    sub_param = spec['sub_param']
    level_str = str(level)
    iid = int(base_inst['instance_id'])
    tier = base_inst.get('config', {}).get('tier', '')
    key = (study, sub_param, level_str, iid, int(run_id), int(seed))
    warn = None
    obj = None
    total_flow = None
    misc = None
    feasible = False
    time_valid = False
    solve_time = None
    sched = None
    result = {'mip_repair_count': 0}
    solver = None
    inst = base_inst
    try:
        random.seed(seed)
        try:
            np.random.seed(int(seed) % (2 ** 31 - 1))
        except Exception:
            pass
        solver = create_sensitivity_solver(
            spec, level, batch_time_limit, mip_sub_time_limit)
        inst = build_sensitivity_instance(base_inst, spec, level)
        inst['_penalty_coeff'] = solver.penalty_coeff
        tc = inst.get('team_config')
        if tc is not None:
            inst['team_config'] = normalize_team_config_keys(tc)
        result = solver.solve(inst)
        sched = result.get('schedule')
        penalty = inst.get('_penalty_coeff', solver.penalty_coeff)
        if sched is not None:
            sched, obj, feasible = report_objective_from_schedule(
                sched, inst, penalty_coeff=penalty)
            if not feasible:
                warn = f'[不可行] Case={iid} level={level_str} run={run_id}'
                obj = None
                total_flow = None
                misc = None
            else:
                total_flow = compute_flow_from_schedule(sched, inst)
                utils = HeuristicScheduler()
                utils._load_team_config(inst)
                misc = compute_misallocation_count_from_schedule(
                    sched, utils._team_skills)
        solve_time_raw = result.get('solve_time')
        time_valid = (
            solve_time_raw is not None
            and float(solve_time_raw) >= 0
            and float(solve_time_raw) <= float(solver.time_limit) * float(
                SENSITIVITY_TIME_CLIP_FACTOR)
        )
        if not time_valid:
            warn = (f'[异常时间] Case={iid}, level={level_str}, '
                    f'run={run_id}, solve_time={solve_time_raw}')
            obj = None
            total_flow = None
            misc = None
        solve_time = sanitize_sensitivity_solve_time(
            solve_time_raw, solver.time_limit)
    except (MemoryError, OSError, BrokenExecutor):
        raise
    except Exception as e:
        if _is_parallel_resource_failure(e):
            raise
        warn = f'异常: {e}'
        obj = None
        total_flow = None
        misc = None
        feasible = False
        time_valid = False
        solve_time = None
        result = {'mip_repair_count': 0}
        sched = None

    row = {
        'study': study,
        'group_label': spec['group_label'],
        'sub_param': sub_param,
        'param_label': spec['param_label'],
        'level_value': level_str,
        'level_label': level_str,
        'instance_id': iid,
        'run_id': run_id,
        'seed': seed,
        'protocol': protocol,
        'num_ships': inst.get('num_ships') if isinstance(inst, dict) else None,
        'num_tasks': inst.get('num_tasks') if isinstance(inst, dict) else None,
        'num_berths': inst.get('num_berths') if isinstance(inst, dict) else None,
        'num_docks': inst.get('num_docks') if isinstance(inst, dict) else None,
        'num_teams': inst.get('num_teams') if isinstance(inst, dict) else None,
        'tier': tier,
        'objective': obj,
        'total_flow': total_flow,
        'misallocation_count': misc,
        'feasible': bool(feasible),
        'time_valid': bool(time_valid),
        'scheduled_task_count': len(sched) if sched else 0,
        'expected_task_count': inst.get('num_tasks') if isinstance(inst, dict) else None,
        'solve_time': solve_time,
        'mip_repair_count': result.get('mip_repair_count') if result else 0,
        'time_limit_used': getattr(solver, 'time_limit', batch_time_limit),
        'mip_sub_used': getattr(solver, 'mip_sub_time_limit', mip_sub_time_limit),
        'penalty_coeff_used': getattr(solver, 'penalty_coeff', None),
        'knowledge_weights_used': ','.join(
            f'{w:.4f}' for w in (getattr(solver, 'knowledge_weights', None) or ())),
        'neighborhood_cap_used': getattr(
            solver, '_neighborhood_cap_override',
            getattr(solver, '_neighborhood_cap', None)) if solver else None,
        'destroy_ratio_fixed': getattr(solver, '_destroy_ratio_fixed', None) if solver else None,
        'actual_destroy_count': getattr(
            solver, '_last_actual_destroy_count', None) if solver else None,
        'actual_destroy_ratio': getattr(
            solver, '_last_actual_destroy_ratio', None) if solver else None,
        'repair_alpha_used': float(
            getattr(solver, 'KNOWLEDGE_REPAIR_ALPHA', 0.05)) if solver else 0.05,
    }
    ok = obj is not None and feasible and time_valid
    try:
        if solver is not None:
            solver._insertion_engine = None
    except Exception:
        pass
    del solver, result, inst, sched
    gc.collect()
    return row, key, ok, warn


def _sensitivity_single_run_worker(args):
    """ProcessPool 子进程：敏感性单次独立运行（须模块级可 pickle）。"""
    (spec, level, base_inst, run_id, seed,
     batch_time_limit, mip_sub_time_limit, protocol) = args
    global GUROBI_SUB_THREADS
    try:
        GUROBI_SUB_THREADS = 1
    except Exception:
        pass
    try:
        return _sensitivity_execute_one_run(
            spec, level, base_inst, run_id, seed,
            batch_time_limit, mip_sub_time_limit, protocol)
    except (MemoryError, OSError, BrokenExecutor):
        gc.collect()
        raise
    except Exception as e:
        if _is_parallel_resource_failure(e):
            gc.collect()
            raise
        gc.collect()
        iid = int(base_inst.get('instance_id', -1))
        key = (spec['study'], spec['sub_param'], str(level), iid,
               int(run_id), int(seed))
        return None, key, False, f'{type(e).__name__}: {e}'


def run_sensitivity_study(output_root=OUTPUT_ROOT, batch_time_limit=None,
                          mip_sub_time_limit=None, specs=None, resume=None,
                          large_exp=None, instance_ids=None, per_tier=None,
                          num_runs=None, protocol=None):
    """批量敏感性分析。

    输出：OUTPUT/sensitivity/<protocol>/
    默认 protocol=SENSITIVITY_PROTOCOL；旧结果在 sensitivity/ 根目录，互不覆盖。
    SENSITIVITY_PARALLEL=True 时按独立 (level, 算例, run) 进程并行。
    """
    batch_time_limit = batch_time_limit or SENSITIVITY_BASE_TIME_LIMIT
    mip_sub_time_limit = mip_sub_time_limit or SENSITIVITY_BASE_MIP_SUB
    per_tier = per_tier or SENSITIVITY_PER_TIER
    instance_ids = instance_ids or SENSITIVITY_INSTANCE_IDS
    num_runs = int(num_runs if num_runs is not None else SENSITIVITY_NUM_RUNS)
    protocol = protocol or SENSITIVITY_PROTOCOL
    resume = SENSITIVITY_RESUME if resume is None else bool(resume)
    all_specs = SENSITIVITY_SPECS
    if specs:
        spec_keys = set(specs)
        all_specs = [s for s in SENSITIVITY_SPECS
                     if s['study'] in spec_keys or
                     f"{s['study']}_{s['sub_param']}" in spec_keys]
    elif SENSITIVITY_STUDIES_TO_RUN:
        spec_keys = set(SENSITIVITY_STUDIES_TO_RUN)
        all_specs = [s for s in SENSITIVITY_SPECS
                     if s['study'] in spec_keys or
                     f"{s['study']}_{s['sub_param']}" in spec_keys]

    sens_dir = os.path.join(output_root, 'sensitivity', protocol)
    large_dir = os.path.join(output_root, 'large_scale')
    if not os.path.isfile(os.path.join(large_dir, 'large_configs.json')):
        for alt_name in ('有效数据', 'valid_data'):
            alt = os.path.join(output_root, alt_name, 'large_scale')
            if os.path.isfile(os.path.join(alt, 'large_configs.json')):
                print(f"[敏感性] 使用算例目录: {alt}", flush=True)
                large_dir = alt
                break
    os.makedirs(sens_dir, exist_ok=True)
    os.makedirs(large_dir, exist_ok=True)
    print(f"\n[敏感性] 启动 — protocol={protocol} | runs×{num_runs} | "
          f"resume={resume}", flush=True)
    print(f"[敏感性] 输出目录: {sens_dir}", flush=True)

    if large_exp is not None and large_exp.instances:
        exp = large_exp
        print(f"\n[敏感性] 复用阶段二算例池 ({len(exp.instances)} 例)")
    else:
        saved_configs = load_large_configs(large_dir)
        if saved_configs:
            configs = saved_configs
            print(f"\n[敏感性] 从 large_configs.json 加载 {len(configs)} 组配置")
        else:
            random.seed(REPRODUCIBILITY_SEED)
            np.random.seed(REPRODUCIBILITY_SEED)
            configs = generate_large_scale_configs(60, seed=REPRODUCIBILITY_SEED)
            save_large_configs(configs, large_dir)
        exp = BatchExperimentGurobi(output_dir=large_dir, heuristic_type='lNS')
        exp.set_experiment_configs(configs)
        print("[敏感性] 加载算例实例...", flush=True)
        exp.generate_instances(force_regen=False, quiet=True)
        if not exp.instances:
            exp.generate_instances(force_regen=False, quiet=False)
        if not glob.glob(os.path.join(large_dir, 'instance_*_tasks.csv')):
            exp.save_instances()
            save_large_configs(configs, large_dir)

    selected = select_sensitivity_instances(
        exp.instances, per_tier=per_tier, instance_ids=instance_ids)
    if not selected:
        print("[敏感性] 无可用算例，退出")
        return [], pd.DataFrame()

    for inst in selected:
        tc = inst.get('team_config')
        if tc is not None:
            inst['team_config'] = normalize_team_config_keys(tc)

    selected_ids = [i['instance_id'] for i in selected]

    results_path = os.path.join(sens_dir, 'sensitivity_results.csv')
    existing_rows = []
    done_keys = set()
    if resume and os.path.isfile(results_path):
        try:
            existing_rows = pd.read_csv(results_path, encoding='utf-8-sig').to_dict('records')
            existing_rows, dropped = filter_sensitivity_rows_to_instances(
                existing_rows, selected_ids)
            if dropped:
                print(f"[敏感性] 丢弃 {dropped} 条非规范算例旧记录", flush=True)
            done_keys = load_completed_sensitivity_keys(sens_dir)
            # 仅保留规范算例上的 done_keys
            allow = set(int(x) for x in selected_ids)
            done_keys = {k for k in done_keys if int(k[3]) in allow}
            print(f"[敏感性] 续跑：已有 {len(done_keys)} 条有效记录")
        except Exception:
            existing_rows = []
            done_keys = set()
    elif not resume and os.path.isfile(results_path):
        print(f"[敏感性] SENSITIVITY_RESUME=False：忽略已有 {results_path}", flush=True)

    all_rows = list(existing_rows) if resume else []
    print(f"[敏感性] 抽样算例 {len(selected)} 个: {selected_ids}")
    write_sensitivity_outputs(
        all_rows, sens_dir, selected_ids, all_specs,
        batch_time_limit, mip_sub_time_limit, large_dir)

    total_jobs = (
        sum(len(s['levels']) for s in all_specs) * len(selected) * num_runs)
    pending_list = []
    for s in all_specs:
        for lv in s['levels']:
            for inst in selected:
                iid = int(inst['instance_id'])
                for run_id in range(num_runs):
                    seed = REPRODUCIBILITY_SEED + iid * 1000 + run_id
                    key = (s['study'], s['sub_param'], str(lv), iid, run_id, seed)
                    if resume and key in done_keys:
                        continue
                    pending_list.append((
                        s, lv, inst, run_id, seed,
                        batch_time_limit, mip_sub_time_limit, protocol,
                    ))
    pending_jobs = len(pending_list)
    parallel = bool(globals().get('SENSITIVITY_PARALLEL', False)) and pending_jobs > 1
    workers = max(1, int(globals().get('SENSITIVITY_PARALLEL_WORKERS', 1) or 1))
    if parallel:
        workers = min(workers, pending_jobs)
    else:
        workers = 1
    wall_factor = (1.0 / workers) if workers > 1 else 1.0
    est_min = pending_jobs * batch_time_limit / 60.0 * wall_factor
    print(f"[敏感性] 规格 {len(all_specs)} × 算例 {len(selected)} × runs {num_runs}, "
          f"待跑 {pending_jobs}/{total_jobs}, 时限={batch_time_limit}s, "
          f"并行×{workers}, 预计≈{est_min:.0f} 分钟 (~{est_min/60:.1f} 小时)")
    if parallel and sys.platform == 'win32':
        print(f"[提示] Windows 敏感性并行={workers}；"
              f"若页面文件/OOM 请设 SENSITIVITY_PARALLEL_WORKERS=1 或 2",
              flush=True)

    def _ingest_sensitivity_result(row, key, ok, warn, tag=''):
        if warn:
            print(f"    {warn}", flush=True)
        if row is None:
            if tag:
                print(f"    [失败] {tag}", flush=True)
            return
        all_rows.append(row)
        if ok and key is not None:
            done_keys.add(key)
        _safe_dataframe_to_csv(pd.DataFrame(all_rows), results_path)

    try:
        leftover = []
        pool_failed = False
        if parallel and workers > 1:
            print(f"\n[敏感性] 启动进程池 max_workers={workers}，"
                  f"共 {pending_jobs} 个独立任务", flush=True)
            print("[敏感性] 若页面文件/RAM 不足将自动改串行续跑（已完成的不重跑）",
                  flush=True)
            gc.collect()
            pool = None
            completed = 0
            try:
                pool = ProcessPoolExecutor(max_workers=workers)
                in_flight = {}
                job_iter = iter(pending_list)

                def _fill_pool():
                    while len(in_flight) < workers:
                        try:
                            a = next(job_iter)
                        except StopIteration:
                            return
                        fut = pool.submit(_sensitivity_single_run_worker, a)
                        in_flight[fut] = a

                _fill_pool()
                while in_flight:
                    done, _ = wait(list(in_flight.keys()),
                                   return_when=FIRST_COMPLETED)
                    for fut in done:
                        args = in_flight.pop(fut)
                        spec, level, base_inst, run_id, seed = args[:5]
                        iid = int(base_inst['instance_id'])
                        tag = (f"{spec['study']}|{spec['sub_param']}={level} "
                               f"Case {iid} run{run_id}/{num_runs}")
                        completed += 1
                        try:
                            row, key, ok, warn = fut.result()
                        except Exception as e:
                            if _is_parallel_resource_failure(e):
                                raise
                            row, key, ok, warn = None, None, False, f'异常: {e}'
                        print(f"  [{completed}/{pending_jobs}] {tag}", flush=True)
                        _ingest_sensitivity_result(row, key, ok, warn, tag=tag)
                        if completed % max(workers, 3) == 0:
                            write_sensitivity_outputs(
                                all_rows, sens_dir, selected_ids, all_specs,
                                batch_time_limit, mip_sub_time_limit, large_dir)
                        _fill_pool()
            except (BrokenExecutor, OSError, MemoryError) as e:
                err_hint = '页面文件/RAM 不足' if '页面文件' in str(e) or 'paging' in str(e).lower() else str(e)
                print(f"[敏感性] 并行失败 ({type(e).__name__}: {err_hint})",
                      flush=True)
                print("[敏感性] 已完成结果保留，未完成任务改串行继续", flush=True)
                pool_failed = True
            except Exception as e:
                print(f"[敏感性] 并行失败 ({type(e).__name__}: {e})", flush=True)
                print("[敏感性] 已完成结果保留，未完成任务改串行继续", flush=True)
                pool_failed = True
            finally:
                if pool is not None:
                    try:
                        pool.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                gc.collect()
            if pool_failed:
                leftover = _sensitivity_unfinished_jobs(
                    pending_list, done_keys, selected_ids, sens_dir)
        else:
            leftover = list(pending_list)

        if leftover:
            if pool_failed:
                print(f"[敏感性] 串行续跑 {len(leftover)} 个未完成任务", flush=True)
            elif not (parallel and workers > 1):
                print(f"[敏感性] 串行求解 {len(leftover)} 个任务", flush=True)
            for i, a in enumerate(leftover, 1):
                spec, level, base_inst, run_id, seed = a[:5]
                iid = int(base_inst['instance_id'])
                print(
                    f"  [{i}/{len(leftover)}] "
                    f"[{spec['study']}|{spec['sub_param']}={level}] "
                    f"Case {iid} run{run_id}/{num_runs} seed={seed}",
                    flush=True)
                try:
                    row, key, ok, warn = _sensitivity_execute_one_run(*a)
                except Exception as e:
                    print(f"    异常(串行): {e}", flush=True)
                    continue
                _ingest_sensitivity_result(row, key, ok, warn)
                if i % 3 == 0:
                    write_sensitivity_outputs(
                        all_rows, sens_dir, selected_ids, all_specs,
                        batch_time_limit, mip_sub_time_limit, large_dir)
    finally:
        df, summary_df, tier_df, results_path, summary_path, tier_path = (
            write_sensitivity_outputs(
                all_rows, sens_dir, selected_ids, all_specs,
                batch_time_limit, mip_sub_time_limit, large_dir))

    print(f"\n[敏感性完成] 明细: {results_path}")
    print(f"             汇总: {summary_path}")
    print(f"             分梯度: {tier_path}")
    if not summary_df.empty:
        print("\n  敏感性汇总（配对 mean/median RPD% | 平均目标）")
        print(f"  {'组别':>14s} | {'参数':>12s} | {'水平':>8s} | "
              f"{'平均目标':>10s} | {'meanRPD':>8s} | {'medRPD':>8s}")
        print(f"  {'-' * 78}")
        for _, r in summary_df.iterrows():
            gap_s = (f"{r.get('mean_rpd_pct', r.get('avg_gap_vs_baseline_pct')):.2f}"
                     if pd.notna(r.get('mean_rpd_pct', r.get('avg_gap_vs_baseline_pct')))
                     else '-')
            med_s = (f"{r.get('median_rpd_pct', r.get('median_gap_vs_baseline_pct')):.2f}"
                     if pd.notna(r.get('median_rpd_pct', r.get('median_gap_vs_baseline_pct')))
                     else '-')
            print(f"  {str(r['group_label'])[:14]:>14s} | {str(r['sub_param'])[:12]:>12s} | "
                  f"{str(r['level_value'])[:8]:>8s} | {r['avg_objective']:>10.1f} | "
                  f"{gap_s:>8s} | {med_s:>8s}")
    else:
        print("  [敏感性] 汇总为空：尚无有效结果行")
    return all_rows, summary_df


def clear_output_root(root_path=OUTPUT_ROOT, clear=True, require_confirm=False):
    """清空输出根目录；resume 模式下不清空已有结果。"""
    root_path = os.path.normpath(root_path)
    if clear and os.path.exists(root_path):
        if require_confirm:
            ans = input(f"将删除整个目录 [{root_path}]，输入 YES 继续: ").strip()
            if ans != 'YES':
                print("[跳过] 未清空输出目录")
                clear = False
        if clear:
            _safe_rmtree(root_path)
    for sub in ('small_scale', 'small_scale_enhanced', 'large_scale',
                'ablation', 'ablation_operators', 'sensitivity',
                'pilot_destroy', 'pilot_test'):
        os.makedirs(os.path.join(root_path, sub), exist_ok=True)
    if clear:
        print(f"\n[输出目录已清空] {root_path}")
    else:
        print(f"\n[保留已有输出] {root_path}")


def clear_small_scale_output(output_root=OUTPUT_ROOT):
    """仅清空阶段一输出（不影响 large_scale / ablation / sensitivity）。"""
    import shutil
    output_root = os.path.normpath(output_root)
    for sub in ('small_scale', 'small_scale_enhanced'):
        path = os.path.join(output_root, sub)
        if os.path.isdir(path):
            _safe_rmtree(path)
        os.makedirs(path, exist_ok=True)
    cfg_path = os.path.join(output_root, 'small_scale_configs.json')
    if os.path.isfile(cfg_path):
        try:
            os.remove(cfg_path)
        except OSError:
            pass
    print(f"\n[阶段一已清空] {output_root}/small_scale/ 与 small_scale_enhanced/", flush=True)


def clear_large_scale_tier(tier_index, output_root=OUTPUT_ROOT, regen_instances=False):
    """清空某一梯度的大规模结果（保留其他梯度 CSV 行；可选删除该梯度算例文件）。"""
    if tier_index not in (0, 1, 2):
        raise ValueError(f"tier_index 须为 0/1/2，收到 {tier_index}")
    tier_label = LARGE_SCALE_TIER_LABELS[tier_index]
    tier_name = LARGE_SCALE_TIER_NAMES[tier_index]
    large_dir = os.path.join(output_root, 'large_scale')
    os.makedirs(large_dir, exist_ok=True)

    removed = 0
    csv_path = os.path.join(large_dir, 'large_scale_results.csv')
    if os.path.isfile(csv_path):
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            if not df.empty and 'tier' in df.columns:
                mask = df['tier'].astype(str) == tier_label
                removed = int(mask.sum())
                kept_df = df[~mask]
                if kept_df.empty:
                    os.remove(csv_path)
                else:
                    _safe_dataframe_to_csv(kept_df, csv_path)
        except Exception as e:
            print(f"[警告] 清理 CSV 失败: {e}", flush=True)

    comp_path = os.path.join(large_dir, 'large_scale_comparison.csv')
    if os.path.isfile(comp_path):
        try:
            df = pd.read_csv(comp_path, encoding='utf-8-sig')
            if not df.empty and 'tier' in df.columns:
                kept_df = df[df['tier'].astype(str) != tier_label]
                if kept_df.empty:
                    os.remove(comp_path)
                else:
                    _safe_dataframe_to_csv(kept_df, comp_path)
        except Exception:
            pass

    configs = load_large_configs(large_dir)
    tier_ids = []
    if configs:
        for i, cfg in enumerate(configs, start=1):
            if cfg.get('tier') == tier_label:
                tier_ids.append(i)
    else:
        tier_ids = list(range(tier_index * 20 + 1, tier_index * 20 + 21))

    if regen_instances and tier_ids:
        for iid in tier_ids:
            for f in glob.glob(os.path.join(large_dir, f'instance_{iid}_*')):
                try:
                    os.remove(f)
                except OSError:
                    pass
            for f in glob.glob(os.path.join(large_dir, f'Case_{iid:02d}_*')):
                try:
                    os.remove(f)
                except OSError:
                    pass

    print(f"\n[梯度{tier_index + 1}已清空] {tier_label}艘 {tier_name}: "
          f"移除结果 {removed} 行（保留其他梯度）", flush=True)
    if regen_instances:
        print(f"  已删除算例 instance_{tier_ids[0]}..{tier_ids[-1]}，将重新生成", flush=True)
    return removed


def save_large_configs(configs, output_dir):
    path = os.path.join(output_dir, 'large_configs.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)
    return path


def load_large_configs(output_dir):
    path = os.path.join(output_dir, 'large_configs.json')
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    # 断点续跑：从已保存算例 metadata 还原配置（避免重新 random 生成不同算例）
    meta_files = sorted(glob.glob(os.path.join(output_dir, 'instance_*_metadata.json')))
    if not meta_files:
        return None
    configs = []
    for mf in meta_files:
        with open(mf, encoding='utf-8') as f:
            meta = json.load(f)
        cfg = meta.get('config')
        if cfg:
            configs.append(cfg)
    if configs:
        print(f"  [断点] 从 {len(configs)} 个 metadata 文件还原算例配置")
    return configs or None


def generate_small_scale_configs(num_cases=15, seed=SMALL_SCALE_CONFIG_SEED):
    """生成阶段一基准+增强算例配置（固定种子，续跑时复用）。"""
    random.seed(seed)
    np.random.seed(seed)
    baseline_configs = []
    for _ in range(num_cases):
        n_ships = random.randint(1, 10)
        baseline_configs.append({
            'num_ships': n_ships,
            'tasks_per_ship_range': (1, 2),
            'arrival_max': n_ships * 5,
            'num_berths': max(2, n_ships // 3 + 1),
            'num_docks': max(1, n_ships // 5 + 1),
            'force_team_count': max(3, n_ships // 3 + 3),
            'num_instances': 1,
            'time_limit': 120,
            'mip_gap': 0.0,
            'group_name': f'Baseline_N={n_ships}',
            'config_type': 'baseline',
        })

    enhanced_configs = []
    for base_cfg in baseline_configs:
        n_ships = base_cfg['num_ships']
        enhanced_configs.append({
            'num_ships': n_ships,
            'tasks_per_ship_range': base_cfg['tasks_per_ship_range'],
            'arrival_max': base_cfg['arrival_max'],
            'num_berths': max(2, n_ships // 2 + 1),
            'num_docks': max(2, n_ships // 4 + 1),
            'force_team_count': max(4, n_ships // 2 + 2),
            'num_instances': 1,
            'time_limit': 120,
            'mip_gap': 0.0,
            'group_name': f'Enhanced_N={n_ships}',
            'config_type': 'enhanced',
        })
    return baseline_configs, enhanced_configs


def load_completed_batch_results(output_dir):
    """从 batch_results.csv 读取已完成算例 ID 及结果行（阶段一断点续跑）。"""
    csv_path = os.path.join(output_dir, 'batch_results.csv')
    if not os.path.isfile(csv_path):
        return set(), []
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        if df.empty or 'instance_id' not in df.columns:
            return set(), []
        records = df.to_dict('records')
        return set(int(x) for x in df['instance_id']), records
    except Exception:
        return set(), []


def load_small_scale_configs(output_root):
    path = os.path.join(output_root, 'small_scale_configs.json')
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return data.get('baseline'), data.get('enhanced')
    except Exception:
        return None, None


def save_small_scale_configs(output_root, baseline_configs, enhanced_configs, seed=None):
    path = os.path.join(output_root, 'small_scale_configs.json')
    payload = {'baseline': baseline_configs, 'enhanced': enhanced_configs}
    if seed is not None:
        payload['seed'] = seed
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def print_pipeline_resume_status(output_root=OUTPUT_ROOT):
    """启动时打印各阶段已完成算例数。"""
    if not AUTO_RESUME or FRESH_START:
        return
    small_dir = os.path.join(output_root, 'small_scale')
    enhanced_dir = os.path.join(output_root, 'small_scale_enhanced')
    large_dir = os.path.join(output_root, 'large_scale')
    bl_ids, _ = load_completed_batch_results(small_dir)
    en_ids, _ = load_completed_batch_results(enhanced_dir)
    lg_ids, _ = load_completed_large_scale_ids(large_dir)
    print("\n" + "-" * 72)
    print("[断点续跑] 检测到已有结果（AUTO_RESUME=True, FRESH_START=False）")
    print(f"  阶段一-基准: {len(bl_ids)}/{SMALL_SCALE_NUM_CASES} 例")
    print(f"  阶段一-增强: {len(en_ids)}/{SMALL_SCALE_NUM_CASES} 例")
    print(f"  阶段二-大规模: {len(lg_ids)}/60 例")
    if bl_ids or en_ids or lg_ids:
        print("  将自动跳过 CSV 中已完成的算例，从未完成处继续")
    print("-" * 72, flush=True)


def _run_small_scale_track(track_label, output_dir, configs, force_regen=False, resume=True):
    """阶段一单轨（基准/增强）求解，支持断点续跑。"""
    os.makedirs(output_dir, exist_ok=True)
    skip_ids, existing = (load_completed_batch_results(output_dir) if resume else (set(), []))
    target = len(configs)

    if len(skip_ids) >= target:
        print(f"\n[阶段一-{track_label}] 已完成 {len(skip_ids)}/{target} 例，跳过", flush=True)
        return list(existing)

    if resume and skip_ids:
        print(f"\n[阶段一-{track_label}] 断点续跑：跳过 {len(skip_ids)} 例，"
              f"待跑 {target - len(skip_ids)} 例", flush=True)

    exp = BatchExperimentGurobi(output_dir=output_dir, heuristic_type='lNS')
    exp.set_experiment_configs(configs)
    exp.generate_instances(force_regen=force_regen)
    if force_regen or not skip_ids:
        exp.save_instances()

    if track_label == '基准':
        print("\n[资源紧张度检查 — 基准配置（适度紧张）]")
        tight_count = 0
        for inst in exp.instances:
            is_tight, details = check_resource_tightness(inst)
            tag = "✓" if is_tight else "⚠"
            print(f"  {tag} 算例 {inst['instance_id']:>2d}: N={inst['num_ships']} "
                  f"泊位/船={details['berth_ratio']:.2f} "
                  f"船坞/船={details['dock_ratio']:.2f} "
                  f"任务/队={details['tasks_per_team']:.1f}")
            if not is_tight:
                for w in details['warnings']:
                    print(f"     警告: {w}")
            else:
                tight_count += 1
        print(f"  资源紧张度合格: {tight_count}/{len(exp.instances)}")

    print(f"\n[阶段一-{track_label}] 批量求解 (Gurobi + 启发式)...")
    results = exp.run_batch(skip_instance_ids=skip_ids, existing_results=existing)
    exp.save_summary()
    exp.save_summary_table()
    exp.save_benchmark_csv()
    return results


def run_small_scale_stage(output_root=OUTPUT_ROOT, force_regen=False, resume=True):
    """阶段一：小规模双轨实验，支持断点续跑。"""
    small_dir = os.path.join(output_root, 'small_scale')
    enhanced_dir = os.path.join(output_root, 'small_scale_enhanced')
    os.makedirs(output_root, exist_ok=True)

    baseline_configs, enhanced_configs = (None, None) if force_regen else load_small_scale_configs(output_root)
    if not baseline_configs or not enhanced_configs:
        # 始终用 SMALL_SCALE_CONFIG_SEED（算法设计8），保证可复现且与设计6不同
        seed = SMALL_SCALE_CONFIG_SEED
        baseline_configs, enhanced_configs = generate_small_scale_configs(
            SMALL_SCALE_NUM_CASES, seed)
        cfg_path = save_small_scale_configs(output_root, baseline_configs, enhanced_configs, seed)
        print(f"\n[阶段一] 生成并保存算例配置（种子={seed}，独立于算法设计6）→ {cfg_path}")
    else:
        print(f"\n[阶段一] 从 small_scale_configs.json 加载算例配置（续跑复用同一批算例）")

    for idx, cfg in enumerate(baseline_configs + enhanced_configs):
        ctype = cfg.get('config_type', '?')
        print(f"  算例 {idx + 1:>2d} [{ctype:>8s}]: N={cfg['num_ships']}  "
              f"berths={cfg['num_berths']}  docks={cfg['num_docks']}  "
              f"teams={cfg['force_team_count']}  "
              f"tasks/ship={cfg['tasks_per_ship_range']}  TL=120s  gap=0")

    small_results = _run_small_scale_track(
        '基准', small_dir, baseline_configs, force_regen=force_regen, resume=resume)
    enhanced_results = _run_small_scale_track(
        '增强', enhanced_dir, enhanced_configs, force_regen=force_regen, resume=resume)

    small_report = {
        '阶段': '小规模 (1-10艘) — 双轨制',
        '实验时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        '基准算例数': len(baseline_configs),
        '增强算例数': len(enhanced_configs),
        '策略': 'Gurobi TimeLimit=120s, MIPGap=0; KG-ALNS (Matheuristic)',
        '资源紧张度标准': {
            '基准': '泊位≈N/3+1, 船坞≈N/5+1, 团队≈N/3+3, 每船1-2任务',
            '增强': '泊位≈N/2+1, 船坞≈N/4+1, 团队≈N/2+2',
            '泊位 <= 8, 船坞 <= 4, 团队 <= 12': '双轨均遵守',
        },
    }
    with open(os.path.join(small_dir, 'experiment_report.json'), 'w', encoding='utf-8') as f:
        json.dump(small_report, f, indent=2)

    return small_results, enhanced_results


def _iter_large_scale_result_csvs(output_dir):
    """主文件 + 占用改存的 large_scale_results_YYYYMMDD_HHMMSS.csv。"""
    paths = []
    main = os.path.join(output_dir, 'large_scale_results.csv')
    if os.path.isfile(main):
        paths.append(main)
    paths.extend(sorted(glob.glob(os.path.join(output_dir, 'large_scale_results_*.csv'))))
    return paths


def load_completed_large_scale_ids(output_dir):
    """从增量 CSV 读取已完成算例 ID 及已有结果行。

    若 Excel/并发占用导致主文件写入失败、结果改存为带时间戳的备份，
    会自动合并这些备份；同一 instance_id 保留更新时间更新的一行。
    """
    candidates = _iter_large_scale_result_csvs(output_dir)
    if not candidates:
        return set(), []
    best_by_id = {}  # instance_id -> (mtime, record)
    for path in candidates:
        try:
            mtime = os.path.getmtime(path)
            df = pd.read_csv(path, encoding='utf-8-sig')
            if df.empty or 'instance_id' not in df.columns:
                continue
            for rec in df.to_dict('records'):
                iid = int(rec['instance_id'])
                prev = best_by_id.get(iid)
                if prev is None or mtime >= prev[0]:
                    best_by_id[iid] = (mtime, rec)
        except Exception:
            continue
    if not best_by_id:
        return set(), []
    records = [best_by_id[i][1] for i in sorted(best_by_id)]
    return set(best_by_id.keys()), records


def sync_large_scale_results_from_backups(output_dir):
    """把主文件与时间戳备份合并回 large_scale_results.csv（续跑前调用）。"""
    ids, records = load_completed_large_scale_ids(output_dir)
    if not records:
        return 0
    main = os.path.join(output_dir, 'large_scale_results.csv')
    try:
        if os.path.isfile(main):
            old = pd.read_csv(main, encoding='utf-8-sig')
            if (not old.empty and 'instance_id' in old.columns
                    and len(old) == len(records)
                    and set(int(x) for x in old['instance_id']) == ids):
                return len(records)
    except Exception:
        pass
    _safe_dataframe_to_csv(pd.DataFrame(records), main)
    print(f"[断点] 已从备份合并 {len(records)} 例 → {main}", flush=True)
    return len(records)


def _round_val(value, digits=2):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def build_unified_experiment_rows(small_results, enhanced_results, large_results):
    """将小规模（基准/增强）与大规模结果整合为统一表格行。"""
    rows = []

    def _append_small(phase_label, config_type, results):
        for r in results or []:
            rows.append({
                '阶段': phase_label,
                '配置类型': config_type,
                '算例编号': r.get('instance_id'),
                '组别': r.get('group_name', ''),
                '梯度': '',
                '船舶数': r.get('num_ships'),
                '任务数': r.get('num_tasks'),
                '泊位数': r.get('num_berths'),
                '干船坞数': r.get('num_docks'),
                '团队数': r.get('num_teams'),
                '目标值(Gurobi)': _round_val(r.get('objective')),
                '目标值(KG-ALNS)': _round_val(r.get('heuristic_obj')),
                '目标值(GA)': None,
                '目标值(ALNS)': None,
                'Gurobi耗时(秒)': _round_val(r.get('solve_time')),
                'KG-ALNS耗时(秒)': _round_val(r.get('heuristic_time'), 6),
                'GA耗时(秒)': None,
                'ALNS耗时(秒)': None,
                'Gap_to_Optimal(%)': _round_val(r.get('gap_to_optimal'), 4),
                '效率(%)': _round_val(r.get('efficiency')),
                '胜负': r.get('heu_superiority', '-'),
                '目标优胜(三方)': '',
                '求解状态': r.get('status', ''),
                '是否可行': '是' if r.get('feasible') else '否',
            })

    _append_small('小规模', '基准', small_results)
    _append_small('小规模', '增强', enhanced_results)

    for r in large_results or []:
        rows.append({
            '阶段': '大规模',
            '配置类型': r.get('group_name', 'baseline'),
            '算例编号': r.get('instance_id'),
            '组别': r.get('group_name', ''),
            '梯度': r.get('tier', ''),
            '船舶数': r.get('num_ships'),
            '任务数': r.get('num_tasks'),
            '泊位数': r.get('num_berths'),
            '干船坞数': r.get('num_docks'),
            '团队数': r.get('num_teams'),
            '目标值(Gurobi)': None,
            '目标值(KG-ALNS)': _round_val(r.get('mip_lns_obj_mean') or r.get('mip_lns_obj_best') or r.get('heuristic_obj')),
            '目标值(GA)': _round_val(r.get('ga_obj_mean') or r.get('ga_obj_best')),
            '目标值(ALNS)': _round_val(r.get('alns_obj_mean') or r.get('alns_obj_best')),
            '目标值std(KG-ALNS)': _round_val(r.get('mip_lns_obj_std')),
            '目标值std(GA)': _round_val(r.get('ga_obj_std')),
            '目标值std(ALNS)': _round_val(r.get('alns_obj_std')),
            '目标值best(KG-ALNS)': _round_val(r.get('mip_lns_obj_best')),
            '目标值best(GA)': _round_val(r.get('ga_obj_best')),
            '目标值best(ALNS)': _round_val(r.get('alns_obj_best')),
            'Gurobi耗时(秒)': None,
            'KG-ALNS总墙钟(秒)': _round_val(r.get('mip_lns_time') or r.get('heuristic_time')),
            'KG-ALNS主搜索(秒)': _round_val(r.get('mip_lns_search_time') or r.get('heuristic_search_time')),
            'KG-ALNS后处理(秒)': _round_val(r.get('mip_lns_postprocess_time') or r.get('heuristic_postprocess_time')),
            'GA耗时(秒)': _round_val(r.get('ga_time')),
            'ALNS耗时(秒)': _round_val(r.get('alns_time')),
            'Gap_to_Optimal(%)': _round_val(r.get('gap_to_alns_pct'), 4),
            '效率(%)': None,
            '胜负': '',
            '目标优胜(三方)': r.get('obj_winner', ''),
            '求解状态': 'KG-ALNS' if r.get('mip_lns_feasible') or r.get('heuristic_feasible') else 'INFEASIBLE',
            '是否可行': '是' if r.get('mip_lns_feasible') or r.get('heuristic_feasible') else '否',
        })

    return rows


def save_unified_excel_report(rows, output_path):
    """将全部实验结果写入单个 Excel 文件（单表汇总）。"""
    if not rows:
        print("没有结果数据，无法生成统一 Excel 报告")
        return None

    if not OPENPYXL_AVAILABLE:
        csv_path = output_path.replace('.xlsx', '.csv')
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"警告: openpyxl 未安装，已改存 CSV: {csv_path}")
        return csv_path

    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows

    export_df = pd.DataFrame(rows)
    wb = Workbook()
    ws = wb.active
    ws.title = "实验结果汇总"

    for col_idx, col_name in enumerate(export_df.columns, 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.value = col_name
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for row_idx, row in enumerate(dataframe_to_rows(export_df, index=False, header=False), 2):
        for col_idx, value in enumerate(row, 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.value = value
            cell.alignment = Alignment(horizontal="center", vertical="center")

    for col_idx, col_name in enumerate(export_df.columns, 1):
        col_values = [str(col_name)] + [str(v) for v in export_df.iloc[:, col_idx - 1].tolist()]
        max_length = max(len(v) for v in col_values)
        col_letter = chr(64 + col_idx) if col_idx <= 26 else None
        if col_letter:
            ws.column_dimensions[col_letter].width = min(max_length + 2, 28)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        wb.save(output_path)
        print(f"\n统一 Excel 汇总已保存: {output_path}  (共 {len(rows)} 行)")
    except PermissionError:
        alt_path = output_path.replace('.xlsx', f"_{datetime.now().strftime('%H%M%S')}.xlsx")
        wb.save(alt_path)
        print(f"原文件被占用，统一 Excel 已保存至: {alt_path}")
        output_path = alt_path
    return output_path


def check_resource_tightness(instance):
    """检查资源配置是否合理（避免资源过剩导致问题平凡化）

    论文 §4.1 修船厂资源紧张环境标准：
      泊位数 ≤ 船舶数 × 0.25（最多8个）
      干船坞数 ≤ 船舶数 × 0.15（最多4个）
      团队数 ≤ 任务数 × 0.15（最多12个）
      每个团队平均服务任务数 ≥ 6（利用率 > 60%）

    返回: (is_tight, details_dict)
    """
    num_ships = instance['num_ships']
    num_tasks = instance['num_tasks']
    num_berths = instance['num_berths']
    num_docks = instance['num_docks']
    num_teams = instance['num_teams']

    details = {
        'berth_ratio': round(num_berths / num_ships, 3) if num_ships > 0 else 0,
        'dock_ratio': round(num_docks / num_ships, 3) if num_ships > 0 else 0,
        'tasks_per_team': round(num_tasks / num_teams, 1) if num_teams > 0 else 0,
        'berths': num_berths,
        'docks': num_docks,
        'teams': num_teams,
    }

    checks = {
        'berth_ratio_ok': details['berth_ratio'] <= 0.25,
        'dock_ratio_ok': details['dock_ratio'] <= 0.15,
        'team_utilization_ok': details['tasks_per_team'] >= 6.0,
        'max_berths_ok': num_berths <= 8,
        'max_docks_ok': num_docks <= 4,
        'max_teams_ok': num_teams <= 12,
    }

    is_tight = all(checks.values())
    details['checks'] = checks
    details['is_tight'] = is_tight

    # 生成警告信息
    warnings = []
    if not checks['berth_ratio_ok']:
        warnings.append(f"泊位/船比例={details['berth_ratio']:.2f}(>0.25)")
    if not checks['dock_ratio_ok']:
        warnings.append(f"船坞/船比例={details['dock_ratio']:.2f}(>0.15)")
    if not checks['team_utilization_ok']:
        warnings.append(f"每队任务数={details['tasks_per_team']:.1f}(<6.0)")
    if not checks['max_berths_ok']:
        warnings.append(f"泊位数={num_berths}(>8)")
    if not checks['max_docks_ok']:
        warnings.append(f"船坞数={num_docks}(>4)")
    if not checks['max_teams_ok']:
        warnings.append(f"团队数={num_teams}(>12)")
    details['warnings'] = warnings

    return is_tight, details


def large_scale_configs_are_current(configs):
    """旧版 30-100 艘配置需重新生成，避免与新版 30-60 梯度混用。"""
    if not configs or len(configs) != 60:
        return False
    tiers = {c.get('tier') for c in configs}
    return tiers == set(LARGE_SCALE_TIER_LABELS)


def run_large_scale_stage(output_root=OUTPUT_ROOT, force_regen=False, resume=False,
                          split_by_tier=None, prompt_between_tiers=None,
                          only_tier_labels=None):
    """阶段二：大规模算例批量求解（可单独调用 / 断点续跑 / 按梯度分 3 批）。

    only_tier_labels: 仅跑指定梯度，如 {'30-40'} 或 {LARGE_SCALE_TIER_LABELS[0]}。
    """
    split_by_tier = LARGE_SPLIT_BY_TIER if split_by_tier is None else split_by_tier
    if only_tier_labels:
        split_by_tier = True
        prompt_between_tiers = False
        only_tier_labels = set(only_tier_labels)
    prompt_between_tiers = (
        PROMPT_BETWEEN_STAGES and split_by_tier
        if prompt_between_tiers is None else prompt_between_tiers
    )
    large_dir = os.path.join(output_root, 'large_scale')
    os.makedirs(large_dir, exist_ok=True)

    # 禁止从算法设计6/7 复制算例：算法设计8 一律用本目录种子重新生成
    if resume:
        # 主文件被 Excel 占用时结果会改存为时间戳备份，先合并再读
        sync_large_scale_results_from_backups(large_dir)
        skip_ids, existing_results = load_completed_large_scale_ids(large_dir)
    else:
        skip_ids, existing_results = set(), []
    skip_ids = skip_ids | set(LARGE_MANUAL_SKIP_IDS)
    if LARGE_MANUAL_SKIP_IDS:
        print(f"\n[阶段二] 手动跳过算例: {sorted(LARGE_MANUAL_SKIP_IDS)}")

    saved_configs = load_large_configs(large_dir)
    if saved_configs and not large_scale_configs_are_current(saved_configs):
        print(f"\n[阶段二] 检测到旧版算例配置（30-100 艘），将重新生成 "
              f"{LARGE_SCALE_SHIP_RANGE[0]}-{LARGE_SCALE_SHIP_RANGE[1]} 艘配置")
        saved_configs = None
        force_regen = True
        skip_ids, existing_results = set(), []
    if saved_configs and not force_regen:
        large_configs = saved_configs
        print(f"\n[阶段二] 从 {large_dir}/large_configs.json 加载 {len(large_configs)} 组配置")
    else:
        random.seed(REPRODUCIBILITY_SEED)
        np.random.seed(REPRODUCIBILITY_SEED)
        large_configs = generate_large_scale_configs(60, seed=REPRODUCIBILITY_SEED)
        save_large_configs(large_configs, large_dir)
        print(f"\n[阶段二] 已用种子 {REPRODUCIBILITY_SEED} 重新生成 {len(large_configs)} 组配置"
              f"（独立于算法设计6/7）", flush=True)

    for i, cfg in enumerate(large_configs):
        print(f"  算例 {i + 1:>2d}: N={cfg['num_ships']}  berths={cfg['num_berths']}  "
              f"docks={cfg['num_docks']}  teams={cfg['force_team_count']}  "
              f"tier={cfg['tier']}  tasks/ship={cfg['tasks_per_ship_range']}")

    large_exp = BatchExperimentGurobi(
        output_dir=large_dir,
        heuristic_type='lNS')
    large_exp.set_experiment_configs(large_configs)

    if resume and skip_ids and not force_regen:
        print(f"\n[阶段二] 断点续跑：跳过已完成的 {len(skip_ids)} 例")
    elif force_regen:
        skip_ids, existing_results = set(), []
        print(f"\n[阶段二] force_regen：清空旧结果索引，全量重生成算例", flush=True)

    print("\n[阶段二] 生成/加载大规模算例...")
    large_exp.generate_instances(force_regen=force_regen)
    if force_regen or not skip_ids:
        large_exp.save_instances()
        save_large_configs(large_configs, large_dir)

    batch_kwargs = dict(
        num_runs=LARGE_MIP_NUM_RUNS,
        ga_num_runs=LARGE_GA_NUM_RUNS,
        alns_num_runs=LARGE_ALNS_NUM_RUNS,
        batch_time_limit=LARGE_BATCH_TIME_LIMIT,
        mip_time_limit=LARGE_MIP_TIME_LIMIT,
        ga_time_limit=LARGE_GA_TIME_LIMIT,
        alns_time_limit=LARGE_ALNS_TIME_LIMIT,
        mip_sub_time_limit=LARGE_MIP_SUB_TIME_LIMIT,
        compare_ga=LARGE_COMPARE_GA,
        compare_alns=LARGE_COMPARE_ALNS,
        parallel_baselines=LARGE_PARALLEL_BASELINES,
        equal_compute_budget=LARGE_EQUAL_COMPUTE_BUDGET,
        primary_metric=LARGE_COMPARE_PRIMARY,
        skip_instance_ids=skip_ids,
        existing_results=existing_results,
    )

    if split_by_tier:
        tier_labels = LARGE_SCALE_TIER_LABELS
        tier_names = LARGE_SCALE_TIER_NAMES
        large_results = list(existing_results)
        user_stopped = False

        if prompt_between_tiers:
            print(f"\n[阶段二] 按梯度分 {len(tier_labels)} 批求解，"
                  f"每批完成后需输入 {PROMPT_YES!r} 才继续下一梯度", flush=True)

        tier_prompted = False
        for ti, tier_label in enumerate(tier_labels):
            if only_tier_labels and tier_label not in only_tier_labels:
                continue
            tier_name = tier_names[ti] if ti < len(tier_names) else ''
            tier_total = sum(
                1 for inst in large_exp.instances
                if inst.get('config', {}).get('tier') == tier_label)
            tier_pending_ids = {
                inst['instance_id'] for inst in large_exp.instances
                if inst.get('config', {}).get('tier') == tier_label
                and inst['instance_id'] not in skip_ids
            }

            if not tier_pending_ids:
                print(f"\n[阶段二] 梯度 {ti + 1}/{len(tier_labels)} "
                      f"({tier_label} {tier_name}) {tier_total} 例均已完成的，跳过", flush=True)
                continue

            tier_title = (f"阶段二·梯度{ti + 1}/{len(tier_labels)} "
                          f"({tier_label}艘 {tier_name}, {len(tier_pending_ids)} 例待跑)")

            if prompt_between_tiers:
                prev_label = None
                if tier_prompted:
                    prev_label = (f"阶段二·梯度{ti} "
                                  f"({tier_labels[ti - 1]} "
                                  f"{tier_names[ti - 1] if ti - 1 < len(tier_names) else ''})")
                if not prompt_stage_continue(prev_label, tier_title, start=not tier_prompted):
                    user_stopped = True
                    break
                tier_prompted = True

            est = estimate_large_batch_minutes(
                len(tier_pending_ids), LARGE_BATCH_TIME_LIMIT,
                LARGE_MIP_NUM_RUNS, LARGE_GA_NUM_RUNS, LARGE_ALNS_NUM_RUNS,
                compare_ga=LARGE_COMPARE_GA, compare_alns=LARGE_COMPARE_ALNS,
                parallel_baselines=LARGE_PARALLEL_BASELINES,
                parallel_triple=globals().get('LARGE_PARALLEL_TRIPLE', False),
                typical_wall_factor=LARGE_TYPICAL_WALL_FACTOR)
            print(f"\n[阶段二] {tier_title}")
            print(f"  批量求解 (待跑 {len(tier_pending_ids)}/{tier_total} 例, "
                  f"严格墙钟={LARGE_BATCH_TIME_LIMIT}s, "
                  f"KG×{LARGE_MIP_NUM_RUNS} GA×{LARGE_GA_NUM_RUNS} "
                  f"ALNS×{LARGE_ALNS_NUM_RUNS}, 主口径={LARGE_COMPARE_PRIMARY}, "
                  f"profile={LARGE_MIP_ALNS_PROFILE}, "
                  f"kg={globals().get('KG_ALNS_PROFILE_VERSION', '')})")
            if est.get('parallel_triple'):
                sched_note = 'KG∥GA∥ALNS 三方并行'
            elif est.get('parallel_baselines'):
                sched_note = 'GA∥ALNS'
            else:
                sched_note = 'GA+ALNS 串行'
            print(f"  每例预算: KG {est['mip_budget_sec']}s / 基线 {est['baseline_budget_sec']}s "
                  f"({sched_note})")
            print(f"  预计本梯度剩余: 典型 ≈ {est['typical_min']:.0f} 分钟 | "
                  f"上界 ≈ {est['worst_min']:.0f} 分钟")

            large_results = large_exp.run_large_scale_batch(
                only_tiers={tier_label},
                **batch_kwargs,
            )
            skip_ids = {int(r['instance_id']) for r in large_results}
            batch_kwargs['skip_instance_ids'] = skip_ids
            batch_kwargs['existing_results'] = large_results
            gc.collect()

        if user_stopped:
            print(f"\n[阶段二] 用户暂停于梯度批次之间，"
                  f"已完成 {len(large_results)}/{len(large_configs)} 例 → "
                  f"{large_dir}/large_scale_results.csv", flush=True)
        large_exp.large_stage_stopped_early = user_stopped
    else:
        pending = len(large_configs) - len(skip_ids)
        est = estimate_large_batch_minutes(
            pending, LARGE_BATCH_TIME_LIMIT,
            LARGE_MIP_NUM_RUNS, LARGE_GA_NUM_RUNS, LARGE_ALNS_NUM_RUNS,
            compare_ga=LARGE_COMPARE_GA, compare_alns=LARGE_COMPARE_ALNS,
            parallel_baselines=LARGE_PARALLEL_BASELINES,
            parallel_triple=globals().get('LARGE_PARALLEL_TRIPLE', False),
            typical_wall_factor=LARGE_TYPICAL_WALL_FACTOR)
        print(f"\n[阶段二] 批量求解 (待跑 {pending} 例, 严格墙钟={LARGE_BATCH_TIME_LIMIT}s, "
              f"KG×{LARGE_MIP_NUM_RUNS} GA×{LARGE_GA_NUM_RUNS} ALNS×{LARGE_ALNS_NUM_RUNS}, "
              f"主口径={LARGE_COMPARE_PRIMARY}, profile={LARGE_MIP_ALNS_PROFILE}, "
              f"kg={globals().get('KG_ALNS_PROFILE_VERSION', '')})")
        if est.get('parallel_triple'):
            sched_note = 'KG∥GA∥ALNS 三方并行'
        elif est.get('parallel_baselines'):
            sched_note = 'GA∥ALNS'
        else:
            sched_note = 'GA+ALNS 串行'
        print(f"  每例预算: KG {est['mip_budget_sec']}s / 基线 {est['baseline_budget_sec']}s "
              f"({sched_note})")
        print(f"  预计剩余: 典型 ≈ {est['typical_min']:.0f} 分钟 | "
              f"上界 ≈ {est['worst_min']:.0f} 分钟")
        large_results = large_exp.run_large_scale_batch(**batch_kwargs)
        large_exp.large_stage_stopped_early = False

    large_report = {
        '阶段': '大规模 (30-60艘)',
        '实验时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        '算例总数': len(large_exp.instances),
        '已完成': len(large_results),
        '策略': '严格墙钟对等(含Polish/Swap)；主口径 run-mean，辅口径 best-of-runs；跨算例 t+Wilcoxon',
        '参数': {
            'batch_time_limit': LARGE_BATCH_TIME_LIMIT,
            'strict_wall_clock': LARGE_STRICT_WALL_CLOCK,
            'mip_num_runs': LARGE_MIP_NUM_RUNS,
            'ga_num_runs': LARGE_GA_NUM_RUNS,
            'alns_num_runs': LARGE_ALNS_NUM_RUNS,
            'compare_primary': LARGE_COMPARE_PRIMARY,
            'mip_sub_time_limit': LARGE_MIP_SUB_TIME_LIMIT,
            'mip_alns_profile': LARGE_MIP_ALNS_PROFILE,
            'kg_alns_profile_version': globals().get(
                'KG_ALNS_PROFILE_VERSION', ''),
            'parallel_baselines': LARGE_PARALLEL_BASELINES,
            'parallel_triple': globals().get('LARGE_PARALLEL_TRIPLE', False),
            'equal_compute_budget': LARGE_EQUAL_COMPUTE_BUDGET,
        },
        '梯度分类': {label: 20 for label in LARGE_SCALE_TIER_LABELS},
        '船舶范围': f'{LARGE_SCALE_SHIP_RANGE[0]}-{LARGE_SCALE_SHIP_RANGE[1]}',
        '规模依据': '对标洋山港四期年约2500艘≈周50艘',
    }
    with open(f"{large_exp.output_dir}/experiment_report.json", 'w', encoding='utf-8') as f:
        json.dump(large_report, f, indent=2)
    return large_exp, large_results


def resolve_large_scale_time_limits(batch_time_limit=None, mip_time_limit=None,
                                    ga_time_limit=None, alns_time_limit=None):
    """解析三算法墙钟上限；返回 dict: default/mip/ga/alns/equal。

    LARGE_EQUAL_COMPUTE_BUDGET=True 时强制三算法同为 batch 墙钟（忽略更长的 quality MIP 时限）。
    """
    base = batch_time_limit if batch_time_limit is not None else LARGE_BATCH_TIME_LIMIT
    if globals().get('LARGE_EQUAL_COMPUTE_BUDGET', False):
        t = int(base)
        return {
            'default': t,
            'mip': t,
            'ga': t,
            'alns': t,
            'equal': True,
        }
    mip = (mip_time_limit if mip_time_limit is not None
           else LARGE_MIP_TIME_LIMIT if LARGE_MIP_TIME_LIMIT is not None else base)
    ga = (ga_time_limit if ga_time_limit is not None
          else LARGE_GA_TIME_LIMIT if LARGE_GA_TIME_LIMIT is not None else base)
    alns = (alns_time_limit if alns_time_limit is not None
            else LARGE_ALNS_TIME_LIMIT if LARGE_ALNS_TIME_LIMIT is not None else base)
    return {
        'default': base,
        'mip': int(mip),
        'ga': int(ga),
        'alns': int(alns),
        'equal': int(mip) == int(ga) == int(alns),
    }


def scale_mip_sub_time_limit(mip_time_limit, equal_budget=None):
    """MIP 子问题时限。

    同等墙钟：≤2s（与 EQUAL_BUDGET_120 的 iter_sub_cap 对齐）。
    解质量优先（长时限）：可给子问题更长时间。
    """
    if equal_budget is None:
        equal_budget = bool(globals().get('LARGE_EQUAL_COMPUTE_BUDGET', False))
    if equal_budget:
        return 1
    if mip_time_limit <= 150:
        return min(5, max(3, int(mip_time_limit * 0.035)))
    if mip_time_limit >= 500:
        return min(40, max(28, int(mip_time_limit * 0.05)))
    if mip_time_limit >= 300:
        return min(35, max(22, int(mip_time_limit * 0.04)))
    return min(26, max(18, int(mip_time_limit * 0.15)))


def get_tier_instance_ids(tier_index, configs=None, output_root=OUTPUT_ROOT):
    """返回某梯度算例 ID 列表（按 instance_id 升序）。"""
    if tier_index not in (0, 1, 2):
        raise ValueError(f"tier_index 须为 0/1/2，收到 {tier_index}")
    tier_label = LARGE_SCALE_TIER_LABELS[tier_index]
    if configs is None:
        large_dir = os.path.join(output_root, 'large_scale')
        configs = load_large_configs(large_dir)
    if not configs:
        configs = generate_large_scale_configs(60)
    return [i + 1 for i, cfg in enumerate(configs) if cfg.get('tier') == tier_label]


def tier_skip_ids_beyond(tier_index, max_cases, configs=None, output_root=OUTPUT_ROOT):
    """仅跑某梯度前 max_cases 例，返回应跳过的 instance_id 集合。"""
    ids = get_tier_instance_ids(tier_index, configs=configs, output_root=output_root)
    if max_cases is None or max_cases >= len(ids):
        return set()
    return set(ids[max_cases:])


def apply_large_tier_experiment_config(
        baseline_time_limit=120,
        mip_time_limit=None,
        alns_time_limit=None,
        ga_time_limit=None,
        num_runs=3,
        compare_ga=True,
        apply_dominance_profile=True,
        quality_first=True,
        time_limit=None):
    """三梯度实验配置。

    quality_first=False / --equal-budget（论文主表推荐）:
      三算法同等墙钟（默认 120s），参数档 EQUAL_BUDGET_120

    quality_first=True:
      GA/ALNS = baseline，KG-ALNS = 更长时限（辅表/上界实验）
    """
    global LARGE_BATCH_TIME_LIMIT, LARGE_MIP_TIME_LIMIT, LARGE_ALNS_TIME_LIMIT
    global LARGE_GA_TIME_LIMIT, LARGE_MIP_SUB_TIME_LIMIT, LARGE_MIP_NUM_RUNS
    global LARGE_GA_NUM_RUNS, LARGE_ALNS_NUM_RUNS, LARGE_EQUAL_COMPUTE_BUDGET
    global LARGE_COMPARE_GA, GUROBI_SUB_THREADS, MIP_ALNS_TIER1_USE_EXTENDED
    global LARGE_TIER_APPLY_DOMINANCE_PROFILE, LARGE_COMPARE_PRIMARY
    global LARGE_QUALITY_FIRST_MODE, LARGE_BASELINE_TIME_LIMIT, LARGE_MIP_QUALITY_TIME_LIMIT

    bl = int(baseline_time_limit)
    if quality_first:
        mip = int(mip_time_limit if mip_time_limit is not None
                  else LARGE_MIP_QUALITY_TIME_LIMIT)
        alns = int(alns_time_limit if alns_time_limit is not None else bl)
        ga = int(ga_time_limit if ga_time_limit is not None else bl)
        equal = False
    else:
        unified = int(time_limit if time_limit is not None else bl)
        mip = int(mip_time_limit if mip_time_limit is not None else unified)
        alns = int(alns_time_limit if alns_time_limit is not None else unified)
        ga = int(ga_time_limit if ga_time_limit is not None else unified)
        equal = (mip == alns == ga)

    runs = max(1, int(num_runs))

    LARGE_BASELINE_TIME_LIMIT = bl
    LARGE_MIP_QUALITY_TIME_LIMIT = mip if quality_first else mip
    LARGE_BATCH_TIME_LIMIT = max(mip, alns, ga)
    LARGE_MIP_TIME_LIMIT = mip
    LARGE_ALNS_TIME_LIMIT = alns
    LARGE_GA_TIME_LIMIT = ga
    LARGE_EQUAL_COMPUTE_BUDGET = equal
    LARGE_MIP_SUB_TIME_LIMIT = scale_mip_sub_time_limit(mip, equal_budget=equal)
    LARGE_MIP_NUM_RUNS = runs
    LARGE_GA_NUM_RUNS = runs
    LARGE_ALNS_NUM_RUNS = runs
    LARGE_COMPARE_GA = bool(compare_ga)
    LARGE_COMPARE_PRIMARY = 'best'
    LARGE_QUALITY_FIRST_MODE = bool(quality_first)
    GUROBI_SUB_THREADS = 1 if equal else 2  # 同等墙钟：少线程，减子问题开销
    MIP_ALNS_TIER1_USE_EXTENDED = (not quality_first and mip >= 150) or mip >= 300
    LARGE_TIER_APPLY_DOMINANCE_PROFILE = bool(apply_dominance_profile)


def apply_tier1_experiment_config(
        baseline_time_limit=120,
        mip_time_limit=None,
        alns_time_limit=None,
        ga_time_limit=None,
        num_runs=3,
        compare_ga=True,
        quality_first=True):
    """向后兼容别名 → apply_large_tier_experiment_config"""
    apply_large_tier_experiment_config(
        baseline_time_limit=baseline_time_limit,
        mip_time_limit=mip_time_limit,
        alns_time_limit=alns_time_limit,
        ga_time_limit=ga_time_limit,
        num_runs=num_runs,
        compare_ga=compare_ga,
        quality_first=quality_first,
    )


def apply_tier1_mip_dominance_config():
    """向后兼容：等同 apply_large_tier_experiment_config() 定稿默认。"""
    apply_large_tier_experiment_config()


def run_large_scale_tier(tier_index, output_root=OUTPUT_ROOT, force_regen=False, resume=True,
                         max_cases=None, extra_skip_ids=None):
    """只跑大规模某一梯度：0=Light(30-40), 1=Medium(41-50), 2=Heavy(51-60)。

    max_cases: 仅跑该梯度前 N 例（调试用）
    extra_skip_ids: 额外跳过的 instance_id 集合
    """
    if tier_index not in (0, 1, 2):
        raise ValueError(f"tier_index 须为 0/1/2，收到 {tier_index}")
    tier_label = LARGE_SCALE_TIER_LABELS[tier_index]
    tier_name = LARGE_SCALE_TIER_NAMES[tier_index]
    skip_extra = set(extra_skip_ids or [])
    if max_cases is not None:
        skip_extra |= tier_skip_ids_beyond(tier_index, max_cases, output_root=output_root)
        print(f"\n[独立运行] 梯度 {tier_label} 仅跑前 {max_cases} 例", flush=True)
    if skip_extra:
        prev = set(globals().get('LARGE_MANUAL_SKIP_IDS') or set())
        globals()['LARGE_MANUAL_SKIP_IDS'] = prev | skip_extra
    print(f"\n[独立运行] 大规模梯度 {tier_index + 1}/3: {tier_label}艘 {tier_name}", flush=True)
    try:
        return run_large_scale_stage(
            output_root=output_root,
            force_regen=force_regen,
            resume=resume,
            only_tier_labels={tier_label},
        )
    finally:
        if skip_extra:
            globals()['LARGE_MANUAL_SKIP_IDS'] = prev


def merge_all_experiment_results(output_root=OUTPUT_ROOT):
    """从各阶段 CSV 汇总为 实验结果汇总.xlsx（各脚本跑完后可单独调用）。"""
    small_dir = os.path.join(output_root, 'small_scale')
    enhanced_dir = os.path.join(output_root, 'small_scale_enhanced')
    large_dir = os.path.join(output_root, 'large_scale')
    _, small_results = load_completed_batch_results(small_dir)
    _, enhanced_results = load_completed_batch_results(enhanced_dir)
    _, large_results = load_completed_large_scale_ids(large_dir)
    unified_rows = build_unified_experiment_rows(small_results, enhanced_results, large_results)
    if not unified_rows:
        print("[汇总] 尚无结果可合并", flush=True)
        return None
    out_path = os.path.join(output_root, '实验结果汇总.xlsx')
    save_unified_excel_report(unified_rows, out_path)
    print(f"[汇总] 已写入 {out_path} ({len(unified_rows)} 行)", flush=True)
    return out_path


def prompt_stage_continue(completed_stage, next_stage, *, start=False):
    """控制台确认是否进入下一阶段。PROMPT_BETWEEN_STAGES=False 时自动继续。"""
    if not PROMPT_BETWEEN_STAGES:
        return True
    print("\n" + "=" * 72, flush=True)
    if start:
        print(f"[准备运行] {next_stage}", flush=True)
    else:
        print(f"[阶段完成] {completed_stage}", flush=True)
        print(f"是否继续下一阶段？", flush=True)
        print(f"  → {next_stage}", flush=True)
    print(f"请输入 {PROMPT_YES!r} 继续，其他任何输入则保存并退出", flush=True)
    if not FRESH_START and AUTO_RESUME:
        print("  （退出后保持 FRESH_START=False 即可断点续跑）", flush=True)
    print("=" * 72, flush=True)
    try:
        ans = input(f"继续? 输入 {PROMPT_YES}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n[中断] 用户取消，已保存的结果不受影响。", flush=True)
        return False
    if ans == PROMPT_YES:
        print(f"[确认] 开始 {next_stage}\n", flush=True)
        gc.collect()
        return True
    label = completed_stage or '当前准备项'
    print(f"[退出] 保留 {label} 的结果 → {OUTPUT_ROOT}", flush=True)
    return False


def _finalize_pipeline_partial(small_results, enhanced_results, large_results,
                               ablation_done=False, sensitivity_done=False):
    """分阶段提前退出时，尽量汇总已有 CSV。"""
    print("\n" + "=" * 72)
    print("[分阶段退出] 已完成阶段的结果已写入磁盘。")
    if small_results or enhanced_results:
        print(f"  阶段一: {OUTPUT_ROOT}/small_scale/ 与 small_scale_enhanced/")
    if large_results:
        print(f"  阶段二: {OUTPUT_ROOT}/large_scale/large_scale_results.csv "
              f"({len(large_results)} 行)")
    if ablation_done:
        print(f"  阶段三: {OUTPUT_ROOT}/ablation/ablation_results.csv")
    if sensitivity_done:
        print(f"  阶段四: {OUTPUT_ROOT}/sensitivity/sensitivity_results.csv")
    print("  续跑: FRESH_START=False, AUTO_RESUME=True，直接再运行 python design4.py", flush=True)
    print("        各阶段自动跳过 batch_results / large_scale_results 中已完成算例", flush=True)
    print("=" * 72, flush=True)
    if small_results or enhanced_results or large_results:
        try:
            unified_rows = build_unified_experiment_rows(
                small_results, enhanced_results, large_results or [])
            if unified_rows:
                save_unified_excel_report(
                    unified_rows, os.path.join(OUTPUT_ROOT, "实验结果汇总.xlsx"))
                print(f"  已更新: {OUTPUT_ROOT}/实验结果汇总.xlsx ({len(unified_rows)} 行)")
        except Exception as e:
            print(f"  [汇总 Excel 跳过] {e}", flush=True)


def generate_pilot_tune_configs(num_per_tier=None, seed=None):
    """生成独立调参算例配置（与正式 60 例种子不同，避免污染测试集）。"""
    num_per_tier = (PILOT_SCREEN_NUM_PER_TIER if num_per_tier is None
                    else int(num_per_tier))
    seed = PILOT_TUNE_SEED if seed is None else int(seed)
    random.seed(seed)
    np.random.seed(seed)
    configs = []
    for tier_min, tier_max, _ in LARGE_SCALE_TIER_SPECS:
        for _ in range(num_per_tier):
            n_ships = random.randint(tier_min, tier_max)
            n_berths = min(8, max(2, n_ships // 5 + 2))
            n_docks = min(4, max(1, n_ships // 10 + 1))
            n_teams = min(12, max(3, n_ships // 3 + 2))
            if n_ships <= 40:
                tasks_per_ship = (2, 3)
            elif n_ships <= 50:
                tasks_per_ship = (2, 3)
            else:
                tasks_per_ship = (2, 4)
            # 略收紧资源，使筛选更能暴露领域修复差异
            if random.random() < 0.45:
                n_berths = max(2, n_berths - 1)
                n_teams = max(3, n_teams - 1)
            configs.append({
                'num_ships': n_ships,
                'tasks_per_ship_range': tasks_per_ship,
                'arrival_max': n_ships * 3,
                'num_berths': n_berths,
                'num_docks': n_docks,
                'force_team_count': n_teams,
                'num_instances': 1,
                'time_limit': 600,
                'mip_gap': 0.0,
                'group_name': f'Tune_{tier_min}-{tier_max}',
                'tier': f'{tier_min}-{tier_max}',
                'config_type': 'pilot_tune',
            })
    return configs


def _pilot_apply_budget(budget_dict):
    """临时切换主表预算档（MatheuristicSolver 运行时读取全局）。"""
    global MIP_ALNS_EQUAL_BUDGET_120
    prev = MIP_ALNS_EQUAL_BUDGET_120
    MIP_ALNS_EQUAL_BUDGET_120 = dict(budget_dict)
    return prev


def _pilot_select_validate_instances(instances, per_tier=None):
    """每梯度选任务数最多的 per_tier 例（资源紧/规模大优先）。"""
    per_tier = (PILOT_VALIDATE_PER_TIER if per_tier is None else int(per_tier))
    selected = []
    for tier in LARGE_SCALE_TIER_LABELS:
        pool = [inst for inst in instances if inst.get('tier') == tier
                or (inst.get('config') or {}).get('tier') == tier]
        if not pool:
            pool = [inst for inst in instances
                    if str(inst.get('group_name', '')).endswith(tier)]
        pool = sorted(
            pool,
            key=lambda x: (int(x.get('num_tasks', 0)), int(x.get('num_ships', 0))),
            reverse=True,
        )
        selected.extend(pool[:per_tier])
    return selected


def _pilot_checkpoint_rows(rows, path):
    if not rows:
        return
    _safe_dataframe_to_csv(pd.DataFrame(rows), path)


def run_pilot_screen_step(output_dir, instances, resume=True):
    """第一步：15 调参算例 × V1/V2/V3 × 3 次 × 60s。返回 (rows, winner_key)。"""
    specs = _pilot_kg_profile_specs()
    screen_csv = os.path.join(output_dir, 'pilot_screen_results.csv')
    rows = []
    done_keys = set()
    if resume and os.path.isfile(screen_csv):
        try:
            old = pd.read_csv(screen_csv, encoding='utf-8-sig')
            rows = old.to_dict('records')
            for r in rows:
                done_keys.add((int(r['instance_id']), str(r['profile'])))
            print(f"  [续跑] 已加载筛选结果 {len(rows)} 条", flush=True)
        except Exception as e:
            print(f"  [续跑] 读取失败，将重跑: {e}", flush=True)
            rows, done_keys = [], set()

    tl = int(PILOT_SCREEN_TIME_LIMIT)
    n_runs = int(PILOT_SCREEN_NUM_RUNS)
    overrides = dict(PILOT_SCREEN_BUDGET_OVERRIDES)
    total = len(instances) * len(specs)
    finished = len(done_keys)
    print(f"\n{'=' * 72}")
    print(f"[试点·第一步] 60s 配置筛选")
    print(f"  算例={len(instances)}, 配置={list(specs)}, 每配置×{n_runs}, 墙钟={tl}s")
    print(f"  预计上界 ≈ {total * n_runs * tl / 3600:.2f} 小时 "
          f"（已完成键 {finished}/{total}）")
    print(f"{'=' * 72}", flush=True)

    exp = BatchExperimentGurobi(output_dir=output_dir, heuristic_type='lNS')
    for profile_key, spec in specs.items():
        budget = dict(spec['budget'])
        budget.update(overrides)
        # V2 只开领域修复，仍锁定小邻域、关闭关键路径
        if profile_key == 'V2':
            budget['critical_path_intensify'] = False
            budget['critical_intensify_rounds'] = 0
            budget['mid_intensify_budget_sec'] = 0
            budget['domain_intensify_max_sec'] = 0.0
            budget['lock_neighborhood'] = True
        prev_budget = _pilot_apply_budget(budget)
        try:
            solver = MatheuristicSolver(
                time_limit=tl,
                mip_sub_time_limit=2,
                max_iters=50000,
                verbose=False,
                ablation_flags=spec['flags'],
                ablation_variant=profile_key,
            )
            for inst in instances:
                iid = int(inst['instance_id'])
                key = (iid, profile_key)
                if key in done_keys:
                    continue
                print(f"\n  >> {profile_key} Case {iid} "
                      f"(N={inst['num_ships']}, T={inst['num_tasks']}, "
                      f"tier={inst.get('tier') or (inst.get('config') or {}).get('tier')})",
                      flush=True)
                inst_cfg = dict(inst)
                inst_cfg['config'] = dict(inst.get('config') or {})
                inst_cfg['config']['time_limit'] = tl
                result = exp._run_mip_lns_multi(solver, inst_cfg, num_runs=n_runs)
                row = {
                    'step': 'screen_60s',
                    'profile': profile_key,
                    'profile_label': spec['label'],
                    'instance_id': iid,
                    'tier': (inst.get('tier')
                             or (inst.get('config') or {}).get('tier', '')),
                    'num_ships': inst['num_ships'],
                    'num_tasks': inst['num_tasks'],
                    'num_berths': inst['num_berths'],
                    'num_docks': inst['num_docks'],
                    'num_teams': inst['num_teams'],
                    'obj_mean': result.get('objective_mean'),
                    'obj_best': result.get('objective_min'),
                    'obj_std': result.get('objective_std'),
                    'obj_runs': ';'.join(
                        f'{x:.4f}' for x in (result.get('objective_runs') or [])),
                    'time_mean': result.get('solve_time'),
                    'time_limit': tl,
                    'num_runs': n_runs,
                }
                rows.append(row)
                done_keys.add(key)
                _pilot_checkpoint_rows(rows, screen_csv)
                gc.collect()
        finally:
            _pilot_apply_budget(prev_budget)

    # 汇总选优：均值最低，并列看标准差
    summary = []
    for profile_key, spec in specs.items():
        sub = [r for r in rows if r.get('profile') == profile_key
               and r.get('obj_mean') is not None]
        if not sub:
            continue
        means = [float(r['obj_mean']) for r in sub]
        stds = [float(r['obj_std'] or 0.0) for r in sub]
        summary.append({
            'profile': profile_key,
            'label': spec['label'],
            'n_cases': len(sub),
            'avg_obj_mean': float(np.mean(means)),
            'avg_obj_std': float(np.mean(stds)),
            'avg_obj_best': float(np.mean([float(r['obj_best']) for r in sub])),
        })
    summary_path = os.path.join(output_dir, 'pilot_screen_summary.csv')
    if summary:
        _safe_dataframe_to_csv(pd.DataFrame(summary), summary_path)
        summary.sort(key=lambda x: (x['avg_obj_mean'], x['avg_obj_std']))
        winner = summary[0]['profile']
        print(f"\n[试点·第一步] 汇总 → {summary_path}", flush=True)
        for s in summary:
            mark = ' ← 最优' if s['profile'] == winner else ''
            print(f"  {s['profile']} ({s['label']}): "
                  f"mean={s['avg_obj_mean']:.1f}, std={s['avg_obj_std']:.1f}, "
                  f"best={s['avg_obj_best']:.1f}{mark}", flush=True)
        print(f"  选定配置: {winner}", flush=True)
        return rows, winner, summary
    print("[试点·第一步] 无有效结果", flush=True)
    return rows, 'V3', summary


def run_pilot_validate_step(output_dir, instances, winner_key, resume=True):
    """第二步：9 例 × KG(winner) vs ALNS × 3 次 × 300s。"""
    specs = _pilot_kg_profile_specs()
    if winner_key not in specs:
        winner_key = 'V3'
    spec = specs[winner_key]
    validate_ids = {int(i['instance_id']) for i in instances}
    val_csv = os.path.join(output_dir, 'pilot_validate_results.csv')
    rows = []
    done_ids = set()
    if resume and os.path.isfile(val_csv):
        try:
            old = pd.read_csv(val_csv, encoding='utf-8-sig')
            rows = old.to_dict('records')
            for r in rows:
                if int(r.get('instance_id', -1)) in validate_ids:
                    done_ids.add(int(r['instance_id']))
            print(f"  [续跑] 已加载验证结果 {len(done_ids)}/{len(instances)} 例", flush=True)
        except Exception as e:
            print(f"  [续跑] 验证 CSV 读取失败，将重跑: {e}", flush=True)
            rows, done_ids = [], set()

    tl = int(PILOT_VALIDATE_TIME_LIMIT)
    n_runs = int(PILOT_VALIDATE_NUM_RUNS)
    print(f"\n{'=' * 72}")
    print(f"[试点·第二步] 300s 小范围验证（KG={winner_key} vs ALNS）")
    print(f"  算例={len(instances)}, 各×{n_runs}, 墙钟={tl}s")
    print(f"  预计上界 ≈ {len(instances) * 2 * n_runs * tl / 3600:.2f} 小时")
    print(f"  通过门槛: KG 至少 {PILOT_VALIDATE_MIN_WINS} 胜，且 mean 目标优于 ALNS")
    print(f"{'=' * 72}", flush=True)

    budget = dict(spec['budget'])
    prev_budget = _pilot_apply_budget(budget)
    exp = BatchExperimentGurobi(output_dir=output_dir, heuristic_type='lNS')
    try:
        kg_solver = MatheuristicSolver(
            time_limit=tl,
            mip_sub_time_limit=3,
            max_iters=50000,
            verbose=False,
            ablation_flags=spec['flags'],
            ablation_variant=winner_key,
        )
        for inst in instances:
            iid = int(inst['instance_id'])
            if iid in done_ids:
                continue
            print(f"\n  >> Validate Case {iid} "
                  f"(N={inst['num_ships']}, T={inst['num_tasks']})", flush=True)
            inst_cfg = dict(inst)
            inst_cfg['config'] = dict(inst.get('config') or {})
            inst_cfg['config']['time_limit'] = tl

            kg = exp._run_mip_lns_multi(kg_solver, inst_cfg, num_runs=n_runs)
            print(f"    Case {iid}: ALNS ×{n_runs}...", flush=True)
            alns_solver = ALNSSolver(time_limit=tl, verbose=False, num_runs=n_runs)
            alns = alns_solver.solve(inst_cfg)
            alns_mean = alns.get('objective')
            alns_best = alns.get('objective_min', alns_mean)
            alns_std = alns.get('objective_std')
            alns_time = alns.get('solve_time')

            kg_mean = kg.get('objective_mean')
            gap = None
            if kg_mean is not None and alns_mean and alns_mean > 0:
                gap = (kg_mean - alns_mean) / alns_mean * 100.0
            winner = '-'
            if kg_mean is not None and alns_mean is not None:
                if kg_mean < alns_mean - 1e-6:
                    winner = 'KG'
                elif alns_mean < kg_mean - 1e-6:
                    winner = 'ALNS'
                else:
                    winner = 'TIE'

            row = {
                'step': 'validate_300s',
                'kg_profile': winner_key,
                'kg_profile_label': spec['label'],
                'instance_id': iid,
                'tier': (inst.get('tier')
                         or (inst.get('config') or {}).get('tier', '')),
                'num_ships': inst['num_ships'],
                'num_tasks': inst['num_tasks'],
                'kg_obj_mean': kg_mean,
                'kg_obj_best': kg.get('objective_min'),
                'kg_obj_std': kg.get('objective_std'),
                'kg_time': kg.get('solve_time'),
                'alns_obj_mean': alns_mean,
                'alns_obj_best': alns_best,
                'alns_obj_std': alns_std,
                'alns_time': alns_time,
                'gap_to_alns_pct_mean': gap,
                'obj_winner': winner,
                'time_limit': tl,
                'num_runs': n_runs,
            }
            rows.append(row)
            done_ids.add(iid)
            _pilot_checkpoint_rows(rows, val_csv)
            if gap is not None:
                print(f"    → KG={kg_mean:.1f} ALNS={alns_mean:.1f} "
                      f"gap={gap:+.2f}% winner={winner}", flush=True)
            else:
                print(f"    → KG={kg_mean} ALNS={alns_mean} winner={winner}",
                      flush=True)
            gc.collect()
    finally:
        _pilot_apply_budget(prev_budget)

    # 判定是否建议全量
    cmp_rows = [r for r in rows if int(r['instance_id']) in validate_ids
                and r.get('kg_obj_mean') is not None
                and r.get('alns_obj_mean') is not None]
    kg_wins = sum(1 for r in cmp_rows if r.get('obj_winner') == 'KG')
    alns_wins = sum(1 for r in cmp_rows if r.get('obj_winner') == 'ALNS')
    ties = sum(1 for r in cmp_rows if r.get('obj_winner') == 'TIE')
    kg_avg = float(np.mean([float(r['kg_obj_mean']) for r in cmp_rows])) if cmp_rows else None
    alns_avg = float(np.mean([float(r['alns_obj_mean']) for r in cmp_rows])) if cmp_rows else None
    pass_wins = kg_wins >= int(PILOT_VALIDATE_MIN_WINS)
    pass_mean = (kg_avg is not None and alns_avg is not None and kg_avg < alns_avg)
    recommend_full = bool(pass_wins and pass_mean)

    decision = {
        'kg_profile': winner_key,
        'n_cases': len(cmp_rows),
        'kg_wins': kg_wins,
        'alns_wins': alns_wins,
        'ties': ties,
        'kg_avg_obj_mean': kg_avg,
        'alns_avg_obj_mean': alns_avg,
        'pass_wins': pass_wins,
        'pass_mean': pass_mean,
        'recommend_full': recommend_full,
        'min_wins_required': int(PILOT_VALIDATE_MIN_WINS),
    }
    decision_path = os.path.join(output_dir, 'pilot_validate_decision.json')
    with open(decision_path, 'w', encoding='utf-8') as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)
    print(f"\n[试点·第二步] 战绩: KG {kg_wins} 胜 / ALNS {alns_wins} 胜 / 平 {ties}",
          flush=True)
    if kg_avg is not None:
        print(f"  平均目标: KG={kg_avg:.1f}, ALNS={alns_avg:.1f}", flush=True)
    if recommend_full:
        print(f"  结论: 通过门槛 → 可将 EXPERIMENT_MODE 改为 'large_only' 跑全量 60 例",
              flush=True)
    else:
        print(f"  结论: 未达门槛（需 ≥{PILOT_VALIDATE_MIN_WINS} 胜且 mean 更优）"
              f" → 建议继续增强后再测，勿直接全量", flush=True)
    print(f"  判定文件: {decision_path}", flush=True)
    return rows, decision


def run_pilot_destroy_test(output_root=OUTPUT_ROOT, force_regen=False, resume=True):
    """轻量 destroy 权重试点：15 例 × 3 profile × 相同种子。

    只改 DESTROY_PROFILE 初始权重；repair / SA / 目标函数不变。
    输出 pilot_destroy_test_results.csv。
    """
    global DESTROY_PROFILE

    pilot_dir = os.path.join(output_root, 'pilot_destroy')
    os.makedirs(pilot_dir, exist_ok=True)
    results_path = os.path.join(pilot_dir, 'pilot_destroy_test_results.csv')

    cfg_path = os.path.join(pilot_dir, 'pilot_destroy_configs.json')
    if force_regen or not os.path.isfile(cfg_path):
        configs = generate_pilot_tune_configs(
            num_per_tier=PILOT_DESTROY_NUM_PER_TIER, seed=PILOT_DESTROY_SEED)
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(configs, f, indent=2, ensure_ascii=False)
        print(f"\n[pilot_destroy] 已生成 {len(configs)} 组配置 "
              f"(seed={PILOT_DESTROY_SEED})", flush=True)
    else:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        print(f"\n[pilot_destroy] 加载配置 {len(configs)} 组", flush=True)

    exp = BatchExperimentGurobi(output_dir=pilot_dir, heuristic_type='lNS')
    exp.set_experiment_configs(configs)
    exp.generate_instances(force_regen=force_regen)
    for inst, cfg in zip(exp.instances, configs):
        inst['tier'] = cfg.get('tier', '')
        inst.setdefault('config', {})['tier'] = cfg.get('tier', '')
        inst['config']['time_limit'] = int(PILOT_DESTROY_TIME_LIMIT)
    if force_regen or not resume:
        exp.save_instances()

    rows = []
    done_keys = set()
    if resume and os.path.isfile(results_path):
        try:
            old = pd.read_csv(results_path, encoding='utf-8-sig')
            rows = old.to_dict('records')
            for r in rows:
                done_keys.add((int(r['instance_id']), str(r['destroy_profile'])))
            print(f"  [续跑] 已有 {len(done_keys)} 条", flush=True)
        except Exception as e:
            print(f"  [续跑] 读取失败，重跑: {e}", flush=True)
            rows, done_keys = [], set()

    tl = int(PILOT_DESTROY_TIME_LIMIT)
    n_runs = int(PILOT_DESTROY_NUM_RUNS)
    profiles = list(PILOT_DESTROY_PROFILES)
    total = len(exp.instances) * len(profiles)
    print(f"\n{'=' * 72}")
    print("[pilot_destroy] Destroy 初始权重对比（不改 repair）")
    print(f"  实例={len(exp.instances)}, profiles={profiles}")
    print(f"  墙钟={tl}s × {n_runs} runs/例")
    print(f"  预计上界 ≈ {total * n_runs * tl / 3600:.1f} 小时 "
          f"（已完成 {len(done_keys)}/{total}）")
    print(f"  权重表: {DESTROY_PROFILE_INIT_WEIGHTS}")
    print(f"{'=' * 72}", flush=True)

    saved_profile = DESTROY_PROFILE
    try:
        for profile in profiles:
            DESTROY_PROFILE = profile
            label = DESTROY_PROFILE_LABELS.get(profile, profile)
            init_w = DESTROY_PROFILE_INIT_WEIGHTS.get(profile, [1.0] * 5)
            print(f"\n>>> Profile {label}: init_weights={init_w}", flush=True)
            solver = MatheuristicSolver(
                time_limit=tl,
                mip_sub_time_limit=3,
                max_iters=50000,
                verbose=False,
                ablation_flags={
                    'use_mip_repair': False,
                    'use_l1_destroy': True,
                    'use_l3_destroy': True,
                    'use_l4_destroy': True,
                    'use_adaptive_weights': True,
                    'use_domain_repair': False,
                    'use_knowledge_repair': True,
                    'use_polish': False,
                    'use_team_swap': False,
                    'use_skill_refine': True,
                    'use_elite_restart': True,
                },
                ablation_variant=f'destroy_{profile}',
            )
            for inst in exp.instances:
                iid = int(inst['instance_id'])
                key = (iid, profile)
                if key in done_keys:
                    continue
                print(f"\n  >> {profile} Case {iid} "
                      f"(N={inst['num_ships']}, T={inst['num_tasks']}, "
                      f"tier={inst.get('tier')})", flush=True)
                # 固定种子：同实例跨 profile 可对照
                random.seed(PILOT_DESTROY_SEED + iid * 97)
                np.random.seed((PILOT_DESTROY_SEED + iid * 97) % (2 ** 31 - 1))
                inst_cfg = dict(inst)
                inst_cfg['config'] = dict(inst.get('config') or {})
                inst_cfg['config']['time_limit'] = tl
                result = exp._run_mip_lns_multi(solver, inst_cfg, num_runs=n_runs)
                row = {
                    'instance_id': iid,
                    'tier': inst.get('tier', ''),
                    'num_ships': inst['num_ships'],
                    'num_tasks': inst['num_tasks'],
                    'num_berths': inst['num_berths'],
                    'num_docks': inst['num_docks'],
                    'num_teams': inst['num_teams'],
                    'destroy_profile': profile,
                    'destroy_profile_label': label,
                    'init_weights': ';'.join(f'{w:.4f}' for w in init_w),
                    'objective_mean': result.get('objective_mean'),
                    'objective_best': result.get('objective_min'),
                    'std': result.get('objective_std'),
                    'runtime': result.get('solve_time'),
                    'time_limit': tl,
                    'num_runs': n_runs,
                    'obj_runs': ';'.join(
                        f'{x:.4f}' for x in (result.get('objective_runs') or [])),
                }
                rows.append(row)
                done_keys.add(key)
                _safe_dataframe_to_csv(pd.DataFrame(rows), results_path)
                gc.collect()
    finally:
        DESTROY_PROFILE = saved_profile

    # 汇总：哪个 profile 平均更好
    summary_rows = []
    if rows:
        df = pd.DataFrame(rows)
        for profile in profiles:
            sub = df[df['destroy_profile'] == profile]
            if sub.empty:
                continue
            summary_rows.append({
                'destroy_profile': profile,
                'label': DESTROY_PROFILE_LABELS.get(profile, profile),
                'n_cases': len(sub),
                'avg_objective_mean': float(sub['objective_mean'].mean()),
                'avg_objective_best': float(sub['objective_best'].mean()),
                'avg_std': float(sub['std'].mean()),
                'avg_runtime': float(sub['runtime'].mean()),
            })
        # 逐例胜负相对 baseline
        if 'baseline' in set(df['destroy_profile']):
            base = df[df['destroy_profile'] == 'baseline'].set_index('instance_id')
            for profile in profiles:
                if profile == 'baseline':
                    continue
                sub = df[df['destroy_profile'] == profile].set_index('instance_id')
                common = base.index.intersection(sub.index)
                wins = int((sub.loc[common, 'objective_mean']
                            < base.loc[common, 'objective_mean'] - 1e-6).sum())
                losses = int((sub.loc[common, 'objective_mean']
                              > base.loc[common, 'objective_mean'] + 1e-6).sum())
                ties = len(common) - wins - losses
                for s in summary_rows:
                    if s['destroy_profile'] == profile:
                        s['wins_vs_baseline'] = wins
                        s['losses_vs_baseline'] = losses
                        s['ties_vs_baseline'] = ties
        summary_path = os.path.join(pilot_dir, 'pilot_destroy_summary.csv')
        _safe_dataframe_to_csv(pd.DataFrame(summary_rows), summary_path)
        print(f"\n[pilot_destroy] 汇总 → {summary_path}", flush=True)
        best = min(summary_rows, key=lambda x: x['avg_objective_mean'])
        for s in summary_rows:
            mark = ' ← 最优mean' if s is best else ''
            vs = ''
            if 'wins_vs_baseline' in s:
                vs = (f", vsA={s['wins_vs_baseline']}W/"
                      f"{s['losses_vs_baseline']}L/{s['ties_vs_baseline']}T")
            print(f"  {s['destroy_profile']}: mean={s['avg_objective_mean']:.1f}, "
                  f"best={s['avg_objective_best']:.1f}, std={s['avg_std']:.1f}, "
                  f"time={s['avg_runtime']:.1f}s{vs}{mark}", flush=True)
        print(f"\n建议: 若 knowledge_high/stable 相对 baseline 均值更低且胜场占优，"
              f"再设 DESTROY_PROFILE 后改 EXPERIMENT_MODE='large_only' 跑全量60例。",
              flush=True)

    print(f"\n[完成] {results_path}")
    return rows, summary_rows


def run_pilot_test_stage(output_root=OUTPUT_ROOT, force_regen=False, resume=True):
    """试点全流程：另生成 15 调参算例 → 60s 筛配置 → 300s 验证。"""
    pilot_dir = os.path.join(output_root, 'pilot_test')
    os.makedirs(pilot_dir, exist_ok=True)

    configs_path_ok = os.path.isfile(os.path.join(pilot_dir, 'pilot_configs.json'))
    if force_regen or not configs_path_ok:
        configs = generate_pilot_tune_configs()
        save_large_configs(configs, pilot_dir)
        # save_large_configs 写的是 large_configs.json；再存一份明确名
        with open(os.path.join(pilot_dir, 'pilot_configs.json'), 'w', encoding='utf-8') as f:
            json.dump(configs, f, indent=2, ensure_ascii=False)
        print(f"\n[试点] 已生成 {len(configs)} 组调参配置 "
              f"(seed={PILOT_TUNE_SEED})", flush=True)
    else:
        with open(os.path.join(pilot_dir, 'pilot_configs.json'), 'r', encoding='utf-8') as f:
            configs = json.load(f)
        print(f"\n[试点] 加载已有调参配置 {len(configs)} 组", flush=True)

    exp = BatchExperimentGurobi(output_dir=pilot_dir, heuristic_type='lNS')
    exp.set_experiment_configs(configs)
    exp.generate_instances(force_regen=force_regen)
    # 把 tier 提到 instance 顶层，便于筛选
    for inst, cfg in zip(exp.instances, configs):
        inst['tier'] = cfg.get('tier', '')
        inst.setdefault('config', {})['tier'] = cfg.get('tier', '')
        inst['config']['time_limit'] = PILOT_SCREEN_TIME_LIMIT
    if force_regen or not resume:
        exp.save_instances()

    print(f"[试点] 算例 {len(exp.instances)} 例 → {pilot_dir}", flush=True)
    for inst in exp.instances:
        print(f"  id={inst['instance_id']:>2d}  N={inst['num_ships']}  "
              f"T={inst['num_tasks']}  tier={inst.get('tier')}  "
              f"B/D/Team={inst['num_berths']}/{inst['num_docks']}/{inst['num_teams']}")

    screen_rows, winner, screen_summary = run_pilot_screen_step(
        pilot_dir, exp.instances, resume=resume and not force_regen)

    validate_insts = _pilot_select_validate_instances(exp.instances)
    print(f"\n[试点] 验证子集 {len(validate_insts)} 例（每梯度任务数最多的 "
          f"{PILOT_VALIDATE_PER_TIER} 例）:", flush=True)
    for inst in validate_insts:
        print(f"  id={inst['instance_id']:>2d}  N={inst['num_ships']}  "
              f"T={inst['num_tasks']}  tier={inst.get('tier')}")

    val_rows, decision = run_pilot_validate_step(
        pilot_dir, validate_insts, winner,
        resume=resume and not force_regen)

    report = {
        '阶段': '试点测试 pilot_test',
        '实验时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        '调参种子': PILOT_TUNE_SEED,
        '筛选': {
            'time_limit': PILOT_SCREEN_TIME_LIMIT,
            'num_runs': PILOT_SCREEN_NUM_RUNS,
            'n_instances': len(exp.instances),
            'winner': winner,
            'summary': screen_summary,
        },
        '验证': decision,
        '输出目录': pilot_dir,
    }
    with open(os.path.join(pilot_dir, 'pilot_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[试点完成] 结果目录: {pilot_dir}/")
    print(f"  - pilot_screen_results.csv / pilot_screen_summary.csv")
    print(f"  - pilot_validate_results.csv / pilot_validate_decision.json")
    print(f"  - pilot_report.json")
    return report


def main():
    mode = EXPERIMENT_MODE
    use_prompts = PROMPT_BETWEEN_STAGES and mode in ('interactive', 'full', 'resume_large')
    pipeline_modes = ('full', 'interactive', 'resume_large', 'large_only')

    if sys.platform == 'win32' and mode == 'full' and not WINDOWS_SAFE_RUN and not use_prompts:
        print("\n" + "!" * 72)
        print("  警告: Windows + full 模式连续跑 4 个阶段，极易黑屏/OOM。")
        print("  建议: WINDOWS_SAFE_RUN=True 或 EXPERIMENT_MODE='resume_large'")
        print("!" * 72 + "\n", flush=True)

    if WINDOWS_SAFE_RUN:
        print("\n" + "=" * 72)
        print("[WINDOWS_SAFE_RUN] 防黑屏模式（无人值守，阶段间不询问）")
        print(f"  模式={mode} | 小规模={RUN_SMALL_SCALE} | "
              f"消融={RUN_ABLATION_AFTER_LARGE} | 敏感性={RUN_SENSITIVITY_AFTER_ABLATION}")
        if FRESH_START:
            print(f"  FRESH_START=True → 从零重跑（清空 {OUTPUT_ROOT}）")
        else:
            print(f"  FRESH_START=False, AUTO_RESUME=True → 中断后从 CSV 断点续跑")
        if mode == 'pilot_destroy':
            print(f"  试点: 15例 × 3 destroy profile（baseline/knowledge_high/stable）")
            print(f"  墙钟={PILOT_DESTROY_TIME_LIMIT}s × {PILOT_DESTROY_NUM_RUNS} runs")
            print(f"  预计约 {15 * 3 * PILOT_DESTROY_NUM_RUNS * PILOT_DESTROY_TIME_LIMIT / 3600:.1f} 小时（不上全量60）")
        elif mode == 'pilot_test':
            print(f"  试点测试: 15例×60s 筛 V1/V2/V3 → 9例×300s 验证 KG vs ALNS")
            print(f"  预计约 5～7 小时（不上全量 60 例）")
        elif mode == 'ablation_and_lambda':
            print("  流程: 消融×3(ablation_x3, run并行) → λ敏感性(sensitivity_lambda)")
            print("  预计约 8h + 11h ≈ 19–20 小时；断点续跑已启用")
            print("  不覆盖: 有效数据/ablation、sensitivity_compact_9x3")
        elif mode == 'sensitivity_lambda':
            print("  单阶段: λ 惩罚系数敏感性 (S4_lambda)")
            print(f"  输出: {OUTPUT_ROOT}/sensitivity/{SENSITIVITY_LAMBDA_PROTOCOL}/")
            print(f"  并行: {SENSITIVITY_PARALLEL_WORKERS} 进程"
                  if SENSITIVITY_PARALLEL else "  并行: 关闭（串行）")
            print("  预计约 3.5–4.5 小时（3 进程）；断点续跑已启用")
            print("  不覆盖: ablation_x3、有效数据、sensitivity_compact_9x3")
        elif mode in ('ablation', 'sensitivity'):
            print(f"  单阶段模式: {mode}")
        else:
            stages = []
            if RUN_SMALL_SCALE:
                stages.append('阶段一小规模(15+15)')
            stages.append('阶段二大规模(60)')
            if RUN_ABLATION_AFTER_LARGE:
                stages.append('消融')
            if RUN_SENSITIVITY_AFTER_ABLATION:
                stages.append('敏感性')
            print(f"  流程: {' → '.join(stages)}"
                  + (" | 大规模分3梯度" if LARGE_SPLIT_BY_TIER else ""))
            print(f"  重复: KG×{LARGE_MIP_NUM_RUNS} GA×{LARGE_GA_NUM_RUNS} ALNS×{LARGE_ALNS_NUM_RUNS}"
                  f" | 主口径={LARGE_COMPARE_PRIMARY}"
                  f" | kg={globals().get('KG_ALNS_PROFILE_VERSION', '')}"
                  f" | destroy={globals().get('DESTROY_PROFILE', 'baseline')}")
        print(f"  低内存: {LARGE_LOW_MEMORY_MODE} | "
              f"基线={'KG∥GA∥ALNS' if globals().get('LARGE_PARALLEL_TRIPLE', False) else ('GA∥ALNS并行' if LARGE_PARALLEL_BASELINES else 'GA→ALNS串行')} | "
              f"Gurobi子问题线程={GUROBI_SUB_THREADS}")
        print(f"  输出: {OUTPUT_ROOT}/")
        print("  运行方式: CMD 执行 python design4.py（勿 PyCharm Debug）")
        print("  运行前: 关闭 Excel 中已打开的结果目录下 CSV/Excel 文件")
        print("=" * 72, flush=True)

    if use_prompts and mode in pipeline_modes:
        print(f"\n[交互模式] 每完成一阶段需在控制台输入 {PROMPT_YES!r} 才继续下一阶段", flush=True)
        print("  同一进程内 large_exp 会传递给消融/敏感性，保证算例连贯\n", flush=True)

    if mode == 'pilot_destroy':
        clear_output_root(OUTPUT_ROOT, clear=CLEAR_OUTPUT_ON_START)
        rows, summary = run_pilot_destroy_test(
            output_root=OUTPUT_ROOT,
            force_regen=FRESH_START,
            resume=(AUTO_RESUME and not FRESH_START),
        )
        print(f"\n[完成] destroy 权重试点结束，共 {len(rows or [])} 条")
        print(f"  输出: {OUTPUT_ROOT}/pilot_destroy/pilot_destroy_test_results.csv")
        return

    if mode == 'pilot_test':
        clear_output_root(OUTPUT_ROOT, clear=CLEAR_OUTPUT_ON_START)
        report = run_pilot_test_stage(
            output_root=OUTPUT_ROOT,
            force_regen=FRESH_START,
            resume=(AUTO_RESUME and not FRESH_START),
        )
        print(f"\n[完成] 试点测试结束")
        print(f"  输出: {OUTPUT_ROOT}/pilot_test/")
        if report and report.get('验证', {}).get('recommend_full'):
            print("  下一步: 将 EXPERIMENT_MODE 改为 'large_only' 再跑全量 60 例")
        return

    if mode == 'ablation_regression':
        clear_output_root(OUTPUT_ROOT, clear=False)
        print("\n[Full回归模式] Case 2/25/43 × 300s，对照 large_scale_results.csv", flush=True)
        run_ablation_full_regression_test(output_root=OUTPUT_ROOT)
        return

    if mode == 'ablation':
        # 不清空整棵输出树；输出 ablation_hard_51_80_x3（不覆盖旧 ablation_final_x3）
        clear_output_root(OUTPUT_ROOT, clear=False)
        print(f"\n[消融模式] ABLATION_STUDY={ABLATION_STUDY} | pilot={ABLATION_PILOT} → "
              f"{ABLATION_OUTPUT_SUBDIR}/ | tiers={ABLATION_TIER_FILTER} | "
              f"×{ABLATION_NUM_RUNS} runs | "
              f"flat∥={ABLATION_FLAT_PARALLEL}×{ABLATION_FLAT_WORKERS}",
              flush=True)
        print("  跑完后统计: write_ablation_stats_postprocess("
              f"'{OUTPUT_ROOT}/{ABLATION_OUTPUT_SUBDIR}')", flush=True)
        run_ablation_study(
            batch_time_limit=LARGE_BATCH_TIME_LIMIT,
            mip_sub_time_limit=LARGE_MIP_SUB_TIME_LIMIT,
            import_full_from_large_scale=ABLATION_IMPORT_FULL_FROM_LARGE,
            resume=(AUTO_RESUME and not FRESH_START),
            output_subdir=ABLATION_OUTPUT_SUBDIR,
            tier_filter=ABLATION_TIER_FILTER,
            sample_per_tier=ABLATION_SAMPLE_PER_TIER,
            num_runs=ABLATION_NUM_RUNS,
            variants=list(ABLATION_VARIANTS_TO_RUN),
        )
        return

    if mode == 'ablation_and_lambda':
        # 联合新增实验：消融×3 → λ敏感性；均写新目录，不碰旧有效数据
        clear_output_root(OUTPUT_ROOT, clear=False)
        print("\n" + "=" * 72, flush=True)
        print("[联合模式] ablation_and_lambda", flush=True)
        print(f"  ① 消融 → {OUTPUT_ROOT}/{ABLATION_OUTPUT_SUBDIR}/ "
              f"(×{ABLATION_NUM_RUNS}, run∥{ABLATION_PARALLEL_WORKERS})", flush=True)
        print(f"  ② λ敏感性 → {OUTPUT_ROOT}/sensitivity/{SENSITIVITY_LAMBDA_PROTOCOL}/ "
              f"(studies={SENSITIVITY_LAMBDA_STUDIES}, "
              f"×{SENSITIVITY_LAMBDA_NUM_RUNS})", flush=True)
        print("  不覆盖: 有效数据/ablation、sensitivity_compact_9x3", flush=True)
        print("=" * 72, flush=True)

        print(f"\n[阶段①/② 消融] ABLATION_STUDY={ABLATION_STUDY} → "
              f"{ABLATION_OUTPUT_SUBDIR}/", flush=True)
        run_ablation_study(
            batch_time_limit=LARGE_BATCH_TIME_LIMIT,
            mip_sub_time_limit=LARGE_MIP_SUB_TIME_LIMIT,
            import_full_from_large_scale=ABLATION_IMPORT_FULL_FROM_LARGE,
            resume=(AUTO_RESUME and not FRESH_START),
            output_subdir=ABLATION_OUTPUT_SUBDIR,
            tier_filter=ABLATION_TIER_FILTER,
            sample_per_tier=ABLATION_SAMPLE_PER_TIER,
            num_runs=ABLATION_NUM_RUNS,
            variants=list(ABLATION_VARIANTS_TO_RUN),
        )

        print(f"\n[阶段②/② λ敏感性] protocol={SENSITIVITY_LAMBDA_PROTOCOL} | "
              f"studies={SENSITIVITY_LAMBDA_STUDIES}", flush=True)
        run_sensitivity_study(
            batch_time_limit=SENSITIVITY_BASE_TIME_LIMIT,
            mip_sub_time_limit=SENSITIVITY_BASE_MIP_SUB,
            resume=SENSITIVITY_LAMBDA_RESUME,
            num_runs=SENSITIVITY_LAMBDA_NUM_RUNS,
            protocol=SENSITIVITY_LAMBDA_PROTOCOL,
            specs=list(SENSITIVITY_LAMBDA_STUDIES),
        )
        print("\n[联合完成] 新增结果目录:", flush=True)
        print(f"  - {OUTPUT_ROOT}/{ABLATION_OUTPUT_SUBDIR}/", flush=True)
        print(f"  - {OUTPUT_ROOT}/sensitivity/{SENSITIVITY_LAMBDA_PROTOCOL}/",
              flush=True)
        return

    if mode == 'sensitivity_lambda':
        # 仅跑惩罚系数 λ；写入独立协议目录，不碰 compact_9x3 / 消融结果
        clear_output_root(OUTPUT_ROOT, clear=False)
        print(f"\n[λ敏感性] protocol={SENSITIVITY_LAMBDA_PROTOCOL} | "
              f"studies={SENSITIVITY_LAMBDA_STUDIES} | "
              f"runs×{SENSITIVITY_LAMBDA_NUM_RUNS} | "
              f"并行×{SENSITIVITY_PARALLEL_WORKERS} | "
              f"resume={SENSITIVITY_LAMBDA_RESUME} | "
              f"时限={SENSITIVITY_BASE_TIME_LIMIT}s", flush=True)
        print(f"  输出: {OUTPUT_ROOT}/sensitivity/{SENSITIVITY_LAMBDA_PROTOCOL}/",
              flush=True)
        print("  不覆盖: ablation_x3、有效数据、sensitivity_compact_9x3",
              flush=True)
        run_sensitivity_study(
            batch_time_limit=SENSITIVITY_BASE_TIME_LIMIT,
            mip_sub_time_limit=SENSITIVITY_BASE_MIP_SUB,
            resume=SENSITIVITY_LAMBDA_RESUME,
            num_runs=SENSITIVITY_LAMBDA_NUM_RUNS,
            protocol=SENSITIVITY_LAMBDA_PROTOCOL,
            specs=list(SENSITIVITY_LAMBDA_STUDIES),
        )
        return

    if mode == 'sensitivity':
        # 不清空整棵输出树：保留 large_scale / ablation；敏感性写入独立协议目录
        clear_output_root(OUTPUT_ROOT, clear=False)
        print(f"\n[敏感性模式] protocol={SENSITIVITY_PROTOCOL} | "
              f"studies={SENSITIVITY_STUDIES_TO_RUN} | "
              f"runs×{SENSITIVITY_NUM_RUNS} | resume={SENSITIVITY_RESUME} | "
              f"时限={SENSITIVITY_BASE_TIME_LIMIT}s", flush=True)
        print(f"  输出: {OUTPUT_ROOT}/sensitivity/{SENSITIVITY_PROTOCOL}/", flush=True)
        run_sensitivity_study(
            batch_time_limit=SENSITIVITY_BASE_TIME_LIMIT,
            mip_sub_time_limit=SENSITIVITY_BASE_MIP_SUB,
            resume=SENSITIVITY_RESUME,
            num_runs=SENSITIVITY_NUM_RUNS,
            protocol=SENSITIVITY_PROTOCOL,
        )
        return

    if mode in ('large_only', 'resume_large') and not use_prompts:
        # FRESH_START 时两种模式都清空目录，避免残留 CSV 污染全量重跑
        clear_output_root(OUTPUT_ROOT, clear=bool(FRESH_START))
        _, large_results = run_large_scale_stage(
            resume=(AUTO_RESUME and not FRESH_START),
            force_regen=bool(FRESH_START or mode == 'large_only'))
        print(f"\n[完成] 大规模阶段结束，共 {len(large_results)} 条结果")
        print(f"  输出: {OUTPUT_ROOT}/large_scale/large_scale_results.csv")
        # 严格胜率：KG_mean < ALNS_mean（不含平局）
        if large_results:
            n = len(large_results)
            wins = sum(
                1 for r in large_results
                if r.get('mip_lns_obj_mean') is not None
                and r.get('alns_obj_mean') is not None
                and float(r['mip_lns_obj_mean']) < float(r['alns_obj_mean']) - 1e-9
            )
            ties = sum(
                1 for r in large_results
                if r.get('mip_lns_obj_mean') is not None
                and r.get('alns_obj_mean') is not None
                and abs(float(r['mip_lns_obj_mean']) - float(r['alns_obj_mean'])) <= 1e-9
            )
            losses = n - wins - ties
            print(f"\n[严格胜率] KG-ALNS_mean < ALNS_mean: "
                  f"{wins}/{n} = {100.0 * wins / n:.1f}% "
                  f"（平局 {ties}，负 {losses}）", flush=True)
        return

    if mode not in pipeline_modes:
        print(f"[错误] 未知 EXPERIMENT_MODE={mode!r}", flush=True)
        return

    clear_on_start = CLEAR_OUTPUT_ON_START and mode in pipeline_modes
    clear_output_root(OUTPUT_ROOT, clear=clear_on_start)
    if FRESH_START and not clear_on_start and mode in pipeline_modes:
        print(f"\n[FRESH_START] 输出目录已手动清空或未配置 CLEAR_OUTPUT_ON_START，"
              f"各阶段将不读断点 CSV", flush=True)

    study_resume = AUTO_RESUME and not FRESH_START
    small_resume = AUTO_RESUME and not FRESH_START
    large_resume = AUTO_RESUME and not FRESH_START and (mode != 'large_only')
    large_force_regen = FRESH_START or (mode == 'large_only')
    small_force_regen = FRESH_START

    print_pipeline_resume_status(OUTPUT_ROOT)

    small_results = []
    enhanced_results = []

    if RUN_SMALL_SCALE:
        if use_prompts:
            bl_ids, _ = load_completed_batch_results(os.path.join(OUTPUT_ROOT, 'small_scale'))
            en_ids, _ = load_completed_batch_results(
                os.path.join(OUTPUT_ROOT, 'small_scale_enhanced'))
            tag = '（从零重跑）' if FRESH_START else ''
            if small_resume and (bl_ids or en_ids):
                tag = (f'（续跑：基准{len(bl_ids)}/{SMALL_SCALE_NUM_CASES}，'
                       f'增强{len(en_ids)}/{SMALL_SCALE_NUM_CASES}已完成）')
            if not prompt_stage_continue(
                    None, f'阶段一：小规模 Gurobi vs KG-ALNS（15+15 例）{tag}', start=True):
                return
        print("=" * 80)
        print("修船厂调度 — 阶段一：小规模算例（15 例）Gurobi vs 启发式 对比")
        print("双轨配置: 基准(资源紧张) + 增强(适度宽松)")
        print("=" * 80)

        small_results, enhanced_results = run_small_scale_stage(
            force_regen=small_force_regen, resume=small_resume)

        if use_prompts:
            if not prompt_stage_continue(
                    '阶段一：小规模 Gurobi vs KG-ALNS',
                    '阶段二：大规模 KG-ALNS vs GA vs ALNS（60 例）'):
                _finalize_pipeline_partial(small_results, enhanced_results, [])
                return
    else:
        print("\n[skip] stage-1 RUN_SMALL_SCALE=False", flush=True)
        if use_prompts:
            hint_parts = []
            if FRESH_START:
                hint_parts.append('从零重跑')
            elif mode in ('resume_large',):
                hint_parts.append('断点续跑')
            if LARGE_SPLIT_BY_TIER:
                hint_parts.append('分3梯度批')
            hint = f"（{'，'.join(hint_parts)}）" if hint_parts else ''
            if not prompt_stage_continue(
                    None, f'阶段二：大规模三方对比{hint}', start=True):
                return

    # ============================================================
    # 阶段二：大规模算例（60 例, 30-60 艘船, 3梯度 × 20组）启发式求解
    # ============================================================
    print("\n\n" + "=" * 80)
    print("修船厂调度 — 阶段二：大规模算例（60 例, 3梯度×20组, 可分批确认）")
    print("=" * 80)

    large_exp, large_results = run_large_scale_stage(
        resume=large_resume,
        force_regen=large_force_regen)
    gc.collect()

    if getattr(large_exp, 'large_stage_stopped_early', False):
        _finalize_pipeline_partial(small_results, enhanced_results, large_results)
        return

    # ============================================================
    # 阶段三：KG-ALNS 消融实验（在大规模算例之后，复用同一批算例）
    # ============================================================
    ablation_results, ablation_summary = None, None
    ablation_done = False
    if RUN_ABLATION_AFTER_LARGE:
        if use_prompts:
            if not prompt_stage_continue(
                    '阶段二：大规模三方对比',
                    '阶段三：KG-ALNS 消融（完整/无MIP/无领域）'):
                _finalize_pipeline_partial(
                    small_results, enhanced_results, large_results)
                return
        print("\n\n" + "=" * 80)
        print("修船厂调度 — 阶段三：KG-ALNS 消融实验")
        print("复用阶段二算例；full 变体优先从 large_scale_results.csv 导入")
        print("=" * 80)
        try:
            ablation_results, ablation_summary = run_ablation_study(
                large_exp=large_exp,
                batch_time_limit=LARGE_BATCH_TIME_LIMIT,
                mip_sub_time_limit=LARGE_MIP_SUB_TIME_LIMIT,
                import_full_from_large_scale=ABLATION_IMPORT_FULL_FROM_LARGE,
                resume=study_resume,
                output_subdir=ABLATION_OUTPUT_SUBDIR,
                tier_filter=ABLATION_TIER_FILTER,
                sample_per_tier=ABLATION_SAMPLE_PER_TIER,
                num_runs=ABLATION_NUM_RUNS,
                variants=list(ABLATION_VARIANTS_TO_RUN),
            )
        except Exception as e:
            print(f"\n[消融] 运行异常（已保存阶段性结果）: {e}", flush=True)
            import traceback
            traceback.print_exc()
        ablation_done = ablation_results is not None

    # ============================================================
    # 阶段四：敏感性分析（六组参数 × 15 抽样算例）
    # ============================================================
    sensitivity_results, sensitivity_summary = None, None
    sensitivity_done = False
    if RUN_SENSITIVITY_AFTER_ABLATION:
        if use_prompts:
            if not prompt_stage_continue(
                    '阶段三：消融实验' if ablation_done else '阶段二：大规模三方对比',
                    '阶段四：敏感性分析（6 组 × 15 抽样算例）'):
                _finalize_pipeline_partial(
                    small_results, enhanced_results, large_results,
                    ablation_done=ablation_done)
                return
        print("\n\n" + "=" * 80)
        print("修船厂调度 — 阶段四：敏感性分析（六组 × 15 抽样算例）")
        print("S1资源紧张度 | S2λ | S3到港 | S4时限 | S5MIP子时限 | S6时间预算")
        print("=" * 80)
        try:
            sensitivity_results, sensitivity_summary = run_sensitivity_study(
                large_exp=large_exp,
                batch_time_limit=SENSITIVITY_BASE_TIME_LIMIT,
                mip_sub_time_limit=SENSITIVITY_BASE_MIP_SUB,
                resume=study_resume,
            )
        except Exception as e:
            print(f"\n[敏感性] 运行异常（已保存阶段性结果）: {e}", flush=True)
            import traceback
            traceback.print_exc()
        sensitivity_done = sensitivity_results is not None

    # ============================================================
    # 综合对比汇总（基准 vs 增强）
    # ============================================================
    print("\n\n" + "=" * 80)
    print("两阶段实验 — 综合对比汇总（基准 vs 增强）")
    print("=" * 80)

    # ---- 基准小规模 ----
    feasible_small = [r for r in small_results if r['feasible']]
    if feasible_small:
        optimal_count = sum(1 for r in feasible_small if r['status'] == 'OPTIMAL')
        win_count = sum(1 for r in small_results if r.get('heu_superiority') == 'WIN')
        tie_count = sum(1 for r in small_results if r.get('heu_superiority') == 'TIE')
        gaps = [r['gap_to_optimal'] for r in small_results if r['gap_to_optimal'] is not None]
        effs = [r['efficiency'] for r in small_results if r['efficiency'] is not None]
        avg_heu_time_small = np.mean([r['heuristic_time'] for r in small_results if r['heuristic_time'] is not None])
        avg_gur_time_small = np.mean([r['solve_time'] for r in feasible_small])
        avg_speedup_small = avg_gur_time_small / avg_heu_time_small if avg_heu_time_small > 0 else 0

        print(f"\n  【阶段一-基准：小规模 (1-10艘, 15例) — 适度紧张】")
        print(f"  {'─' * 60}")
        print(f"  可行率:            {len(feasible_small)}/{len(small_results)} ({len(feasible_small)/len(small_results)*100:.1f}%)")
        print(f"  达到最优 (Gurobi): {optimal_count}/{len(feasible_small)}")
        print(f"  Gurobi 平均耗时:   {avg_gur_time_small:.2f}s")
        print(f"  Heuristic 平均耗时:{avg_heu_time_small:.6f}s")
        print(f"  平均加速比:        {avg_speedup_small:.0f}x")
        print(f"  平均 Gap:          {np.mean(gaps):.2f}% (max={max(gaps):.2f}%, min={min(gaps):.2f}%)" if gaps else "")
        print(f"  平均效率值:        {np.mean(effs):.2f}%" if effs else "")
        print(f"  Heuristic WIN:     {win_count}/{len(small_results)} 例")
        print(f"  Heuristic TIE:     {tie_count}/{len(small_results)} 例")

    # ---- 增强小规模 ----
    feasible_enhanced = [r for r in enhanced_results if r['feasible']]
    if feasible_enhanced:
        e_optimal = sum(1 for r in feasible_enhanced if r['status'] == 'OPTIMAL')
        e_win = sum(1 for r in enhanced_results if r.get('heu_superiority') == 'WIN')
        e_tie = sum(1 for r in enhanced_results if r.get('heu_superiority') == 'TIE')
        e_gaps = [r['gap_to_optimal'] for r in enhanced_results if r['gap_to_optimal'] is not None]
        e_effs = [r['efficiency'] for r in enhanced_results if r['efficiency'] is not None]
        avg_heu_enh = np.mean([r['heuristic_time'] for r in enhanced_results if r['heuristic_time'] is not None])
        avg_gur_enh = np.mean([r['solve_time'] for r in feasible_enhanced])

        print(f"\n  【阶段一-增强：小规模 (1-10艘, 15例) — 适度宽松】")
        print(f"  {'─' * 60}")
        print(f"  可行率:            {len(feasible_enhanced)}/{len(enhanced_results)} ({len(feasible_enhanced)/len(enhanced_results)*100:.1f}%)")
        print(f"  达到最优 (Gurobi): {e_optimal}/{len(feasible_enhanced)}")
        print(f"  Gurobi 平均耗时:   {avg_gur_enh:.2f}s")
        print(f"  Heuristic 平均耗时:{avg_heu_enh:.6f}s")
        print(f"  平均 Gap:          {np.mean(e_gaps):.2f}% (max={max(e_gaps):.2f}%, min={min(e_gaps):.2f}%)" if e_gaps else "")
        print(f"  平均效率值:        {np.mean(e_effs):.2f}%" if e_effs else "")

        # 对比
        if gaps and e_gaps:
            gap_delta = np.mean(gaps) - np.mean(e_gaps)
            print(f"\n  【基准 vs 增强 对比】")
            print(f"  {'─' * 60}")
            print(f"  基准平均 Gap:  {np.mean(gaps):.2f}%")
            print(f"  增强平均 Gap:  {np.mean(e_gaps):.2f}%")
            print(f"  Gap 改善:      {gap_delta:.2f} 百分点")
            if gap_delta > 0:
                print(f"  结论: 增强配置使 Gap 降低 {gap_delta:.1f} 个百分点，资源适度增加有效缩小启发式与最优解的差距")

    large_df = pd.DataFrame(large_results)
    print(f"\n  【阶段二：大规模 (30-60艘, 60例) — KG-ALNS vs GA vs ALNS】")
    print(f"  {'─' * 60}")
    print(f"  算例总数:          {len(large_results)}")
    print(f"  平均船舶数:        {large_df['num_ships'].mean():.1f}")
    print(f"  平均任务数:        {large_df['num_tasks'].mean():.1f}")
    mip_avg = (large_df['mip_lns_obj_mean'].mean() if 'mip_lns_obj_mean' in large_df.columns
               else large_df['heuristic_obj'].mean())
    ga_avg = large_df['ga_obj_mean'].mean() if 'ga_obj_mean' in large_df.columns else None
    alns_avg = large_df['alns_obj_mean'].mean() if 'alns_obj_mean' in large_df.columns else None
    mip_best_avg = large_df['mip_lns_obj_best'].mean() if 'mip_lns_obj_best' in large_df.columns else None
    print(f"  主口径 run-mean (KG×{LARGE_MIP_NUM_RUNS}, GA×{LARGE_GA_NUM_RUNS}, ALNS×{LARGE_ALNS_NUM_RUNS})")
    if ga_avg is not None and not pd.isna(ga_avg):
        print(f"  平均目标 mean — KG: {mip_avg:.1f} | GA: {ga_avg:.1f} | "
              f"ALNS: {alns_avg:.1f}" if alns_avg is not None and not pd.isna(alns_avg)
              else f"  平均目标 mean — KG: {mip_avg:.1f} | GA: {ga_avg:.1f}")
    else:
        print(f"  平均目标 mean — KG: {mip_avg:.1f}")
    if mip_best_avg is not None and not pd.isna(mip_best_avg):
        print(f"  辅口径 best-of-runs — KG均值: {mip_best_avg:.1f}")
    print(f"  平均总墙钟(KG-ALNS): {large_df['mip_lns_time'].mean():.2f}s "
          f"(上限 {LARGE_BATCH_TIME_LIMIT}s, 含后处理)")
    if 'mip_lns_search_time' in large_df.columns:
        print(f"    └ 主搜索: {large_df['mip_lns_search_time'].mean():.2f}s | "
              f"后处理: {large_df['mip_lns_postprocess_time'].mean():.2f}s")
    print(f"  平均总墙钟(GA):      {large_df['ga_time'].mean():.2f}s")
    print(f"  平均总墙钟(ALNS):    {large_df['alns_time'].mean():.2f}s")
    wins_ga = int(large_df['mip_lns_better_ga'].sum()) if 'mip_lns_better_ga' in large_df.columns else 0
    wins_alns = int(large_df['mip_lns_better_alns'].sum()) if 'mip_lns_better_alns' in large_df.columns else 0
    wins_ga_best = int(large_df['mip_lns_better_ga_best'].sum()) if 'mip_lns_better_ga_best' in large_df.columns else 0
    wins_alns_best = int(large_df['mip_lns_better_alns_best'].sum()) if 'mip_lns_better_alns_best' in large_df.columns else 0
    obj_wins = int((large_df['obj_winner'] == 'KG-ALNS').sum()) if 'obj_winner' in large_df.columns else 0
    obj_wins_best = int((large_df['obj_winner_best'] == 'KG-ALNS').sum()) if 'obj_winner_best' in large_df.columns else 0
    print(f"  KG-ALNS mean胜 GA:       {wins_ga}/{len(large_results)} 例")
    print(f"  KG-ALNS mean胜 ALNS:     {wins_alns}/{len(large_results)} 例")
    print(f"  KG-ALNS 三方 mean最优:   {obj_wins}/{len(large_results)} 例")
    if 'mip_lns_better_ga_best' in large_df.columns:
        print(f"  (辅) best胜 GA/ALNS/三方: {wins_ga_best}/{wins_alns_best}/{obj_wins_best} 例")

    # 梯度统计
    print(f"\n  【梯度分类统计 — 三方 run-mean 平均目标】")
    print(f"  {'梯度':>12s} | {'数量':>5s} | {'KG-ALNS':>10s} | {'GA':>10s} | {'ALNS':>10s} | "
          f"{'KG(s)':>8s} | {'GA(s)':>8s} | {'ALN(s)':>8s}")
    print(f"  {'─' * 90}")
    for tier in LARGE_SCALE_TIER_LABELS:
        tdf = large_df[large_df['tier'] == tier]
        if len(tdf) > 0:
            mip_t = (tdf['mip_lns_obj_mean'].mean() if 'mip_lns_obj_mean' in tdf.columns
                     else tdf['heuristic_obj'].mean())
            ga_t = tdf['ga_obj_mean'].mean() if 'ga_obj_mean' in tdf.columns else None
            alns_t = tdf['alns_obj_mean'].mean() if 'alns_obj_mean' in tdf.columns else None
            ga_s = f"{ga_t:.1f}" if ga_t is not None and not pd.isna(ga_t) else "N/A"
            alns_s = f"{alns_t:.1f}" if alns_t is not None and not pd.isna(alns_t) else "N/A"
            print(f"  {tier:>12s} | {len(tdf):>5d} | {mip_t:>10.1f} | {ga_s:>10s} | {alns_s:>10s} | "
                  f"{tdf['mip_lns_time'].mean():>8.2f} | {tdf['ga_time'].mean():>8.2f} | "
                  f"{tdf['alns_time'].mean():>8.2f}")

    # 跨阶段对比
    if feasible_small:
        avg_heu_time_large = large_df['heuristic_time'].mean()
        avg_n_small = np.mean([r['num_ships'] for r in small_results])
        avg_n_large = large_df['num_ships'].mean()
        n_ratio = avg_n_large / avg_n_small
        t_ratio = large_df['heuristic_time'].mean() / avg_heu_time_small if avg_heu_time_small > 0 else 0

        print(f"\n  【跨阶段对比 — 启发式算法可扩展性】")
        print(f"  {'─' * 60}")
        print(f"  小规模平均 N={avg_n_small:.1f} → 启发式耗时 {avg_heu_time_small:.6f}s")
        print(f"  大规模平均 N={avg_n_large:.1f} → 启发式耗时 {avg_heu_time_large:.4f}s")
        print(f"  船舶数增长: {n_ratio:.1f}x | 耗时增长: {t_ratio:.1f}x")

        # 保存综合报告（含双轨对比数据）
        cross_report = {
            '实验时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            '阶段一_基准': {
                '算例数': len(small_results),
                '船舶范围': '1-10',
                '配置类型': 'baseline (适度紧张)',
                'Gurobi平均耗时': f"{avg_gur_time_small:.2f}s",
                'Heuristic平均耗时': f"{avg_heu_time_small:.6f}s",
                '平均加速比': f"{avg_speedup_small:.0f}x",
                '平均Gap': f"{np.mean(gaps):.2f}%" if gaps else "N/A",
                'Heuristic_WIN': win_count,
                'Heuristic_TIE': tie_count,
            },
            '阶段一_增强': {
                '算例数': len(enhanced_results),
                '船舶范围': '1-10',
                '配置类型': 'enhanced (适度宽松)',
                'Gurobi平均耗时': f"{avg_gur_enh:.2f}s",
                'Heuristic平均耗时': f"{avg_heu_enh:.6f}s",
                '平均Gap': f"{np.mean(e_gaps):.2f}%" if e_gaps else "N/A",
            } if feasible_enhanced else None,
            '阶段二': {
                '算例数': len(large_results),
                '船舶范围': f'{LARGE_SCALE_SHIP_RANGE[0]}-{LARGE_SCALE_SHIP_RANGE[1]}',
                '配置类型': 'baseline',
                '梯度': {label: 20 for label in LARGE_SCALE_TIER_LABELS},
                '规模依据': '对标洋山港四期年约2500艘≈周50艘',
                'Heuristic平均耗时': f"{large_df['heuristic_time'].mean():.4f}s",
                '总耗时': f"{large_df['heuristic_time'].sum():.4f}s",
            },
        }
        report_path = os.path.join(OUTPUT_ROOT, "cross_phase_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(cross_report, f, indent=2, ensure_ascii=False)

    unified_rows = build_unified_experiment_rows(small_results, enhanced_results, large_results)
    save_unified_excel_report(unified_rows, os.path.join(OUTPUT_ROOT, "实验结果汇总.xlsx"))

    # ============================================================
    # 文件清单
    # ============================================================
    print("\n\n" + "=" * 80)
    print("实验全部完成！")
    print("=" * 80)
    print(f"\n结果根目录: {OUTPUT_ROOT}/")
    print(f"\n  【统一 Excel 汇总】")
    print(f"    - 实验结果汇总.xlsx            (小规模基准+增强 + 大规模，共 {len(unified_rows)} 行)")
    print(f"\n  【阶段一 — 小规模 (small_scale/ & small_scale_enhanced/)】")
    print(f"    - summary_table.csv / comparison_15_cases.csv / 算例与调度明细 CSV")
    print(f"\n  【阶段二 — 大规模 (large_scale/)】")
    print(f"    - large_scale_results.csv / large_scale_comparison.csv")
    if RUN_ABLATION_AFTER_LARGE:
        print(f"\n  【阶段三 — 消融 (ablation/)】")
        print(f"    - ablation_results.csv / ablation_summary.csv")
        print(f"    - ablation_summary_by_tier.csv / ablation_report.xlsx")
    if RUN_SENSITIVITY_AFTER_ABLATION:
        print(f"\n  【阶段四 — 敏感性 (sensitivity/)】")
        print(f"    - sensitivity_results.csv / sensitivity_summary.csv")
        print(f"    - sensitivity_summary_by_tier.csv / sensitivity_report.xlsx")
    print(f"\n  【综合】")
    print(f"    - cross_phase_report.json        (跨阶段对比报告)")
    print("=" * 80)

if __name__ == "__main__":
    main()
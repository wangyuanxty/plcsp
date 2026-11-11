"""
1、一个node如果有很多个入边，说明有很多个先后约束
2、这么多先后约束，我们只能选结束的最晚的时间（=上一步的ft+对应的运输t），也就是当前节点最大的开始时间（因为有约束存在啊！所以必须选最大）
"""
import numpy as np

try:  # scripts add ``plcsp_v5/`` to ``sys.path``
    from agv.state import idle_cost, resolve_idle_mode
except ImportError:  # repository root only
    from plcsp_v5.agv.state import idle_cost, resolve_idle_mode


"""def find_max_st_in_inedge(G, taskid, edge_init_weight):
    inedge_list = list(G.in_edges(taskid)) # 找到所有入边
    max_list = []
    for i in range(len(inedge_list)):  # 遍历所有入边！
        prev_nodeid, _ = list(G.in_edges(taskid))[i] #第一个入边的节点id
        edge_weight = edge_init_weight[G.nodes[prev_nodeid]['machine'], G.nodes[taskid]['machine']]  #计算入边节点到当前节点的运输时间t

        # 修改BUG，防止src节点的m_id=-1查表导致错误，应该是边权重为0，但是查表得到了，担心（-1，x）的情形吧
        # 当前节点task_id不会出现虚拟的sink节点的，总共就几个子任务就运行几遍，所以不用担心（x，-1）的情形
        if G.nodes[prev_nodeid]['machine'] < 0:
            edge_weight = 0  # 此时没有运输时间！！！

        max_val = G.nodes[prev_nodeid]['finish_time'] + edge_weight  #入边节点的完工时间+运输时间 = 当前taskid节点的开始时间
        max_list.append(max_val)
    max_st_currentTask = max(max_list)  #必定要选最大值，因为是有先后约束的，不能选小的先开始的，约束不满足！

    return max_st_currentTask"""

"""
1、本质：寻找当前节点的所有的入边（都是先后约束），最晚结束的入边 = 当前节点的最早开始时间 = 当前节点的最晚的到达时间
2、node图 + 当前节点 + 传输t能力矩阵 + job个数 + machine个数 + 传输时间次数和累加的统计 + 用来记录已经查到过的平行边的传输时间t
3、需要判断：1、前置节点不是src源节点，（-1,1）会倒着查表；2、同一个job的子任务之间才存在运输时间
4、machine_num 严格等于 每一个job的子任务数量（正方形矩阵）
5、energy_transport 为了统计有几个传输时间，累加的传输时间是多少

BUG：
# 修改BUG，防止src节点的m_id=-1查表导致错误，应该是边权重为0，但是查表得到了，担心（-1，x）的情形吧
# 当前节点task_id不会出现虚拟的sink节点的，总共就几个子任务就运行几遍，所以不用担心（x，-1）的情形

6、避免被反复查运输t（平行边）的最简单方法：画完图，每个平行边和其出发节点的weight和duration之间的差值之和就行（或者画平行边的时候，再记录，次数 = jobN*（machineN-1））！！！
7、一个是有结果之后的查询，一个是在调度过程中的统计（前者验证后者，保证调度没问题）
"""

"""
根据当前节点（`taskid`）的入边和其他条件找到最晚到达时间
1、`G.in_edges('D')`的返回值将是`[('B', 'D'), ('C', 'D')]`。其中，每个元组表示一条边，第一个元素是起始节点，第二个元素是终点节点。
2、肯定选最大值作为当前节点的st，因为入边都是先后顺序的约束
"""
def find_max_arrivaTime_for_currentNode(G, taskid, edge_init_weight, configs):
    # [v5] In AGV mode, a scheduled node already stores the exact start time.
    # Recomputing it from in-edges would lose the AGV wait that was paid when
    # this node was scheduled. This branch is inactive when no AGV override
    # is set, so the no-AGV baseline keeps the upstream behaviour.
    wait_overrides = getattr(G, "graph", {}).get("transport_wait_override", None)
    if wait_overrides is not None and G.nodes[taskid].get("scheduled"):
        return float(G.nodes[taskid]["start_time"])

    inedge_list = list(G.in_edges(taskid))  # 找到所有入边, 返回list中存有n个元组（起始节点，taskID节点）
    max_list = []
    overrides = getattr(G, "graph", {}).get("transport_override", None)
    for i in range(len(inedge_list)):  # 遍历所有入边！
        prev_nodeid, _ = inedge_list[i]  # 入边的节点id
        transport_t = edge_init_weight[G.nodes[prev_nodeid]['machine'], G.nodes[taskid]['machine']]  # 计算入边节点到当前节点的运输时间t

        # [v5] Use the AGV loaded travel time when this step has an override.
        # The table value also contains the fixed cross-workshop delay; the
        # wait override below adds that delay once. Using both would charge
        # the delay twice.
        if isinstance(overrides, dict):
            override = overrides.get((int(prev_nodeid), int(taskid)))
            if override is not None:
                transport_t = float(override)

        if G.nodes[prev_nodeid]['machine'] < 0:  # 排除src的节点，m_id=-1
            transport_t = 0  # 此时没有运输时间！！！

        same, same_job_id = find_2nodeID_in_same_job(prev_nodeid, taskid, configs.n_job, configs.n_machine)  # 判断输入的两个节点是否位于同一个job,和是哪一个job

        if not same:  # 不是同一个job中
            transport_t = 0

        # [v5] AGV wait delay: start time only, never TEC / edge weight.
        if same and G.nodes[prev_nodeid]['machine'] >= 0 and isinstance(wait_overrides, dict):
            wait = wait_overrides.get((int(prev_nodeid), int(taskid)))
            if wait is not None:
                transport_t = float(transport_t) + float(wait)

        max_val = G.nodes[prev_nodeid]['finish_time'] + transport_t  # 入边节点的完工时间+运输时间 = 当前taskid节点的开始时间
        max_list.append(max_val)
        # print(exist_list)
    st_currentTask = max(max_list)  # 必定要选最大值，因为是有先后约束的，不能选小的先开始的，约束不满足！

    # [v5 M2] Earliest-start override for a task that has no AGV transport
    # edge, i.e. the first operation of a job entering at a load/unload
    # station.  Inactive unless the caller sets the override.
    ready_overrides = getattr(G, "graph", {}).get("ready_override", None)
    if isinstance(ready_overrides, dict):
        value = ready_overrides.get(int(taskid))
        if value is not None:
            st_currentTask = max(float(st_currentTask), float(value))

    return st_currentTask


"""
# 查找两个节点是否位于同一个job中，笨方法：
1、按照行j查找，查j次，两个节点分别记录在哪一行
2、同行，返回true和job的行数，否则不同行
"""
def find_2nodeID_in_same_job(prev_nodeid, taskid, nj, nm) -> (bool, int):
    n_job = nj
    n_machine = nm
    total_task = n_job * n_machine
    task_list = np.arange(1, (total_task + 1)).reshape(n_job, -1)  # -1不指定维度  j*m的taskid的矩阵
    # print("task_list", task_list)

    job_current = 0
    job_prev = 0  # 虽然初始化了，for之后就会重新赋值了

    same_job_id = 0

    for i in range(task_list.shape[0]):
        if taskid in task_list[i]:
            job_current = i
        if prev_nodeid in task_list[i]:
            job_prev = i

    if job_current == job_prev:
        same_job = True  # 位于同一个job，有运输t
        same_job_id = job_current  # 是第几个job的平行边
    else:
        same_job = False

    return same_job, same_job_id


"""
# 查表node1到node2之间的运输时间t，并判断是否有效（同一个job）
# energy_transport用来记录运输了几次 + exist_list用来记录已经查到过的平行边的传输时间t

1、先通过task的id找到都是选定的哪些m，然后查表m之间是否有transT
2、还要判断这些taskid是否是同一个m的，不然也没有transT
"""
def find_transportT(G, prev_node1, curr_node2, edge_init_weight, configs):
    # [v5] AGV travel-time override for one specific job edge (set by
    # Parallel_env for one step). TEC and edge weights use the travel time
    # only. The AGV wait is a scheduling delay, not transport time; it is
    # handled in find_max_arrivaTime_for_currentNode.
    overrides = getattr(G, "graph", {}).get("transport_override", None)
    if isinstance(overrides, dict):
        key = (int(prev_node1), int(curr_node2))
        if key in overrides:
            return float(overrides[key])
    # print(G.nodes[node1]['machine'])
    # print(G.nodes[node2]['machine'])  # 为什么node的machine的属性会变成浮点数？？？？？？？
    """"""
    """
    因为现在初始化存在m_id=-1小于0的task节点，所以这些节点没必要判断运输时间（__schedule之前都会先更新当前task节点的属性信息：m_id + color + dur！！！！）
    """
    if G.nodes[curr_node2]['machine'] >= 0:  # 防止初始化load_instance的时候无效判断transT
        transport_t = edge_init_weight[G.nodes[prev_node1]['machine'], G.nodes[curr_node2]['machine']]  # 查表分配的m——id得要新增的运输时间
    else:
        transport_t = 0

    same, same_job_id = find_2nodeID_in_same_job(prev_node1, curr_node2, configs.n_job, configs.n_machine)  # 判断输入的两个节点是否位于同一个job，和是哪一个job

    if G.nodes[prev_node1]['machine'] < 0:  # 排除src的节点，m_id=-1
        transport_t = 0  # 此时没有运输时间！！！

    if not same:
        transport_t = 0

    return transport_t


"""
1、需要反馈参数（y，width，left） 对应 node[m_id+1, ft-st, st]：直接从machien_route里边获取
2、可以基于当前的machien_route来输出实时的idle指标 + 最终的指标

machine_route里边的task是有顺序的，代表当前m被选取后的task的先后执行顺序
    machien_routes = {0: array([], dtype=int64), 1: array([3, 4, 1, 2])}  字典形式：存有同一个m中的task的加工顺序！

总体理解：
1、每个节点的开始-上一个节点的结束
2、首节点就是开始-0
3、不断累加其中的空白！
4、其中必然会包含transT的时间，有的时候是别的task的dur，但是说实际的，有这时间必须要等待的（无非就是最后不算it？）
"""
def calculate_idle_t_for_each_machine(G, machineRoute, p2_ins=None,
                                      idle_mode=None):
    """Machine-idle term from the current machine routes.

    ``idle_mode="gap"`` is the old reading: leading idle plus the gaps
    between consecutive operations.  ``idle_mode="trailing"`` (default)
    is ``sum_k power_k * (MKT - P_k)``, which also charges every machine
    from its last operation to the makespan.  The value is recomputed
    from the routes on every call, so no accumulator is involved.

    ``p2_ins`` is the optional idle-power row ``(1, n_machine)``; v5 uses
    all ones.
    """
    mode = resolve_idle_mode(idle_mode)

    def power(m_index):
        if p2_ins is None:
            return 1.0
        return float(p2_ins[0][m_index])

    sum_idle = 0.0
    proc = [0.0] * len(machineRoute)
    makespan = 0.0
    for m_index, route in machineRoute.items():
        m_index = int(m_index)
        p = power(m_index)
        route_proc = 0.0
        for task in route:
            route_proc += float(G.nodes[task]['duration'])
            ft = float(G.nodes[task]['finish_time'])
            if ft > makespan:
                makespan = ft
        proc[m_index] = route_proc

        if len(route) == 1:
            blank = G.nodes[route[0]]['start_time'] - 0
            sum_idle = sum_idle + blank * p
        elif len(route) > 1:
            blank = G.nodes[route[0]]['start_time'] - 0
            sum_idle = sum_idle + blank * p
            for i in range(len(route) - 1):
                blank = (G.nodes[route[i + 1]]['start_time']
                         - G.nodes[route[i]]['finish_time'])
                sum_idle = sum_idle + blank * p

    idle_power = [power(k) for k in range(len(machineRoute))]
    return idle_cost(mode, sum_idle, proc, makespan, idle_power)


def machine_context_features(G, machine_routes, task_idx, dur_row, tt, n_machine):
    """ST-only per-machine context for the current task.

    Returns (n_machine, 3): queue length / n_machine, earliest feasible
    start (gap-aware, divided by makespan), and utilization.

    The current task is 0-based.  Its job predecessor is the node whose id
    equals ``task_idx`` (node ids are 1-based).  The machine route is the
    ordered task list already assigned to the machine.
    """
    task_idx = int(task_idx)
    if task_idx % n_machine == 0:
        prev_m, prev_ft = -1, 0.0
    else:
        prev = G.nodes[task_idx]
        prev_m = int(prev.get('machine', -1))
        prev_ft = (float(prev.get('finish_time') or 0.0)
                   if prev.get('scheduled') else 0.0)
        if prev_m < 0:
            prev_ft = 0.0

    makespan = 0.0
    for _, data in G.nodes(data=True):
        ft = data.get('finish_time')
        if ft is not None and float(ft) > makespan:
            makespan = float(ft)

    out = np.zeros((n_machine, 3), dtype=np.float32)
    for m in range(n_machine):
        route = list(machine_routes.get(m, []))
        queue_len = len(route)
        busy = sum(float(G.nodes[t].get('duration', 0) or 0) for t in route)
        util = busy / makespan if makespan > 1e-9 else 0.0

        dur = float(dur_row[m])
        if dur <= 0:
            earliest = 0.0
        else:
            ready = 0.0
            if prev_m >= 0 and prev_ft > 0:
                ready = prev_ft + float(tt[prev_m][m])
            if not route:
                earliest = ready
            else:
                first_st = float(G.nodes[route[0]].get('start_time') or 0.0)
                if ready + dur <= first_st + 1e-9:
                    earliest = ready
                else:
                    earliest = None
                    for a, b in zip(route, route[1:]):
                        a_ft = float(G.nodes[a].get('finish_time') or 0.0)
                        b_st = float(G.nodes[b].get('start_time') or 0.0)
                        start = max(ready, a_ft)
                        if start + dur <= b_st + 1e-9:
                            earliest = start
                            break
                    if earliest is None:
                        last_ft = float(
                            G.nodes[route[-1]].get('finish_time') or 0.0)
                        earliest = max(ready, last_ft)
            if makespan > 1e-9:
                earliest = earliest / makespan

        out[m, 0] = queue_len / max(1, n_machine)
        out[m, 1] = earliest
        out[m, 2] = util
    return out


def split_edge_matrices(G, adj_wrk):
    """Split ``adj_wrk - 1`` into job-edge transport and machine-edge idle.

    ``adj_wrk[i, j]`` is the weight of edge j -> i (0 = no edge,
    1 = edge with zero gap).  Returns two (N, N) float arrays.
    """
    a = np.asarray(adj_wrk)
    trans = np.zeros(a.shape, dtype=np.float32)
    idle = np.zeros(a.shape, dtype=np.float32)
    n = a.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j or a[i, j] <= 0:
                continue
            edge = G.get_edge_data(j + 1, i + 1)
            if edge is None:
                continue
            gap = max(0.0, float(a[i, j]) - 1.0)
            if edge.get('job_edge', True):
                trans[i, j] = gap
            else:
                idle[i, j] = gap
    return trans, idle


def task_context_features(G, machine_routes, t_ins, p_ins, tt, n_machine):
    """ST-only per-task context (N, 13).

    Adds operation index, eligible-machine count, min/mean/max t and p,
    transport-aware earliest start, global progress, and job planned finish.
    """
    n_task = t_ins.shape[0]
    makespan = 0.0
    for _, data in G.nodes(data=True):
        ft = data.get('finish_time')
        if ft is not None:
            makespan = max(makespan, float(ft))

    scheduled = np.zeros(n_task, dtype=bool)
    finish = np.zeros(n_task, dtype=np.float32)
    for t in range(n_task):
        data = G.nodes[t + 1]
        if data.get('scheduled'):
            scheduled[t] = True
            finish[t] = float(data.get('finish_time') or 0.0)

    busy = np.zeros(n_machine, dtype=np.float32)
    for m, route in machine_routes.items():
        busy[m] = sum(float(G.nodes[t].get('duration', 0) or 0)
                      for t in route)
    avg_util = float(np.mean(busy / max(makespan, 1.0)))
    sched_ratio = float(scheduled.mean())
    makespan_norm = makespan / (1.0 + makespan)

    scale_t = max(1.0, float(np.max(t_ins[t_ins > 0])) if np.any(t_ins > 0) else 1.0)
    scale_p = max(1.0, float(np.max(p_ins[p_ins > 0])) if np.any(p_ins > 0) else 1.0)

    out = np.zeros((n_task, 13), dtype=np.float32)
    earliest_arr = np.zeros(n_task, dtype=np.float32)
    planned_arr = np.zeros(n_task, dtype=np.float32)
    for t in range(n_task):
        eligible = t_ins[t] > 0
        count = int(eligible.sum())
        tvals = t_ins[t][eligible]
        pvals = p_ins[t][eligible]
        op = t % n_machine

        earliest = 0.0
        if op != 0:
            prev = G.nodes[t]
            prev_m = int(prev.get('machine', -1))
            if prev.get('scheduled') and prev_m >= 0 and count:
                prev_ft = float(prev.get('finish_time') or 0.0)
                earliest = prev_ft + min(
                    float(tt[prev_m][m]) for m in range(n_machine)
                    if eligible[m])

        job = t // n_machine
        job_tasks = range(job * n_machine, (job + 1) * n_machine)
        job_ft = max((float(finish[q]) for q in job_tasks if scheduled[q]),
                     default=0.0)
        rem_min = sum(
            float(np.min(t_ins[q][t_ins[q] > 0]))
            for q in job_tasks
            if not scheduled[q] and np.any(t_ins[q] > 0))
        planned = job_ft + rem_min
        earliest_arr[t] = earliest
        planned_arr[t] = planned

        out[t] = [
            op / max(1, n_machine - 1),
            count / max(1, n_machine),
            float(np.min(tvals)) / scale_t if count else 0.0,
            float(np.mean(tvals)) / scale_t if count else 0.0,
            float(np.max(tvals)) / scale_t if count else 0.0,
            float(np.min(pvals)) / scale_p if count else 0.0,
            float(np.mean(pvals)) / scale_p if count else 0.0,
            float(np.max(pvals)) / scale_p if count else 0.0,
            0.0,
            makespan_norm,
            sched_ratio,
            avg_util,
            0.0,
        ]
    planned_scale = max(1.0, float(planned_arr.max()))
    out[:, 8] = earliest_arr / planned_scale
    out[:, 12] = planned_arr / planned_scale
    return out


def task_context_features_fast(G, machine_routes, t_ins, p_ins, tt, n_machine):
    """Vectorized twin of ``task_context_features``; same (N, 13) output."""
    n_task = t_ins.shape[0]
    m = int(n_machine)

    sched = np.zeros(n_task, dtype=bool)
    fin = np.zeros(n_task, dtype=np.float32)
    dur = np.zeros(n_task, dtype=np.float32)
    mach = np.full(n_task, -1, dtype=np.int64)
    for t in range(n_task):
        data = G.nodes[t + 1]
        if data.get('scheduled'):
            sched[t] = True
            fin[t] = float(data.get('finish_time') or 0.0)
        dur[t] = float(data.get('duration', 0) or 0)
        mach[t] = int(data.get('machine', -1))

    makespan = 0.0
    for _, data in G.nodes(data=True):
        ft = data.get('finish_time')
        if ft is not None:
            makespan = max(makespan, float(ft))

    dur_node = np.zeros(n_task + 2, dtype=np.float32)
    dur_node[1:n_task + 1] = dur
    busy = np.zeros(m, dtype=np.float32)
    for mm, route in machine_routes.items():
        if len(route):
            busy[mm] = float(dur_node[np.asarray(route, dtype=np.int64)].sum())
    avg_util = float(np.mean(busy / max(makespan, 1.0)))
    sched_ratio = float(sched.mean())
    makespan_norm = makespan / (1.0 + makespan)

    pos_t = t_ins[t_ins > 0]
    scale_t = max(1.0, float(pos_t.max())) if pos_t.size else 1.0
    pos_p = p_ins[p_ins > 0]
    scale_p = max(1.0, float(pos_p.max())) if pos_p.size else 1.0

    idx = np.arange(n_task)
    op = idx % m
    job = idx // m
    elig = t_ins > 0
    cnt = elig.sum(axis=1)
    has = cnt > 0
    safe_cnt = np.maximum(cnt, 1)

    t_min = np.where(elig, t_ins, np.inf).min(axis=1)
    t_max = np.where(elig, t_ins, -np.inf).max(axis=1)
    t_sum = np.where(elig, t_ins, 0.0).sum(axis=1)
    p_min = np.where(elig, p_ins, np.inf).min(axis=1)
    p_max = np.where(elig, p_ins, -np.inf).max(axis=1)
    p_sum = np.where(elig, p_ins, 0.0).sum(axis=1)

    earliest = np.zeros(n_task, dtype=np.float32)
    prev_ok = (op > 0) & has
    if prev_ok.any():
        prev = idx - 1
        prev_m = mach[prev]
        ok = prev_ok & sched[prev] & (prev_m >= 0)
        if ok.any():
            tt_rows = tt[np.clip(prev_m, 0, m - 1)]
            mn = np.where(elig, tt_rows, np.inf).min(axis=1)
            earliest = np.where(ok, fin[prev] + mn, 0.0).astype(np.float32)

    fin_s = np.where(sched, fin, -np.inf).reshape(-1, m)
    jf = fin_s.max(axis=1)
    jf = np.where(np.isfinite(jf), jf, 0.0).astype(np.float32)

    tmin = np.where(has, t_min, 0.0)
    rem_t = np.where(~sched & has, tmin, 0.0).reshape(-1, m).sum(axis=1)
    planned = jf[job] + rem_t[job]

    out = np.zeros((n_task, 13), dtype=np.float32)
    out[:, 0] = op / max(1, m - 1)
    out[:, 1] = cnt / max(1, m)
    out[:, 2] = np.where(has, t_min / scale_t, 0.0)
    out[:, 3] = np.where(has, (t_sum / safe_cnt) / scale_t, 0.0)
    out[:, 4] = np.where(has, t_max / scale_t, 0.0)
    out[:, 5] = np.where(has, p_min / scale_p, 0.0)
    out[:, 6] = np.where(has, (p_sum / safe_cnt) / scale_p, 0.0)
    out[:, 7] = np.where(has, p_max / scale_p, 0.0)
    out[:, 9] = makespan_norm
    out[:, 10] = sched_ratio
    out[:, 11] = avg_util
    planned_scale = max(1.0, float(planned.max())) if n_task else 1.0
    out[:, 8] = earliest / planned_scale
    out[:, 12] = planned / planned_scale
    return out


def machine_context_features_full(G, machine_routes, task_idx, dur_row, tt,
                                  n_machine, p2=None, t_ins=None):
    """ST-only machine context (M, 8): the 3 base features plus five more."""
    out = machine_context_features(
        G, machine_routes, task_idx, dur_row, tt, n_machine)
    full = np.zeros((n_machine, 8), dtype=np.float32)
    full[:, :3] = out

    makespan = 0.0
    for _, data in G.nodes(data=True):
        ft = data.get('finish_time')
        if ft is not None:
            makespan = max(makespan, float(ft))

    scheduled = np.zeros(t_ins.shape[0], dtype=bool) if t_ins is not None else None
    if t_ins is not None:
        for t in range(t_ins.shape[0]):
            scheduled[t] = bool(G.nodes[t + 1].get('scheduled'))

    p2_max = max(1.0, float(np.max(p2))) if p2 is not None else 1.0
    for m in range(n_machine):
        route = list(machine_routes.get(m, []))
        busy = sum(float(G.nodes[t].get('duration', 0) or 0) for t in route)
        last_ft = (float(G.nodes[route[-1]].get('finish_time') or 0.0)
                   if route else 0.0)
        p2v = float(p2[0][m]) / p2_max if p2 is not None else 0.0
        wait = max(0.0, makespan - last_ft) / max(makespan, 1.0)
        if t_ins is not None:
            cand = sum(1 for q in range(t_ins.shape[0])
                       if not scheduled[q] and t_ins[q][m] > 0)
            cand = cand / max(1, t_ins.shape[0])
        else:
            cand = 0.0
        crit = last_ft / max(makespan, 1.0)
        full[m, 3:] = [p2v, busy / max(makespan, 1.0),
                       wait, cand, crit]
    return full


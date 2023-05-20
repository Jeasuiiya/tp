"""
Filename: tf_pl_rewarder.py
"""
from .tf_rew import Simulator, get_op_costs


# 该类继承了tf_sim.py中的模拟器
class ImportantOpsRewarder(Simulator):
    """
    ImportantOpsRewarder类
    """

    def __init__(self, nx_graph, devices):
        self.nx_graph = nx_graph
        self.f = None
        # 获取每一步骤的耗时
        cost_d = get_op_costs(nx_graph)
        out_dict = {}
        for node in self.nx_graph.nodes():
            outedge = self.nx_graph.out_edges(node)
            output_memorys = []
            for edge in outedge:
                output_memorys.append(self.nx_graph[edge[0]][edge[1]]["weight"])
            out_dict[node] = output_memorys
        # 将底层模拟器进行初始化，输入包括原始的tf元数据，耗时，输出大小以及设备
        Simulator.__init__(self, nx_graph, cost_d, out_dict, devices)

    def simulate(self, pl, sim_mem_usage=False, sim_com_usage=False):
        pl_ = pl.copy()
        for k, v in pl.items():
            pl[k] = self.devices[int(v)]
        r, f = Simulator.simulate(self, pl)
        self.f = f
        start_t = {}
        for node in self.nx_graph.nodes():
            n = node
            start_t[n] = f[n].start_time
        if sim_mem_usage:
            mem_q = []
            for n, t in start_t.items():
                mem = sum(self.output_dict[n])
                dev = self.devices.index(f[n].device)
                mem_q.append((t, "+", mem, dev))
                t_out_done = t
                for c in f[n].children:
                    t_out_done = max(t_out_done, int(f[c].start_time) + int(f[c].compute_cost) - 1)

                mem_q.append((t_out_done, "-", -mem, dev))
            mem_q.sort()
            mem_utils = [0] * len(self.devices)
            peak_utils = [0] * len(self.devices)
            for t, _, mem, dev in mem_q:
                mem_utils[dev] += mem
                if mem_utils[dev] > peak_utils[dev]:
                    peak_utils[dev] = mem_utils[dev]
            if sim_com_usage:
                comm_utils = [0] * len(self.devices)
                for k, v in self.node_dict.items():
                    dev = pl_[k]
                    out_size = v.output_memory
                    if not out_size:
                        continue
                    for name in v.children:
                        if dev != pl_[name]:
                            comm_utils[dev] += sum(out_size)
                return r, peak_utils, mem_utils, comm_utils
            return r, peak_utils, mem_utils
        return r

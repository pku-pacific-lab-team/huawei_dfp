"""Analytical Macro actions, separate from transport and cross-Macro reduction.

Counts are physical full-Macro actions (tails are zero padded). They are not
switching activity or measured circuit energy. Timing is optional and local.
"""
from ...hardware.macro import CIMMacroSpec

# 本文件只负责一个 Macro 的局部动作及权重槽位状态。
# 跨 Macro 的广播、K 归约和系统执行时间由上层另行建模。


def positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def check_mode(mode):
    if mode not in {"count", "timing"}:
        raise ValueError("mode must be count or timing")


class CIMMacroModel:
    def __init__(self, spec: CIMMacroSpec):
        self.spec = spec

    def compute(self, rows, k=16, n=32, *, mode="count", data_format="INT8"):
        check_mode(mode)
        if data_format != "INT8":
            raise ValueError("CIM Macro supports INT8 operands and INT32 results only")
        for name, value in (("rows", rows), ("k", k), ("n", n)):
            positive_integer(value, name)
        spec = self.spec
        if k > spec.block_k_dim or n > spec.block_n_dim:
            raise ValueError("compute shape exceeds one Macro")
        bits = spec.values["activation_bits"]
        # k/n 表示尾块的有效范围；硬件仍发出完整 K16×N32 动作，不默认获得门控收益。
        # bit_serial_cycles 是局部逐位动作数；local_tree_adds 每位每列需合并 K 个贡献。
        counts = {
            "useful_mac": rows * k * n,
            "issued_mac": rows * spec.macs_per_launch,
            "launches": rows,
            "bit_serial_cycles": rows * bits,
            "activation_capture_bytes": rows * spec.block_k_dim,
            "result_bytes": rows * spec.block_n_dim * 4,
            "weight_word_bit_products": rows * spec.macs_per_launch * bits,
            "local_tree_adds": rows * spec.block_n_dim * (spec.block_k_dim - 1) * bits,
            "shift_accumulates": rows * spec.block_n_dim * bits,
        }
        cycles = None
        if mode == "timing":
            first, = spec.require_timing("compute_first_cycles")
            # 第一行经历完整首结果延迟，后续行按输入启动间隔 II 连续到达。
            cycles = first + (rows - 1) * spec.values["input_ii_cycles"]
        return {"counts": counts, "cycles": cycles, "energy_pj": None, "area_mm2": None}

    def program_context(self, *, mode="count"):
        check_mode(mode)
        spec = self.spec
        # 编程一套完整权重：512 byte / 每组 32 byte = 16 组 wordline 动作。
        words = spec.context_bytes // spec.values["write_wordline_bytes"]
        cycles = None
        if mode == "timing":
            first, ii = spec.require_timing("write_first_cycles", "write_wordline_ii_cycles")
            cycles = first + (words - 1) * ii
        return {"counts": {"program_bytes": spec.context_bytes, "program_wordlines": words},
                "cycles": cycles, "energy_pj": None, "area_mm2": None}

    def switch_context(self, *, mode="count"):
        check_mode(mode)
        cycles = None
        if mode == "timing":
            cycles, = self.spec.require_timing("context_switch_cycles")
        return {"counts": {"context_switches": 1}, "cycles": cycles,
                "energy_pj": None, "area_mm2": None}


class MacroContexts:
    """Tags only, called at drained task boundaries; no cycle/ISA state machine.

    Replacement is round-robin over inactive contexts. Programming an active
    context is forbidden. Numerical SRAM data and in-flight work are not modeled.
    """
    def __init__(self, spec: CIMMacroSpec):
        # tags 保存各槽的权重身份，active 指当前选中槽，next_victim 指下次替换起点。
        # 这里只存标签，不保存 SRAM 数值，也不追踪仍在执行中的任务。
        self.tags = [None] * spec.values["weight_contexts"]
        self.active = None
        self.next_victim = 0

    def _index(self, index):
        if type(index) is not int or not 0 <= index < len(self.tags):
            raise ValueError("invalid context index")

    @staticmethod
    def _tag(tag):
        if tag is None:
            raise ValueError("weight tag cannot be None")
        try:
            hash(tag)
        except TypeError as exc:
            raise ValueError("weight tag must be hashable") from exc

    def load(self, index, tag):
        self._index(index)
        self._tag(tag)
        if index == self.active:
            raise ValueError("cannot program the active context")
        self.tags[index] = tag

    def activate(self, index):
        self._index(index)
        if self.tags[index] is None:
            raise ValueError("cannot activate an unloaded context")
        changed = self.active != index
        self.active = index
        return changed

    def ensure(self, tag):
        self._tag(tag)
        # 命中只需选择已有槽；“无需编程”与“无需切换”是两个不同判断。
        if tag in self.tags:
            index = self.tags.index(tag)
            loaded = False
        else:
            index = self.next_victim
            # 按轮转顺序找可写槽，跳过当前 active，避免覆盖正在选用的权重。
            if index == self.active:
                index = (index + 1) % len(self.tags)
            self.load(index, tag)
            self.next_victim = (index + 1) % len(self.tags)
            loaded = True
        # 两个返回值分别驱动上层的编程计数与 context 切换计数。
        return loaded, self.activate(index)

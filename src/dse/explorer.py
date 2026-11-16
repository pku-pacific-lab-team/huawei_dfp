"""
CIM Design Space Explorer

设计空间探索引擎，用于搜索最优硬件配置
"""
import itertools
import traceback
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd

from ..hardware.macro import CIMMacroSpec
from ..core.simulator import CIMCoreSimulator
from ..utils.report import ResultReporter


class CIMDesignExplorer:
    """
    CIM设计空间探索器

    搜索最优的硬件配置，包括：
    - L1 Buffer大小
    - Macro并行划分策略
    - 循环顺序
    - 数据格式
    """

    def __init__(
        self,
        macro_spec: CIMMacroSpec,
        l0_params: Dict[str, int],
        energy_params: Dict[str, float],
        workload: Tuple[int, int, int],
        fixed_bw: Optional[Dict[str, float]] = None
    ):
        """
        初始化DSE引擎

        Args:
            macro_spec: CIM Macro规格
            l0_params: L0 Buffer参数 {'input': size, 'output': size}
            energy_params: 能量参数字典
            workload: 工作负载 (M, K, N)
            fixed_bw: 固定带宽参数（可选）
        """
        self.macro_spec = macro_spec
        self.l0_params = l0_params
        self.energy_params = energy_params
        self.workload = workload

        # 固定带宽参数
        self.fixed_bw = fixed_bw or {
            "noc_to_l1": 64.0,
            "l1_to_l0": 128.0
        }

        # 结果报告器
        self.reporter = ResultReporter()

    def run_dse_hybrid(
        self,
        l1_size_candidates: List[int],
        loop_orders: List[str],
        data_format: str,
        total_macros: int = 32,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        运行混合设计空间探索

        Args:
            l1_size_candidates: L1 Buffer大小候选列表（字节）
            loop_orders: 循环顺序候选列表
            data_format: 数据格式
            total_macros: 总Macro数量
            verbose: 是否显示详细输出

        Returns:
            结果列表，每个元素是一个配置的性能指标
        """
        M, K, N = self.workload
        results_db = []

        # 生成有效的并行划分策略
        valid_partitions = []
        for i in range(1, total_macros + 1):
            if total_macros % i == 0:
                pk = i
                pn = total_macros // i
                valid_partitions.append((pk, pn))

        # 生成搜索空间
        search_space = list(itertools.product(
            l1_size_candidates,
            valid_partitions,
            loop_orders
        ))

        total_configs = len(search_space)
        print(f"  Exploring {total_configs} configurations...", end=" ", flush=True)

        count = 0
        failed_count = 0

        for l1_size, strategy, order in search_space:
            # 实例化仿真器
            sim = CIMCoreSimulator(
                n_macros=total_macros,
                macro_spec=self.macro_spec,
                clock_freq_ghz=1.0,
                size_l1_input_buffer_bytes=l1_size,
                size_l1_output_buffer_bytes=l1_size,
                size_l0_input_buffer_bytes=self.l0_params['input'],
                size_l0_output_buffer_bytes=self.l0_params['output'],
                bw_noc_to_l1_gbps=self.fixed_bw['noc_to_l1'],
                bw_l1_to_l0_gbps=self.fixed_bw['l1_to_l0'],
                bw_l0_to_l1_output_gbps=self.fixed_bw.get('l0_to_l1_output'),
                throughput_vector_unit_gops=128,
                energy_params=self.energy_params,
                data_format=data_format,
                custom_overheads=None,
                verbose=verbose
            )

            try:
                # 运行仿真
                res = sim.simulate(M, K, N, strategy, order)

                edp = res['total_energy_pj'] * res['total_latency_cycles']

                dram_energy = res['energies_pj'].get('dram_read', 0) + res['energies_pj'].get('dram_write', 0)
                compute_energy = res['total_energy_pj'] - dram_energy

                entry = {
                    "Data_Format": data_format,
                    "L1_Size_KB": l1_size / 1024,
                    "Strategy": str(strategy),
                    "Loop_Order": order,
                    "Latency_Cyc": res['total_latency_cycles'],
                    "Energy_pJ": res['total_energy_pj'],
                    "Compute_Energy_pJ": compute_energy,
                    "Throughput_GOPS": res['throughput_gops'],
                    "EDP": edp,
                    "Bottleneck": res['bottleneck']
                }
                results_db.append(entry)
                count += 1

            except Exception as e:
                failed_count += 1
                if verbose:
                    print(f"\n[ERROR] Config failed: L1={l1_size}, Strategy={strategy}, Order={order}")
                    print(f"Error details: {e}")
                    traceback.print_exc()
                    print("-" * 30)
                break

        if failed_count == 0:
            print(f"Done! ({count} valid)")
        else:
            print(f"Done with errors! (Valid: {count}, Failed: {failed_count})")
        return results_db

    def analyze_results(self, results_db: List[Dict[str, Any]], show_summary: bool = True) -> pd.DataFrame:
        """
        分析DSE结果

        Args:
            results_db: 结果列表
            show_summary: 是否显示摘要

        Returns:
            包含所有结果的DataFrame
        """
        if not results_db:
            print("\n[WARNING] No valid results found!")
            return pd.DataFrame()

        try:
            df = pd.DataFrame(results_db)

            if show_summary:
                # 使用表格形式显示 Top 5 配置
                self.reporter.print_dse_summary(results_db, top_n=5)

            return df

        except ImportError:
            print("Pandas not found, utilizing basic sort.")
            sorted_res = sorted(results_db, key=lambda x: x.get('EDP', float('inf')))
            if sorted_res:
                print("Best EDP:", sorted_res[0])
            return sorted_res

    def reproduce_specific_config(self, config_row: pd.Series, total_macros: int = 32) -> None:
        """
        复现特定配置的详细报告

        Args:
            config_row: 配置行（pandas Series）
            total_macros: 总Macro数量
        """
        import ast

        # 1. 参数还原
        data_format = config_row['Data_Format']
        l1_size_bytes = int(config_row['L1_Size_KB'] * 1024)
        strat_raw = config_row['Strategy']

        try:
            strategy = ast.literal_eval(strat_raw)
        except (ValueError, SyntaxError):
            strategy = strat_raw

        loop_order = config_row['Loop_Order']

        print(f"\n{'='*60}")
        print(f"  BEST CONFIGURATION DETAILS")
        print(f"{'='*60}")
        print(f"  Format: {data_format} | L1: {config_row['L1_Size_KB']:.0f} KB | "
              f"Strategy: {strategy} | Order: {loop_order}")
        print(f"{'='*60}")

        # 2. 重新实例化仿真器 (不打印初始化信息)
        sim = CIMCoreSimulator(
            n_macros=total_macros,
            macro_spec=self.macro_spec,
            clock_freq_ghz=1.0,
            size_l1_input_buffer_bytes=l1_size_bytes,
            size_l1_output_buffer_bytes=l1_size_bytes,
            size_l0_input_buffer_bytes=self.l0_params['input'],
            size_l0_output_buffer_bytes=self.l0_params['output'],
            bw_noc_to_l1_gbps=self.fixed_bw['noc_to_l1'],
            bw_l1_to_l0_gbps=self.fixed_bw['l1_to_l0'],
            bw_l0_to_l1_output_gbps=self.fixed_bw.get('l0_to_l1_output'),
            throughput_vector_unit_gops=128,
            energy_params=self.energy_params,
            data_format=data_format,
            custom_overheads=None,
            verbose=False  # 不打印初始化信息
        )

        # 3. 运行仿真
        M, K, N = self.workload
        res = sim.simulate(M, K, N, strategy, loop_order)

        # 4. 打印详细报告
        self.reporter.print_simulation_result(res)

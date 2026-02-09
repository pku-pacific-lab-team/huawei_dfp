"""
Result Reporter

格式化输出仿真结果
"""
from typing import Dict, Any, List, Optional


class ResultReporter:
    """
    结果报告生成器

    格式化打印仿真结果和DSE摘要
    """

    def print_simulation_result(self, results: Dict[str, Any]) -> None:
        """
        打印单次仿真结果

        Args:
            results: 仿真结果字典
        """
        # 1. 流水线状态
        print(f"\n[Pipeline Latency]")
        if 'latencies' in results:
            for k, v in results['latencies'].items():
                bar_len = int(v / max(results['latencies'].values()) * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"  {k:<12} {bar} {v:>12,} cyc")

        print(f"\n  Bottleneck: {results.get('bottleneck', 'N/A')}")

        # 2. 能耗分解 — 按物理资源分组
        print(f"\n[Energy Breakdown by Resource]")
        total_e = results['total_energy_pj']
        e = results['energies_pj']

        # 分组定义
        groups = {
            "DRAM":       ["dram_read", "dram_write"],
            "L1 SRAM":    ["load_X_NoC", "load_W", "read_L1", "write_l1_output"],
            "L0 SRAM":    ["write_L0", "read_L0", "write_psums", "read_psums"],
            "Compute":    ["compute_macro", "local_readout"],
            "Pre/Post":   ["pre_process", "accum_vector_unit"],
            "Weight Prog": ["weight_prog"],
        }

        for group_name, keys in groups.items():
            group_e = sum(e.get(k, 0) for k in keys)
            if group_e <= 0:
                continue
            ratio = (group_e / total_e * 100) if total_e > 0 else 0
            bar_len = int(ratio / 5)
            bar = "█" * bar_len
            # 打印分组行
            print(f"  {group_name:<14} {bar:<10} {ratio:5.1f}%  ({group_e/1e9:,.3f} mJ)")
            # 打印子项
            for k in keys:
                v = e.get(k, 0)
                if v <= 0:
                    continue
                sub_ratio = (v / total_e * 100) if total_e > 0 else 0
                print(f"    {k:<16}          {sub_ratio:5.1f}%  ({v/1e9:,.3f} mJ)")

        # 3. 总体性能（含DRAM vs 不含DRAM对比）
        total_e = results['total_energy_pj']
        dram_e = results['energies_pj'].get('dram_read', 0) + results['energies_pj'].get('dram_write', 0)
        compute_e = total_e - dram_e
        total_latency = results['total_latency_cycles']
        total_ops = results['throughput_gops'] * total_latency / results.get('clock_freq_ghz', 1.0) if results['throughput_gops'] > 0 else 0

        print(f"\n[Performance Summary]")
        print(f"  {'':30} {'w/ DRAM':>14} {'w/o DRAM':>14}")
        print(f"  {'─'*58}")
        print(f"  {'Total Latency':<30} {total_latency:>12,} Cycles")
        print(f"  {'Total Energy':<30} {total_e/1e9:>12,.2f} mJ {compute_e/1e9:>12,.2f} mJ")

        eff_full = (total_ops / 1e12) / (total_e / 1e12) if total_e > 0 else 0
        eff_no_dram = (total_ops / 1e12) / (compute_e / 1e12) if compute_e > 0 else 0
        print(f"  {'Throughput':<30} {results['throughput_gops']:>12,.2f} GOPS")
        print(f"  {'Energy Efficiency':<30} {eff_full:>12,.2f} TOPS/W {eff_no_dram:>12,.2f} TOPS/W")
        print(f"  {'DRAM Energy':<30} {dram_e/1e9:>12,.2f} mJ  ({dram_e/total_e*100:.1f}% of total)" if total_e > 0 else "")
        print("")

    def print_dse_summary(
        self,
        results_db: List[Dict[str, Any]],
        top_n: int = 5
    ) -> None:
        """
        打印DSE结果摘要

        Args:
            results_db: 结果列表
            top_n: 显示前N个最优配置
        """
        if not results_db:
            print("[WARNING] No results to summarize.")
            return

        # 按EDP排序
        sorted_by_edp = sorted(results_db, key=lambda x: x.get('EDP', float('inf')))

        print(f"\n{'='*100}")
        print(f"  DSE RESULTS - Top {top_n} Configurations (by EDP)")
        print(f"{'='*100}")
        print(f"{'Rank':<5} {'Format':<10} {'L1(KB)':<8} {'Strategy':<12} "
              f"{'Order':<7} {'Throughput':>12} {'Energy(mJ)':>12} {'Bottleneck':<12}")
        print("-" * 100)

        for i, entry in enumerate(sorted_by_edp[:top_n]):
            print(f"{i+1:<5} {entry['Data_Format']:<10} {entry['L1_Size_KB']:<8.0f} "
                  f"{entry['Strategy']:<12} {entry['Loop_Order']:<7} "
                  f"{entry['Throughput_GOPS']:>12.2f} {entry['Energy_pJ']/1e9:>12.2f} "
                  f"{entry['Bottleneck']:<12}")

        print("-" * 100)

    def format_config_summary(self, config: Dict[str, Any]) -> str:
        """
        格式化配置摘要

        Args:
            config: 配置字典

        Returns:
            格式化的字符串
        """
        lines = []
        lines.append("Configuration Summary:")
        lines.append("-" * 40)

        for key, value in config.items():
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for k, v in value.items():
                    lines.append(f"    {k}: {v}")
            else:
                lines.append(f"  {key}: {value}")

        lines.append("-" * 40)
        return "\n".join(lines)

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

        # 2. 能耗分解 (只显示主要项)
        print(f"\n[Energy Breakdown]")
        total_e = results['total_energy_pj']

        # 按能耗排序，只显示前5项
        sorted_energy = sorted(results['energies_pj'].items(), key=lambda x: -x[1])
        for k, v in sorted_energy[:6]:
            ratio = (v / total_e * 100) if total_e > 0 else 0
            bar_len = int(ratio / 5)  # 每5%一个块
            bar = "█" * bar_len
            print(f"  {k:<20} {bar:<8} {ratio:5.1f}%  ({v/1e6:,.1f} mJ)")

        other_energy = sum(v for k, v in sorted_energy[6:])
        if other_energy > 0:
            other_ratio = (other_energy / total_e * 100)
            print(f"  {'(others)':<20} {'':8} {other_ratio:5.1f}%  ({other_energy/1e6:,.1f} mJ)")

        # 3. 总体性能
        print(f"\n[Performance Summary]")
        print(f"  Total Latency : {results['total_latency_cycles']:>12,} Cycles")
        print(f"  Total Energy  : {results['total_energy_pj']/1e6:>12,.2f} mJ")
        print(f"  Throughput    : {results['throughput_gops']:>12,.2f} GOPS")
        print(f"  Efficiency    : {results['energy_efficiency_tops_per_watt']:>12,.2f} TOPS/W")
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
                  f"{entry['Throughput_GOPS']:>12.2f} {entry['Energy_pJ']/1e6:>12.2f} "
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

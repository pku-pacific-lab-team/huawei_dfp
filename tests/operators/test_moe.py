"""Fixed routing, native expert work, payloads and global MoE storage contracts."""
import copy
from dataclasses import replace
import json
import unittest

from src.hardware.blocks import BlockExecution, load_block_execution
from src.hardware.tpu import TPUSpec
from src.hardware.tsp import TSPSpec
from src.operators import (MoERoutingPlan, MoEExperts, TensorSpec, SwiGLUFFN, Residual,
                           simulate_moe_experts, simulate_block, plan_connection)
from src.utils.operator_workload import parse_moe_workload, parse_block_workload
from tests.operators.test_gemm import specs
from tests.operators.test_mla import actions, config
from tests.operators.test_vector import raw


def task(assignments=((0, 2), (0, 1), (0, 2)), experts=5, hidden=5, intermediate=7, shared=2):
    tensor = lambda n, s: TensorSpec(n, s, "INT8")
    h, i, e = hidden, intermediate, experts
    weights = (tensor("shared_gate", (h, shared * i)), tensor("shared_up", (h, shared * i)),
               tensor("shared_down", (shared * i, h))) if shared else (None,) * 3
    return MoEExperts(tensor("x", (len(assignments), h)), tensor("gate", (e, h, i)), tensor("up", (e, h, i)),
                       tensor("down", (e, i, h)), MoERoutingPlan(e, assignments), shared, *weights)


class MoETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = specs()

    def run_moe(self, op=None, arch="tpu", **options):
        return simulate_moe_experts(op or task(), arch, self.hardware[arch], include_details=True, **options)

    def test_four_architectures_counts_timing_shapes_and_serial_children(self):
        for arch in self.hardware:
            for shared in (0, 2):
                with self.subTest(arch=arch, shared=shared):
                    a, b = (self.run_moe(task(shared=shared), arch, mode=m, tile_elements=3) for m in ("count", "timing"))
                    self.assertEqual(a.action_ledger, b.action_ledger)
                    self.assertEqual(a.native_report["memory"], b.native_report["memory"])
                    self.assertIsNone(a.cycle_count)
                    self.assertGreater(b.latency_s, 0)
                    self.assertEqual(b.cycle_count, sum(n["cycle_count_per_repetition"] for n in b.native_report["schedule"]["nodes"]))
                    self.assertEqual([t.shape for t in a.operator.outputs], [(3, 2, 5)] + ([(3, 5)] if shared else []))
                    self.assertTrue(all(t.dtype == "FP32" for t in a.operator.outputs))
                    self.assertFalse(b.boundary["chip_execution_included"])
                    json.dumps(b.to_detailed_dict(), allow_nan=False)

    def test_hand_macs_nonlinear_work_and_explicit_fp32_output_boundary(self):
        for arch in self.hardware:
            r = self.run_moe(arch=arch)
            logical = r.work_counts["logical"]
            # Routed: 6 rows * 3 * 5 * 7; shared: 3 rows * 3 * 5 * 14.
            self.assertEqual(logical["useful_macs"], 1260)
            self.assertEqual(logical["operations"], 2 * 1260 + 504)
            self.assertEqual(logical["operation_counts"], {"fp32_mul": 252, "fp32_exp": 84, "fp32_add": 84, "fp32_rcp": 84})
            self.assertEqual(r.work_counts["internal"]["boundary_elements"], {"boundary_to_fp32": 213, "boundary_to_int8": 84})
            self.assertEqual(logical["output_bytes"], 180)
            self.assertFalse(dict(r.operator.attributes)["weighted_combine_included"])
            self.assertFalse(dict(r.operator.attributes)["routing_compute_included"])

    def test_hand_tpu_payloads_and_global_storage_peak(self):
        r = self.run_moe()
        self.assertEqual(actions(r, "hbm_read_bytes"), 1725)
        self.assertEqual(actions(r, "hbm_write_bytes"), 1266)
        logical = r.work_counts["logical"]
        for name, value in (("route_index_read_bytes", 24), ("dispatch_payload_bytes", 30),
                            ("publish_payload_bytes", 120), ("declared_weight_logical_bytes", 735),
                            ("active_weight_logical_bytes", 525)):
            self.assertEqual(logical[name], value)
        life = r.native_report["memory"]["lifetime_validation"]
        self.assertEqual(life["global_allocations"], {"input": 15, "route_indices": 24, "routed_output": 120,
            "reserved": 0, "routed_gate_weight": 175, "routed_up_weight": 175, "routed_down_weight": 175,
            "shared_gate_weight": 70, "shared_up_weight": 70, "shared_down_weight": 70, "shared_output": 60})
        self.assertEqual(life["persistent_bytes_before_shared_output"], 894)
        self.assertEqual(r.resource_requirements["hbm"]["required_bytes"], 1272)

    def test_routing_conservation_slot_order_and_empty_expert_skip(self):
        r = self.run_moe()
        route, schedule = r.native_report["routing"], r.native_report["schedule"]
        self.assertEqual(route["counts"], [3, 1, 2, 0, 0])
        self.assertEqual(sum(route["counts"]), route["total_assignments"])
        self.assertEqual(schedule["skipped_experts"], [3, 4])
        self.assertEqual([e["token_count"] for e in schedule["experts"]], [3, 1, 2])
        self.assertEqual(schedule["experts"][2]["destination_slots"], [[0, 1], [2, 1]])
        slots = [tuple(slot) for e in schedule["experts"] for slot in e["destination_slots"]]
        self.assertEqual(set(slots), {(b, k) for b in range(3) for k in range(2)})
        self.assertEqual(len(slots), len(set(slots)))
        self.assertFalse(any(n["id"].startswith(("expert_3", "expert_4")) for n in schedule["nodes"]))
        for e in (3, -1, True, 5):
            with self.assertRaises(ValueError):
                task().expert(e)

    def test_inactive_weights_consume_capacity_but_do_not_add_work_or_traffic(self):
        for arch in self.hardware:
            a, b = (self.run_moe(task(experts=e), arch) for e in (5, 9))
            self.assertEqual(a.action_ledger, b.action_ledger)
            self.assertEqual(a.work_counts["logical"]["useful_macs"], b.work_counts["logical"]["useful_macs"])
            life = a.native_report["memory"]["lifetime_validation"]
            space = "mem" if arch in {"tsp", "tsp_cim"} else "hbm"
            self.assertEqual(b.resource_requirements[space]["required_bytes"] - a.resource_requirements[space]["required_bytes"],
                             4 * sum(life["routed_weight_bytes_per_expert"]))

    def test_load_imbalance_uses_actual_expert_rows_and_distinct_weight_reads(self):
        a = self.run_moe(mode="timing")
        b = self.run_moe(task(assignments=((0, 1),) * 3), mode="timing")
        self.assertEqual(a.work_counts["logical"]["operations"], b.work_counts["logical"]["operations"])
        self.assertEqual(actions(a, "hbm_read_bytes") - actions(b, "hbm_read_bytes"), 3 * 5 * 7)
        self.assertNotEqual(a.cycle_count, b.cycle_count)
        self.assertEqual([e["token_count"] for e in b.native_report["schedule"]["experts"]], [3, 3])

    def test_native_ffn_actions_and_roles_preserved_for_every_active_expert(self):
        op = task()
        for arch, hw in self.hardware.items():
            r = self.run_moe(op, arch, mode="timing")
            nodes = {n["id"]: n["native_report"] for n in r.native_report["schedule"]["nodes"]}
            experts = [(f"expert_{e}.ffn", op.expert(e)) for e, n in enumerate(op.routing.counts) if n] + [("shared.ffn", op.shared())]
            for name, expert in experts:
                standalone = simulate_block(expert, arch, hw, mode="timing", include_details=True)
                self.assertEqual(nodes[name]["action_counts"], standalone.native_report["action_counts"])
                self.assertEqual(nodes[name]["work_counts"], standalone.work_counts)
                for action in standalone.action_ledger.entries:
                    composed = next(a for a in r.action_ledger.entries if a.id == name + "." + action.id)
                    self.assertEqual((composed.role, composed.count, composed.unit), (action.role, action.count, action.unit))

    def test_lifetimes_release_packed_rows_before_publish_and_delay_shared_output(self):
        life = self.run_moe().native_report["memory"]["lifetime_validation"]
        rows = {row["node"]: row for row in life["steps"]}
        self.assertEqual(rows["expert_0.dispatch"]["storage_bytes"], 909)
        self.assertEqual(rows["expert_0.ffn.activation"]["storage_bytes"], 1098)
        self.assertEqual(rows["expert_0.publish"]["storage_bytes"], 954)
        self.assertEqual(rows["shared.ffn.activation"]["storage_bytes"], 1272)
        self.assertEqual(rows["shared.ffn.output"]["storage_bytes"], 1014)
        self.assertEqual(life["peak_bytes"], max(row["peak_bytes"] for row in rows.values()))

    def test_global_hbm_capacity_exact_boundary_and_inactive_experts(self):
        cfg = raw("tpu")
        cfg["parameters"]["hbm_usable_bytes"]["value"] = 1271
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_moe_experts(task(), "tpu", TPUSpec(cfg))
        cfg["parameters"]["hbm_usable_bytes"]["value"] = 1272
        hw = TPUSpec(cfg)
        simulate_moe_experts(task(), "tpu", hw)
        simulate_block(task().expert(0), "tpu", hw)
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_moe_experts(task(experts=6), "tpu", hw)

    def test_resident_striping_reservation_and_minimum_scratch_capacity(self):
        for arch in ("tsp", "tsp_cim"):
            r = self.run_moe(arch=arch, mem_slices=[0, 1, 2], streams=2, reserved_bytes=17, tile_elements=2, hemisphere=1)
            life = r.native_report["memory"]["lifetime_validation"]
            self.assertEqual(life["global_allocations"]["reserved"], 18)
            self.assertTrue(all(row["storage_bytes"] % 3 == 0 for row in life["steps"]))
            self.assertTrue(any(row["scratch_bytes"] > 0 for row in life["steps"]))
            self.assertEqual(life["peak_bytes"], max(row["peak_bytes"] for row in life["steps"]))
        hw = copy.deepcopy(self.hardware["tsp"])
        op = task(hidden=17, intermediate=33, shared=1)
        small = simulate_moe_experts(op, "tsp", hw, mem_slices=[0], tile_elements=1)
        hw.mem_total_bytes = hw.mem_n_slices * small.resource_requirements["mem"]["required_bytes"]
        bounded = simulate_moe_experts(op, "tsp", hw, mem_slices=[0], tile_elements=64, include_details=True)
        self.assertLessEqual(bounded.resource_requirements["mem"]["required_bytes"], hw.mem_slice_capacity_bytes)
        hw.mem_total_bytes = hw.mem_n_slices * 64
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_moe_experts(op, "tsp", hw, mem_slices=[0])

    def test_shared_widened_ffn_and_routed_only_contract(self):
        r = self.run_moe(task(shared=0))
        self.assertEqual(len(r.operator.outputs), 1)
        self.assertFalse(any(n["id"].startswith("shared") for n in r.native_report["schedule"]["nodes"]))
        self.assertEqual(r.work_counts["logical"]["useful_macs"], 630)
        shared = task(shared=2).shared()
        self.assertEqual(shared.gate_weight.shape, (5, 14))
        self.assertEqual(shared.x.shape, (3, 5))
        self.assertEqual(shared.spec.outputs[0].dtype, "FP32")

    def test_batch_topk_scaling_and_single_token_all_experts(self):
        for assignments, shared in ((((0,),), 0), (((0, 1, 2, 3, 4),), 1), (((0, 1),) * 7, 2)):
            op = task(assignments=assignments, shared=shared)
            r = self.run_moe(op)
            b, k = len(assignments), len(assignments[0])
            self.assertEqual(r.work_counts["logical"]["useful_macs"], 3 * 5 * 7 * b * (k + shared))
            self.assertEqual(r.work_counts["logical"]["dispatch_payload_bytes"], b * k * 5)
            self.assertEqual(r.work_counts["logical"]["publish_payload_bytes"], b * k * 5 * 4)

    def test_copy_tail_tiles_do_not_duplicate_or_drop_payload_bytes(self):
        a, b = (self.run_moe(tile_elements=t) for t in (1, 3))
        for suffix in ("hbm_read_bytes", "hbm_write_bytes"):
            self.assertEqual(actions(a, suffix), actions(b, suffix))
        nodes = {n["id"]: n["native_report"] for n in b.native_report["schedule"]["nodes"]}
        for name, size in (("route_read", 24), ("expert_0.dispatch", 15), ("expert_0.publish", 60)):
            self.assertEqual(nodes[name]["work_counts"]["logical"]["read_payload_bytes"], size)
            self.assertEqual(sum(t["bytes"] * t["repetitions"] for t in nodes[name]["schedule"]["tile_groups"]), size)

    def test_ffn_fp32_option_reuses_gemms_without_final_int8_conversion(self):
        f = task().expert(0)
        for arch, hw in self.hardware.items():
            a = simulate_block(replace(f, output_dtype="INT8"), arch, hw, include_details=True)
            b = simulate_block(f, arch, hw, include_details=True)
            self.assertEqual(a.work_counts["logical"]["useful_macs"], b.work_counts["logical"]["useful_macs"])
            self.assertEqual(a.work_counts["internal"]["boundary_elements"]["boundary_to_int8"] -
                             b.work_counts["internal"]["boundary_elements"]["boundary_to_int8"], 15)
            self.assertEqual(a.native_report["schedule"]["nodes"][:4], b.native_report["schedule"]["nodes"][:4])
        cfg = config("swiglu_ffn")
        cfg["output_dtype"] = "FP32"
        self.assertEqual(parse_block_workload(cfg).spec.outputs[0].dtype, "FP32")
        with self.assertRaises(ValueError):
            replace(f, output_dtype="BF16")

    def test_unknown_cost_and_clock_are_not_silently_zero(self):
        cfg = copy.deepcopy(load_block_execution().config)
        cfg["parameters"]["to_fp32_cost"].update(value=None, status="unknown")
        block = BlockExecution(cfg)
        self.run_moe(block_execution=block)
        with self.assertRaisesRegex(ValueError, "to_fp32_cost"):
            self.run_moe(block_execution=block, mode="timing")
        cfg = raw("tsp")
        cfg["parameters"]["clock_ghz"].update(value=None, status="unknown")
        r = simulate_moe_experts(task(), "tsp", TSPSpec(cfg), mode="timing")
        self.assertGreater(r.cycle_count, 0)
        self.assertEqual(r.timing_status, "cycles_only")
        self.assertIsNone(r.latency_s)

    def test_fixed_assignments_are_detached_and_invalid_routes_rejected(self):
        assignments = [[0, 2], [0, 1]]
        routing = MoERoutingPlan(3, assignments)
        assignments[0][0] = 1
        self.assertEqual(routing.assignments, ((0, 2), (0, 1)))
        for e, rows in ((0, [[0]]), (True, [[0]]), (3, []), (3, [[]]), (3, [[0], [1, 2]]),
                        (3, [[0, 0]]), (3, [[3]]), (3, [[-1]]), (3, [[True]]), (3, [[0.0]]), (3, None)):
            with self.subTest(e=e, rows=rows), self.assertRaises(ValueError):
                MoERoutingPlan(e, rows)

    def test_input_shapes_storage_aliases_and_shared_consistency(self):
        op = task()
        for changes in ({"x": TensorSpec("x", (4, 5), "INT8")}, {"x": TensorSpec("x", (3, 5), "FP32")},
                        {"gate_weight": TensorSpec("gate", (4, 5, 7), "INT8")}, {"down_weight": TensorSpec("down", (5, 5, 7), "INT8")},
                        {"routing": MoERoutingPlan(5, op.routing.assignments, "x")}, {"shared_experts": 0}, {"shared_experts": True},
                        {"shared_gate_weight": None}, {"shared_down_weight": TensorSpec("sd", (7, 5), "INT8")},
                        {"output_names": ["x", "new"]}, {"output_names": ["same", "same"]},
                        {"output_names": [None, "shared"]}, {"output_names": ["only_one"]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(op, **changes)

    def test_strict_workload_fields_and_routed_only_parsing(self):
        for key, value in (("schema_version", True), ("routing", {}), ("shared_experts", -1), ("route_scale", 2.5),
                           ("description", ""), ("x", None), ("output_names", [])):
            cfg = config("moe_experts")
            cfg[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                parse_moe_workload(cfg)
        cfg = config("moe_experts")
        for key in ("shared_experts", "shared_gate_weight", "shared_up_weight", "shared_down_weight", "output_names"):
            cfg.pop(key)
        self.assertEqual(len(parse_moe_workload(cfg).spec.outputs), 1)

    def test_shared_output_connection_retains_fp32_and_explicit_placement(self):
        r = self.run_moe()
        shared = r.operator.outputs[1]
        residual = simulate_block(Residual(shared, TensorSpec("residual", shared.shape, "FP32")), "tpu", self.hardware["tpu"])
        edge = plan_connection(r, residual, output_index=1)
        self.assertEqual(edge["conversion"]["status"], "not_required")
        self.assertFalse(edge["residency"]["reuse_assumed"])


if __name__ == "__main__":
    unittest.main()

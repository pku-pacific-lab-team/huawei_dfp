"""Independent MoE work/traffic oracles and global lifetime boundaries."""
import copy
from dataclasses import replace
import json
import unittest

from src.hardware.blocks import BlockExecution, load_block_execution
from src.hardware.vector import VectorExecution, load_vector_execution
from src.hardware.tpu import TPUSpec
from src.operators import (MoE, MoERouter, MoECombine, TensorSpec, simulate_moe, simulate_moe_router,
                           simulate_moe_combine, simulate_moe_experts)
from src.utils.operator_workload import parse_moe_block_workload
from tests.operators.test_gemm import specs
from tests.operators.test_moe import task
from tests.operators.test_mla import actions, config
from tests.operators.test_vector import raw


def whole(shared=2, assignments=((0, 2), (0, 1), (0, 2)), **router_options):
    e = task(experts=6, shared=shared, assignments=assignments)
    r = MoERouter(e.x, TensorSpec("router_weight", (5, 6), "INT8"), e.routing.top_k,
                  **{**dict(groups=3, selected_groups=2, bias=TensorSpec("bias", (6,), "FP32"), route_scale=2.5), **router_options})
    return MoE(r, e)


class MoEBlockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = specs()

    def run_op(self, op=None, arch="tpu", **kw):
        op = op or whole()
        sim = {MoE: simulate_moe, MoERouter: simulate_moe_router, MoECombine: simulate_moe_combine}[type(op)]
        return sim(op, arch, self.hardware[arch], include_details=True, **kw)

    def test_four_architectures_count_timing_and_serial_native_children(self):
        for arch in self.hardware:
            for op in (whole(), whole().router, whole().combine()):
                with self.subTest(arch=arch, operation=op.operation):
                    a, b = (self.run_op(op, arch, mode=m, tile_elements=3) for m in ("count", "timing"))
                    self.assertEqual(a.action_ledger, b.action_ledger)
                    self.assertEqual(a.native_report["memory"], b.native_report["memory"])
                    self.assertIsNone(a.cycle_count)
                    self.assertGreater(b.latency_s, 0)
                    self.assertEqual(b.cycle_count, sum(n["cycle_count_per_repetition"] for n in b.native_report["schedule"]["nodes"]))
                    self.assertFalse(b.boundary["numeric_values_simulated"])
                    json.dumps(b.to_detailed_dict(), allow_nan=False)

    def test_hand_router_work_and_hbm_payload(self):
        r = self.run_op(whole().router)
        # B=3,H=5,E=6,k=2,G=3,L=2. Per token: group maxima 3,
        # group selection 2+1, expert selection 3+2 => 11 comparisons.
        self.assertEqual(r.work_counts["logical"], {"operations": 330, "useful_macs": 90,
            "operation_counts": {"fp32_mul": 30, "fp32_exp": 18, "fp32_add": 48, "fp32_rcp": 21, "fp32_max": 33}, "output_bytes": 48})
        self.assertEqual(r.work_counts["internal"]["boundary_elements"], {"boundary_to_fp32": 18})
        # Gate reads 15+30, writes 72. Materialized selection: 332/228 B per row.
        self.assertEqual(actions(r, "hbm_read_bytes"), 45 + 3*332)
        self.assertEqual(actions(r, "hbm_write_bytes"), 72 + 3*228)
        self.assertEqual(r.resource_requirements["hbm"]["required_bytes"], 477)

    def test_hand_combine_broadcast_reduction_shared_and_tail(self):
        for dtype in ("INT8", "FP32"):
            r = self.run_op(replace(whole().combine(), output_dtype=dtype), tile_elements=3)
            self.assertEqual(r.work_counts["logical"]["operation_counts"], {"fp32_mul": 30, "fp32_add": 30})
            self.assertEqual(actions(r, "hbm_read_bytes"), 120 + 60 + 3*2*2*4)
            self.assertEqual(actions(r, "hbm_write_bytes"), 15 if dtype == "INT8" else 60)
            self.assertEqual(r.work_counts["internal"]["boundary_elements"], {"boundary_to_int8": 15} if dtype == "INT8" else {})
        r = self.run_op(replace(whole(shared=0).combine(), shared=None, output_dtype="FP32"), tile_elements=1)
        self.assertEqual(r.work_counts["logical"]["operation_counts"], {"fp32_mul": 30, "fp32_add": 15})
        self.assertEqual(actions(r, "hbm_read_bytes"), 120 + 3*2*5*4)

    def test_hand_whole_work_traffic_and_global_peak(self):
        r = self.run_op(tile_elements=3)
        self.assertEqual(r.work_counts["logical"]["useful_macs"], 1350)
        self.assertEqual(r.work_counts["logical"]["operations"], 3414)
        self.assertEqual(r.work_counts["internal"]["boundary_elements"], {"boundary_to_fp32": 231, "boundary_to_int8": 99})
        self.assertEqual(actions(r, "hbm_read_bytes"), 1041 + 1725 + 228)
        self.assertEqual(actions(r, "hbm_write_bytes"), 756 + 1266 + 15)
        life = r.native_report["memory"]["lifetime_validation"]
        self.assertEqual(sum(life["persistent_allocations"].values()), 909)
        self.assertEqual(life["peak_bytes"], 1455)
        self.assertEqual(life["steps"][1]["peak_bytes"], 1317)
        self.assertEqual(life["steps"][-1]["storage_bytes"], 1128)
        self.assertNotIn("route_indices", life["steps"][-1]["phase_allocations"])
        standalone = simulate_moe_experts(whole().experts, "tpu", self.hardware["tpu"])
        self.assertEqual(standalone.resource_requirements["hbm"]["required_bytes"], 1377)
        self.assertGreater(life["peak_bytes"], standalone.resource_requirements["hbm"]["required_bytes"])

    def test_whole_exact_capacity_not_max_of_standalone_peaks(self):
        for capacity in (1377, 1454, 1455):
            cfg = raw("tpu")
            cfg["parameters"]["hbm_usable_bytes"]["value"] = capacity
            if capacity < 1455:
                with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
                    simulate_moe(whole(), "tpu", TPUSpec(cfg))
            else:
                self.assertEqual(simulate_moe(whole(), "tpu", TPUSpec(cfg)).resource_requirements["hbm"]["required_bytes"], capacity)

    def test_selection_only_bias_and_original_weight_source(self):
        for bias in (None, TensorSpec("bias", (6,), "FP32")):
            r = self.run_op(whole(bias=bias).router)
            select = r.native_report["schedule"]["nodes"][1]["native_report"]
            policy = select["selection_policy"]
            self.assertEqual(policy["weights_from"], "original_sigmoid_scores")
            self.assertEqual(policy["group_score"], "max" if bias is None else "top2_sum")
            phases = [s["phase"] for s in select["schedule"]["stages"]]
            self.assertEqual("selection_only.add_bias" in phases, bias is not None)
            self.assertIn("gather_original_sigmoid_scores", phases)
            self.assertEqual(select["workload"]["inputs"][0]["dtype"], "INT32")
            self.assertEqual(r.work_counts["logical"]["operation_counts"]["fp32_add"], 21 if bias is None else 48)

    def test_ungrouped_and_single_expert_reduce_edges(self):
        op = whole(shared=0, assignments=((1,),), groups=1, selected_groups=1, bias=None)
        r = self.run_op(op.router)
        self.assertEqual(r.work_counts["logical"]["operation_counts"], {"fp32_mul": 8, "fp32_exp": 6,
            "fp32_add": 6, "fp32_rcp": 7, "fp32_max": 5})
        combined = self.run_op(op.combine())
        self.assertEqual(combined.work_counts["logical"]["operation_counts"], {"fp32_mul": 5})
        router = MoERouter(TensorSpec("x", (1, 5), "INT8"), TensorSpec("w", (5, 1), "INT8"), 1)
        self.assertNotIn("fp32_max", self.run_op(router).work_counts["logical"]["operation_counts"])

    def test_batch_scaling_and_distribution_scenario_not_numerical_routing(self):
        a = self.run_op(whole(assignments=((0, 2),)))
        b = self.run_op(whole(assignments=((0, 2), (0, 2))))
        for key in ("useful_macs", "operations", "output_bytes"):
            self.assertEqual(b.work_counts["logical"][key], 2*a.work_counts["logical"][key])
        self.assertEqual(b.native_report["routing"]["counts"], [2, 0, 2, 0, 0, 0])
        self.assertFalse(b.native_report["routing"]["numeric_routing_verified"])
        self.assertTrue(b.native_report["routing"]["router_work_included"])

    def test_resident_reservations_striping_and_no_hbm(self):
        for arch in ("tsp", "tsp_cim"):
            r = self.run_op(arch=arch, mem_slices=[0, 1, 2], streams=2, reserved_bytes=17, hemisphere=1, tile_elements=3)
            life = r.native_report["memory"]["lifetime_validation"]
            self.assertEqual(life["persistent_allocations"]["reserved"], 18)
            self.assertTrue(all(v % 3 == 0 for v in life["persistent_allocations"].values()))
            self.assertEqual(actions(r, "hbm_read_bytes"), 0)
            self.assertEqual(r.boundary["selected_hemisphere"], 1)
            self.assertEqual(max(s["peak_bytes"] for s in life["steps"]), r.resource_requirements["mem"]["required_bytes"])

    def test_local_workspace_capacity_and_combine_tile_fallback(self):
        cfg = raw("tpu")
        cfg["parameters"]["vmem_usable_bytes"]["value"] = 71
        with self.assertRaises(ValueError):
            simulate_moe_router(whole().router, "tpu", TPUSpec(cfg))
        cfg["parameters"]["vmem_usable_bytes"]["value"] = 30
        r = simulate_moe_combine(whole().combine(), "tpu", TPUSpec(cfg), tile_elements=100, include_details=True)
        self.assertEqual(r.native_report["schedule"]["nodes"][0]["native_report"]["execution_contract"]["selected_tile_elements"], 2)

    def test_profiles_unknown_cost_count_and_timing_failure(self):
        block = copy.deepcopy(load_block_execution().config)
        block["parameters"]["fp32_rcp_cost"].update(value=None, status="unknown")
        self.run_op(whole().router, block_execution=BlockExecution(block), mode="count")
        with self.assertRaises(ValueError):
            self.run_op(whole().router, block_execution=BlockExecution(block), mode="timing")
        baseline = self.run_op(mode="timing")
        block["parameters"]["fp32_rcp_cost"].update(value=700, status="assumed")
        slow = self.run_op(mode="timing", block_execution=BlockExecution(block))
        self.assertEqual(baseline.action_ledger, slow.action_ledger)
        self.assertGreater(slow.cycle_count, baseline.cycle_count)

    def test_nested_native_actions_keep_counts_roles_and_unique_attributes(self):
        r = self.run_op()
        standalone = simulate_moe_experts(whole().experts, "tpu", self.hardware["tpu"], include_details=True)
        by_id = {a.id: a for a in r.action_ledger.entries}
        for a in standalone.action_ledger.entries:
            b = by_id["experts."+a.id]
            self.assertEqual((a.count, a.role), (b.count, b.role))
            self.assertEqual(b.summarizes, tuple("experts."+s for s in a.summarizes))
            self.assertEqual(len(b.attributes), len(dict(b.attributes)))

    def test_internal_names_and_reservation_labels_do_not_alias_user_tensors(self):
        op = whole()
        renamed = replace(op.experts, gate_weight=replace(op.experts.gate_weight, name="router.logits"))
        r = self.run_op(replace(op, experts=renamed))
        gate_output = r.native_report["schedule"]["nodes"][0]["operator"]["outputs"][0]["name"]
        self.assertNotEqual(gate_output, "router.logits")
        router = replace(op.router, weight=replace(op.router.weight, name="reserved_bytes"))
        self.assertEqual(self.run_op(router).resource_requirements, self.run_op(op.router).resource_requirements)

    def test_selection_comparison_cost_is_explicit_and_missing_clock_rejects_timing(self):
        profile = copy.deepcopy(load_vector_execution("tpu").config)
        # Composite parameters carry separate provenance; hardware values stay fixed.
        profile["composite"]["parameters"]["fp32_max_cost"]["value"] = 1000000
        slow = self.run_op(whole().router, mode="timing", execution=VectorExecution(profile, "tpu"))
        baseline = self.run_op(whole().router, mode="timing")
        self.assertEqual(baseline.action_ledger, slow.action_ledger)
        self.assertGreater(slow.cycle_count, baseline.cycle_count)
        cfg = raw("tpu")
        cfg["parameters"]["clock_hz"].update(value=None, status="unknown")
        # Per-second memory services require a clock: missing timing data must fail,
        # while pure work/capacity remains available.
        simulate_moe(whole(), "tpu", TPUSpec(cfg), mode="count")
        with self.assertRaises(ValueError):
            simulate_moe(whole(), "tpu", TPUSpec(cfg), mode="timing")

    def test_strict_shapes_names_groups_and_scenarios(self):
        r, c = whole().router, whole().combine()
        for change in ({"top_k": True}, {"top_k": 5}, {"groups": 4}, {"selected_groups": 4},
                       {"route_scale": float("nan")}, {"route_scale": 0}, {"output_names": ["x", "w"]},
                       {"bias": TensorSpec("bias", (5,), "FP32")}, {"output_names": ["same", "same"]},
                       {"weight": TensorSpec("w", (6, 5), "INT8")}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(r, **change)
        for change in ({"weights": TensorSpec("w", (3, 2, 5), "FP32")}, {"output_dtype": "INT32"},
                       {"output_name": c.experts.name}, {"shared": TensorSpec("s", (2, 5), "FP32")}):
            with self.assertRaises(ValueError):
                replace(c, **change)
        with self.assertRaises(ValueError):
            whole(selected_groups=1)
        with self.assertRaises(ValueError):
            replace(whole(), router=replace(r, output_names=("wrong", "route_weights")))

    def test_public_option_validation(self):
        for kwargs in ({"mem_slices": [0]}, {"reserved_bytes": 2}, {"hemisphere": 1},
                       {"mode": "invalid"}, {"tile_elements": 0}, {"_parent_bytes": 1}, {"block_execution": {}}):
            with self.assertRaises(ValueError):
                self.run_op(**kwargs)
        with self.assertRaises(ValueError):
            simulate_moe_experts(whole().experts, "tpu", self.hardware["tpu"], _parent_bytes=1)

    def test_parser_fixtures_and_reject_unknown_or_inconsistent_contract(self):
        for kind in ("moe_router", "moe_combine", "moe"):
            op = parse_moe_block_workload(config(kind))
            self.assertEqual(op.operation, kind)
            for change in ({"schema_version": True}, {"unknown": 1}, {"description": ""}):
                with self.assertRaises(ValueError):
                    parse_moe_block_workload({**config(kind), **change})
        cfg = config("moe")
        cfg["router"]["output_names"] = ["bad", "route_weights"]
        with self.assertRaises(ValueError):
            parse_moe_block_workload(cfg)


if __name__ == "__main__":
    unittest.main()

# Method scope

Primary method: supervised behavior cloning and DAgger. Separate KAN actors and optimizers are retained. CBF, spline planning and CAN execution have separate focused repositories.

Primary source entries:

- `scripts/closed_loop_smoke.py`
- `scripts/collect_dagger.py`
- `scripts/collect_dagger_randomized.py`
- `scripts/generate_stage1_data.py`
- `scripts/generate_stage2_batch.py`
- `scripts/generate_stage2_data.py`
- `scripts/generate_tiny_data.py`
- `scripts/run_dagger_h64.py`
- `scripts/tiny_overfit.py`
- `scripts/train_all_kan_actor.py`
- `scripts/train_curriculum.py`
- `scripts/train_dagger_h64.py`
- `scripts/train_h128.py`
- `scripts/train_left_collision_head.py`
- `scripts/train_left_h64.py`
- `scripts/train_paper_matched_actor.py`
- `scripts/train_parallel_h64.py`
- `scripts/train_right_h64.py`
- `scripts/train_single_h64.py`
- `scripts/train_stage2_h64.py`
- `src/data/__init__.py`
- `src/data/collision_counterfactual.py`
- `src/data/dagger_buffer.py`
- `src/data/dataset.py`
- `src/data/generator.py`
- `src/data/left_collision_boundary.py`
- `src/data/normalization.py`
- `src/data/splits.py`
- `src/data/stage2_generator.py`
- `src/data/strict_expert.py`
- `src/env/__init__.py`
- `src/env/collision_monitor.py`
- `src/env/dual_arm_scene.py`
- `src/env/state_reader.py`
- `src/evaluation/__init__.py`
- `src/evaluation/collision_boundary.py`
- `src/evaluation/goal_manager.py`
- `src/evaluation/metrics.py`
- `src/evaluation/plotting.py`
- `src/evaluation/rollout.py`
- `src/models/__init__.py`
- `src/models/collision_head.py`
- `src/models/full_kan.py`
- `src/models/kan.py`
- `src/models/mlp.py`
- `src/models/tokenizer.py`
- `src/models/trajectory_model.py`
- `src/models/transformer.py`
- `src/training/__init__.py`
- `src/training/dagger.py`
- `src/training/losses.py`
- `src/training/trainer.py`

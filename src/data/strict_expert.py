"""Strict current-scene collection helpers; never repair or shorten a task."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from src.data.generator import Stage1DataGenerator
from src.data.dagger_buffer import ActionHistory
from src.geometry.periodic import periodic_delta
from src.planning.collision_aware_teacher import PhysXSingleArmTeacher
from src.planning.rrt_connect import interpolate_periodic
from src.planning.trajectory_retimer import retime_rrt_path, select_training_frames

BANDS = {'short': (.35, 1.5), 'medium': (1.5, 3.5), 'long': (3.5, 7.0)}

def band_accepts(start, goal, band):
    distance = float(np.linalg.norm(periodic_delta(goal, start)))
    lo, hi = BANDS[band]
    return lo <= distance < hi

def allocate_quotas(total):
    if total < 20 or total % 20:
        raise ValueError('episodes must be a positive multiple of 20, at least 20')
    # 20/45/35 distance mix, 40% independently verified non-direct tasks.
    return {'short_direct':total//20*4,
            'medium_direct':total//20*4, 'medium_detour':total//20*5,
            'long_direct':total//20*4, 'long_detour':total//20*3}

def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8'); tmp.replace(path)

class StableOracle:
    """Two observations of one pose; disagreement is UNKNOWN, never safe."""
    def __init__(self, scene):
        self.scene = scene
        self.zero = torch.zeros(8, device=scene.device)
        self.queries = 0
        self.disagreements = 0

    def label(self, q):
        values = []
        for steps in (3, 2):
            self.scene.teleport(torch.as_tensor(q, dtype=torch.float32, device=self.scene.device), self.zero, settle_steps=steps)
            lab = self.scene.monitor.labels()
            values.append(np.asarray([lab['left_self'], lab['left_environment'] or lab['inter_arm']], dtype=np.uint8))
        self.queries += 1
        uncertain = values[0] != values[1]
        self.disagreements += int(uncertain.any())
        return np.where(uncertain, 2, values[1]).astype(np.uint8)

    def safe(self, q): return bool(np.all(self.label(q) == 0))

class StrictTeacher(PhysXSingleArmTeacher):
    def __init__(self, scene, oracle, seed, iterations):
        super().__init__(scene, seed=seed, max_iterations=iterations)
        self.oracle = oracle
        self.query_budget = 4000

    def state_valid(self, q):
        key = np.asarray(q, np.float32).tobytes()
        if key not in self._validity_cache:
            if len(self._validity_cache) >= self.query_budget:
                raise RuntimeError('Expert planning exhausted 4000 unique configuration checks; task not simplified')
            # Search is a prefilter; all accepted states/edges are independently
            # rechecked by the 3+2-step stable oracle in replay_expert.
            self._validity_cache[key] = not any(self.labels(q, settle_steps=2))
        return self._validity_cache[key]

def replay_expert(scene, oracle, positions, seed, episode, scene_hash):
    """Fresh FK/history, dense action-edge checks, native H128, exact endpoints."""
    path = retime_rrt_path(positions, max_action_rad=np.deg2rad(3.5), terminal_transitions=12)
    if not np.allclose(periodic_delta(path[-1], positions[-1]), 0, atol=1e-5):
        raise RuntimeError('Retimer changed the requested goal')
    start, goal = path[0], path[-1]
    if not oracle.safe(goal): raise RuntimeError('Goal failed independent replay audit')
    goal_snapshot = scene.state.snapshot('left')
    if not oracle.safe(start): raise RuntimeError('Start failed independent replay audit')
    start_snapshot = scene.state.snapshot('left')
    builder = Stage1DataGenerator(scene, '.', history_length=128, seed=seed)
    history = ActionHistory(128); samples={'left':[]}; distances=[]; max_error=0.
    for step, (q, next_q) in enumerate(zip(path[:-1], path[1:])):
        if not oracle.safe(q): raise RuntimeError(f'Unsafe replay state {step}')
        current = scene.state.snapshot('left')
        error = float(np.max(np.abs(periodic_delta(current['q'][0].cpu().numpy(), q))))
        max_error=max(max_error,error)
        if error > 1e-5: raise RuntimeError('Kinematic joint write error')
        action = periodic_delta(next_q, q).astype(np.float32)
        # Retime can cross RRT corners; verify the actually commanded segment.
        for middle in interpolate_periodic(q, next_q, .025)[1:]:
            if not oracle.safe(middle): raise RuntimeError(f'Unsafe executed edge {step}')
        builder._append_sample(samples,'left',start,goal,start_snapshot,goal_snapshot,current,history,action,episode,step)
        row=samples['left'][-1]
        row.update(collision_query_action=action, action_loss_mask=np.float32(1),
                   simulation_mode=np.str_('kinematic'), scene_sha256=np.str_(scene_hash),
                   collision_mode=np.str_(scene.collision_mode),
                   sample_kind=np.str_('fresh_strict_expert'), q_next=next_q.copy())
        distance=float(torch.linalg.vector_norm(current['tcp_pose9_local'][0,:3]-goal_snapshot['tcp_pose9_local'][0,:3]))
        row['tcp_goal_distance_m']=np.float32(distance); distances.append(distance)
        history.append(action)
    selected = select_training_frames(periodic_delta(path[1:], path[:-1]), np.asarray(distances), max_gap=2)
    arrays={k:np.stack([samples['left'][i][k] for i in selected]) for k in samples['left'][0]}
    return arrays, {'control_frames':len(path)-1,'samples':len(selected),'max_q_write_error_rad':max_error,
                    'precision':{str(mm):int(np.sum(arrays['tcp_goal_distance_m'] < mm/1000)) for mm in (30,10,3)}}

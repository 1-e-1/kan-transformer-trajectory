"""PhysX contact reports with explicit collider pairs and impulses."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from omni.physx import get_physx_simulation_interface
from omni.physx.bindings._physx import ContactEventType
from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdPhysics
from src.geometry.contact_labels import penetrating_contact_index


@dataclass(frozen=True)
class ContactRecord:
    collider_a: str
    collider_b: str
    impulse: float
    position: tuple[float, float, float] | None
    kind: str


class CollisionMonitor:
    def __init__(self, stage: Usd.Stage, left_prefix: str, right_prefix: str, left_tool: str, right_tool: str, collision_mode: str = 'impulse'):
        if collision_mode not in ('impulse','separation'):
            raise ValueError('collision_mode must be impulse or separation')
        self.collision_mode = collision_mode
        self.stage = stage
        self.left_prefix = left_prefix
        self.right_prefix = right_prefix
        self.left_tool = left_tool
        self.right_tool = right_tool
        self.active: dict[tuple[str, str], ContactRecord] = {}
        self.recent: list[ContactRecord] = []
        self._subscription = None
        self._tensor_views = []
        self._tensor_filter_paths: dict[int, list[str]] = {}

    def _side(self, path: str) -> str | None:
        if path.startswith(self.left_prefix) or path.startswith(self.left_tool):
            return "left"
        if path.startswith(self.right_prefix) or path.startswith(self.right_tool):
            return "right"
        return None

    def _kind(self, a: str, b: str) -> str:
        side_a, side_b = self._side(a), self._side(b)
        if side_a is not None and side_a == side_b:
            return f"{side_a}_self"
        if {side_a, side_b} == {"left", "right"}:
            return "inter_arm"
        if side_a is not None:
            return f"{side_a}_environment"
        if side_b is not None:
            return f"{side_b}_environment"
        return "environment"

    def install(self) -> None:
        # Contact reporting is authored only into the session layer and never saved to the original USD.
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            for prim in self.stage.TraverseAll():
                path = str(prim.GetPath())
                if prim.HasAPI(UsdPhysics.RigidBodyAPI) and self._side(path) is not None:
                    api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                    api.CreateThresholdAttr().Set(0.0)
        self._subscription = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contacts)

    def begin_step(self) -> None:
        self.recent.clear()

    def initialize_tensor_views(self, physics_sim_view, arm_body_paths: list[str], fixed_body_paths: list[str] | None = None) -> None:
        """Create pair-filtered PhysX tensor views as the headless authoritative source."""
        filters = list(dict.fromkeys([*arm_body_paths, self.left_tool, self.right_tool, *(fixed_body_paths or [])]))
        arm_names = "|".join([f"a{i}" for i in range(1, 9)] + [f"b{i}" for i in range(1, 9)])
        # Arm bodies are direct children; tool bodies require a second sensor view.
        self._tensor_views = [
            physics_sim_view.create_rigid_contact_view(
                f"/World/Base_*/zbot_8s_new_*/({arm_names})", filter_patterns=filters, max_contact_data_count=8192
            ),
            physics_sim_view.create_rigid_contact_view(
                "/World/(cixi_01|cixi_00)", filter_patterns=filters, max_contact_data_count=1024
            ),
        ]
        for view in self._tensor_views:
            if view.sensor_count == 0 or view.filter_count == 0:
                raise RuntimeError(
                    f"Invalid rigid contact view: sensors={view.sensor_count}, filters={view.filter_count}"
                )
            if view.filter_count != len(filters):
                raise RuntimeError(
                    f"Contact filter resolution mismatch: filter_count={view.filter_count}, configured={len(filters)}, backend_paths={view.filter_paths}"
                )
            self._tensor_filter_paths[id(view)] = filters

    def _on_contacts(self, headers, data) -> None:
        self._consume(headers, data)

    def poll(self) -> None:
        """Poll the immediate report too; headless Fabric may not dispatch callbacks."""
        headers, data = get_physx_simulation_interface().get_contact_report()
        self._consume(headers, data)
        if self._tensor_views:
            self._poll_tensor_views()

    def _poll_tensor_views(self) -> None:
        tensor_records: dict[tuple[str, str], ContactRecord] = {}
        for view in self._tensor_views:
            if self.collision_mode == 'separation':
                tensor_records.update(self._geometric_records(view))
                continue
            forces = view.get_contact_force_matrix(dt=1.0).reshape(view.sensor_count, view.filter_count, 3)
            magnitudes = torch.linalg.vector_norm(forces, dim=-1)
            if self.collision_mode == 'impulse' and not torch.any(magnitudes > 1e-6):
                continue
            _, points, _, separations, counts, starts = view.get_contact_data(dt=1.0)
            counts = counts.reshape(view.sensor_count, view.filter_count)
            starts = starts.reshape(view.sensor_count, view.filter_count)
            nonzero = torch.nonzero(counts > 0 if self.collision_mode == 'separation' else magnitudes > 1e-6, as_tuple=False)
            separations = separations.reshape(-1)
            sensor_paths = [str(path) for path in view.sensor_paths]
            # Backend filter_paths exposes one entry per pattern group on this API;
            # exact configured filters retain the actual force-matrix column order.
            filter_paths = self._tensor_filter_paths[id(view)]
            for sensor_index, filter_index in nonzero.cpu().tolist():
                a, b = sensor_paths[sensor_index], filter_paths[filter_index]
                if a == b:
                    continue
                pair = tuple(sorted((a, b)))
                count = int(counts[sensor_index, filter_index].item())
                start = int(starts[sensor_index, filter_index].item())
                if self.collision_mode == 'separation':
                    # Aggregate impulse depends on solver history and can cancel.
                    # Positive-gap proximity contacts are not geometric overlap.
                    point_index = penetrating_contact_index(separations,start,count)
                    if point_index is None:
                        continue
                    start = point_index
                point = tuple(float(x) for x in points[start].cpu().tolist()) if count > 0 else None
                tensor_records[pair] = ContactRecord(
                    a,
                    b,
                    float(magnitudes[sensor_index, filter_index].item()),
                    point,
                    self._kind(a, b),
                )
        # Tensor views are authoritative for arm-arm pairs. The dt=1 query returns
        # impulses directly (the API divides impulse by the requested dt).
        authoritative_kinds = {"left_self", "right_self", "inter_arm", "left_environment", "right_environment"}
        for pair in [pair for pair, record in self.active.items() if record.kind in authoritative_kinds]:
            self.active.pop(pair, None)
        self.active.update(tensor_records)
        self.recent.extend(tensor_records.values())

    def _geometric_records(self, view) -> dict:
        """Bulk transfers avoid one GPU synchronization per contact attribute."""
        forces=view.get_contact_force_matrix(dt=1.0).reshape(view.sensor_count,view.filter_count,3)
        magnitudes=torch.linalg.vector_norm(forces,dim=-1).cpu()
        _,points,_,separations,counts,starts=view.get_contact_data(dt=1.0)
        counts=counts.reshape(view.sensor_count,view.filter_count).cpu()
        starts=starts.reshape(view.sensor_count,view.filter_count).cpu()
        separations=separations.reshape(-1).cpu();points=points.cpu()
        sensors=list(map(str,view.sensor_paths));filters=self._tensor_filter_paths[id(view)]
        records={}
        for a,b in torch.nonzero(counts>0,as_tuple=False).tolist():
            left,right=sensors[a],filters[b]
            if left==right:continue
            index=penetrating_contact_index(separations,int(starts[a,b]),int(counts[a,b]))
            if index is None:continue
            records[tuple(sorted((left,right)))]=ContactRecord(left,right,float(magnitudes[a,b]),tuple(map(float,points[index].tolist())),self._kind(left,right))
        return records

    def _consume(self, headers, data) -> None:
        for header in headers:
            a = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
            b = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
            pair = tuple(sorted((a, b)))
            if header.type == ContactEventType.CONTACT_LOST:
                self.active.pop(pair, None)
                continue
            impulses = []
            positions = []
            begin = header.contact_data_offset
            end = begin + header.num_contact_data
            for index in range(begin, end):
                impulse = data[index].impulse
                try:
                    impulses.append(float(impulse.GetLength()))
                except AttributeError:
                    try:
                        impulses.append(float(sum(float(x) ** 2 for x in impulse) ** 0.5))
                    except TypeError:
                        impulses.append(float(impulse))
                positions.append(tuple(float(x) for x in data[index].position))
            record = ContactRecord(a, b, max(impulses, default=0.0), positions[0] if positions else None, self._kind(a, b))
            self.active[pair] = record
            self.recent.append(record)

    def forbidden_contacts(self, include_active: bool = True) -> list[ContactRecord]:
        records = list(self.active.values()) if include_active else self.recent
        return [
            record for record in records
            if record.kind in {"left_self", "right_self", "inter_arm", "left_environment", "right_environment"}
        ]

    def labels(self) -> dict[str, bool]:
        kinds = {record.kind for record in self.active.values()}
        return {
            "left_self": "left_self" in kinds,
            "right_self": "right_self" in kinds,
            "inter_arm": "inter_arm" in kinds,
            "left_environment": "left_environment" in kinds,
            "right_environment": "right_environment" in kinds,
        }

    def serializable_contacts(self) -> list[dict]:
        return [asdict(record) for record in self.active.values()]

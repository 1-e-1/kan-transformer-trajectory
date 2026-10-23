"""Kinematic contact classification without solver-impulse dependence."""
import torch

def penetrating_contact_index(separations, start, count):
    if count <= 0: return None
    patch=separations.reshape(-1)[start:start+count]
    if start<0 or len(patch)!=count or not torch.isfinite(patch).all():
        raise RuntimeError('Invalid contact separation buffer')
    index=int(torch.argmin(patch).item())
    return start+index if float(patch[index])<=0.0 else None

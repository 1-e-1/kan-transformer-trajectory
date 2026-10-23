"""All-dense KAN conversion, including attention Q/K/V and output projections.

New spline edges use an identity base branch so the original affine mapping is
exactly representable at initialization. This is a residual spline KAN, not a
claim that KAN contains no matrix multiplication. Existing SiLU-base KAN stays
unchanged. Attention softmax, normalization and embeddings are retained.
"""
import torch
from torch import nn
from .kan import BSplineKANLayer


def dense_kan(in_features, out_features, grid_size=4, spline_order=3):
    return BSplineKANLayer(in_features, out_features, grid_size, spline_order, base_activation='identity')


def replace_linears(module, grid_size=4, spline_order=3):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            layer=dense_kan(child.in_features,child.out_features,grid_size,spline_order).to(child.weight)
            with torch.no_grad():
                layer.base_weight.copy_(child.weight)
                layer.bias.copy_(child.bias) if child.bias is not None else layer.bias.zero_()
                layer.spline_weight.zero_()
            setattr(module,name,layer)
        else:replace_linears(child,grid_size,spline_order)
    return module


class KANSelfAttention(nn.Module):
    def __init__(self,width,heads,dropout,grid,order):
        super().__init__()
        assert width%heads==0
        self.heads=heads;self.width=width;self.dropout=dropout
        self.qkv=dense_kan(width,3*width,grid,order)
        self.out_proj=dense_kan(width,width,grid,order)

    def forward(self,x,mask):
        batch,length,width=x.shape
        q,k,v=self.qkv(x).reshape(batch,length,3,self.heads,width//self.heads).permute(2,0,3,1,4).unbind(0)
        bias=None if mask is None else torch.zeros(batch,1,1,length,device=x.device,dtype=x.dtype).masked_fill(mask[:,None,None,:],float('-inf'))
        attended=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=self.dropout if self.training else 0.0)
        return self.out_proj(attended.transpose(1,2).reshape(batch,length,width))


class KANTransformerLayer(nn.Module):
    def __init__(self,width,heads,ffn,dropout,grid,order):
        super().__init__()
        self.self_attn=KANSelfAttention(width,heads,dropout,grid,order)
        self.linear1=dense_kan(width,ffn,grid,order)
        self.linear2=dense_kan(ffn,width,grid,order)
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.dropout=nn.Dropout(dropout);self.dropout1=nn.Dropout(dropout);self.dropout2=nn.Dropout(dropout)

    def forward(self,x,mask):
        x=x+self.dropout1(self.self_attn(self.norm1(x),mask))
        return x+self.dropout2(self.linear2(self.dropout(torch.nn.functional.gelu(self.linear1(self.norm2(x))))))


class KANTransformerEncoder(nn.Module):
    def __init__(self,width,heads,layers,ffn,dropout,grid,order):
        super().__init__()
        self.layers=nn.ModuleList([KANTransformerLayer(width,heads,ffn,dropout,grid,order) for _ in range(layers)])
        self.norm=nn.LayerNorm(width)

    def forward(self,tokens,src_key_padding_mask=None):
        # Masked history cannot influence any valid token. At single-robot
        # inference, omit those padded tokens instead of evaluating large KAN
        # FFNs on them. Position encodings have already been added upstream.
        if not self.training and tokens.shape[0]==1 and src_key_padding_mask is not None:
            tokens=tokens[:,~src_key_padding_mask[0]]
            src_key_padding_mask=None
        for layer in self.layers:tokens=layer(tokens,src_key_padding_mask)
        return self.norm(tokens)


def warm_start_from_dense(student,teacher):
    """Copy every shared tensor and convert every affine map, with an audit."""
    original=teacher.state_dict();target=student.state_dict()
    copied={key:value for key,value in original.items() if key in target and target[key].shape==value.shape}
    student.load_state_dict(copied,strict=False)
    modules=dict(teacher.named_modules());converted=[];covered=set(copied)
    with torch.no_grad():
        for name,layer in student.named_modules():
            if not isinstance(layer,BSplineKANLayer) or layer.base_activation!='identity':continue
            source=modules.get(name)
            if isinstance(source,nn.Linear):
                weight,bias=source.weight,source.bias
                covered.add(name+'.weight')
                if bias is not None:covered.add(name+'.bias')
            elif name.endswith('.qkv') and isinstance(modules.get(name[:-4]),nn.MultiheadAttention):
                source=modules[name[:-4]];weight,bias=source.in_proj_weight,source.in_proj_bias
                covered.add(name[:-4]+'.in_proj_weight');covered.add(name[:-4]+'.in_proj_bias')
            else:raise ValueError(f'No dense teacher for {name}')
            layer.base_weight.copy_(weight)
            layer.bias.copy_(bias) if bias is not None else layer.bias.zero_()
            layer.spline_weight.zero_();converted.append(name)
    missing=set(original)-covered
    if missing:raise ValueError(f'Unmapped teacher tensors: {sorted(missing)}')
    assert_no_dense(student)
    return {'converted_maps':converted,'shared_tensors':len(copied),'unmapped_teacher_tensors':[]}


def assert_no_dense(model):
    remaining=[name for name,module in model.named_modules() if isinstance(module,(nn.Linear,nn.MultiheadAttention))]
    if remaining:raise AssertionError(f'Remaining dense modules: {remaining}')
    return sum(isinstance(m,BSplineKANLayer) for m in model.modules())

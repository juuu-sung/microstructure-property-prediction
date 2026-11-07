"""Process-local memory scheduling; no feature, support, weight or loss changes."""
POLICY = {
    'version': 2,
    'saved_backward_tensors': 'torch.autograd.graph.save_on_cpu(pin_memory=False)',
    'best_weights': 'CPU exact-dtype copies',
    'independent_batch_chunk': 32,
    'pointwise_mlp_row_chunk': 32,
    'suboperation_checkpoint': 'nonreentrant, preserve RNG',
    'prediction_support_chunk': 0,
    'autogluon_features': 'IdentityFeatureGenerator; preserve486 columns',
    'automatic_context_reduction': False,
}


def install():
    import torch
    import einops
    from torch.utils.checkpoint import checkpoint
    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D, Layer, MultiheadAttention
    from mitra_finetune import speed
    if getattr(Tab2D, '_raw486_memory_installed', False):
        raise RuntimeError('Raw486 memory runtime already installed')
    original = Tab2D.forward

    def call(fn, *args):
        if torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    attention = MultiheadAttention.forward
    def batched_attention(self, query, key, value, cu_seqlens_q=None,
                          cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None):
        if cu_seqlens_q is not None or cu_seqlens_k is not None:
            raise ValueError('Raw486 runtime requires the pinned dense SDPA path')
        # Split independent batch entries, NEVER the keys/values sequence.
        chunk = POLICY['independent_batch_chunk']
        def part(q,k,v):
            return attention(self,q,k,v)
        return torch.cat([call(part, query[a:a+chunk], key[a:a+chunk], value[a:a+chunk])
                          for a in range(0, query.shape[0], chunk)], dim=0)
    MultiheadAttention.forward = batched_attention

    def mlp(x, norm, first, second):
        def part(t):
            return second(torch.nn.functional.gelu(first(norm(t))))
        chunk = POLICY['pointwise_mlp_row_chunk']
        return torch.cat([call(part,x[:,a:a+chunk]) for a in range(0,x.shape[1],chunk)],dim=1)

    def layer_forward(self, support, query__, padder_support, padder_query__, batch_size=None,
                      padding_obs_support=None, padding_obs_query__=None, padding_features=None):
        if padder_support is not None or padder_query__ is not None:
            raise ValueError('Raw486 runtime requires dense tensors')
        b = support.shape[0]
        s=einops.rearrange(self.layer_norm1(support),'b s f d -> (b f) s d')
        q=einops.rearrange(self.layer_norm1(query__),'b s f d -> (b f) s d')
        sa=self.attention1(s,s,s); qa=self.attention1(q,s,s)
        support=support+einops.rearrange(sa,'(b f) s d -> b s f d',b=b)
        query__=query__+einops.rearrange(qa,'(b f) s d -> b s f d',b=b)
        del s,q,sa,qa
        support=support+mlp(support,self.layer_norm2,self.linear1,self.linear2)
        query__=query__+mlp(query__,self.layer_norm2,self.linear1,self.linear2)
        s=einops.rearrange(self.layer_norm3(support),'b s f d -> (b s) f d')
        q=einops.rearrange(self.layer_norm3(query__),'b s f d -> (b s) f d')
        sa=self.attention2(s,s,s); qa=self.attention2(q,q,q)
        support=support+einops.rearrange(sa,'(b s) f d -> b s f d',b=b)
        query__=query__+einops.rearrange(qa,'(b s) f d -> b s f d',b=b)
        del s,q,sa,qa
        support=support+mlp(support,self.layer_norm4,self.linear3,self.linear4)
        query__=query__+mlp(query__,self.layer_norm4,self.linear3,self.linear4)
        return support,query__
    Layer.forward=layer_forward

    def forward(self, *args, **kwargs):
        if torch.is_grad_enabled():
            with torch.autograd.graph.save_on_cpu(pin_memory=False):
                return original(self, *args, **kwargs)
        return original(self, *args, **kwargs)

    Tab2D.forward = forward
    Tab2D._raw486_memory_installed = True

    class CPUCheckpoint:
        def reset(self, net):
            self.curr_best_loss = float('inf')
            self.best_model = {k: v.detach().to('cpu', copy=True) for k, v in net.state_dict().items()}

        def __call__(self, net, loss):
            if loss < self.curr_best_loss:
                self.curr_best_loss = loss
                with torch.no_grad():
                    for k, v in net.state_dict().items():
                        self.best_model[k].copy_(v)

        def set_to_best(self, net):
            net.load_state_dict(self.best_model)

    speed.DeviceCheckpoint = CPUCheckpoint

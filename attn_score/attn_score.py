import torch
import attn_score_cuda

class AttentionScore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, att_query, att_keys, bias, linear_att, tile_x=8, tile_y=8, ilp=1):
        score = attn_score_cuda.forward(att_query, att_keys, bias, linear_att)
        ctx.save_for_backward(att_query, att_keys, linear_att, torch.IntTensor([ilp]))
        return score

    @staticmethod
    def backward(ctx, grad_output):
        att_query, att_keys, linear_att, ilp = ctx.saved_variables
        grad_query, grad_kerys, grad_bias, grad_linear_att = attn_score_cuda.backward(grad_output, att_query, att_keys, bias, linear_att)
        return grad_query, grad_kerys, grad_bias, grad_linear_att


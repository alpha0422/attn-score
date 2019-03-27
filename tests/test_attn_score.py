import os
import time
import torch
import unittest
import attn_score
import attn_score_cuda

def calc_score_ref(att_query, att_keys, normalize_bias, linear_att):
    b, t_k, n = att_keys.size()
    t_q = att_query.size(1)

    att_query = att_query.unsqueeze(2).expand(b, t_q, t_k, n)
    att_keys = att_keys.unsqueeze(1).expand(b, t_q, t_k, n)

    sum_qk = att_query + att_keys
    sum_qk = sum_qk + normalize_bias

    out = torch.tanh(sum_qk).matmul(linear_att)
    return out

calc_score_jit = torch.jit.script(calc_score_ref)
calc_score_tst = attn_score.AttentionScore.apply

class AttentionScoreTest(unittest.TestCase):
    def setUp(self):
        torch.cuda.manual_seed(1234)

    def gen_test_inputs(self):
        options = {'device': 'cuda:0', 'dtype': torch.float16, 'requires_grad': True}
        batch_size, t_q, t_k, hidden_size = 32, 33, 34, 1024

        grads = torch.randn([batch_size, t_q, t_k], **options)

        att_query_ref = torch.randn([batch_size, t_q, hidden_size], **options)
        att_keys_ref = torch.randn([batch_size, t_k, hidden_size], **options)
        normalize_bias_ref = torch.randn([hidden_size], **options)
        linear_att_ref = torch.randn([hidden_size], **options)

        att_query_tst = att_query_ref.clone().detach().requires_grad_()
        att_keys_tst = att_keys_ref.clone().detach().requires_grad_()
        normalize_bias_tst = normalize_bias_ref.clone().detach().requires_grad_()
        linear_att_tst = linear_att_ref.clone().detach().requires_grad_()

        return (att_query_ref, att_keys_ref, normalize_bias_ref, linear_att_ref), \
            (att_query_tst, att_keys_tst, normalize_bias_tst, linear_att_tst), grads

    def test_attn_score_function(self):
        inputs_ref, inputs_tst, grads = self.gen_test_inputs()

        for i in range(4):
            score_ref = calc_score_ref(*inputs_ref)
            score_tst = calc_score_ref(*inputs_tst)

            self.assertTrue(torch.allclose(score_ref, score_tst))

            #score_ref.backward(grads)
            #score_tst.backward(grads)
           
            #for t_ref, t_tst in zip(inputs_ref, inputs_tst):
            #    self.assertTrue(torch.allclose(t_ref.grad, t_tst.grad))

    def test_attn_score_perf(self):
        num_iters = 1000
        inputs_ref, inputs_tst, grads = self.gen_test_inputs()

        torch.cuda.synchronize()
        ts_ref = time.time()
        for i in range(num_iters):
            score_ref = calc_score_ref(*inputs_ref)
            #score_ref.backward(grads)
        torch.cuda.synchronize()
        td_ref = time.time()

        torch.cuda.synchronize()
        ts_jit = time.time()
        for i in range(num_iters):
            score_jit = calc_score_jit(*inputs_ref)
            #score_jit.backward(grads)
        torch.cuda.synchronize()
        td_jit = time.time()

        torch.cuda.synchronize()
        ts_tst = time.time()
        for i in range(num_iters):
            score_tst = calc_score_tst(*inputs_tst)
            #score_tst.backward(grads)
        torch.cuda.synchronize()
        td_tst = time.time()

        print("Ref time {:.2f} s elapsed for {} iterations".format(
            td_ref - ts_ref, num_iters))
        print("JIT time {:.2f} s elapsed for {} iterations".format(
            td_jit - ts_jit, num_iters))
        print("Tst time {:.2f} s elapsed for {} iterations".format(
            td_tst - ts_tst, num_iters))

if __name__ == '__main__':
    script_path = os.path.dirname(os.path.realpath(__file__))
    unittest.main()


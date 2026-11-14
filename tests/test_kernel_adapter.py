"""Tests for the psi adapter delta_t(x) = A_t psi_t(B_t x) (models/kernel_adapter.py). Small CPU tensors only."""
import math

import pytest
import torch

from models.kernel_adapter import KINDS, TaskFeatureMap, psi_config, psi_tag

vit_inflora = pytest.importorskip('models.vit_inflora')


def _attn(kind, dim=24, heads=2, r=3, n_tasks=3, m=64, sigma_mult=1.0):
    torch.manual_seed(0)
    psi = None if kind is None else {'kind': kind, 'm': m, 'sigma_mult': sigma_mult}
    attn = vit_inflora.Attention_LoRA(dim, num_heads=heads, qkv_bias=True, r=r, n_tasks=n_tasks, psi=psi)
    attn.init_param()
    for t in range(n_tasks):  # orthonormal rows, like InfLoRA's U[:, :r].T / sqrt(3)
        q, _ = torch.linalg.qr(torch.randn(dim, r))
        attn.lora_A_k[t].weight.data.copy_(q.T / math.sqrt(3))
        attn.lora_A_v[t].weight.data.copy_(q.T / math.sqrt(3))
        attn.lora_B_k[t].weight.data.normal_()
        attn.lora_B_v[t].weight.data.normal_()
    return attn


@pytest.mark.parametrize('kind', KINDS)
def test_psi_of_zero_is_zero(kind):
    psi = TaskFeatureMap(kind, r=4, m=32, n_tasks=2)
    assert torch.equal(psi(torch.zeros(5, 2, 4)), torch.zeros(5, 2, 32))


@pytest.mark.parametrize('kind', ['linear', 'relu'])
def test_psi_preserves_scale(kind):
    torch.manual_seed(0)
    psi = TaskFeatureMap(kind, r=6, m=8192, n_tasks=1)
    u = torch.randn(200, 1, 6)
    ratio = psi(u).pow(2).sum(-1) / u.pow(2).sum(-1)
    assert abs(ratio.mean().item() - 1) < 0.05


def test_rff_is_linear_scale_for_small_inputs_and_saturates_for_large():
    torch.manual_seed(0)
    psi = TaskFeatureMap('rff', r=6, m=8192, n_tasks=1)
    psi.sigma[0] = 10.0
    u = torch.randn(200, 1, 6)
    small = psi(0.1 * u).pow(2).sum(-1) / (0.1 * u).pow(2).sum(-1)
    assert abs(small.mean().item() - 1) < 0.05
    # ||psi(u)||^2 -> 2 s^2 (1 - exp(-||u||^2 / 2 s^2)) <= 2 s^2: bounded, unlike a linear map
    big = psi(1000 * u).pow(2).sum(-1)
    assert big.max().item() < 2.2 * 10.0 ** 2


def test_rff_inner_product_approximates_centred_gaussian_kernel():
    torch.manual_seed(0)
    s = 2.0
    psi = TaskFeatureMap('rff', r=5, m=16384, n_tasks=1)
    psi.sigma[0] = s
    a, b = torch.randn(1, 5), torch.randn(1, 5)
    k = lambda x, y: math.exp(-(x - y).pow(2).sum().item() / (2 * s ** 2))
    z = torch.zeros(1, 5)
    want = s ** 2 * (k(a, b) - k(a, z) - k(z, b) + 1)
    got = (psi(a[:, None]) * psi(b[:, None])).sum().item()
    assert abs(got - want) < 0.05 * s ** 2


def test_inputs_orthogonal_to_every_B_t_get_no_update():
    for kind in KINDS:
        attn = _attn(kind)
        basis = torch.cat([attn.lora_A_k[t].weight for t in range(3)]).T  # (dim, 3r) spans all B_t rows
        q, _ = torch.linalg.qr(basis)
        x = torch.randn(2, 5, 24)
        x = x - x @ q @ q.T
        delta = attn._psi_delta(x, 2, attn.lora_A_k, attn.lora_B_k, attn.psi_k)
        assert delta.abs().max().item() < 1e-5, kind


def test_batched_delta_equals_sum_over_tasks():
    attn = _attn('rff')
    attn.psi_k.sigma.copy_(torch.tensor([0.5, 1.0, 2.0]))
    x = torch.randn(2, 5, 24)
    got = attn._psi_delta(x, 2, attn.lora_A_k, attn.lora_B_k, attn.psi_k)
    want = 0
    for t in range(3):
        u = attn.lora_A_k[t](x)
        s = attn.psi_k.sigma[t]
        z = u @ attn.psi_k.omega[t].T / s
        feats = s * math.sqrt(2 / 64) * torch.cat([torch.cos(z) - 1, torch.sin(z)], -1)
        want = want + attn.lora_B_k[t](feats)
    assert torch.allclose(got, want, atol=1e-5)


def test_linear_psi_is_still_a_rank_r_linear_adapter():
    attn = _attn('linear')
    x = torch.randn(2, 5, 24)
    got = attn._psi_delta(x, 0, attn.lora_A_k, attn.lora_B_k, attn.psi_k)
    w = attn.lora_B_k[0].weight @ attn.psi_k.omega[0] / math.sqrt(64) @ attn.lora_A_k[0].weight
    assert torch.allclose(got, x @ w.T, atol=1e-5)
    assert torch.linalg.matrix_rank(w).item() == 3


def test_no_psi_keeps_the_original_merged_forward():
    attn = _attn(None)
    assert attn.psi_k is None and attn.lora_B_k[0].weight.shape == (24, 3)
    x = torch.randn(2, 5, 24)
    out = attn(x, 1)
    w_k = sum(attn.lora_B_k[t].weight @ attn.lora_A_k[t].weight for t in range(2))
    w_v = sum(attn.lora_B_v[t].weight @ attn.lora_A_v[t].weight for t in range(2))
    qkv = attn.qkv(x).reshape(2, 5, 3, 2, 12).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    k = k + (x @ w_k.T).reshape(2, 5, 2, 12).permute(0, 2, 1, 3)
    v = v + (x @ w_v.T).reshape(2, 5, 2, 12).permute(0, 2, 1, 3)
    a = ((q @ k.transpose(-2, -1)) * attn.scale).softmax(-1)
    want = attn.proj((a @ v).transpose(1, 2).reshape(2, 5, 24))
    assert torch.allclose(out, want, atol=1e-5)


def test_calibrate_sets_bandwidth_from_second_moment():
    attn = _attn('rff', sigma_mult=0.5)
    x = torch.randn(400, 24) * torch.linspace(0.2, 3, 24)
    attn.cur_matrix = x.T @ x / len(x)  # what get_cur_feat accumulates
    attn.calibrate_psi(1)
    u = attn.lora_A_k[1](x)
    want = 0.5 * u.pow(2).sum(-1).mean().sqrt()
    assert torch.isclose(attn.psi_k.sigma[1], want, rtol=1e-4)
    assert attn.psi_k.sigma[0].item() == 1.0  # other tasks untouched


def test_only_the_current_A_t_gets_gradients():
    attn = _attn('relu')
    for p in attn.parameters():
        p.requires_grad_(False)
    attn.lora_B_k[2].weight.requires_grad_(True)
    attn(torch.randn(2, 5, 24), 2).sum().backward()
    assert attn.lora_B_k[2].weight.grad is not None and attn.lora_B_k[2].weight.grad.abs().sum() > 0
    assert attn.lora_B_k[1].weight.grad is None


def test_checkpointed_delta_gives_the_same_gradients():
    attn = _attn('rff')
    x = torch.randn(2, 5, 24, requires_grad=True)
    grads = []
    for fn in (attn._psi_delta, attn._psi_delta_eager):
        attn.zero_grad()
        x.grad = None
        fn(x, 2, attn.lora_A_k, attn.lora_B_k, attn.psi_k).pow(2).sum().backward()
        grads.append((x.grad.clone(), attn.lora_B_k[2].weight.grad.clone()))
    assert torch.allclose(grads[0][0], grads[1][0], atol=1e-5)
    assert torch.allclose(grads[0][1], grads[1][1], atol=1e-5)


def test_psi_draws_leave_the_global_rng_alone():
    torch.manual_seed(3)
    want = torch.rand(4)
    torch.manual_seed(3)
    a, b = TaskFeatureMap('relu', 4, 8, 2), TaskFeatureMap('relu', 4, 8, 2)
    assert torch.equal(torch.rand(4), want)  # same data order / init as a run without psi
    assert not torch.equal(a.omega, b.omega)  # each layer and k/v gets its own draw


@pytest.mark.parametrize('kind', KINDS)
def test_scale_multiplies_output_and_scales_the_adam_step(kind):
    torch.manual_seed(0)
    a, b = TaskFeatureMap(kind, 4, 16, 1), TaskFeatureMap(kind, 4, 16, 1, scale=0.25)
    b.omega.copy_(a.omega)
    u = torch.randn(3, 1, 4)
    assert torch.allclose(b(u), 0.25 * a(u))
    # one Adam step on A from zero: the step on A is scale-free, so the change of A psi(u) scales with psi_scale
    outs = []
    for fm in (a, b):
        w = torch.zeros(5, 16, requires_grad=True)
        opt = torch.optim.Adam([w], lr=1e-2)
        (fm(u) @ w.T).sum().backward()
        opt.step()
        outs.append(fm(u) @ w.detach().T)
    assert torch.allclose(outs[1], 0.25 * outs[0], atol=1e-6)


def test_merged_matrix_helpers_refuse_psi():
    attn = _attn('relu')
    with pytest.raises(NotImplementedError):
        attn.get_matrix(0)


def test_config_and_tag():
    assert psi_config({}) is None and psi_tag({'psi': 'none'}) == ''
    assert psi_config({'psi': 'RFF', 'psi_dim': 64, 'psi_sigma': 2}) == {'kind': 'rff', 'm': 64, 'sigma_mult': 2.0,
                                                                          'scale': 1.0}
    assert psi_tag({'psi': 'rff', 'psi_dim': 64, 'psi_sigma': 0.5}) == '_psi-rff64_s0.5'
    assert psi_tag({'psi': 'relu', 'psi_dim': 128}) == '_psi-relu128'
    assert psi_config({'psi': 'linear', 'psi_dim': 40, 'rank': 10, 'psi_scale': 'auto'})['scale'] == 0.5
    assert psi_tag({'psi': 'linear', 'psi_dim': 128, 'psi_scale': 'auto'}) == '_psi-linear128_xauto'
    assert psi_tag({'psi': 'relu', 'psi_dim': 32, 'psi_scale': 0.25}) == '_psi-relu32_x0.25'
    with pytest.raises(ValueError):
        TaskFeatureMap('poly', 3, 8, 1)
    with pytest.raises(ValueError):
        TaskFeatureMap('rff', 3, 7, 1)


def test_set_overrides():
    main = pytest.importorskip('main')
    got = main.parse_overrides(['psi=rff', 'psi_dim=128', 'psi_sigma=0.5', 'seed=[0]', 'data_path=/a=b'])
    assert got == {'psi': 'rff', 'psi_dim': 128, 'psi_sigma': 0.5, 'seed': [0], 'data_path': '/a=b'}
    with pytest.raises(ValueError):
        main.parse_overrides(['psi'])

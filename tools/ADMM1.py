from PoseNet import PoseNet
from config import cfg
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# ==================== RT2dq =========================
def matrix_to_quaternion(R):
    # Convert a 3x3 rotation matrix to a quaternion [w, x, y, z]
    m = R
    trace = np.trace(m)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m[2,1] - m[1,2]) / s
        y = (m[0,2] - m[2,0]) / s
        z = (m[1,0] - m[0,1]) / s
    else:
        if (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
            s = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2
            w = (m[2,1] - m[1,2]) / s
            x = 0.25 * s
            y = (m[0,1] + m[1,0]) / s
            z = (m[0,2] + m[2,0]) / s
        elif m[1,1] > m[2,2]:
            s = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2
            w = (m[0,2] - m[2,0]) / s
            x = (m[0,1] + m[1,0]) / s
            y = 0.25 * s
            z = (m[1,2] + m[2,1]) / s
        else:
            s = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2
            w = (m[1,0] - m[0,1]) / s
            x = (m[0,2] + m[2,0]) / s
            y = (m[1,2] + m[2,1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])

def quat_multiply(q1, q2):
    # Quaternion multiplication
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def rt2dq(R, t):
    # Convert rotation and translation to dual quaternion representation
    q_r = matrix_to_quaternion(R)
    t_quat = np.concatenate(([0.0], t))
    q_d = 0.5 * quat_multiply(t_quat, q_r)
    return q_r, q_d

# ==================== Dual Quaternion Constraint Loss =========================
def dq_constraint_loss(v):
    real = v[:, :4]
    dual = v[:, 4:]
    real_norm = torch.linalg.norm(real, dim=1)
    dot_rd = (real * dual).sum(dim=1)
    loss_unit_norm = ((real_norm - 1) ** 2).sum()
    loss_orth = (dot_rd ** 2).sum()
    return loss_unit_norm + loss_orth

def get_trainable_param(module):
    return filter(lambda p: p.requires_grad, module.parameters())

def heatmap_to_prob(hm: torch.Tensor) -> torch.Tensor:
    """(N, 1, K, H, W) → (softmax)"""
    N = hm.size(0)
    hm_flat = hm.view(N, -1)
    hm_prob = F.softmax(hm_flat, dim=1)
    return hm_prob.view_as(hm)

def marginalize_heatmaps(hm_prob: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Σ w_i · P(H | B_i)"""
    w = w.view(-1, 1, 1, 1, 1)
    return (w * hm_prob).sum(dim=0)

def mmd_distance(x: torch.Tensor,
                 y: torch.Tensor,
                 sigma: float = 1.0) -> torch.Tensor:
    """RBF-MMD²"""
    def _kernel(a, b):
        a_sq = (a**2).sum(dim=1, keepdim=True)
        b_sq = (b**2).sum(dim=1, keepdim=True).t()
        dist2 = a_sq + b_sq - 2 * a @ b.t()
        return torch.exp(-dist2 / (2 * sigma**2))
    xx = _kernel(x, x)
    yy = _kernel(y, y)
    xy = _kernel(x, y)
    return xx.mean() + yy.mean() - 2 * xy.mean()
class DomainHeatmapAlignmentLoss(nn.Module):
    """
    P(Hs) vs P(Ht) —— L1 / KL / MMD
        Hs_raw : (Ns, 1, K, H, W)
        Ht_raw : (Nt, 1, K, H, W)
    """
    def __init__(self, distance_type='mmd', mmd_sigma=1.0):
        super().__init__()
        assert distance_type in {'l1', 'kl', 'mmd'}
        self.distance = distance_type
        self.sigma = mmd_sigma

    def forward(self, Hs_raw, Ht_raw,
                w_s: torch.Tensor | None = None,
                w_t: torch.Tensor | None = None):
        Hs = heatmap_to_prob(Hs_raw)
        Ht = heatmap_to_prob(Ht_raw)

        Ns, Nt = Hs.size(0), Ht.size(0)
        if w_s is None:
            w_s = torch.full((Ns,), 1. / Ns, device=Hs.device)
        else:
            w_s = w_s / w_s.sum()
        if w_t is None:
            w_t = torch.full((Nt,), 1. / Nt, device=Ht.device)
        else:
            w_t = w_t / w_t.sum()

        P_Hs = marginalize_heatmaps(Hs, w_s)
        P_Ht = marginalize_heatmaps(Ht, w_t)

        if self.distance == 'l1':
            return F.l1_loss(P_Hs, P_Ht)
        if self.distance == 'kl':
            eps = 1e-8
            return (P_Hs * (P_Hs.add(eps).log() - P_Ht.add(eps).log())).sum()
        # MMD
        x, y = P_Hs.view(1, -1), P_Ht.view(1, -1)
        return mmd_distance(x, y, sigma=self.sigma)
def get_trainable_param(module):
    return filter(lambda p: p.requires_grad, module.parameters())
# ==================== ADMM Optimization Main Function =========================
def admm_update_w1(posenet,
                   lambda_dq=1.0, mu=0.1, num_admm_iterations=1,
                   lr_theta_gd=0.01, n_inner_v=10, lr_v=0.01):
    # Get shape and flatten w1

    w1 = posenet.get_first_layer_weights()
    device = w1.device if hasattr(w1, "device") else torch.device("cpu")
    w1_flat = w1.contiguous().view(-1).to(device)
    v = w1_flat.clone().detach().to(device)
    u = torch.zeros_like(w1_flat, device=device)

    w_orig = w1_flat.clone()
    params = get_trainable_param(posenet.model)
    print(params)
    optimizer = torch.optim.Adam(params, lr=lr_theta_gd)
    align_loss_fn = DomainHeatmapAlignmentLoss('mmd').to(device)
    for k in range(num_admm_iterations):
        posenet.prepare_grad_only_first()
        # === w1 step ===
        w1_flat_update = w1_flat.clone().detach().requires_grad_(True)
        for step in range(5):
            w1 = posenet.get_first_layer_weights()
            w1_flat = w1.contiguous().view(-1).to(device)
            print("w1_flat: " + str(w1_flat))
            f_tgt = posenet.get_first_layer_output(posenet.target_data_loader)
            f_src = posenet.get_first_layer_output(posenet.source_data_loader)
            #→ (N,1,K,H,W)
            f_tgt = torch.stack(f_tgt, dim=0).unsqueeze(1).to(device)
            f_src = torch.stack(f_src, dim=0).unsqueeze(1).to(device)
            feature_loss = align_loss_fn(f_src, f_tgt)
            penalty = (mu / 2.0) * ((w1_flat_update - v + u) ** 2).sum()
            total_loss = feature_loss + penalty
            print("feature_loss: " + str(feature_loss))
            print("total_loss: " + str(total_loss))
            total_loss.backward()
            optimizer.step()
            if (step % 100 == 0):
                posenet.save_model("intermediate.pth.tar")
            # print(step)

    
        w1 = posenet.get_first_layer_weights()
        w1_flat = w1.contiguous().view(-1).to(device)
        w1_flat = w1_flat.detach()

        # === v step (Dual Quaternion + remaining part, each with their own loss) ===
        R, T = posenet.get_output_rt()
        N = len(R)
        dq_targets = []
        for i in range(N):
            Ri = R[i]
            Ti = T[i]
            q_r_np, q_t_np = rt2dq(Ri, Ti)  # each: (4,)
            dq = np.concatenate([q_r_np, q_t_np])  # (8,)
            dq_targets.append(dq)
        dq_targets = torch.tensor(np.stack(dq_targets), dtype=torch.float32, device=device)#[N,8]'''

        v_update = v.clone().detach().requires_grad_(True)
        optimizer_v = torch.optim.Adam([v_update], lr=lr_v)
        for step in range(n_inner_v):
            optimizer_v.zero_grad()
            #dqcon_loss = dq_constraint_loss(dq_targets)
            loss_sqr = ((v_update - w1_flat) ** 2).sum() / w1_flat.numel()#w1_flat
            penalty1 = (mu / 2.0) * ((w1_flat - v_update + u) ** 2).sum()
            v_total_loss = lambda_dq * loss_sqr + penalty1
            print("loss_sqr: " + str(loss_sqr))
            print("penalty1: " + str(penalty1))
            print("v_total_loss: " + str(v_total_loss))
            v_total_loss.backward()
            optimizer_v.step()
        v = v_update.detach()
        u = u + (w1_flat - v)
        ADMM_TOTAL_LOSS=feature_loss + lambda_dq * loss_sqr + (mu / 2.0) * ((w1_flat - v_update + u) ** 2).sum()
        print("ADMM_TOTAL_LOSS: " + str(ADMM_TOTAL_LOSS))
        print(f"[ADMM {k+1}] feature_loss={feature_loss.item():.4f}, loss_sqr={loss_sqr.item():.4f}")
    
    posenet.save_model("ladmm_model_best.pth.tar")
    return w1_flat

# ========== RUN ==========
print("Before")
posenet = PoseNet(cfg)
w1_opt = admm_update_w1(posenet)
#print("w1 updated shape:", w1_opt.shape)
#print("Max abs diff in w1:", (w1_opt - w1).abs().max().item())
#print("norm:", torch.norm(w1_opt - w1).item())

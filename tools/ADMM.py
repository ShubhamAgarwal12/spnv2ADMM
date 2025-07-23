from PoseNet import PoseNet
from config import cfg
import torch
import numpy as np
import torch.nn.functional as F


def coral_loss_2d_list(src_list, tgt_list, eps=1e-5):
    """
    CORAL loss for a list of [B, C, H, W] tensors (e.g., from intermediate layers)

    Args:
        src_list: List[Tensor], each [B, C, H, W] from source domain
        tgt_list: List[Tensor], each [B, C, H, W] from target domain
        eps: float, small value to avoid division by zero

    Returns:
        torch.Tensor: scalar CORAL loss
    """
    # Concatenate features along the batch dimension
    src_feat = torch.cat(src_list, dim=0)  # [B_total, C, H, W]
    tgt_feat = torch.cat(tgt_list, dim=0)

    B, C, H, W = src_feat.shape
    N = B * H * W  # Total number of feature vectors

    # Reshape to [N, C] — feature vectors across all spatial locations and batches
    src = src_feat.permute(0, 2, 3, 1).reshape(-1, C)
    tgt = tgt_feat.permute(0, 2, 3, 1).reshape(-1, C)

    # Centered features
    src_centered = src - src.mean(dim=0, keepdim=True)
    tgt_centered = tgt - tgt.mean(dim=0, keepdim=True)

    # Covariance matrices
    src_cov = (src_centered.T @ src_centered) / (N - 1 + eps)
    tgt_cov = (tgt_centered.T @ tgt_centered) / (N - 1 + eps)

    # CORAL loss: Frobenius norm of the difference
    loss = ((src_cov - tgt_cov) ** 2).sum()
    loss = loss / (4 * C * C)

    return loss


# ==================== RT2dq =========================
def matrix_to_quaternion_torch(R):
    # R: [..., 3, 3]
    # Returns: [..., 4] (w, x, y, z)
    m = R
    batch = R.shape[:-2]
    q = torch.zeros(batch + (4,), dtype=R.dtype, device=R.device)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]

    mask = trace > 0.0
    s = torch.zeros_like(trace)
    s[mask] = torch.sqrt(trace[mask] + 1.0) * 2
    s[~mask] = 2 * torch.sqrt(1.0 + m[~mask, 2, 2] - m[~mask, 0, 0] - m[~mask, 1, 1])
    # w, x, y, z
    q[mask, 0] = 0.25 * s[mask]
    q[mask, 1] = (m[mask, 2, 1] - m[mask, 1, 2]) / s[mask]
    q[mask, 2] = (m[mask, 0, 2] - m[mask, 2, 0]) / s[mask]
    q[mask, 3] = (m[mask, 1, 0] - m[mask, 0, 1]) / s[mask]
    # fallback
    idx = (~mask).nonzero(as_tuple=True)[0]
    for i in idx:
        if (m[i, 0, 0] > m[i, 1, 1]) and (m[i, 0, 0] > m[i, 2, 2]):
            s = torch.sqrt(1.0 + m[i, 0, 0] - m[i, 1, 1] - m[i, 2, 2]) * 2
            q[i, 0] = (m[i, 2, 1] - m[i, 1, 2]) / s
            q[i, 1] = 0.25 * s
            q[i, 2] = (m[i, 0, 1] + m[i, 1, 0]) / s
            q[i, 3] = (m[i, 0, 2] + m[i, 2, 0]) / s
        elif m[i, 1, 1] > m[i, 2, 2]:
            s = torch.sqrt(1.0 + m[i, 1, 1] - m[i, 0, 0] - m[i, 2, 2]) * 2
            q[i, 0] = (m[i, 0, 2] - m[i, 2, 0]) / s
            q[i, 1] = (m[i, 0, 1] + m[i, 1, 0]) / s
            q[i, 2] = 0.25 * s
            q[i, 3] = (m[i, 1, 2] + m[i, 2, 1]) / s
        else:
            s = torch.sqrt(1.0 + m[i, 2, 2] - m[i, 0, 0] - m[i, 1, 1]) * 2
            q[i, 0] = (m[i, 1, 0] - m[i, 0, 1]) / s
            q[i, 1] = (m[i, 0, 2] + m[i, 2, 0]) / s
            q[i, 2] = (m[i, 1, 2] + m[i, 2, 1]) / s
            q[i, 3] = 0.25 * s
    return q

def quat_multiply_torch(q1, q2):
    # q1, q2: [N, 4]
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=1)

def rt2dq_torch(R, T):
    # Ensure inputs are PyTorch tensors
    R = torch.as_tensor(R, dtype=torch.float32)
    T = torch.as_tensor(T, dtype=torch.float32)

    # Convert rotation matrix to quaternion
    q_r = matrix_to_quaternion_torch(R)  # shape: [N, 4]

    # Reshape translation
    t = T.view(-1, 3)

    # Form dual quaternion's dual part
    zeros = torch.zeros_like(q_r[:, :1])
    t_quat = torch.cat([zeros, t], dim=1)
    q_d = 0.5 * quat_multiply_torch(t_quat, q_r)

    return q_r, q_d

def binary_entropy(mask):
    """
    Compute Shannon entropy of a binary mask (numpy array with 0 and 1)
    """
    # Flatten to 1D
    values = mask.flatten()
    # Count probabilities
    p0 = np.mean(values == 0)
    p1 = 1 - p0
    # Avoid log(0)
    entropy = 0
    if p0 > 0:
        entropy -= p0 * np.log2(p0)
    if p1 > 0:
        entropy -= p1 * np.log2(p1)
    return entropy

def compute_entropy_list(mask_list):
    """
    Compute average entropy across a list of binary masks
    """
    entropies = [binary_entropy(mask) for mask in mask_list]
    return np.mean(entropies)


def get_trainable_param(module):
    return filter(lambda p: p.requires_grad, module.parameters())


def pairwise_dist(x, y):  # x:[N,D], y:[M,D]
    # return [N,M] L2^2 distance
    x_norm = x.pow(2).sum(dim=1, keepdim=True)   # [N,1]
    y_norm = y.pow(2).sum(dim=1, keepdim=True)   # [M,1]
    dist = x_norm + y_norm.t() - 2.0 * (x @ y.t())
    return dist.clamp(min=0)

def mmd_rbf(src, tgt, sigmas=(1, 2, 4, 8)):
    D_ss = pairwise_dist(src, src)
    D_tt = pairwise_dist(tgt, tgt)
    D_st = pairwise_dist(src, tgt)
    mmd2 = 0.0
    for s in sigmas:
        K_ss = torch.exp(-D_ss / (2*s*s))
        K_tt = torch.exp(-D_tt / (2*s*s))
        K_st = torch.exp(-D_st / (2*s*s))
        mmd2 += K_ss.mean() + K_tt.mean() - 2*K_st.mean()
    return mmd2 / len(sigmas)


# ==================== ADMM Optimization Main Function =========================
def admm_update_w1(posenet,
                   lambda_dq=1.0, mu=0.01, num_admm_iterations=5,
                   lr_theta_gd=0.001, n_inner_v=10, lr_v=0.001):
    # Get shape and flatten w1
    w1 = posenet.get_first_layer_weights()
    #C, Cin, kH, kW = w1.shape
    device = w1.device if hasattr(w1, "device") else torch.device("cpu")
    w1_flat = w1.contiguous().view(-1).to(device)
    v = w1_flat.clone().detach().to(device)
    u = torch.zeros_like(w1_flat, device=device)
    w_orig = w1_flat.clone()
    params = get_trainable_param(posenet.model)
    print(params)
    optimizer = torch.optim.Adam(params, lr=lr_theta_gd)
    
    for k in range(num_admm_iterations):
        posenet.prepare_grad_only_first()
        # === w1 (theta) step ===
        w1_flat_update = w1_flat.clone().detach().requires_grad_(True)
        for step in range(15):
            w1 = posenet.get_first_layer_weights()
            w1_flat = w1.contiguous().view(-1).to(device)
            #print("w1_flat: " + str(w1_flat))
            #fsrc, losses = posenet.get_first_layer_output(posenet.source_data_loader)
            ftgt, losses = posenet.get_first_layer_output(posenet.target_data_loader)
            ftgt = [x.to(device) for x in ftgt]
            #fsrc = [x.to(device) for x in fsrc]
            print(losses)
            feature_loss = torch.stack(losses).mean()
            ###
            #src_vec = fsrc_tensor.squeeze(1).mean(dim=[2,3])
            #tgt_vec = ftgt_tensor.squeeze(1).mean(dim=[2,3])   
            #feature_loss = ((ftgt_mean-fsrc_mean)**2).sum()/ftgt_mean.numel() 
            #feature_loss = mmd_rbf(src_vec, tgt_vec, sigmas=(1, 2, 4, 8))
            penalty = (mu / 2.0) * ((w1_flat_update - v + u) ** 2).sum() #w1_flat

            R, T  = posenet.get_output_rt()  # R: [N,3,3] torch tensor, T: [N,3] torch tensor
            real, dual = rt2dq_torch(R, T)  # torch
            real_norm = torch.linalg.norm(real, dim=1)              # [N]
            dot_rd    = (real * dual).sum(dim=1)                    # [N]
            loss_unit_norm = (real_norm - 1) ** 2                   # [N]
            loss_orth      = dot_rd ** 2                            # [N]
            total_dq_loss  = loss_unit_norm + loss_orth             # [N]
            dq_constraints_loss = total_dq_loss.mean()
            total_loss = feature_loss + penalty + dq_constraints_loss
            print("feature_loss: " + str(feature_loss))
            print("dq_constraints_loss: " + str(dq_constraints_loss))
            print("total_loss: " + str(total_loss))
            total_loss.backward()
            optimizer.step()
            posenet.save_model("intermediate.pth.tar")
            # print(step)
        w1 = posenet.get_first_layer_weights()
        w1_flat = w1.contiguous().view(-1).to(device)
        w1_flat = w1_flat.detach()

        # === v step (Dual Quaternion + remaining part, each with their own loss) ===
        v_update = v.clone().detach().requires_grad_(True)
        optimizer_v = torch.optim.Adam([v_update], lr=lr_v)
        for step in range(n_inner_v):
            optimizer_v.zero_grad()
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
        ADMM_TOTAL_LOSS=feature_loss + dq_constraints_loss + lambda_dq * loss_sqr + (mu / 2.0) * ((w1_flat - v_update + u) ** 2).sum()
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
import time
import torch
from tqdm import tqdm
import torch.nn.functional as F

def svd_decomposition(matrix, rank):
    U, S, Vh = torch.pca_lowrank(matrix, q=rank)
    return U @ torch.diag_embed(S),  Vh.T

def low_rank_decomposition(W, rank, learning_rate=0.01, max_iterations=500, tolerance=1e-5, X = None):
    L = torch.rand((W.shape[0], rank), device=W.device)
    R = torch.rand((rank, W.shape[1]), device=W.device)
    tick = time.time()
    early_stop = False
    if X is None:
        for i in range(max_iterations):
            # Calculate the difference between the original and reconstructed matrices
            diff_part1 = W
            diff_part2 = L @ R
            difference = W - L @ R
            # Calculate the gradients
            gradient_L = -2 * (difference @ R.T)
            gradient_R = -2 * (L.T @ difference)
            L -= learning_rate * gradient_L
            R -= learning_rate * gradient_R
            if F.mse_loss(diff_part1, diff_part2) < tolerance:
                early_stop = True
                break
    else:
        for i in range(max_iterations):
            diff_part1 = W@X
            diff_part2 = L @ R @ X
            diff = diff_part1 - diff_part2
            gradient_L = -2 * (diff @ ((R@X).T))
            gradient_R = -2 * (L.T @ diff @ X.T)
            L -= learning_rate * gradient_L
            R -= learning_rate * gradient_R
            if F.mse_loss(diff_part1, diff_part2) < tolerance:
                early_stop = True
                break
    return L, R

def torch_autograd(W, X, rank, lr, steps):
    L = torch.rand((W.shape[0], rank), device=W.device, requires_grad=True)
    R = torch.rand((rank, W.shape[1]), device=W.device, requires_grad=True)
    optimizer = torch.optim.SGD([L, R], lr=lr)
    for _ in tqdm(range(steps)):
        optimizer.zero_grad()
        output = L @ R @ X
        target = W @ X
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
    return L, R

if __name__=="__main__":
    FULL_RANK = 128
    LOW_RANK = 16
    TARGET_SIZE = 2

    W = torch.rand((FULL_RANK, FULL_RANK))
    input_matrix = torch.rand((FULL_RANK, TARGET_SIZE))
    output_matrix = W @ input_matrix
    
    L_sensitive, R_sensitive = low_rank_decomposition(
        W,
        LOW_RANK,
        learning_rate=1e-6,
        max_iterations=1000,
        X=input_matrix
    )
    L_noinput, R_noinput = low_rank_decomposition(
        W,
        LOW_RANK,
        learning_rate=1e-6,
        max_iterations=1000,
    )
    L_autograd, R_autograd = torch_autograd(W, input_matrix, LOW_RANK, 1e-6, 1000)

    reconstructed_matrix = L_sensitive @ R_sensitive @ input_matrix
    reconstructed_matrix_pca = L_autograd @ R_autograd @ input_matrix
    reconstructed_matrix_noinput = L_noinput @ R_noinput @ input_matrix
    print("difference:")
    print(f"gd: {F.mse_loss(output_matrix, reconstructed_matrix)}")
    print(f"autograd: {F.mse_loss(output_matrix, reconstructed_matrix_pca)}")
    print(f"noinput gd: {F.mse_loss(output_matrix, reconstructed_matrix_noinput)}")
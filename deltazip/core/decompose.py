import time
import torch
from tqdm import tqdm
import torch.nn.functional as F
from typing import List

def svd_decomposition(matrix, rank):
    U, S, Vh = torch.pca_lowrank(matrix, q=rank)
    return U @ torch.diag_embed(S), Vh.T


def low_rank_decomposition(
    W, rank, X=None, learning_rate=0.01, max_iterations=500, tolerance=1e-5, 
):
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
            diff_part1 = W @ X
            diff_part2 = L @ R @ X
            diff = diff_part1 - diff_part2
            gradient_L = -2 * (diff @ ((R @ X).T))
            gradient_R = -2 * (L.T @ diff @ X.T)
            L -= learning_rate * gradient_L
            R -= learning_rate * gradient_R
            if F.mse_loss(diff_part1, diff_part2) < tolerance:
                early_stop = True
                break
    return L, R

def matrix_factorization(W: torch.Tensor, X: torch.Tensor, rank=32, lr=1e-5, steps=10000):
    
    # L = torch.normal(mean=0, std=1, size=(W.shape[0], rank), device=W.device, requires_grad=True)
    L = torch.rand((W.shape[0], rank), device=W.device, requires_grad=True)
    # R = torch.normal(
    #     mean=0, std=1, size=(rank, W.shape[1]), device=W.device, requires_grad=True
    # )
    R = torch.zeros(
        (rank, W.shape[1]), device=W.device, requires_grad=True
    )
    optimizer = torch.optim.SGD([L, R], lr=lr)
    pbar = tqdm(range(steps))
    for _ in  pbar:
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(L @ R @ X, W @ X)
        loss.backward()
        pbar.set_description(f"loss={loss}")
        optimizer.step()
    return L, R

def batched_matrix_factorization(W: torch.Tensor, X: List[torch.Tensor], rank=32, lr=1e-5, steps=10000, batch_size=2,pad_token_id=0):
    assert len(X) % batch_size == 0, f"batch size must be a divisor of len(X), got {len(X)}/{batch_size}"
    input_size = W.shape[0]
    
    L = torch.rand((W.shape[0], rank), device=W.device, requires_grad=True, dtype=torch.float16)
    R = torch.zeros(
        (rank, W.shape[1]), device=W.device, requires_grad=True, dtype=torch.float16
    )

    optimizer = torch.optim.SGD([L, R], lr=lr)
    pbar = tqdm(range(steps))
    for _ in pbar:
        optimizer.zero_grad()
        for i in range(0, len(X), batch_size):
            # left pad the input batch if necessary
            input_batch = [F.pad(x, ( 0, 0, input_size - x.shape[1], 0), 'constant', pad_token_id) for x in X[i:i+batch_size]]
            input_batch = torch.stack(input_batch)
            loss = torch.nn.functional.mse_loss(L @ R @ input_batch, W @ input_batch)
            del input_batch
        loss.backward()
        pbar.set_description(f"loss={loss}")
        optimizer.step()
    return L, R

def calculate_factorization_loss(W, L, R, X):
    return F.mse_loss(W @ X, L @ R @ X)

if __name__ == "__main__":
    FULL_RANK = 4096
    LOW_RANK = 16
    TARGET_SIZE = 1024
    LEARNING_RATE = 1e-6
    MAX_ITERATION = 500
    
    W = torch.rand((FULL_RANK, FULL_RANK))
    input_matrix = torch.rand((FULL_RANK, TARGET_SIZE))
    output_matrix = W @ input_matrix

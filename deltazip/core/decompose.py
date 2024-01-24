import time
import torch
from tqdm import tqdm
import torch.nn.functional as F
from typing import List
from loguru import logger
from deltazip.core.sparsity_utils import hard_threshold

def svd_decomposition(matrix, rank):
    U, S, Vh = torch.pca_lowrank(matrix, q=rank)
    return U @ torch.diag_embed(S), Vh.T


def low_rank_decomposition(
    W,
    rank,
    X=None,
    learning_rate=0.01,
    max_iterations=500,
    tolerance=1e-5,
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

def matrix_factorization(
    W: torch.Tensor, X: torch.Tensor, rank=32, lr=1e-5, steps=10000
):
    # L = torch.normal(mean=0, std=1, size=(W.shape[0], rank), device=W.device, requires_grad=True)
    L = torch.rand((W.shape[0], rank), device=W.device, requires_grad=True)
    # R = torch.normal(
    #     mean=0, std=1, size=(rank, W.shape[1]), device=W.device, requires_grad=True
    # )
    R = torch.zeros((rank, W.shape[1]), device=W.device, requires_grad=True)
    optimizer = torch.optim.SGD([L, R], lr=lr)
    pbar = tqdm(range(steps))
    for _ in pbar:
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(L @ R @ X, W @ X)
        loss.backward()
        pbar.set_description(f"loss={loss}")
        optimizer.step()
    return L, R

# temporarily disable the inference mode
@torch.inference_mode(False)
def batched_matrix_factorization(
    W: torch.Tensor,
    X: List[torch.Tensor],
    rank=32,
    lr=1e-5,
    steps=1000,
    batch_size=8,
    pad_token_id=0,
):
    assert (
        len(X) % batch_size == 0
    ), f"batch size must be a divisor of len(X), got {len(X)}/{batch_size}"
    input_size = W.shape[0]
    L = torch.rand(
        (W.shape[0], rank), device=W.device, requires_grad=True, dtype=torch.float16
    )
    R = torch.zeros(
        (rank, W.shape[1]), device=W.device, requires_grad=True, dtype=torch.float16
    )
    X = [torch.squeeze(x, dim=0) for x in X]
    optimizer = torch.optim.SGD([L, R], lr=lr)
    pbar = tqdm(range(steps))
    logger.info(f"length of X: {len(X)}, requires grad? {X[0].requires_grad}, shape: {X[1].shape}")
    logger.info(f"W shape: {W.shape}, requires grad? {W.requires_grad}")
    
    for _ in pbar:
        for i in range(0, len(X), batch_size):
            optimizer.zero_grad()
            # left pad the input batch if necessary
            input_batch = torch.stack([
                F.pad(
                    x, 
                    (0, 0, input_size - x.shape[0], 0), 
                    "constant", 
                    pad_token_id
                )
                for x in X[i : i + batch_size]
            ])
            pred =  R @ input_batch
            print(pred)
            logger.info(f"pred requires grad? {pred.requires_grad} <- {L.requires_grad}, {R.requires_grad}, {input_batch.requires_grad}")
            exit()
            gt = W @ input_batch
            logger.info(f"pred shape: {pred.shape} {pred.requires_grad}, gt shape: {gt.shape}, {gt.requires_grad}")
            loss = F.mse_loss(
                pred, gt
            )
            print(loss)
            loss.backward()
            optimizer.step()
        pbar.set_description(f"loss={loss}")
    return L, R, 0

def calculate_factorization_loss(W, L, R, X):
    return F.mse_loss(W @ X, L @ R @ X)


if __name__ == "__main__":
    FULL_RANK = 4096
    LOW_RANK = 32
    TARGET_SIZE = 1024
    LEARNING_RATE = 1e-5
    MAX_ITERATION = 10000
    BATCH = 32
    BATCH_SIZE=2

    W = torch.rand((FULL_RANK, FULL_RANK), dtype=torch.float16).to(torch.device("cuda"))
    input_matrix = torch.rand((FULL_RANK, TARGET_SIZE), dtype=torch.float16).to(torch.device("cuda"))
    # batched_input_matrix = torch.rand((BATCH, FULL_RANK, TARGET_SIZE), dtype=torch.float16).to(torch.device("cuda"))

    batched_input_matrix = [torch.rand((1, FULL_RANK-i, TARGET_SIZE), dtype=torch.float16).to(torch.device("cuda")) for i in range(BATCH)]

    output_matrix = W @ input_matrix
    batched_input_matrix = [hard_threshold(x, 0.75) for x in batched_input_matrix]
    L, R = batched_matrix_factorization(W, batched_input_matrix, lr=1e-4)
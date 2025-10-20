import torch
from typing import List, Tuple
from torch import Tensor


def hello():
    print('Hello from pytorch101.py!')


def create_sample_tensor() -> Tensor:
    # 创建一个 3×2 的全零张量
    x = torch.zeros((3, 2))
    # 设置指定位置
    x[0, 1] = 10
    x[1, 0] = 100
    return x


def mutate_tensor(
    x: Tensor, indices: List[Tuple[int, int]], values: List[float]
) -> Tensor:
    # 遍历索引和值，直接修改 x
    for (i, j), v in zip(indices, values):
        x[i, j] = v
    return x


def count_tensor_elements(x: Tensor) -> int:
    # 手动计算所有维度的乘积
    num_elements = 1
    for d in x.shape:
        num_elements *= d
    return num_elements


def create_tensor_of_pi(M: int, N: int) -> Tensor:
    # 创建填充 3.14 的张量
    x = torch.full((M, N), 3.14)
    return x


def multiples_of_ten(start: int, stop: int) -> Tensor:
    multiples = [i for i in range(start, stop + 1) if i % 10 == 0]
    if len(multiples) == 0:
        return torch.tensor([], dtype=torch.float64)
    return torch.tensor(multiples, dtype=torch.float64)


def slice_indexing_practice(x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    last_row = x[-1]
    third_col = x[:, 2:3]
    first_two_rows_three_cols = x[:2, :3]
    even_rows_odd_cols = x[::2, 1::2]
    return last_row, third_col, first_two_rows_three_cols, even_rows_odd_cols


def slice_assignment_practice(x: Tensor) -> Tensor:
    # 第一块 [0 1 2 2 2 2]
    x[0:2, 0:3] = torch.tensor([[0, 1, 2], [0, 1, 2]])
    x[0:2, 3:6] = 2
    x[2:4, 0:2] = torch.tensor([[3, 4], [3, 4]])
    x[2:4, 2:4] = torch.tensor([[3, 4], [3, 4]])
    x[2:4, 4:6] = 5
    return x


def shuffle_cols(x: Tensor) -> Tensor:
    # 重排列列索引：[0, 0, 2, 1]
    y = x[:, [0, 0, 2, 1]]
    return y


def reverse_rows(x: Tensor) -> Tensor:
    # 反转行顺序
    y = x[torch.arange(x.shape[0] - 1, -1, -1)]
    return y


def take_one_elem_per_col(x: Tensor) -> Tensor:
    y = x[[1, 0, 3], [0, 1, 2]]
    return y


def make_one_hot(x: List[int]) -> Tensor:
    N, C = len(x), max(x) + 1
    y = torch.zeros((N, C), dtype=torch.float32)
    y[torch.arange(N), x] = 1
    return y


def sum_positive_entries(x: Tensor) -> int:
    pos_sum = int(x[x > 0].sum())
    return pos_sum


def reshape_practice(x: Tensor) -> Tensor:
    y = x.view(2, 3, 4).permute(1, 0, 2).contiguous().view(3, 8)
    return y


def zero_row_min(x: Tensor) -> Tensor:
    y = x.clone()
    mins = x.min(dim=1).values
    y[torch.arange(x.shape[0]), x.argmin(dim=1)] = 0
    return y


def batched_matrix_multiply_loop(x: Tensor, y: Tensor) -> Tensor:
    B = x.shape[0]
    outs = []
    for i in range(B):
        outs.append(x[i] @ y[i])
    return torch.stack(outs)


def batched_matrix_multiply_noloop(x: Tensor, y: Tensor) -> Tensor:
    return torch.bmm(x, y)


def normalize_columns(x: Tensor) -> Tensor:
    M = x.shape[0]
    mu = x.sum(0) / M
    sigma = torch.sqrt(((x - mu) ** 2).sum(0) / M)
    y = (x - mu) / sigma
    return y


def mm_on_cpu(x: Tensor, w: Tensor) -> Tensor:
    return x.mm(w)


def mm_on_gpu(x: Tensor, w: Tensor) -> Tensor:
    x_gpu, w_gpu = x.cuda(), w.cuda()
    y_gpu = x_gpu @ w_gpu
    return y_gpu.cpu()


def challenge_mean_tensors(xs: List[Tensor], ls: Tensor) -> Tensor:
    all_concat = torch.cat(xs)
    cs = torch.cumsum(ls, dim=0)
    ss = torch.cat((torch.tensor([0], device=cs.device), cs[:-1]))
    means = torch.stack([(all_concat[s:e].sum() / (e - s)) for s, e in zip(ss, cs)])
    return means


def challenge_get_uniques(x: torch.Tensor) -> Tuple[Tensor, Tensor]:
    uniques, indices = torch.unique(x, return_inverse=False, return_counts=False, return_index=True)
    return uniques, indices

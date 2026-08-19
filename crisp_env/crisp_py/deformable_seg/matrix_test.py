import time

import numpy as np

import torch



if __name__ == "__main__":

    for i in range(10):
        print(f"Iteration {i+1}/10")
        N = 300
        M = 300

        A = np.random.rand(4, N, M).astype(np.float64)
        B = np.random.rand(4, M, N).astype(np.float64)
        

        A_torch = torch.from_numpy(A).to(device='cuda')
        B_torch = torch.from_numpy(B).to(device='cuda')


        start_time = time.time()
        # C = np.dot(A, B)
        C_torch = torch.bmm(A_torch, B_torch)
        end_time = time.time()

        print(f"Matrix multiplication of {N}x{M} and {M}x{N} took {end_time - start_time:.6f} seconds.")

        # print(f"Resulting matrix shape: {C_torch.shape}")
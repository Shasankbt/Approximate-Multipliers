#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <cstdint>
#include <vector>
#include <algorithm>

#include "multiplier.h"

torch::Tensor pbo_product_tensor(torch::Tensor a, torch::Tensor b_t) {
    TORCH_CHECK(a.dim() == 3, "a must be [B, M, K]");
    TORCH_CHECK(b_t.dim() == 3, "b_t must be [B, N, K]");
    TORCH_CHECK(a.size(0) == b_t.size(0), "batch mismatch");
    TORCH_CHECK(a.size(2) == b_t.size(2), "K mismatch");
    TORCH_CHECK(a.scalar_type() == torch::kFloat16, "a must be float16");
    TORCH_CHECK(b_t.scalar_type() == torch::kFloat16, "b_t must be float16");

    a = a.contiguous();
    b_t = b_t.contiguous();

    const int64_t batch_size = a.size(0);
    const int64_t M = a.size(1);
    const int64_t K = a.size(2);
    const int64_t N = b_t.size(1);

    auto result = torch::zeros({batch_size, M, N}, a.options());

    const at::Half* A_ptr = a.data_ptr<at::Half>();
    const at::Half* B_ptr = b_t.data_ptr<at::Half>();
    at::Half* C_ptr = result.data_ptr<at::Half>();

    at::parallel_for(0, batch_size, 1, [&](int64_t b0, int64_t b1) {
        std::vector<float> acc(N);

        for (int64_t b = b0; b < b1; ++b) {
            const int64_t A_base = b * M * K;
            const int64_t B_base = b * N * K;
            const int64_t C_base = b * M * N;

            for (int64_t i = 0; i < M; ++i) {
                std::fill(acc.begin(), acc.end(), 0.0f);

                const at::Half* A_row = A_ptr + A_base + i * K;

                for (int64_t j = 0; j < N; ++j) {
                    const at::Half* B_row = B_ptr + B_base + j * K;

                    float sum = 0.0f;
                    for (int64_t k = 0; k < K; ++k) {
                        sum += A_row[k] * B_row[k]; // approx_half_scalar(A_row[k], B_row[k]);
                    }
                    acc[j] = sum;
                }

                at::Half* C_row = C_ptr + C_base + i * N;
                for (int64_t j = 0; j < N; ++j) {
                    C_row[j] = static_cast<at::Half>(acc[j]);
                }
            }
        }
    });

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pbo_product", &pbo_product_tensor,
          "PBOM8-73Y approximate FP16 matmul (CPU)");
}
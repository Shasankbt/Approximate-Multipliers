#include <torch/extension.h>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

// ==========================================
// PART 1: The "Broken Chip" Logic (PBOM8-73Y)
// ==========================================
inline uint8_t pbom8_mantissa_mult(uint8_t a, uint8_t b) {
    // Split B into nibbles (no cast needed, uint8_t is fine)
    const uint8_t b_low = b & 0x0F;
    const uint8_t b_high = b >> 4;  // Already 4 bits, no mask needed

    // Multiplier-1 (Lower 4 bits)
    const uint16_t m1_exact = (a * b_low) & 0xFFF8;
    const uint16_t m1_approx = (a & 0x07) | (b_low & 0x07);
    const uint16_t m1_out = m1_exact | m1_approx;

    // Multiplier-2 (Upper 4 bits)
    const uint16_t m2_exact = (a * b_high) & 0xFF00;
    const uint16_t m2_approx = a | b_high;  // Simplified
    const uint16_t m2_out = m2_exact | m2_approx;

    // Final computation - combine and extract top 8 bits
    return (uint8_t)((m1_out | (m2_out << 4)) >> 8);
}

// ==========================================
// PART 2: FP16 Approximate Multiplication
// ==========================================
// Union for safe bit manipulation
union Float16Bits {
    at::Half h;
    uint16_t u;
};

// Approximate FP16 multiplication
// Returns float for accurate accumulation
inline float approx_half_scalar(at::Half h_a, at::Half h_b) {
    Float16Bits a, b, result;
    a.h = h_a;
    b.h = h_b;

    // Extract sign (1 bit)
    const uint16_t sign_a = (a.u >> 15) & 0x1;
    const uint16_t sign_b = (b.u >> 15) & 0x1;
    const uint16_t sign_out = sign_a ^ sign_b;

    // Extract exponent (5 bits, bias = 15)
    const uint16_t exp_a = (a.u >> 10) & 0x1F;
    const uint16_t exp_b = (b.u >> 10) & 0x1F;

    // Handle zeros early
    if (__builtin_expect((exp_a == 0) | (exp_b == 0), 0)) {
        return 0.0f;
    }

    // Extract mantissa (10 bits)
    const uint16_t man_a = a.u & 0x03FF;
    const uint16_t man_b = b.u & 0x03FF;

    // Add hidden bit (bit 10 for normalized numbers)
    const uint16_t full_man_a = 0x0400 | man_a;  // Add implicit 1
    const uint16_t full_man_b = 0x0400 | man_b;

    // Truncate to 8 bits for PBOM8 (keep top 8 bits of 11-bit mantissa)
    const uint8_t man_a_8 = full_man_a >> 3;
    const uint8_t man_b_8 = full_man_b >> 3;

    // Approximate mantissa multiplication
    uint8_t prod_8 = pbom8_mantissa_mult(man_a_8, man_b_8);

    // Compute output exponent
    int exp_out = int(exp_a) + int(exp_b) - 15;  // Subtract bias

    // Normalize if needed (check if top bit is set)
    if (!(prod_8 & 0x80)) {
        prod_8 <<= 1;
        exp_out -= 1;
    }

    // Clamp exponent to valid FP16 range [0, 31]
    if (exp_out <= 0) {
        exp_out = 0;
        prod_8 = 0;  // Flush to zero
    }
    if (exp_out >= 31) {
        exp_out = 31;  // Infinity
        prod_8 = 0;
    }

    // Remove hidden bit and shift to mantissa position
    const uint16_t man_out = (uint16_t(prod_8 & 0x7F)) << 3;

    // Pack back to FP16
    result.u = (sign_out << 15) | (uint16_t(exp_out) << 10) | (man_out & 0x03FF);

    // Convert to float for accumulation (prevents precision loss)
    return float(result.h);
}

// ==========================================
// PART 3: OPTIMIZED Batched Matrix Multiplication
// ==========================================
torch::Tensor pbo_product_tensor(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dim() == 3, "Input a must be 3D");
    TORCH_CHECK(b.dim() == 3, "Input b must be 3D");
    TORCH_CHECK(a.size(0) == b.size(0), "Batch sizes must match");
    TORCH_CHECK(a.size(2) == b.size(1), "Inner dimensions must match");
    TORCH_CHECK(a.scalar_type() == torch::kFloat16, "Input must be Float16");
    TORCH_CHECK(b.scalar_type() == torch::kFloat16, "Input must be Float16");
    
    a = a.contiguous();
    b = b.contiguous();
    
    const int B = a.size(0);  // batch size
    const int M = a.size(1);  // rows of a
    const int K = a.size(2);  // cols of a, rows of b
    const int N = b.size(2);  // cols of b
    
    // Create output as Float16
    auto result = torch::zeros({B, M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(a.device()));
    
    // Get raw pointers
    at::Half* __restrict__ a_ptr = a.data_ptr<at::Half>();
    at::Half* __restrict__ b_ptr = b.data_ptr<at::Half>();
    at::Half* __restrict__ res_ptr = result.data_ptr<at::Half>();
    
    // Parallelize over batch dimension
    #pragma omp parallel for schedule(dynamic)
    for(int batch = 0; batch < B; batch++) {
        const int batch_offset_a = batch * M * K;
        const int batch_offset_b = batch * K * N;
        const int batch_offset_res = batch * M * N;
        
        // Loop reordering (i-k-j) for better cache locality
        for(int i = 0; i < M; i++) {
            const int row_offset_a = batch_offset_a + i * K;
            const int row_offset_res = batch_offset_res + i * N;
            
            // Accumulate in float for precision, then convert back
            std::vector<float> row_acc(N, 0.0f);
            
            for(int k = 0; k < K; k++) {
                const at::Half a_ik = a_ptr[row_offset_a + k];
                const int col_offset_b = batch_offset_b + k * N;
                
                // Manual loop unrolling (process 4 elements at a time)
                int j = 0;
                for(; j + 3 < N; j += 4) {
                    row_acc[j + 0] += approx_half_scalar(a_ik, b_ptr[col_offset_b + j + 0]);
                    row_acc[j + 1] += approx_half_scalar(a_ik, b_ptr[col_offset_b + j + 1]);
                    row_acc[j + 2] += approx_half_scalar(a_ik, b_ptr[col_offset_b + j + 2]);
                    row_acc[j + 3] += approx_half_scalar(a_ik, b_ptr[col_offset_b + j + 3]);
                }
                
                // Handle remaining elements
                for(; j < N; j++) {
                    row_acc[j] += approx_half_scalar(a_ik, b_ptr[col_offset_b + j]);
                }
            }
            
            // Convert accumulated floats back to FP16
            for(int j = 0; j < N; j++) {
                res_ptr[row_offset_res + j] = at::Half(row_acc[j]);
            }
        }
    }
    
    return result;
}

// ==========================================
// PART 4: Python Bindings
// ==========================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pbo_product", &pbo_product_tensor, 
          "PBOM8-73Y Optimized Batched Matrix Multiplication (FP16)");
}
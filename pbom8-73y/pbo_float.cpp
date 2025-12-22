#include <torch/extension.h>
#include <cmath>
#include <cstdint>
#include <vector>

// ==========================================
// PART 1: The "Broken Chip" Logic (PBOM8-73Y)
// ==========================================
// Simulates the hardware behavior on 8-bit integers.
// Input: Two 8-bit integers (representing the Mantissa)
// Output: 8-bit approximate product
inline uint8_t pbom8_mantissa_mult(uint8_t val_a, uint8_t val_b) {
    // Cast to int to prevent overflow during intermediate calcs
    int a = (int)val_a;
    int b = (int)val_b;

    // --- PBOM8-73Y ARCHITECTURE ---
    // Ref: Kumari et al. (2025)
    
    // Split B into nibbles (4 bits each)
    int b_low = b & 0x0F;
    int b_high = (b >> 4) & 0x0F;

    // Multiplier-1 (Lower 4 bits) 
    // Rule: Columns 2-3 are OR gates. Top bits are Exact.
    int m1_exact = (a * b_low) & 0xFFF8;       // Keep top 5 bits exact
    int m1_approx = (a & 0x07) | (b_low & 0x07); // Bottom 3 bits are OR'd
    int m1_out = m1_exact | m1_approx;

    // Multiplier-2 (Upper 4 bits)
    // Rule: Columns 2-7 are OR gates. Only the very top bit is safe-ish.
    int m2_exact = (a * b_high) & 0xFF00;      // Keep top bits exact
    int m2_approx = (a & 0xFF) | (b_high & 0xFF); // Everything else is OR'd
    int m2_out = m2_exact | m2_approx;

    // Final Glue (The 'Y' factor)
    // Rule: The final adder is replaced by an OR gate
    int product = m1_out | (m2_out << 4);
    
    // NORMALIZE FOR BFLOAT16 MANTISSA
    // We multiplied two 8-bit numbers (result is ~16 bits).
    // We need to return the top 8 bits to fit back into the Mantissa slot.
    return (uint8_t)(product >> 8); 
}

// ==========================================
// PART 2: The Bit-Hacking (Float -> Int -> Float)
// ==========================================
// Takes a standard float, treats it as BFloat16, runs the broken math.
float approx_bfloat16_scalar(float f_a, float f_b) {
    // 1. Bit Cast to see raw bits
    uint32_t bits_a = *reinterpret_cast<uint32_t*>(&f_a);
    uint32_t bits_b = *reinterpret_cast<uint32_t*>(&f_b);

    // 2. Extract Components (Standard Float32 Layout)
    // Sign (1 bit), Exponent (8 bits), Mantissa (23 bits)
    
    uint32_t sign_a = (bits_a >> 31) & 0x1;
    uint32_t exp_a  = (bits_a >> 23) & 0xFF;
    // We only grab the top 7 bits of the mantissa (matching BFloat16 precision)
    uint32_t man_a  = (bits_a >> 16) & 0x7F; 
    
    uint32_t sign_b = (bits_b >> 31) & 0x1;
    uint32_t exp_b  = (bits_b >> 23) & 0xFF;
    uint32_t man_b  = (bits_b >> 16) & 0x7F;

    // 3. Handle Special Zero Case (Avoid garbage output)
    if (exp_a == 0 || exp_b == 0) return 0.0f;

    // 4. Calculate Exact Sign and Exponent
    uint32_t sign_out = sign_a ^ sign_b;
    uint32_t exp_out = exp_a + exp_b - 127; // Subtract Bias

    // 5. Approximate Mantissa Multiplication
    // Add the "Hidden 1" to make it a full 8-bit number
    uint8_t val_a = (1 << 7) | man_a; 
    uint8_t val_b = (1 << 7) | man_b;

    // *** INJECT BROKEN LOGIC HERE ***
    uint8_t approx_man_full = pbom8_mantissa_mult(val_a, val_b);

    // Remove the hidden 1 (mask 0x7F) to store it back
    uint32_t man_out = approx_man_full & 0x7F;

    // 6. Repack into Float32
    uint32_t bits_out = (sign_out << 31) | (exp_out << 23) | (man_out << 16);
    return *reinterpret_cast<float*>(&bits_out);
}

// ==========================================
// PART 3: The Tensor Matrix Multiplication
// ==========================================
// Performs actual matrix multiplication: [M, K] @ [K, N] -> [M, N]
torch::Tensor pbo_product_tensor(torch::Tensor a, torch::Tensor b) {
    // Ensure inputs are contiguous and 2D
    TORCH_CHECK(a.dim() == 2, "Input a must be 2D");
    TORCH_CHECK(b.dim() == 2, "Input b must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "Inner dimensions must match");
    
    a = a.contiguous();
    b = b.contiguous();
    
    int M = a.size(0);  // rows of a
    int K = a.size(1);  // cols of a, rows of b
    int N = b.size(1);  // cols of b
    
    // Create output tensor [M, N]
    auto result = torch::zeros({M, N}, a.options());
    
    // Get accessors
    auto a_acc = a.accessor<float, 2>();
    auto b_acc = b.accessor<float, 2>();
    auto res_acc = result.accessor<float, 2>();
    
    // Perform matrix multiplication with approximate multiplier
    // C[i,j] = sum(A[i,k] * B[k,j]) for k=0 to K-1
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            float sum = 0.0f;
            for(int k = 0; k < K; k++) {
                // Use approximate multiplication for each element
                sum += approx_bfloat16_scalar(a_acc[i][k], b_acc[k][j]);
            }
            res_acc[i][j] = sum;
        }
    }
    
    return result;
}

// ==========================================
// PART 4: The Python Binding
// ==========================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // We bind "pbo_product" to the TENSOR function, not the scalar one!
  m.def("pbo_product", &pbo_product_tensor, "PBOM8-73Y Tensor Product");
}
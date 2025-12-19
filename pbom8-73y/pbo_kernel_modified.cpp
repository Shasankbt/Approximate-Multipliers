#include <torch/extension.h>
#include <vector>

// The C++ implementation of your approximate multiplier
// This runs on the CPU, but MUCH faster than Python loops
torch::Tensor pbo_product_cpu(torch::Tensor a, torch::Tensor b) {
    
    // Ensure inputs are Int32
    auto a_acc = a.accessor<int, 2>(); // Assuming 2D tensor for simplicity
    auto b_acc = b.accessor<int, 2>();
    
    int rows = a.size(0);
    int cols = a.size(1);
    
    auto result = torch::zeros_like(a);
    auto res_acc = result.accessor<int, 2>();

    // The loops here are compiled to machine code! 
    // No Python overhead per iteration.
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            
            int val_a = a_acc[i][j];
            int val_b = b_acc[i][j];
            
            // --- PBOM8-73Y LOGIC (Same as before, but in C++) ---
            int abs_a = std::abs(val_a);
            int abs_b = std::abs(val_b);
            int sign = ((val_a ^ val_b) < 0) ? -1 : 1; // Fast XOR sign check

            int b_low = abs_b & 0x0F;
            int b_high = (abs_b >> 4) & 0x0F;

            // Multiplier-1 (Lower)
            int m1_exact = (abs_a * b_low) & 0xFFF8;
            int m1_approx = (abs_a & 0x07) | (b_low & 0x07);
            int m1_out = m1_exact | m1_approx;

            // Multiplier-2 (Upper)
            int m2_exact = (abs_a * b_high) & 0xFF00;
            int m2_approx = (abs_a & 0xFF) | (b_high & 0xFF);
            int m2_out = m2_exact | m2_approx;

            int final_abs = m1_out | (m2_out << 4);
            
            res_acc[i][j] = final_abs * sign;
        }
    }
    return result;
}

// The Glue (PyBind11)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pbo_product", &pbo_product_cpu, "PBOM8-73Y Product (CPU)");
}
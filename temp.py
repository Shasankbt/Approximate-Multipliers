from torch.utils.cpp_extension import load
import torch

# 1. Compile
# Make sure the file above is saved as "pbom8-73y/pbo_float.cpp"
pbo_cpp = load(name="pbo_extension", sources=["pbom8-73y/pbo_float.cpp"], verbose=True)

# # 2. Test Data (2D Tensors)
# a = torch.tensor([[2.2, 4.5], [10.0, 3.4]], dtype=torch.float32)
# b = torch.tensor([[1.5, 2.0], [5.0, 3.2]], dtype=torch.float32)

# # 3. Run
# print("Exact Math:\n", a.matmul(b))
# approx_output = pbo_cpp.pbo_product(a, b)
# print("\nApproximate BFloat16 Math:\n", approx_output)

from evaluate import run_evaluation

from approximate_multiplier import ApproximateMultiplier

am = ApproximateMultiplier(pbo_cpp.pbo_product)

am.enable()

run_evaluation()
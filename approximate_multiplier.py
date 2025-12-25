import torch
import torch.nn.functional as F
import time

'''
Matrix Multiplication Operations in PyTorch - Complete Coverage

The types of matrix operations are:

1. torch.matmul(A, B) : Two-argument function for general matrix multiplication
   - Handles 2D, 3D, and higher dimensional batched operations
   - Example: torch.matmul(x, weight.t())

2. torch.Tensor.__matmul__(self, other) : One-arg member function of a tensor (@ operator)
   - Called when using the @ operator: A @ B
   - Example: Q @ K.transpose(-2, -1)

3. torch.mm(A, B) : Two-argument function for strict 2D matrix multiplication
   - Only works with 2D tensors [M, K] @ [K, N] -> [M, N]
   - Example: torch.mm(x, weight.t())

4. torch.bmm(A, B) : Two-argument function for batched matrix multiplication
   - Only works with 3D tensors [B, M, K] @ [B, K, N] -> [B, M, N]
   - Example: torch.bmm(Q, K.transpose(-2, -1))

5. F.linear(input, weight, bias) : High-level linear transformation
   - Internally does: input @ weight.t() + bias
   - Used by nn.Linear layers
   - Example: F.linear(x, self.weight, self.bias)

Additional related operations NOT covered here (element-wise, not matrix mult):
- torch.mul() / * operator : Element-wise multiplication, not matrix multiplication
- torch.dot() : 1D dot product only
- torch.mv() : Matrix-vector product

Coverage Strategy:
- By replacing functions 1-5 above, we capture ALL matrix multiplications in PyTorch
- This includes operations inside nn.Linear, nn.MultiheadAttention, and custom code
- Both explicit (torch.matmul) and implicit (@ operator) operations are covered
'''
class ApproximateMultiplier:
    """
    Simple class that replaces ALL matrix multiplications with approximate ones.
    Just call .enable() before running your model.
    """
    def __init__(self, pbo_function):
        self.enabled = False
        self.originals = {}
        self.pbo_fn = pbo_function
        
    def enable(self):
        """Turn on approximate multiplication everywhere"""
        if self.enabled:
            return
            
        # Save original functions
        self.originals = {
            'matmul': torch.matmul,
            'bmm': torch.bmm,
            'mm': torch.mm,
            'linear': F.linear,
            'tensor_matmul': torch.Tensor.__matmul__,
        }
        
        # Replace them with approximate versions
        torch.matmul = lambda a, b: self._approx_matmul(a, b)
        torch.bmm = lambda a, b: self._approx_matmul(a, b)
        torch.mm = lambda a, b: self._approx_matmul(a, b)
        F.linear = lambda input, weight, bias=None: self._approx_linear(input, weight, bias)
        torch.Tensor.__matmul__ = lambda self_tensor, other: self._approx_matmul(self_tensor, other)
        
        self.enabled = True
        print("<-- Approximate multiplication enabled -->")
        
    def disable(self):
        """Turn off approximate multiplication (back to exact)"""
        if not self.enabled:
            return
            
        torch.matmul = self.originals['matmul']
        torch.bmm = self.originals['bmm']
        torch.mm = self.originals['mm']
        F.linear = self.originals['linear']
        torch.Tensor.__matmul__ = self.originals['tensor_matmul']
        
        self.enabled = False
        print("<-- Exact multiplication restored -->")
    
    def _approx_matmul(self, a, b):
        """Approximate version of matmul"""
        # print(f"shapes: {a.shape}, {b.shape}")
        if a.dim() == 2 and b.dim() == 2:
            a_batched = a.unsqueeze(0)  # [1, M, K]
            b_batched = b.unsqueeze(0)  # [1, K, N]
            return self._pbom8_2d(a_batched, b_batched).squeeze(0)
        
        elif a.dim() == 3 and b.dim() == 2:
            B, M, K = a.shape
            K2, N = b.shape
            assert K == K2
            
            # Expand b to match batch size: [K, N] -> [B, K, N]
            b_batched = b.unsqueeze(0).expand(B, K, N)
            result = self._pbom8_2d(a, b_batched)
            return result  # [B, M, N]
        elif a.dim() == 3 and b.dim() == 3:
            assert a.shape[2] == b.shape[1]
            return self._pbom8_2d(a, b)
        elif a.dim() == 4 and b.dim() == 4:
            B, H, M, K = a.shape
            B2, H2, K2, N = b.shape
            assert B == B2 and H == H2 and K == K2
            
            # Reshape: [B, H, M, K] -> [B*H, M, K]
            a_3d = a.reshape(B * H, M, K)
            b_3d = b.reshape(B * H, K, N)
            
            result_3d = self._pbom8_2d(a_3d.contiguous(), b_3d.contiguous())
            
            # Reshape back: [B*H, M, N] -> [B, H, M, N]
            return result_3d.reshape(B, H, M, N)
        else:
            # Fallback for weird shapes
            print("using original")
            return self.originals['matmul'](a, b)
    
    def _approx_linear(self, input, weight, bias=None):
        """Approximate version of F.linear"""
        output = self._approx_matmul(input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    def _make_approx_tensor_matmul(multiplier_instance):
        def _approx_tensor_matmul(self_tensor, other):
            return multiplier_instance._approx_matmul(self_tensor, other)
        return _approx_tensor_matmul
    
    def _pbom8_2d(self, a, b):
        # print(f"shape of inputs : {a.shape}, {b.shape}")
        # print(f"dtype of tensors: {a.dtype}")
        t = time.time()

        B, M, K = a.shape
        B2, K2, N = b.shape
        assert B == B2, f"Batch sizes must must match: {B} != {B2}"
        assert K == K2, f"Inner dimensions must match: {K} != {K2}"
        
        tensor_device = a.device
        tensor_dtype = a.dtype
        a = a.to("cpu") 
        b = b.to("cpu")
        a = a.to(torch.float16)
        b = b.to(torch.float16)

        # print(f"conversion time: {time.time() - t}")
        
        result = self.pbo_fn(a, torch.transpose(b, 1, 2))

        result = result.to(tensor_device)
        result = result.to(tensor_dtype)

        # print(f"time elapsed: {time.time() - t}")
        # print(f"shape of result: {result.shape}")
        # print(f"tensor: {result}")
        return result
    
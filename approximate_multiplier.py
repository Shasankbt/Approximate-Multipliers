import torch
import torch.nn.functional as F

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
        """
        Universal Approximate Matmul.
        Handles 2D, 3D, 4D, and Broadcasting (e.g., 3D @ 2D).
        """
        # 1. Handle cases where inputs are vectors (1D)
        # torch.matmul handles these by temporarily adding dimensions
        is_a_vec = a.dim() == 1
        is_b_vec = b.dim() == 1
        
        if is_a_vec: a = a.unsqueeze(0)
        if is_b_vec: b = b.unsqueeze(1)

        # 2. Get Matrix Dimensions (Last 2 dims)
        M, K = a.shape[-2:]
        K2, N = b.shape[-2:]
        
        if K != K2:
            # Let PyTorch raise the error or handle it (fallback)
            return self.originals['matmul'](a, b)

        # 3. Handle Batch Dimensions (Broadcasting)
        # Example: a=[32, 10, 64], b=[64, 128] -> Batch dims: [32, 10] and []
        batch_a = a.shape[:-2]
        batch_b = b.shape[:-2]
        
        try:
            # Calculate the combined batch shape (e.g. broadcasting [32, 10] + [] -> [32, 10])
            common_batch_shape = torch.broadcast_shapes(batch_a, batch_b)
        except RuntimeError:
            print("Broadcasting failed, using original")
            return self.originals['matmul'](a, b)
            
        # 4. Expand and Reshape to 3D: [Total_Batch, M, K] @ [Total_Batch, K, N]
        # We merge all batch dimensions (Batch, Heads, Seq...) into one 'B' dimension
        a_expanded = a.expand(common_batch_shape + (M, K))
        b_expanded = b.expand(common_batch_shape + (K, N))
        
        a_flat = a_expanded.reshape(-1, M, K)
        b_flat = b_expanded.reshape(-1, K, N)
        
        total_batch = a_flat.shape[0]
        results = []

        # 5. Run the Approximation Loop
        # Note: This loop effectively processes [Batch * Heads * Seq] items
        for i in range(total_batch):
            results.append(self._pbom8_2d(a_flat[i], b_flat[i]))
            
        # 6. Reassemble
        res_stacked = torch.stack(results, dim=0) # [Total_Batch, M, N]
        
        # Reshape back to [Common_Batch, M, N]
        final_shape = common_batch_shape + (M, N)
        res_shaped = res_stacked.view(final_shape)
        
        # 7. Remove dummy dimensions if inputs were vectors
        if is_a_vec: res_shaped = res_shaped.squeeze(-2)
        if is_b_vec: res_shaped = res_shaped.squeeze(-1)
            
        return res_shaped
    
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
        # print(f"shape of inputs : {a}, {b}")

        M, K = a.shape
        K2, N = b.shape
        assert K == K2, f"Inner dimensions must match: {K} != {K2}"
        
        tensor_device = a.device
        a = a.to("cpu") 
        b = b.to("cpu")
        result = self.pbo_fn(a, b)
        result = result.to(tensor_device)
        # print(f"shape of result: {result.shape}")
        # print(f"tensor: {result}")
        return result
    
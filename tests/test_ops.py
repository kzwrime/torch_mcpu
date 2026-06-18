# Owner(s): ["module: PrivateUse1"]

import collections
import functools
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch_mcpu  # noqa: F401
from torch.nn.attention import SDPBackend
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


SDPAShape = collections.namedtuple(
    "Sdpa_Shape", ["batch", "num_heads", "seq_len", "head_dim"]
)


class TestFactory(TestCase):
    def test_empty(self):
        """Test empty tensor creation"""
        x = torch.empty(3, device="mcpu")
        self.assertEqual(x.device.type, "mcpu")
        self.assertEqual(x.shape, torch.Size([3]))

        x = torch.empty([2, 3, 4, 5], device="mcpu", names=["N", "C", "H", "W"])
        self.assertEqual(x.device.type, "mcpu")
        self.assertEqual(x.shape, torch.Size([2, 3, 4, 5]))

        with torch._subclasses.fake_tensor.FakeTensorMode():
            x = torch.empty(3, 3, device="mcpu")
            y = torch.empty(3, 3, device="mcpu:0")
            z = x + y
            self.assertEqual(z.device.type, "mcpu")
            self.assertEqual(z.shape, torch.Size([3, 3]))

    def test_zeros(self):
        """Test zeros tensor creation"""
        y = torch.zeros(3, device="mcpu")
        self.assertEqual(y.device.type, "mcpu")
        self.assertEqual(y.shape, torch.Size([3]))

    def test_tensor(self):
        """Test tensor creation from empty tuple"""
        z = torch.tensor((), device="mcpu")
        self.assertEqual(z.device.type, "mcpu")
        self.assertEqual(z.shape, torch.Size([0]))


class TestCopy(TestCase):
    def test_copy_same_device(self):
        """Test copy operation on same device"""
        a = torch.ones(10, device="mcpu").clone()
        self.assertEqual(a.cpu(), torch.ones(10))

    def test_copy_same_device_last_dim_contiguous_views(self):
        """Test same-device copy for views with contiguous innermost dim."""
        src_cpu = torch.randn(3, 7, 5)
        dst_cpu = torch.empty(3, 7, 5)
        src = torch.empty(3, 14, 5, device="mcpu")
        dst = torch.empty(3, 14, 5, device="mcpu")
        src_view = src[:, ::2, :]
        dst_view = dst[:, 1::2, :]
        src_view.copy_(src_cpu.to("mcpu"))
        dst_view.copy_(src_view)
        dst_cpu.copy_(dst_view.cpu())
        self.assertEqual(dst_cpu, src_cpu)

    def test_copy_same_device_coalesces_inner_dims(self):
        """Test copy where multiple trailing dims are contiguous."""
        src_cpu = torch.randn(2, 3, 4)
        src = torch.empty(2, 6, 4, device="mcpu")
        dst = torch.empty(2, 6, 4, device="mcpu")
        src_view = src[:, ::2, :]
        dst_view = dst[:, ::2, :]
        src_view.copy_(src_cpu.to("mcpu"))
        dst_view.copy_(src_view)
        self.assertEqual(dst_view.cpu(), src_cpu)

    def test_copy_same_device_non_contiguous_last_dim_fallback(self):
        """Test copy fallback still handles non-contiguous innermost dim."""
        src_cpu = torch.randn(4, 3)
        dst_cpu = torch.empty(4, 3)
        src = src_cpu.t().contiguous().to("mcpu").t()
        dst_base = torch.empty(3, 4, device="mcpu")
        dst = dst_base.t()
        dst.copy_(src)
        dst_cpu.copy_(dst.cpu())
        self.assertEqual(dst_cpu, src_cpu)

    def test_copy_same_device_overlapping_view_fallback(self):
        """Test same-storage overlapping views keep copy_ overlap semantics."""
        base = torch.arange(6.0, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "single memory location"):
            base[1:].copy_(base[:-1])

    def test_cross_device_copy(self):
        """Test copy operation across CPU and mcpu"""
        a = torch.rand(10)
        b = a.to(device="mcpu").add(2).to(device="cpu")
        self.assertEqual(b, a + 2)

    def test_cross_diff_devices_copy(self):
        """Test copy operation across different mcpu devices"""
        a = torch.ones(10, device="mcpu:0").to(device="mcpu:1").to(device="cpu")
        self.assertEqual(a, torch.ones(10))


class TestOps(TestCase):
    def test_masked_select(self):
        """Test masked_select does not silently fallback to CPU."""
        tensor_cpu = torch.randn(10)
        tensor_mcpu = tensor_cpu.to(device="mcpu")
        mask = tensor_mcpu.gt(0)
        with self.assertRaisesRegex(RuntimeError, "aten::masked_select"):
            torch.masked_select(tensor_mcpu, mask)

    def test_expand(self):
        """Test tensor expand operation"""
        x = torch.tensor([[1], [2], [3]], device="mcpu")
        y = x.expand(3, 2)
        self.assertEqual(y.to(device="cpu"), torch.tensor([[1, 1], [2, 2], [3, 3]]))
        self.assertEqual(x.data_ptr(), y.data_ptr())

    def test_resize(self):
        """Test tensor resize operation"""
        tensor_cpu = torch.randn([4, 4])

        tensor_mcpu = tensor_cpu.mcpu()
        self.assertTrue(tensor_mcpu.size() == torch.Size([4, 4]))

        storage_mcpu = tensor_mcpu.storage()
        self.assertTrue(storage_mcpu.size() == 16)

        tensor_mcpu.resize_(2, 2, 2, 2)
        self.assertTrue(tensor_mcpu.size() == torch.Size([2, 2, 2, 2]))

        storage_mcpu = tensor_mcpu.storage()
        self.assertTrue(storage_mcpu.size() == 16)

    def test_printing(self):
        """Test tensor printing"""
        a = torch.ones(20, device="mcpu")
        print(a.cpu())

    def test_fill_scalar_last_dim_contiguous_view(self):
        base = torch.zeros(2, 6, 4, dtype=torch.float32, device="mcpu")
        view = base[:, ::2, :]

        view.fill_(3.5)

        expected = torch.zeros(2, 6, 4, dtype=torch.float32)
        expected[:, ::2, :] = 3.5
        self.assertEqual(base.cpu(), expected)


class TestSTUB(TestCase):
    def test_backend_dispatchstub(self):
        """Test backend dispatch stub for abs operation"""
        x_cpu = torch.randn(2, 2, 3, dtype=torch.float32, device="cpu")
        x_mcpu = x_cpu.to("mcpu")

        y_cpu = torch.abs(x_cpu)
        y_mcpu = torch.abs(x_mcpu)
        self.assertEqual(y_cpu, y_mcpu.cpu())

        o_cpu = torch.randn(2, 2, 6, dtype=torch.float32, device="cpu")
        o_mcpu = o_cpu.to("mcpu")
        # output operand with resize flag is False in TensorIterator.
        torch.abs(x_cpu, out=o_cpu[:, :, 0:6:2])
        torch.abs(x_mcpu, out=o_mcpu[:, :, 0:6:2])
        self.assertEqual(o_cpu, o_mcpu.cpu())

        # output operand with resize flag is True in TensorIterator and
        # convert output to contiguous tensor in TensorIterator.
        torch.abs(x_cpu, out=o_cpu[:, :, 0:6:3])
        torch.abs(x_mcpu, out=o_mcpu[:, :, 0:6:3])
        self.assertEqual(o_cpu, o_mcpu.cpu())


class TestQuantization(TestCase):
    def test_quantize(self):
        """Test quantization per tensor"""
        x = torch.randn(3, 4, 5, dtype=torch.float32, device="mcpu")
        quantized_tensor = torch.quantize_per_tensor(x, 0.1, 10, torch.qint8)
        self.assertEqual(quantized_tensor.device, torch.device("mcpu:0"))
        self.assertEqual(quantized_tensor.dtype, torch.qint8)


class TestAutogradFunction(TestCase):
    def test_compile_autograd_function_returns_self(self):
        """Test compiled autograd function that returns self"""
        in_ref = torch.randn(4, device="mcpu", requires_grad=True)
        out_ref = torch.ops.mcpu.custom_autograd_fn_returns_self(in_ref)
        out_ref.sum().backward()

        in_test = in_ref.detach().clone().requires_grad_(True)
        # TODO(FFFrog): Need to support inductor for OpenReg first.
        out_test = torch.compile(backend="aot_eager")(
            torch.ops.mcpu.custom_autograd_fn_returns_self
        )(in_test)
        out_test.sum().backward()

        self.assertEqual(out_ref.cpu(), out_test.cpu())
        self.assertEqual(in_ref.grad.cpu(), in_test.grad.cpu())

    @skipIfTorchDynamo("Temporary disabled due to torch._ops.OpOverloadPacket")
    def test_compile_autograd_function_aliasing(self):
        """Test compiled autograd function with aliasing"""
        in_ref = torch.randn(4, device="mcpu", requires_grad=True)
        out_ref = torch.ops.mcpu.custom_autograd_fn_aliasing(in_ref)
        out_ref.sum().backward()

        in_test = in_ref.detach().clone().requires_grad_(True)
        # TODO(FFFrog): Need to support inductor for OpenReg first.
        out_test = torch.compile(backend="aot_eager")(
            torch.ops.mcpu.custom_autograd_fn_aliasing
        )(in_test)
        out_test.sum().backward()

        self.assertEqual(out_ref.cpu(), out_test.cpu())
        self.assertEqual(in_ref.grad.cpu(), in_test.grad.cpu())


class TestFallback(TestCase):
    def test_unimplemented_factory_does_not_fallback_to_cpu(self):
        with self.assertRaisesRegex(
            RuntimeError, "aten::triu_indices.*docs/how_to_impl_aten_ops.md"
        ):
            torch.triu_indices(3, 3, device="mcpu")

    def test_tensor_ops_use_explicit_mcpu_kernels(self):
        x = torch.Tensor([[1, 2, 3], [2, 3, 4]]).to("mcpu")
        y = torch.Tensor([1, 0, 2]).to("mcpu")
        self.assertTrue(x.device.type, "mcpu")
        self.assertFalse(x.is_cpu)

        z_cpu = torch.Tensor([[0, 2, 1], [1, 3, 2]])
        z = torch.sub(x, y)
        self.assertEqual(z_cpu, z.cpu())

        z_cpu = torch.Tensor([3, 1])
        y = torch.Tensor([1, 0]).long().to("mcpu")
        z = x[y, y]
        self.assertEqual(z_cpu, z.cpu())

    def test_explicit_forward_ops_do_not_fallback(self):
        x = torch.tensor([[1, 2, 3], [4, 5, 6]], device="mcpu", dtype=torch.float32)
        idx = torch.tensor([1, 0], device="mcpu", dtype=torch.long)

        arange_res = torch.arange(1, 7, 2, device="mcpu", dtype=torch.int64)
        torch.mcpu.synchronize()
        self.assertEqual(arange_res.cpu(), torch.tensor([1, 3, 5], dtype=torch.int64))
        self.assertEqual(arange_res.device.type, "mcpu")

        cumsum_res = torch.cumsum(x, dim=1)
        self.assertEqual(
            cumsum_res.cpu(),
            torch.tensor([[1, 3, 6], [4, 9, 15]], dtype=torch.float32),
        )
        self.assertEqual(cumsum_res.device.type, "mcpu")

        int_cumsum = torch.cumsum(
            torch.tensor([1, 1, 1], device="mcpu", dtype=torch.int8),
            dim=0,
        )
        self.assertEqual(int_cumsum.dtype, torch.int64)
        self.assertEqual(int_cumsum.cpu(), torch.tensor([1, 2, 3], dtype=torch.int64))

        bool_cumsum = torch.cumsum(
            torch.tensor([True, True, False], device="mcpu"),
            dim=0,
        )
        self.assertEqual(bool_cumsum.dtype, torch.int64)
        self.assertEqual(bool_cumsum.cpu(), torch.tensor([1, 2, 2], dtype=torch.int64))

        uniform_res = torch.empty(8, device="mcpu")
        self.assertIs(uniform_res.uniform_(0.0, 1.0), uniform_res)
        uniform_cpu = uniform_res.cpu()
        self.assertTrue((uniform_cpu >= 0.0).all())
        self.assertTrue((uniform_cpu < 1.0).all())

        uniform_func = torch.ops.aten.uniform.default(uniform_res, 0.0, 1.0)
        uniform_func_cpu = uniform_func.cpu()
        self.assertEqual(uniform_func.device.type, "mcpu")
        self.assertTrue((uniform_func_cpu >= 0.0).all())
        self.assertTrue((uniform_func_cpu < 1.0).all())

        uniform_out = torch.empty_like(uniform_res)
        self.assertIs(
            torch.ops.aten.uniform.out(uniform_res, 0.0, 1.0, out=uniform_out),
            uniform_out,
        )
        uniform_out_cpu = uniform_out.cpu()
        self.assertTrue((uniform_out_cpu >= 0.0).all())
        self.assertTrue((uniform_out_cpu < 1.0).all())

        rand_res = torch.rand(8, device="mcpu")
        rand_cpu = rand_res.cpu()
        self.assertEqual(rand_res.device.type, "mcpu")
        self.assertTrue((rand_cpu >= 0.0).all())
        self.assertTrue((rand_cpu < 1.0).all())

        rand_out = torch.empty(8, device="mcpu")
        self.assertIs(torch.rand(8, out=rand_out), rand_out)
        rand_out_cpu = rand_out.cpu()
        self.assertTrue((rand_out_cpu >= 0.0).all())
        self.assertTrue((rand_out_cpu < 1.0).all())

        rand_like = torch.rand_like(uniform_res)
        rand_like_cpu = rand_like.cpu()
        self.assertEqual(rand_like.device.type, "mcpu")
        self.assertTrue((rand_like_cpu >= 0.0).all())
        self.assertTrue((rand_like_cpu < 1.0).all())

        sigmoid_res = torch.sigmoid(x)
        self.assertEqual(sigmoid_res.cpu(), torch.sigmoid(x.cpu()))
        self.assertEqual(sigmoid_res.device.type, "mcpu")

        int_sigmoid = torch.sigmoid(
            torch.tensor([0, 1], device="mcpu", dtype=torch.int32)
        )
        self.assertEqual(int_sigmoid.dtype, torch.float32)
        self.assertEqual(
            int_sigmoid.cpu(),
            torch.sigmoid(torch.tensor([0, 1], dtype=torch.int32)),
        )

        silu_res = torch.nn.functional.silu(x)
        self.assertEqual(silu_res.cpu(), torch.nn.functional.silu(x.cpu()))
        self.assertEqual(silu_res.device.type, "mcpu")

        filled = torch.empty(2, 3, device="mcpu", dtype=torch.float32)
        filled.fill_(7)
        torch.mcpu.synchronize()
        self.assertEqual(filled.cpu(), torch.full((2, 3), 7, dtype=torch.float32))
        self.assertEqual(filled.device.type, "mcpu")

        indexed = x[idx, idx]
        self.assertEqual(indexed.cpu(), torch.tensor([5, 1], dtype=torch.float32))
        self.assertEqual(indexed.device.type, "mcpu")

        bad_out = torch.empty(1, device="mcpu", dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "aten::cumsum.out"):
            torch.cumsum(x, dim=1, out=bad_out)

        bad_index_out = torch.empty(1, device="mcpu", dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "aten::index.Tensor_out"):
            torch.ops.aten.index.Tensor_out(x, [idx, idx], out=bad_index_out)

    def test_explicit_forward_ops_batch_2(self):
        x = torch.tensor([[4.0, 6.0], [7.0, 5.0]], device="mcpu")
        y = torch.tensor([[10.0, 20.0], [31.0, 41.0]], device="mcpu")

        add_out = torch.empty_like(x)
        torch.add(x, y, out=add_out)
        self.assertEqual(add_out.cpu(), torch.tensor([[14.0, 26.0], [38.0, 46.0]]))

        div_out = torch.empty_like(x)
        torch.div(y, x, out=div_out)
        self.assertEqual(
            div_out.cpu(),
            torch.tensor([[2.5, 20.0 / 6.0], [31.0 / 7.0, 8.2]]),
        )

        mul_out = torch.empty_like(x)
        torch.mul(x, y, out=mul_out)
        self.assertEqual(mul_out.cpu(), torch.tensor([[40.0, 120.0], [217.0, 205.0]]))

        remainder_out = torch.empty_like(y)
        torch.remainder(y, x, out=remainder_out)
        self.assertEqual(
            remainder_out.cpu(),
            torch.tensor([[2.0, 2.0], [3.0, 1.0]]),
        )

        sub_res = torch.sub(y, x)
        self.assertEqual(sub_res.cpu(), torch.tensor([[6.0, 14.0], [24.0, 36.0]]))

        cos_out = torch.empty_like(x)
        torch.cos(x, out=cos_out)
        self.assertEqual(cos_out.cpu(), torch.cos(x.cpu()))

        sin_out = torch.empty_like(x)
        torch.sin(x, out=sin_out)
        self.assertEqual(sin_out.cpu(), torch.sin(x.cpu()))

        reciprocal_out = torch.empty_like(x)
        torch.reciprocal(x, out=reciprocal_out)
        self.assertEqual(reciprocal_out.cpu(), torch.reciprocal(x.cpu()))

        neg_out = torch.empty_like(x)
        torch.neg(x, out=neg_out)
        self.assertEqual(neg_out.cpu(), torch.neg(x.cpu()))

        sigmoid_out = torch.empty_like(x)
        torch.sigmoid(x, out=sigmoid_out)
        self.assertEqual(sigmoid_out.cpu(), torch.sigmoid(x.cpu()))

        sigmoid_inplace = x.clone()
        sigmoid_inplace.sigmoid_()
        self.assertEqual(sigmoid_inplace.cpu(), x.cpu().sigmoid_())

        silu_out = torch.empty_like(x)
        torch.ops.aten.silu.out(x, out=silu_out)
        self.assertEqual(silu_out.cpu(), torch.nn.functional.silu(x.cpu()))

        silu_inplace = x.clone()
        torch.ops.aten.silu_.default(silu_inplace)
        self.assertEqual(
            silu_inplace.cpu(),
            torch.ops.aten.silu_.default(x.cpu()),
        )

        clamp_out = torch.empty_like(x)
        torch.clamp(x, min=5.0, max=6.0, out=clamp_out)
        self.assertEqual(clamp_out.cpu(), torch.clamp(x.cpu(), min=5.0, max=6.0))

        clamp_tensor_min = torch.tensor([[5.0, 6.0], [4.0, 6.0]], device="mcpu")
        clamp_tensor_out = torch.empty_like(x)
        torch.clamp(x, min=clamp_tensor_min, out=clamp_tensor_out)
        self.assertEqual(
            clamp_tensor_out.cpu(),
            torch.clamp(x.cpu(), min=clamp_tensor_min.cpu()),
        )

        bad_unary_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::neg.out"):
            torch.neg(x, out=bad_unary_out)

        with self.assertRaisesRegex(RuntimeError, "aten::sigmoid.out"):
            torch.sigmoid(x, out=bad_unary_out)

        with self.assertRaisesRegex(RuntimeError, "aten::silu.out"):
            torch.ops.aten.silu.out(x, out=bad_unary_out)

        with self.assertRaisesRegex(RuntimeError, "aten::clamp.out"):
            torch.clamp(x, min=5.0, max=6.0, out=bad_unary_out)

        with self.assertRaisesRegex(RuntimeError, "aten::clamp.Tensor_out"):
            torch.clamp(x, min=clamp_tensor_min, out=bad_unary_out)

        scalar_pow_out = torch.empty_like(x)
        torch.pow(2.0, x, out=scalar_pow_out)
        self.assertEqual(scalar_pow_out.cpu(), torch.pow(2.0, x.cpu()))

        cat_out = torch.empty(4, 2, device="mcpu")
        torch.cat([x, y], dim=0, out=cat_out)
        self.assertEqual(cat_out.cpu(), torch.cat([x.cpu(), y.cpu()], dim=0))

        empty_1d = torch.empty(0, device="mcpu")
        cat_empty_out = torch.empty(2, 2, device="mcpu")
        torch.cat([empty_1d, x], dim=0, out=cat_empty_out)
        self.assertEqual(cat_empty_out.cpu(), x.cpu())

        vals = torch.empty(2, 1, device="mcpu")
        inds = torch.empty(2, 1, device="mcpu", dtype=torch.long)
        torch.topk(x, 1, dim=1, largest=True, sorted=True, out=(vals, inds))
        ref_vals, ref_inds = torch.topk(x.cpu(), 1, dim=1, largest=True, sorted=True)
        self.assertEqual(vals.cpu(), ref_vals)
        self.assertEqual(inds.cpu(), ref_inds)

        target = torch.zeros(3, 3, device="mcpu")
        row = torch.tensor([0, 2], device="mcpu", dtype=torch.long)
        col = torch.tensor([1, 0], device="mcpu", dtype=torch.long)
        vals = torch.tensor([5.0, 7.0], device="mcpu")
        torch.ops.aten._index_put_impl_(target, [row, col], vals, False, False)
        self.assertEqual(
            target.cpu(),
            torch.tensor([[0.0, 5.0, 0.0], [0.0, 0.0, 0.0], [7.0, 0.0, 0.0]]),
        )

        target.zero_()
        self.assertEqual(target.cpu(), torch.zeros(3, 3))

    def test_explicit_forward_ops_batch_3(self):
        x = torch.tensor([[0.0, 2.0], [3.0, 0.0]], device="mcpu")
        mask = torch.tensor([[True, False], [False, True]], device="mcpu")

        x.masked_fill_(mask, 9.0)
        self.assertEqual(x.cpu(), torch.tensor([[9.0, 2.0], [3.0, 9.0]]))

        ne_out = torch.empty_like(x, dtype=torch.bool)
        torch.ne(x, 9.0, out=ne_out)
        self.assertEqual(ne_out.cpu(), torch.ne(x.cpu(), 9.0))

        gt_scalar = torch.gt(x, 2.0)
        self.assertEqual(gt_scalar.cpu(), torch.gt(x.cpu(), 2.0))

        gt_scalar_out = torch.empty_like(x, dtype=torch.bool)
        torch.gt(x, 2.0, out=gt_scalar_out)
        self.assertEqual(gt_scalar_out.cpu(), torch.gt(x.cpu(), 2.0))

        gt_other = torch.tensor([[8.0], [2.0]], device="mcpu")
        gt_tensor = torch.gt(x, gt_other)
        self.assertEqual(gt_tensor.cpu(), torch.gt(x.cpu(), gt_other.cpu()))

        gt_tensor_out = torch.empty(2, 2, device="mcpu", dtype=torch.bool)
        torch.gt(x, gt_other, out=gt_tensor_out)
        self.assertEqual(gt_tensor_out.cpu(), torch.gt(x.cpu(), gt_other.cpu()))

        bits = torch.tensor([[0, 1], [2, 3]], device="mcpu", dtype=torch.int32)
        self.assertEqual(torch.bitwise_not(bits).cpu(), torch.bitwise_not(bits.cpu()))

        bitwise_not_out = torch.empty_like(bits)
        torch.bitwise_not(bits, out=bitwise_not_out)
        self.assertEqual(bitwise_not_out.cpu(), torch.bitwise_not(bits.cpu()))

        bitwise_not_inplace = bits.clone()
        bitwise_not_inplace.bitwise_not_()
        self.assertEqual(bitwise_not_inplace.cpu(), bits.cpu().bitwise_not_())

        bad_binary_out = torch.empty(1, device="mcpu", dtype=torch.bool)
        with self.assertRaisesRegex(RuntimeError, "aten::gt.Scalar_out"):
            torch.gt(x, 2.0, out=bad_binary_out)

        with self.assertRaisesRegex(RuntimeError, "aten::gt.Tensor_out"):
            torch.gt(x, gt_other, out=bad_binary_out)

        bad_bitwise_not_out = torch.empty(1, device="mcpu", dtype=bits.dtype)
        with self.assertRaisesRegex(RuntimeError, "aten::bitwise_not.out"):
            torch.bitwise_not(bits, out=bad_bitwise_not_out)

        nz = torch.nonzero(x)
        self.assertEqual(nz.cpu(), torch.nonzero(x.cpu()))

        sum_out = torch.empty(2, device="mcpu")
        torch.sum(x, dim=[1], out=sum_out)
        self.assertEqual(sum_out.cpu(), torch.sum(x.cpu(), dim=[1]))

        bad_sum_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::sum.IntList_out"):
            torch.sum(x, dim=[1], out=bad_sum_out)

    def test_explicit_forward_ops_batch_4(self):
        x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], device="mcpu")

        max_values = torch.empty(2, 1, device="mcpu")
        max_indices = torch.empty(2, 1, device="mcpu", dtype=torch.long)
        torch.max(x, dim=1, keepdim=True, out=(max_values, max_indices))
        ref_values, ref_indices = torch.max(x.cpu(), dim=1, keepdim=True)
        self.assertEqual(max_values.cpu(), ref_values)
        self.assertEqual(max_indices.cpu(), ref_indices)

        mean_res = torch.empty(2, 1, device="mcpu")
        torch.mean(x, dim=[1], keepdim=True, out=mean_res)
        self.assertEqual(mean_res.cpu(), torch.mean(x.cpu(), dim=[1], keepdim=True))

        scatter_self = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mcpu")
        scatter_index = torch.tensor([[0, 1], [1, 0]], device="mcpu", dtype=torch.long)
        scatter_src = torch.tensor([[10.0, 20.0], [30.0, 40.0]], device="mcpu")
        scatter_out = torch.empty_like(scatter_self)
        torch.scatter_add(scatter_self, 1, scatter_index, scatter_src, out=scatter_out)
        self.assertEqual(
            scatter_out.cpu(),
            torch.scatter_add(scatter_self.cpu(), 1, scatter_index.cpu(), scatter_src.cpu()),
        )

        softmax_out = torch.empty_like(x)
        torch.ops.aten._softmax.out(x, 1, False, out=softmax_out)
        self.assertEqual(
            softmax_out.cpu(),
            torch.ops.aten._softmax.out(x.cpu(), 1, False, out=torch.empty_like(x.cpu())),
        )

        exp_sample = torch.empty(8, device="mcpu")
        exp_sample.exponential_(2.0)
        self.assertEqual(exp_sample.device.type, "mcpu")
        self.assertTrue(torch.isfinite(exp_sample.cpu()).all())
        self.assertTrue((exp_sample.cpu() >= 0).all())

        uniform_sample = torch.empty(16, device="mcpu")
        uniform_result = uniform_sample.uniform_(-1.0, 2.0)
        uniform_cpu = uniform_sample.cpu()
        self.assertIs(uniform_result, uniform_sample)
        self.assertEqual(uniform_sample.device.type, "mcpu")
        self.assertTrue(torch.isfinite(uniform_cpu).all())
        self.assertTrue((uniform_cpu >= -1.0).all())
        self.assertTrue((uniform_cpu < 2.0).all())

        uniform_out = torch.empty_like(uniform_sample)
        torch.ops.aten.uniform.out(uniform_sample, -1.0, 2.0, out=uniform_out)
        uniform_out_cpu = uniform_out.cpu()
        self.assertEqual(uniform_out.shape, uniform_sample.shape)
        self.assertEqual(uniform_out.device.type, "mcpu")
        self.assertTrue(torch.isfinite(uniform_out_cpu).all())
        self.assertTrue((uniform_out_cpu >= -1.0).all())
        self.assertTrue((uniform_out_cpu < 2.0).all())

        bad_uniform_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::uniform.out"):
            torch.ops.aten.uniform.out(
                uniform_sample, -1.0, 2.0, out=bad_uniform_out
            )

        normal_sample = torch.empty(8, device="mcpu")
        normal_result = normal_sample.normal_(0.0, 1.0)
        self.assertIs(normal_result, normal_sample)
        self.assertEqual(normal_sample.device.type, "mcpu")
        self.assertTrue(torch.isfinite(normal_sample.cpu()).all())

        normal_out = torch.empty_like(x)
        torch.ops.aten.normal.out(x, 0.0, 1.0, out=normal_out)
        self.assertEqual(normal_out.shape, x.shape)
        self.assertEqual(normal_out.device.type, "mcpu")
        self.assertTrue(torch.isfinite(normal_out.cpu()).all())

        normal_factory = torch.normal(0.0, 1.0, (2, 3), device="mcpu")
        self.assertEqual(normal_factory.shape, torch.Size([2, 3]))
        self.assertEqual(normal_factory.device.type, "mcpu")
        self.assertTrue(torch.isfinite(normal_factory.cpu()).all())

        mean = torch.zeros(2, 1, device="mcpu")
        std = torch.ones(1, 3, device="mcpu")
        normal_mean = torch.normal(mean, 1.0)
        self.assertEqual(normal_mean.shape, mean.shape)
        self.assertEqual(normal_mean.device.type, "mcpu")
        self.assertTrue(torch.isfinite(normal_mean.cpu()).all())

        normal_std = torch.normal(0.0, std)
        self.assertEqual(normal_std.shape, std.shape)
        self.assertEqual(normal_std.device.type, "mcpu")
        self.assertTrue(torch.isfinite(normal_std.cpu()).all())

        normal_tensor = torch.normal(mean, std)
        self.assertEqual(normal_tensor.shape, torch.Size([2, 3]))
        self.assertEqual(normal_tensor.device.type, "mcpu")
        self.assertTrue(torch.isfinite(normal_tensor.cpu()).all())

        normal_tensor_out = torch.empty(2, 3, device="mcpu")
        torch.normal(mean, std, out=normal_tensor_out)
        self.assertEqual(normal_tensor_out.shape, torch.Size([2, 3]))
        self.assertEqual(normal_tensor_out.device.type, "mcpu")
        self.assertTrue(torch.isfinite(normal_tensor_out.cpu()).all())

        normal_float_float_out = torch.empty(2, 3, device="mcpu")
        torch.ops.aten.normal.float_float_out(
            0.0, 1.0, [2, 3], out=normal_float_float_out
        )
        self.assertEqual(normal_float_float_out.shape, torch.Size([2, 3]))
        self.assertEqual(normal_float_float_out.device.type, "mcpu")
        self.assertTrue(torch.isfinite(normal_float_float_out.cpu()).all())

        argmax_out = torch.empty(2, 1, device="mcpu", dtype=torch.long)
        torch.argmax(x, dim=1, keepdim=True, out=argmax_out)
        self.assertEqual(argmax_out.cpu(), torch.argmax(x.cpu(), dim=1, keepdim=True))

        bad_max_values = torch.empty(1, device="mcpu")
        bad_max_indices = torch.empty(1, device="mcpu", dtype=torch.long)
        with self.assertRaisesRegex(RuntimeError, "aten::max.dim_max"):
            torch.max(x, dim=1, keepdim=True, out=(bad_max_values, bad_max_indices))

        bad_mean_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::mean.out"):
            torch.mean(x, dim=[1], keepdim=True, out=bad_mean_out)

        bad_scatter_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::scatter_add.out"):
            torch.scatter_add(scatter_self, 1, scatter_index, scatter_src, out=bad_scatter_out)

        bad_softmax_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::_softmax.out"):
            torch.ops.aten._softmax.out(x, 1, False, out=bad_softmax_out)

        bad_argmax_out = torch.empty(1, device="mcpu", dtype=torch.long)
        with self.assertRaisesRegex(RuntimeError, "aten::argmax.out"):
            torch.argmax(x, dim=1, keepdim=True, out=bad_argmax_out)

        bad_normal_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::normal.out"):
            torch.ops.aten.normal.out(x, 0.0, 1.0, out=bad_normal_out)

        with self.assertRaisesRegex(RuntimeError, "aten::normal.Tensor_Tensor_out"):
            torch.normal(mean, std, out=bad_normal_out)

    def test_explicit_gather_out(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mcpu")
        index = torch.tensor([[2, 1], [0, 2]], device="mcpu", dtype=torch.long)
        out = torch.empty(2, 2, device="mcpu")

        torch.gather(x, 1, index, out=out)
        self.assertEqual(
            out.cpu(),
            torch.gather(x.cpu(), 1, index.cpu()),
        )
        self.assertEqual(out.device.type, "mcpu")

        bad_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::gather.out"):
            torch.gather(x, 1, index, out=bad_out)

    def test_explicit_index_select_and_index_copy(self):
        x = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mcpu"
        )
        index = torch.tensor([2, 0], device="mcpu", dtype=torch.long)

        selected = torch.index_select(x, 1, index)
        self.assertEqual(selected.cpu(), torch.index_select(x.cpu(), 1, index.cpu()))
        self.assertEqual(selected.device.type, "mcpu")

        selected_out = torch.empty(2, 2, device="mcpu")
        torch.index_select(x, 1, index, out=selected_out)
        self.assertEqual(
            selected_out.cpu(), torch.index_select(x.cpu(), 1, index.cpu())
        )

        bad_out = torch.empty(1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::index_select.out"):
            torch.index_select(x, 1, index, out=bad_out)

        target = torch.zeros(2, 3, device="mcpu")
        source = torch.tensor(
            [[10.0, 20.0], [30.0, 40.0]], device="mcpu"
        )
        result = target.index_copy_(1, index, source)
        self.assertIs(result, target)
        self.assertEqual(
            target.cpu(),
            torch.zeros(2, 3).index_copy_(1, index.cpu(), source.cpu()),
        )

    def test_tensorlist_op_does_not_fallback_to_cpu(self):
        v_mcpu = torch.Tensor([1, 2, 3]).to("mcpu")
        x = (v_mcpu, v_mcpu)
        y = (v_mcpu, v_mcpu)

        self.assertTrue(v_mcpu.device.type == "mcpu")
        self.assertFalse(v_mcpu.is_cpu)

        z = torch._foreach_add(x, y)
        self.assertEqual(z[0].device.type, "mcpu")
        self.assertEqual(z[1].device.type, "mcpu")


class TestSDPA(NNTestCase):
    @skipIfTorchDynamo()
    def test_fused_sdp_choice_privateuseone(self):
        """Test fused SDP choice for privateuse1 backend"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        q_privateuse1 = q_cpu.to("mcpu")
        k_privateuse1 = k_cpu.to("mcpu")
        v_privateuse1 = v_cpu.to("mcpu")
        assert (
            torch._fused_sdp_choice(q_privateuse1, k_privateuse1, v_privateuse1)
            == SDPBackend.OVERRIDEABLE.value
        )

    def test_scaled_dot_product_fused_attention_overrideable(self):
        """Test scaled dot product fused attention overrideable forward"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        q_privateuse1 = q_cpu.to("mcpu")
        k_privateuse1 = k_cpu.to("mcpu")
        v_privateuse1 = v_cpu.to("mcpu")
        torch.nn.functional.scaled_dot_product_attention(
            q_privateuse1, k_privateuse1, v_privateuse1, attn_mask=None, dropout_p=0.0
        )

    def test_scaled_dot_product_fused_attention_overrideable_backward(self):
        """Test scaled dot product fused attention overrideable backward"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(
            torch.rand, device="cpu", dtype=torch.float16, requires_grad=True
        )
        shape = (batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        attn_mask = make_tensor((batch_size, num_heads, seq_len, seq_len))
        q_privateuse1 = q_cpu.to("mcpu")
        k_privateuse1 = k_cpu.to("mcpu")
        v_privateuse1 = v_cpu.to("mcpu")
        attn_mask_privateuse1 = attn_mask.to("mcpu")
        (
            output,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            philox_seed,
            philox_offset,
            _debug_attn_mask,
        ) = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
            q_privateuse1, k_privateuse1, v_privateuse1, attn_bias=attn_mask_privateuse1
        )

        rand_upward = torch.rand(
            shape, device="cpu", dtype=torch.float16, requires_grad=False
        )
        rand_upward_privateuse1 = rand_upward.to("mcpu")
        grad_input_mask = [True, True, True, True]
        _grad_q, _grad_k, _grad_v, _grad_attn_mask = (
            torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward(
                rand_upward_privateuse1,
                q_privateuse1,
                k_privateuse1,
                v_privateuse1,
                attn_mask_privateuse1,
                grad_input_mask,
                output,
                logsumexp,
                cum_seq_q,
                cum_seq_k,
                max_q,
                max_k,
                dropout_p=0.0,
                is_causal=False,
                philox_seed=philox_seed,
                philox_offset=philox_offset,
            )
        )


class TestFactoryExtended(TestCase):
    def test_empty_with_memory_format(self):
        """Test empty tensor creation with memory format"""
        x = torch.empty(1, 2, 3, 4, device="mcpu", memory_format=torch.channels_last)
        self.assertEqual(x.device.type, "mcpu")
        self.assertEqual(x.shape, torch.Size([1, 2, 3, 4]))

        x = torch.empty(
            2, 3, 4, device="mcpu", memory_format=torch.contiguous_format
        )
        self.assertEqual(x.device.type, "mcpu")
        self.assertTrue(x.is_contiguous())

    def test_empty_strided(self):
        """Test empty_strided tensor creation"""
        size = (3, 4)
        stride = (4, 1)
        x = torch.empty_strided(size, stride, device="mcpu")
        self.assertEqual(x.device.type, "mcpu")
        self.assertEqual(x.shape, torch.Size(size))
        self.assertEqual(x.stride(), stride)

    def test_ones(self):
        """Test ones tensor creation"""
        x = torch.ones(3, 4, device="mcpu")
        self.assertEqual(x.device.type, "mcpu")
        self.assertEqual(x.shape, torch.Size([3, 4]))
        self.assertTrue(torch.all(x.cpu() == 1))

    def test_ones_like(self):
        """Test ones_like tensor creation"""
        x = torch.randn(3, 4, device="mcpu")
        y = torch.ones_like(x)
        self.assertEqual(y.device.type, "mcpu")
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.all(y.cpu() == 1))

    def test_randn(self):
        """Test randn tensor creation"""
        x = torch.randn(3, 4, device="mcpu")
        self.assertEqual(x.device.type, "mcpu")
        self.assertEqual(x.shape, torch.Size([3, 4]))

    def test_full(self):
        """Test full tensor creation"""
        x = torch.full((3, 4), 5.0, device="mcpu")
        self.assertEqual(x.device.type, "mcpu")
        self.assertEqual(x.shape, torch.Size([3, 4]))
        self.assertTrue(torch.all(x.cpu() == 5.0))


class TestCopyExtended(TestCase):
    def test_copy_different_dtypes(self):
        """Test copy with different dtypes"""
        x = torch.randn(3, 4, dtype=torch.float32, device="mcpu")
        y = torch.empty(3, 4, dtype=torch.float64, device="mcpu")
        y.copy_(x)
        self.assertEqual(y.dtype, torch.float64)
        self.assertEqual(y.cpu(), x.cpu().double())

    def test_clone(self):
        """Test tensor clone"""
        x = torch.randn(3, 4, device="mcpu")
        y = x.clone()
        self.assertEqual(y.device.type, "mcpu")
        self.assertEqual(y.cpu(), x.cpu())
        self.assertNotEqual(y.data_ptr(), x.data_ptr())

    def test_copy_non_blocking(self):
        """Test non-blocking copy"""
        x = torch.randn(3, 4, device="mcpu")
        y = torch.empty(3, 4, device="mcpu")
        y.copy_(x, non_blocking=True)
        self.assertEqual(y.cpu(), x.cpu())


class TestOpsExtended(TestCase):
    def test_view(self):
        """Test tensor view operation"""
        x = torch.randn(2, 3, 4, device="mcpu")
        y = x.view(6, 4)
        self.assertEqual(y.device.type, "mcpu")
        self.assertEqual(y.shape, torch.Size([6, 4]))
        self.assertEqual(x.data_ptr(), y.data_ptr())

    def test_reshape(self):
        """Test tensor reshape operation"""
        x = torch.randn(2, 3, 4, device="mcpu")
        y = x.reshape(6, 4)
        self.assertEqual(y.device.type, "mcpu")
        self.assertEqual(y.shape, torch.Size([6, 4]))

    def test_as_strided(self):
        """Test as_strided operation"""
        x = torch.randn(3, 4, device="mcpu")
        y = torch.as_strided(x, (2, 2), (4, 1), 1)
        self.assertEqual(y.device.type, "mcpu")
        self.assertEqual(y.shape, torch.Size([2, 2]))

    def test_unfold(self):
        """Test unfold view operation"""
        x_cpu = torch.arange(2 * 3 * 4 * 5 * 6, dtype=torch.float32).reshape(
            2, 3, 4, 5, 6
        )
        x = x_cpu.to(device="mcpu")

        y = x.unfold(2, 2, 2).unfold(3, 3, 1).unfold(4, 2, 2)
        y_cpu = x_cpu.unfold(2, 2, 2).unfold(3, 3, 1).unfold(4, 2, 2)

        self.assertEqual(y.device.type, "mcpu")
        self.assertEqual(y.shape, y_cpu.shape)
        self.assertEqual(y.stride(), y_cpu.stride())
        self.assertEqual(y.cpu(), y_cpu)
        self.assertEqual(x.data_ptr(), y.data_ptr())

    def test_local_scalar_dense(self):
        """Test local scalar dense extraction"""
        x = torch.tensor([5.0], device="mcpu")
        scalar = x.item()
        self.assertEqual(scalar, 5.0)

    def test_set_tensor(self):
        """Test set_ operation with tensor source"""
        x = torch.randn(3, 4, device="mcpu")
        y = torch.empty(3, 4, device="mcpu")
        y.set_(x)
        self.assertEqual(y.cpu(), x.cpu())

    def test_set_storage(self):
        """Test set_ operation with storage source"""
        x = torch.randn(3, 4, device="mcpu")
        storage = x.storage()
        y = torch.empty(3, 4, device="mcpu")
        y.set_(storage, 0, y.size())
        self.assertEqual(y.cpu(), x.cpu())


class TestSTUBExtended(TestCase):
    def test_abs_contiguous(self):
        """Test abs operation with contiguous tensor"""
        x = torch.randn(2, 3, dtype=torch.float32, device="mcpu")
        y = torch.abs(x)
        self.assertEqual(y.device.type, "mcpu")
        self.assertTrue(torch.all(y.cpu() >= 0))
        self.assertEqual(y.shape, x.shape)

    def test_abs_non_contiguous(self):
        """Test abs operation with non-contiguous tensor"""
        x = torch.randn(2, 3, dtype=torch.float32, device="mcpu")
        x_t = x.t()  # Transpose makes it non-contiguous
        y = torch.abs(x_t)
        self.assertEqual(y.device.type, "mcpu")
        self.assertTrue(torch.all(y.cpu() >= 0))

    def test_custom_abs(self):
        """Test custom abs operation"""
        x = torch.randn(2, 3, dtype=torch.float32, device="mcpu")
        y = torch.ops.mcpu.custom_abs(x)
        self.assertEqual(y.device.type, "mcpu")
        self.assertTrue(torch.all(y.cpu() >= 0))
        self.assertEqual(y.shape, x.shape)

    def test_abs_out(self):
        """Test abs with output tensor"""
        x = torch.randn(2, 3, dtype=torch.float32, device="mcpu")
        out = torch.empty_like(x)
        torch.abs(x, out=out)
        self.assertEqual(out.device.type, "mcpu")
        self.assertTrue(torch.all(out.cpu() >= 0))
        self.assertEqual(out.cpu(), torch.abs(x.cpu()))


@unittest.skip("Skipping all quantization tests for mcpu backend")
class TestQuantizationExtended(TestCase):
    def test_quantize_per_tensor_different_scales(self):
        """Test quantization with different scales"""
        x = torch.randn(3, 4, 5, dtype=torch.float32, device="mcpu")

        scale = 0.1
        zero_point = 10
        quantized = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
        self.assertEqual(quantized.device.type, "mcpu")
        self.assertEqual(quantized.dtype, torch.qint8)
        self.assertEqual(quantized.q_scale(), scale)
        self.assertEqual(quantized.q_zero_point(), zero_point)

    def test_quantize_per_tensor_quint8(self):
        """Test quantization with quint8 dtype"""
        x = torch.randn(3, 4, dtype=torch.float32, device="mcpu")
        quantized = torch.quantize_per_tensor(x, 0.1, 128, torch.quint8)
        self.assertEqual(quantized.device.type, "mcpu")
        self.assertEqual(quantized.dtype, torch.quint8)

    def test_dequantize(self):
        """Test dequantization"""
        x = torch.randn(3, 4, dtype=torch.float32, device="mcpu")
        quantized = torch.quantize_per_tensor(x, 0.1, 10, torch.qint8)
        dequantized = quantized.dequantize()
        self.assertEqual(dequantized.device.type, "mcpu")
        self.assertEqual(dequantized.dtype, torch.float32)


class TestFallbackExtended(TestCase):
    def test_contiguous_add_sub_use_raw_kernel(self):
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4).to("mcpu")
        y = (torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1).to("mcpu")

        torch.mcpu.reset_kernel_timing()
        torch.mcpu.set_kernel_timing_enabled(True)
        try:
            add_res = torch.add(x, y)
            sub_res = torch.sub(y, x)
            torch.mcpu.synchronize()
            timing = torch.mcpu.get_kernel_timing()
        finally:
            torch.mcpu.set_kernel_timing_enabled(False)
            torch.mcpu.reset_kernel_timing()

        self.assertEqual(add_res.cpu(), x.cpu() + y.cpu())
        self.assertEqual(sub_res.cpu(), y.cpu() - x.cpu())
        event_names = [
            event.get("name")
            for thread in timing
            for event in thread.get("events", [])
        ]
        self.assertIn("mcpu::aten::add.raw", event_names)
        self.assertIn("mcpu::aten::sub.raw", event_names)

    def test_add_sub_raw_kernel_fallback_cases(self):
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4).to("mcpu")
        y = torch.arange(4, dtype=torch.float32).to("mcpu")

        broadcast_res = torch.add(x, y)
        self.assertEqual(broadcast_res.cpu(), x.cpu() + y.cpu())

        alpha_res = torch.sub(x, x, alpha=0.5)
        self.assertEqual(alpha_res.cpu(), x.cpu() - x.cpu() * 0.5)

        torch.mcpu.reset_kernel_timing()
        torch.mcpu.set_kernel_timing_enabled(True)
        try:
            base = torch.arange(5, dtype=torch.float32, device="mcpu")
            torch.add(base[:-1], base[:-1], out=base[1:])
            torch.mcpu.synchronize()
            timing = torch.mcpu.get_kernel_timing()
        finally:
            torch.mcpu.set_kernel_timing_enabled(False)
            torch.mcpu.reset_kernel_timing()

        event_names = [
            event.get("name")
            for thread in timing
            for event in thread.get("events", [])
        ]
        self.assertNotIn("mcpu::aten::add.raw", event_names)

    def test_scalar_gt_bitwise_not_and_index_put_use_raw_kernels(self):
        x = torch.tensor([[0.0, 2.0], [3.0, 4.0]], device="mcpu")
        gt_out = torch.empty_like(x, dtype=torch.bool)
        bits = torch.tensor([[0, 1], [2, 3]], device="mcpu", dtype=torch.int32)
        bitwise_not_out = torch.empty_like(bits)
        target = torch.zeros(3, 3, device="mcpu")
        row = torch.tensor([0, 2], device="mcpu", dtype=torch.long)
        col = torch.tensor([1, 0], device="mcpu", dtype=torch.long)
        vals = torch.tensor([5.0, 7.0], device="mcpu")

        torch.mcpu.reset_kernel_timing()
        torch.mcpu.set_kernel_timing_enabled(True)
        try:
            torch.gt(x, 2.0, out=gt_out)
            torch.bitwise_not(bits, out=bitwise_not_out)
            torch.ops.aten._index_put_impl_(target, [row, col], vals, False, False)
            torch.mcpu.synchronize()
            timing = torch.mcpu.get_kernel_timing()
        finally:
            torch.mcpu.set_kernel_timing_enabled(False)
            torch.mcpu.reset_kernel_timing()

        self.assertEqual(gt_out.cpu(), torch.gt(x.cpu(), 2.0))
        self.assertEqual(bitwise_not_out.cpu(), torch.bitwise_not(bits.cpu()))
        self.assertEqual(
            target.cpu(),
            torch.tensor([[0.0, 5.0, 0.0], [0.0, 0.0, 0.0], [7.0, 0.0, 0.0]]),
        )
        event_names = [
            event.get("name")
            for thread in timing
            for event in thread.get("events", [])
        ]
        self.assertIn("mcpu::aten::gt.Scalar.raw", event_names)
        self.assertIn("mcpu::aten::bitwise_not.raw", event_names)
        self.assertIn("mcpu::aten::_index_put_impl_.raw", event_names)

    def test_abs_uses_explicit_mcpu_kernel(self):
        x = torch.tensor([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], device="mcpu")
        y = torch.abs(x)
        self.assertEqual(y.device.type, "mcpu")

        out = torch.empty_like(x)
        torch.abs(x, out=out)
        self.assertEqual(out.device.type, "mcpu")

    def test_registered_binary_operations_stay_on_mcpu(self):
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4).to("mcpu")
        y = (torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1).to("mcpu")

        z = torch.add(x, y)
        self.assertEqual(z.device.type, "mcpu")

        z = torch.sub(x, y)
        self.assertEqual(z.device.type, "mcpu")

    def test_registered_scalar_operations_stay_on_mcpu(self):
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4).to("mcpu")
        y = x + 1.0
        self.assertEqual(y.device.type, "mcpu")

        y = x - 2.0
        self.assertEqual(y.device.type, "mcpu")


class TestSDPAExtended(NNTestCase):
    @skipIfTorchDynamo()
    def test_fused_sdp_choice_with_mask(self):
        """Test fused SDP choice with attention mask"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        attn_mask = make_tensor((batch_size, num_heads, seq_len, seq_len))

        q_privateuse1 = q_cpu.to("mcpu")
        k_privateuse1 = k_cpu.to("mcpu")
        v_privateuse1 = v_cpu.to("mcpu")
        attn_mask_privateuse1 = attn_mask.to("mcpu")

        backend = torch._fused_sdp_choice(
            q_privateuse1, k_privateuse1, v_privateuse1, attn_mask_privateuse1
        )
        self.assertEqual(backend, SDPBackend.OVERRIDEABLE.value)

    @skipIfTorchDynamo()
    def test_scaled_dot_product_attention_with_dropout(self):
        """Test scaled dot product attention with dropout"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)

        q_privateuse1 = q_cpu.to("mcpu")
        k_privateuse1 = k_cpu.to("mcpu")
        v_privateuse1 = v_cpu.to("mcpu")

        output = torch.nn.functional.scaled_dot_product_attention(
            q_privateuse1,
            k_privateuse1,
            v_privateuse1,
            attn_mask=None,
            dropout_p=0.1,
            is_causal=False,
        )
        self.assertEqual(output.device.type, "mcpu")
        self.assertEqual(output.shape, shape)

    @skipIfTorchDynamo()
    def test_scaled_dot_product_attention_is_causal(self):
        """Test scaled dot product attention with causal mask"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)

        q_privateuse1 = q_cpu.to("mcpu")
        k_privateuse1 = k_cpu.to("mcpu")
        v_privateuse1 = v_cpu.to("mcpu")

        output = torch.nn.functional.scaled_dot_product_attention(
            q_privateuse1,
            k_privateuse1,
            v_privateuse1,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        self.assertEqual(output.device.type, "mcpu")
        self.assertEqual(output.shape, shape)


class TestCopyFromAndResize(TestCase):
    # def test_copy_from_and_resize(self):
    #     """Test _copy_from_and_resize operation"""
    #     x = torch.randn(3, 4, device="mcpu")
    #     y = torch.empty(2, 2, device="mcpu")
    #     result = torch.ops.aten._copy_from_and_resize(x, y)
    #     self.assertEqual(result.device.type, "mcpu")
    #     self.assertEqual(result.shape, x.shape)

    def test_copy_from_same_device(self):
        """Test _copy_from operation on same device"""
        x = torch.randn(3, 4, device="mcpu")
        y = torch.empty(3, 4, device="mcpu")
        result = torch.ops.aten._copy_from(x, y, non_blocking=False)
        self.assertEqual(result.device.type, "mcpu")
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.cpu(), x.cpu())

    def test_copy_from_cross_device(self):
        """Test _copy_from operation across devices"""
        x = torch.randn(3, 4, device="cpu")
        y = torch.empty(3, 4, device="mcpu")
        result = torch.ops.aten._copy_from(x, y, non_blocking=False)
        self.assertEqual(result.device.type, "mcpu")
        self.assertEqual(result.cpu(), x)

    def test_copy_from_host_device_non_blocking_uses_raw_memcpy(self):
        """Test host-device _copy_from avoids copy_ for memcpy-compatible layouts."""
        cpu_src = torch.randn(3, 7, 5)
        mcpu_dst = torch.empty_like(cpu_src, device="mcpu")
        mcpu_src = cpu_src.to("mcpu")
        cpu_dst = torch.empty_like(cpu_src)

        torch.mcpu.reset_kernel_timing()
        torch.mcpu.set_kernel_timing_enabled(True)
        try:
            torch.ops.aten._copy_from(cpu_src, mcpu_dst, non_blocking=True)
            torch.ops.aten._copy_from(mcpu_src, cpu_dst, non_blocking=True)
            torch.mcpu.synchronize()
            timing = torch.mcpu.get_kernel_timing()
        finally:
            torch.mcpu.set_kernel_timing_enabled(False)
            torch.mcpu.reset_kernel_timing()

        self.assertEqual(mcpu_dst.cpu(), cpu_src)
        self.assertEqual(cpu_dst, cpu_src)
        event_names = [
            event.get("name")
            for thread in timing
            for event in thread.get("events", [])
        ]
        self.assertGreaterEqual(
            event_names.count("mcpu::_copy_from.host_device.memcpy"),
            2,
            event_names,
        )

    def test_copy_from_non_blocking(self):
        """Test _copy_from with non_blocking=True"""
        x = torch.randn(3, 4, device="mcpu")
        y = torch.empty(3, 4, device="mcpu")
        result = torch.ops.aten._copy_from(x, y, non_blocking=True)
        self.assertEqual(result.device.type, "mcpu")
        self.assertEqual(result.cpu(), x.cpu())

    def test_reshape_alias(self):
        """Test _reshape_alias operation"""
        x = torch.randn(2, 3, 4, device="mcpu")
        new_size = (6, 4)
        new_stride = (4, 1)
        y = torch.ops.aten._reshape_alias(x, new_size, new_stride)
        self.assertEqual(y.device.type, "mcpu")
        self.assertEqual(y.shape, torch.Size(new_size))
        self.assertEqual(y.stride(), new_stride)

    def test_set_storage_storage_offset(self):
        """Test set_ operation with storage and storage offset"""
        x = torch.randn(3, 4, device="mcpu")
        storage = x.storage()
        y = torch.empty(2, 2, device="mcpu")
        result = torch.ops.aten.set_.source_Storage_storage_offset(
            y, storage, 0, (2, 2), (2, 1)
        )
        self.assertEqual(result.device.type, "mcpu")
        self.assertEqual(result.shape, torch.Size([2, 2]))

    def test_set_storage_storage_offset_with_offset(self):
        """Test set_ operation with non-zero storage offset"""
        x = torch.randn(4, 4, device="mcpu")
        storage = x.storage()
        y = torch.empty(2, 2, device="mcpu")
        # Use storage offset to skip first 4 elements
        result = torch.ops.aten.set_.source_Storage_storage_offset(
            y, storage, 4, (2, 2), (2, 1)
        )
        self.assertEqual(result.device.type, "mcpu")
        self.assertEqual(result.shape, torch.Size([2, 2]))


class TestCustomAutogradFunctions(TestCase):
    def test_custom_autograd_fn_returns_self_basic(self):
        """Test basic usage of custom_autograd_fn_returns_self"""
        x = torch.randn(4, device="mcpu", requires_grad=True)
        y = torch.ops.mcpu.custom_autograd_fn_returns_self(x)

        # Should return the same tensor
        self.assertEqual(x.cpu(), y.cpu())
        self.assertTrue(y.requires_grad)

        # Test backward
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        # Gradient should be 0.5 * 1.0 = 0.5
        self.assertTrue(torch.allclose(x.grad.cpu(), torch.ones_like(x.cpu()) * 0.5))

    def test_custom_autograd_fn_aliasing_basic(self):
        """Test basic usage of custom_autograd_fn_aliasing"""
        x = torch.randn(4, device="mcpu", requires_grad=True)
        y = torch.ops.mcpu.custom_autograd_fn_aliasing(x)

        # Should return a view of the same tensor
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(y.requires_grad)

        # Test backward
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        # Gradient should be 0.5 * 1.0 = 0.5
        self.assertTrue(torch.allclose(x.grad.cpu(), torch.ones_like(x.cpu()) * 0.5))

    def test_custom_autograd_fn_returns_self_no_grad(self):
        """Test custom_autograd_fn_returns_self without requires_grad"""
        x = torch.randn(4, device="mcpu", requires_grad=False)
        y = torch.ops.mcpu.custom_autograd_fn_returns_self(x)
        self.assertEqual(x.cpu(), y.cpu())
        self.assertFalse(y.requires_grad)

    def test_custom_autograd_fn_aliasing_no_grad(self):
        """Test custom_autograd_fn_aliasing without requires_grad"""
        x = torch.randn(4, device="mcpu", requires_grad=False)
        y = torch.ops.mcpu.custom_autograd_fn_aliasing(x)
        self.assertEqual(x.shape, y.shape)
        self.assertFalse(y.requires_grad)


class TestMM(TestCase):
    def test_mm_basic(self):
        """Test basic mm operation"""
        a = torch.randn(3, 4, device="mcpu")
        b = torch.randn(4, 5, device="mcpu")
        c = torch.mm(a, b)
        self.assertEqual(c.device.type, "mcpu")
        self.assertEqual(c.shape, torch.Size([3, 5]))
        self.assertEqual(c.cpu(), torch.mm(a.cpu(), b.cpu()))

    def test_mm_out(self):
        """Test mm with output tensor"""
        a = torch.randn(3, 4, device="mcpu")
        b = torch.randn(4, 5, device="mcpu")
        out = torch.empty(3, 5, device="mcpu")
        torch.mm(a, b, out=out)
        self.assertEqual(out.cpu(), torch.mm(a.cpu(), b.cpu()))

    def test_mm_out_wrong_size(self):
        """Test mm.out rejects wrong output size"""
        a = torch.randn(3, 4, device="mcpu")
        b = torch.randn(4, 5, device="mcpu")
        bad_out = torch.empty(1, 1, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "aten::mm.out"):
            torch.mm(a, b, out=bad_out)

    def test_mm_square(self):
        """Test mm with square matrices"""
        a = torch.randn(4, 4, device="mcpu")
        b = torch.randn(4, 4, device="mcpu")
        c = torch.mm(a, b)
        self.assertEqual(c.cpu(), torch.mm(a.cpu(), b.cpu()))

    def test_addmm_basic(self):
        """Test basic addmm operation"""
        bias = torch.randn(3, 5, device="mcpu")
        a = torch.randn(3, 4, device="mcpu")
        b = torch.randn(4, 5, device="mcpu")
        c = torch.addmm(bias, a, b)
        self.assertEqual(c.device.type, "mcpu")
        self.assertEqual(c.shape, torch.Size([3, 5]))
        self.assertEqual(c.cpu(), torch.addmm(bias.cpu(), a.cpu(), b.cpu()))

    def test_addmm_alpha_beta(self):
        """Test addmm with non-default alpha and beta"""
        bias = torch.randn(3, 5, device="mcpu")
        a = torch.randn(3, 4, device="mcpu")
        b = torch.randn(4, 5, device="mcpu")
        c = torch.addmm(bias, a, b, beta=0.5, alpha=2.0)
        expected = torch.addmm(bias.cpu(), a.cpu(), b.cpu(), beta=0.5, alpha=2.0)
        self.assertEqual(c.cpu(), expected)

    def test_addmm_out(self):
        """Test addmm with output tensor"""
        bias = torch.randn(3, 5, device="mcpu")
        a = torch.randn(3, 4, device="mcpu")
        b = torch.randn(4, 5, device="mcpu")
        out = torch.empty(3, 5, device="mcpu")
        torch.addmm(bias, a, b, out=out)
        self.assertEqual(out.cpu(), torch.addmm(bias.cpu(), a.cpu(), b.cpu()))

    def test_addmm_broadcast_bias(self):
        """Test addmm with broadcastable bias"""
        bias = torch.randn(5, device="mcpu")
        a = torch.randn(3, 4, device="mcpu")
        b = torch.randn(4, 5, device="mcpu")
        c = torch.addmm(bias, a, b)
        self.assertEqual(c.cpu(), torch.addmm(bias.cpu(), a.cpu(), b.cpu()))


if __name__ == "__main__":
    run_tests()

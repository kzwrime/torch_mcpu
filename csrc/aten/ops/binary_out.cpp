#include "Common.h"

#include <ATen/ExpandUtils.h>
#include <ATen/ops/add.h>
#include <ATen/ops/div.h>
#include <ATen/ops/eq.h>
#include <ATen/ops/ge.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/le.h>
#include <ATen/ops/less.h>
#include <ATen/ops/less_equal.h>
#include <ATen/ops/lt.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/remainder.h>
#include <ATen/ops/sub.h>
#include <c10/core/DefaultDtype.h>
#include <torch/library.h>

#include <memory>
#include <utility>

namespace at::mcpu {
namespace {

enum class ArithmeticOp {
  Add,
  Sub,
  Mul,
  Div,
};

enum class CompareOp {
  Eq,
  Ne,
  Lt,
  Le,
  Gt,
  Ge,
};

at::Tensor empty_binary_mcpu(const at::Tensor& self, const at::Tensor& other) {
  auto out_sizes = at::infer_size(self.sizes(), other.sizes());
  return at::empty(
      out_sizes, self.options().dtype(at::result_type(self, other)));
}

at::Tensor empty_like_mcpu_result(
    const at::Tensor& self,
    const at::Scalar& other) {
  return at::empty(
      self.sizes(), self.options().dtype(at::result_type(self, other)));
}

at::Tensor empty_comparison_mcpu(
    const at::Tensor& self,
    at::IntArrayRef out_sizes) {
  return at::empty(out_sizes, self.options().dtype(at::kBool));
}

bool is_cpu_scalar_tensor(const at::Tensor& tensor) {
  return tensor.is_cpu() && tensor.dim() == 0;
}

struct ArithmeticTensorKernelArgs {
  ops::TensorViewSpec self;
  ops::TensorViewSpec other;
  ops::TensorViewSpec out;
  at::Scalar alpha;
  at::Tensor self_owner;
  at::Tensor other_owner;
};

struct TensorTripletKernelArgs {
  ops::TensorViewSpec self;
  ops::TensorViewSpec other;
  ops::TensorViewSpec out;
  at::Tensor self_owner;
  at::Tensor other_owner;
};

struct TensorScalarKernelArgs {
  ops::TensorViewSpec self;
  ops::TensorViewSpec out;
  at::Scalar other;
  at::Scalar alpha;
  at::Tensor self_owner;
};

struct PowScalarOutKernelArgs {
  at::Scalar self;
  ops::TensorViewSpec exponent;
  ops::TensorViewSpec out;
  at::Tensor exponent_owner;
};

template <ArithmeticOp op>
at::ScalarType arithmetic_result_type(
    const at::Tensor& self,
    const at::Tensor& other) {
  auto result_type = at::result_type(self, other);
  if constexpr (op == ArithmeticOp::Div) {
    if (c10::isIntegralType(result_type, /*includeBool=*/true)) {
      return c10::get_default_dtype_as_scalartype();
    }
  }
  return result_type;
}

template <ArithmeticOp op>
at::Tensor empty_arithmetic_mcpu(
    const at::Tensor& self,
    const at::Tensor& other) {
  auto out_sizes = at::infer_size(self.sizes(), other.sizes());
  return at::empty(
      out_sizes, self.options().dtype(arithmetic_result_type<op>(self, other)));
}

template <ArithmeticOp op>
at::Tensor& cpu_arithmetic_tensor_out(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  if constexpr (op == ArithmeticOp::Add) {
    return at::add_out(out, self, other, alpha);
  } else if constexpr (op == ArithmeticOp::Sub) {
    return at::sub_out(out, self, other, alpha);
  } else if constexpr (op == ArithmeticOp::Mul) {
    return at::mul_out(out, self, other);
  } else {
    return at::div_out(out, self, other);
  }
}

template <ArithmeticOp op>
at::Tensor& cpu_arithmetic_scalar_out(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  if constexpr (op == ArithmeticOp::Add) {
    return at::add_out(out, self, other, alpha);
  } else if constexpr (op == ArithmeticOp::Sub) {
    return at::sub_out(out, self, other, alpha);
  } else if constexpr (op == ArithmeticOp::Mul) {
    return at::mul_out(out, self, other);
  } else {
    return at::div_out(out, self, other);
  }
}

template <ArithmeticOp op>
at::Tensor& arithmetic_Scalar_out_impl(
    const char* op_name,
    const char* record_name,
    const at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha,
    at::Tensor& out) {
  ops::check_out_sizes(op_name, out, self.sizes());

  auto args = std::make_shared<TensorScalarKernelArgs>(TensorScalarKernelArgs{
      ops::make_cpu_view_spec(self),
      ops::make_cpu_view_spec(out),
      other,
      alpha,
      self});

  MCPU_LAUNCH_TIMED_KERNEL(
      record_name, ([ args = std::move(args), record_name ]), {
        KernelPointerMemoryGuard guard({args->self.data, args->out.data});
        auto cpu_self = ops::cpu_view_from_spec(args->self);
        auto cpu_out = ops::cpu_view_from_spec(args->out);
        cpu_arithmetic_scalar_out<op>(
            cpu_out, cpu_self, args->other, args->alpha);
      });
  return out;
}

template <ArithmeticOp op>
at::Tensor& arithmetic_Tensor_out_impl(
    const char* op_name,
    const char* record_name,
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& out) {
  if (is_cpu_scalar_tensor(other)) {
    return arithmetic_Scalar_out_impl<op>(
        op_name, record_name, self, other.item(), alpha, out);
  }

  auto expected_sizes = at::infer_size(self.sizes(), other.sizes());
  ops::check_out_sizes(op_name, out, expected_sizes);

  auto args =
      std::make_shared<ArithmeticTensorKernelArgs>(ArithmeticTensorKernelArgs{
          ops::make_cpu_view_spec(self),
          ops::make_cpu_view_spec(other),
          ops::make_cpu_view_spec(out),
          alpha,
          self,
          other});

  MCPU_LAUNCH_TIMED_KERNEL(
      record_name, ([ args = std::move(args), record_name ]), {
        KernelPointerMemoryGuard guard(
            {args->self.data, args->other.data, args->out.data});
        auto cpu_self = ops::cpu_view_from_spec(args->self);
        auto cpu_other = ops::cpu_view_from_spec(args->other);
        auto cpu_out = ops::cpu_view_from_spec(args->out);
        cpu_arithmetic_tensor_out<op>(
            cpu_out, cpu_self, cpu_other, args->alpha);
      });
  return out;
}

template <ArithmeticOp op>
at::Tensor arithmetic_Tensor_impl(
    const char* op_name,
    const char* record_name,
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  auto out = empty_arithmetic_mcpu<op>(self, other);
  arithmetic_Tensor_out_impl<op>(op_name, record_name, self, other, alpha, out);
  return out;
}

template <ArithmeticOp op>
at::Tensor& arithmetic_Tensor__impl(
    const char* op_name,
    const char* record_name,
    at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  arithmetic_Tensor_out_impl<op>(
      op_name, record_name, self, other, alpha, self);
  return self;
}

at::Tensor& add_out(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& out) {
  return arithmetic_Tensor_out_impl<ArithmeticOp::Add>(
      "aten::add.out", "mcpu::aten::add", self, other, alpha, out);
}

at::Tensor add_Tensor(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  return arithmetic_Tensor_impl<ArithmeticOp::Add>(
      "aten::add.out", "mcpu::aten::add", self, other, alpha);
}

at::Tensor& add_Tensor_(
    at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  return arithmetic_Tensor__impl<ArithmeticOp::Add>(
      "aten::add_.Tensor", "mcpu::aten::add_.Tensor", self, other, alpha);
}

at::Tensor add_Scalar(
    const at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  auto out = empty_like_mcpu_result(self, other);
  arithmetic_Scalar_out_impl<ArithmeticOp::Add>(
      "aten::add.Scalar_out",
      "mcpu::aten::add.Scalar",
      self,
      other,
      alpha,
      out);
  return out;
}

at::Tensor& add_Scalar_(
    at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  auto self_spec = ops::make_cpu_view_spec(self);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::add_.Scalar",
      ([ self_spec = std::move(self_spec), other, alpha ]),
      {
        KernelPointerMemoryGuard guard({self_spec.data});
        auto cpu_self = ops::cpu_view_from_spec(self_spec);
        at::add_out(cpu_self, cpu_self, other, alpha);
      });
  return self;
}

at::Tensor& div_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  return arithmetic_Tensor_out_impl<ArithmeticOp::Div>(
      "aten::div.out", "mcpu::aten::div.out", self, other, at::Scalar(1), out);
}

at::Tensor div_Tensor(const at::Tensor& self, const at::Tensor& other) {
  return arithmetic_Tensor_impl<ArithmeticOp::Div>(
      "aten::div.out", "mcpu::aten::div.Tensor", self, other, at::Scalar(1));
}

at::Tensor& div_Tensor_(at::Tensor& self, const at::Tensor& other) {
  return arithmetic_Tensor__impl<ArithmeticOp::Div>(
      "aten::div_.Tensor",
      "mcpu::aten::div_.Tensor",
      self,
      other,
      at::Scalar(1));
}

template <CompareOp op>
at::Tensor& cpu_compare_tensor_out(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Tensor& other) {
  if constexpr (op == CompareOp::Eq) {
    return at::eq_out(out, self, other);
  } else if constexpr (op == CompareOp::Ne) {
    return at::ne_out(out, self, other);
  } else if constexpr (op == CompareOp::Lt) {
    return at::lt_out(out, self, other);
  } else if constexpr (op == CompareOp::Le) {
    return at::le_out(out, self, other);
  } else if constexpr (op == CompareOp::Gt) {
    return at::gt_out(out, self, other);
  } else {
    return at::ge_out(out, self, other);
  }
}

template <CompareOp op>
at::Tensor& cpu_compare_scalar_out(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Scalar& other) {
  if constexpr (op == CompareOp::Eq) {
    return at::eq_out(out, self, other);
  } else if constexpr (op == CompareOp::Ne) {
    return at::ne_out(out, self, other);
  } else if constexpr (op == CompareOp::Lt) {
    return at::lt_out(out, self, other);
  } else if constexpr (op == CompareOp::Le) {
    return at::le_out(out, self, other);
  } else if constexpr (op == CompareOp::Gt) {
    return at::gt_out(out, self, other);
  } else {
    return at::ge_out(out, self, other);
  }
}

template <CompareOp op>
at::Tensor& compare_Scalar_out_impl(
    const char* op_name,
    const char* record_name,
    const at::Tensor& self,
    const at::Scalar& other,
    at::Tensor& out);

template <CompareOp op>
at::Tensor& compare_Tensor_out_impl(
    const char* op_name,
    const char* record_name,
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  if (is_cpu_scalar_tensor(other)) {
    return compare_Scalar_out_impl<op>(
        op_name, record_name, self, other.item(), out);
  }

  auto expected_sizes = at::infer_size(self.sizes(), other.sizes());
  ops::check_out_sizes(op_name, out, expected_sizes);

  auto args = std::make_shared<TensorTripletKernelArgs>(TensorTripletKernelArgs{
      ops::make_cpu_view_spec(self),
      ops::make_cpu_view_spec(other),
      ops::make_cpu_view_spec(out),
      self,
      other});

  MCPU_LAUNCH_TIMED_KERNEL(
      record_name, ([ args = std::move(args), record_name ]), {
        KernelPointerMemoryGuard guard(
            {args->self.data, args->other.data, args->out.data});
        auto cpu_self = ops::cpu_view_from_spec(args->self);
        auto cpu_other = ops::cpu_view_from_spec(args->other);
        auto cpu_out = ops::cpu_view_from_spec(args->out);
        cpu_compare_tensor_out<op>(cpu_out, cpu_self, cpu_other);
      });
  return out;
}

template <CompareOp op>
at::Tensor compare_Tensor_impl(
    const char* op_name,
    const char* record_name,
    const at::Tensor& self,
    const at::Tensor& other) {
  auto out_sizes = at::infer_size(self.sizes(), other.sizes());
  auto out = empty_comparison_mcpu(self, out_sizes);
  compare_Tensor_out_impl<op>(op_name, record_name, self, other, out);
  return out;
}

template <CompareOp op>
at::Tensor& compare_Scalar_out_impl(
    const char* op_name,
    const char* record_name,
    const at::Tensor& self,
    const at::Scalar& other,
    at::Tensor& out) {
  ops::check_out_sizes(op_name, out, self.sizes());

  auto args = std::make_shared<TensorScalarKernelArgs>(TensorScalarKernelArgs{
      ops::make_cpu_view_spec(self),
      ops::make_cpu_view_spec(out),
      other,
      at::Scalar(1),
      self});

  MCPU_LAUNCH_TIMED_KERNEL(
      record_name, ([ args = std::move(args), record_name ]), {
        KernelPointerMemoryGuard guard({args->self.data, args->out.data});
        auto cpu_self = ops::cpu_view_from_spec(args->self);
        auto cpu_out = ops::cpu_view_from_spec(args->out);
        cpu_compare_scalar_out<op>(cpu_out, cpu_self, args->other);
      });
  return out;
}

template <CompareOp op>
at::Tensor compare_Scalar_impl(
    const char* op_name,
    const char* record_name,
    const at::Tensor& self,
    const at::Scalar& other) {
  auto out = empty_comparison_mcpu(self, self.sizes());
  compare_Scalar_out_impl<op>(op_name, record_name, self, other, out);
  return out;
}

template <CompareOp op>
at::Tensor& compare_Tensor__impl(
    const char* op_name,
    const char* record_name,
    at::Tensor& self,
    const at::Tensor& other) {
  compare_Tensor_out_impl<op>(op_name, record_name, self, other, self);
  return self;
}

template <CompareOp op>
at::Tensor& compare_Scalar__impl(
    const char* op_name,
    const char* record_name,
    at::Tensor& self,
    const at::Scalar& other) {
  compare_Scalar_out_impl<op>(op_name, record_name, self, other, self);
  return self;
}

at::Tensor& mul_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  return arithmetic_Tensor_out_impl<ArithmeticOp::Mul>(
      "aten::mul.out", "mcpu::aten::mul.out", self, other, at::Scalar(1), out);
}

at::Tensor mul_Tensor(const at::Tensor& self, const at::Tensor& other) {
  return arithmetic_Tensor_impl<ArithmeticOp::Mul>(
      "aten::mul.out", "mcpu::aten::mul.Tensor", self, other, at::Scalar(1));
}

at::Tensor& mul_Tensor_(at::Tensor& self, const at::Tensor& other) {
  return arithmetic_Tensor__impl<ArithmeticOp::Mul>(
      "aten::mul_.Tensor",
      "mcpu::aten::mul_.Tensor",
      self,
      other,
      at::Scalar(1));
}

at::Tensor& remainder_Tensor_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  if (is_cpu_scalar_tensor(other)) {
    ops::check_out_sizes("aten::remainder.Tensor_out", out, self.sizes());

    auto args = std::make_shared<TensorScalarKernelArgs>(TensorScalarKernelArgs{
        ops::make_cpu_view_spec(self),
        ops::make_cpu_view_spec(out),
        other.item(),
        at::Scalar(1),
        self});

    MCPU_LAUNCH_TIMED_KERNEL(
        "mcpu::aten::remainder.Tensor_out", ([args = std::move(args)]), {
          KernelPointerMemoryGuard guard({args->self.data, args->out.data});
          auto cpu_self = ops::cpu_view_from_spec(args->self);
          auto cpu_out = ops::cpu_view_from_spec(args->out);
          at::remainder_out(cpu_out, cpu_self, args->other);
        });
    return out;
  }

  auto expected_sizes = at::infer_size(self.sizes(), other.sizes());
  ops::check_out_sizes("aten::remainder.Tensor_out", out, expected_sizes);

  auto args = std::make_shared<TensorTripletKernelArgs>(TensorTripletKernelArgs{
      ops::make_cpu_view_spec(self),
      ops::make_cpu_view_spec(other),
      ops::make_cpu_view_spec(out),
      self,
      other});

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::remainder.Tensor_out", ([args = std::move(args)]), {
        KernelPointerMemoryGuard guard(
            {args->self.data, args->other.data, args->out.data});
        auto cpu_self = ops::cpu_view_from_spec(args->self);
        auto cpu_other = ops::cpu_view_from_spec(args->other);
        auto cpu_out = ops::cpu_view_from_spec(args->out);
        at::remainder_out(cpu_out, cpu_self, cpu_other);
      });
  return out;
}

at::Tensor sub_Tensor(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  return arithmetic_Tensor_impl<ArithmeticOp::Sub>(
      "aten::sub.out", "mcpu::aten::sub.Tensor", self, other, alpha);
}

at::Tensor& sub_out(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& out) {
  return arithmetic_Tensor_out_impl<ArithmeticOp::Sub>(
      "aten::sub.out", "mcpu::aten::sub.out", self, other, alpha, out);
}

at::Tensor& sub_Tensor_(
    at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  return arithmetic_Tensor__impl<ArithmeticOp::Sub>(
      "aten::sub_.Tensor", "mcpu::aten::sub_.Tensor", self, other, alpha);
}

at::Tensor sub_Scalar(
    const at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  auto out = empty_like_mcpu_result(self, other);
  arithmetic_Scalar_out_impl<ArithmeticOp::Sub>(
      "aten::sub.Scalar_out",
      "mcpu::aten::sub.Scalar",
      self,
      other,
      alpha,
      out);
  return out;
}

at::Tensor& sub_Scalar_(
    at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  auto self_spec = ops::make_cpu_view_spec(self);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::sub_.Scalar",
      ([ self_spec = std::move(self_spec), other, alpha ]),
      {
        KernelPointerMemoryGuard guard({self_spec.data});
        auto cpu_self = ops::cpu_view_from_spec(self_spec);
        at::sub_out(cpu_self, cpu_self, other, alpha);
      });
  return self;
}

at::Tensor& pow_Scalar_out(
    const at::Scalar& self,
    const at::Tensor& exponent,
    at::Tensor& out) {
  ops::check_out_sizes("aten::pow.Scalar_out", out, exponent.sizes());

  auto args = std::make_shared<PowScalarOutKernelArgs>(PowScalarOutKernelArgs{
      self,
      ops::make_cpu_view_spec(exponent),
      ops::make_cpu_view_spec(out),
      exponent});

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::pow.Scalar_out", ([args = std::move(args)]), {
        KernelPointerMemoryGuard guard({args->exponent.data, args->out.data});
        auto cpu_exponent = ops::cpu_view_from_spec(args->exponent);
        auto cpu_out = ops::cpu_view_from_spec(args->out);
        at::pow_out(cpu_out, args->self, cpu_exponent);
      });
  return out;
}

#define DEFINE_COMPARE_WRAPPERS(name, op)                                     \
  at::Tensor& name##_Tensor_out(                                              \
      const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {     \
    return compare_Tensor_out_impl<op>(                                       \
        "aten::" #name ".Tensor_out",                                         \
        "mcpu::aten::" #name ".Tensor_out",                                   \
        self,                                                                 \
        other,                                                                \
        out);                                                                 \
  }                                                                           \
                                                                              \
  at::Tensor name##_Tensor(const at::Tensor& self, const at::Tensor& other) { \
    return compare_Tensor_impl<op>(                                           \
        "aten::" #name ".Tensor_out",                                         \
        "mcpu::aten::" #name ".Tensor",                                       \
        self,                                                                 \
        other);                                                               \
  }                                                                           \
                                                                              \
  at::Tensor& name##_Scalar_out(                                              \
      const at::Tensor& self, const at::Scalar& other, at::Tensor& out) {     \
    return compare_Scalar_out_impl<op>(                                       \
        "aten::" #name ".Scalar_out",                                         \
        "mcpu::aten::" #name ".Scalar_out",                                   \
        self,                                                                 \
        other,                                                                \
        out);                                                                 \
  }                                                                           \
                                                                              \
  at::Tensor name##_Scalar(const at::Tensor& self, const at::Scalar& other) { \
    return compare_Scalar_impl<op>(                                           \
        "aten::" #name ".Scalar_out",                                         \
        "mcpu::aten::" #name ".Scalar",                                       \
        self,                                                                 \
        other);                                                               \
  }                                                                           \
                                                                              \
  at::Tensor& name##_Tensor_(at::Tensor& self, const at::Tensor& other) {     \
    return compare_Tensor__impl<op>(                                          \
        "aten::" #name "_.Tensor",                                            \
        "mcpu::aten::" #name "_.Tensor",                                      \
        self,                                                                 \
        other);                                                               \
  }                                                                           \
                                                                              \
  at::Tensor& name##_Scalar_(at::Tensor& self, const at::Scalar& other) {     \
    return compare_Scalar__impl<op>(                                          \
        "aten::" #name "_.Scalar",                                            \
        "mcpu::aten::" #name "_.Scalar",                                      \
        self,                                                                 \
        other);                                                               \
  }

DEFINE_COMPARE_WRAPPERS(eq, CompareOp::Eq)
DEFINE_COMPARE_WRAPPERS(ne, CompareOp::Ne)
DEFINE_COMPARE_WRAPPERS(lt, CompareOp::Lt)
DEFINE_COMPARE_WRAPPERS(le, CompareOp::Le)
DEFINE_COMPARE_WRAPPERS(gt, CompareOp::Gt)
DEFINE_COMPARE_WRAPPERS(ge, CompareOp::Ge)
DEFINE_COMPARE_WRAPPERS(less, CompareOp::Lt)
DEFINE_COMPARE_WRAPPERS(less_equal, CompareOp::Le)

#undef DEFINE_COMPARE_WRAPPERS

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", &add_Tensor);
  m.impl("add.out", &add_out);
  m.impl("add_.Tensor", &add_Tensor_);
  m.impl("add.Scalar", &add_Scalar);
  m.impl("add_.Scalar", &add_Scalar_);
  m.impl("div.Tensor", &div_Tensor);
  m.impl("div.out", &div_out);
  m.impl("div_.Tensor", &div_Tensor_);
  m.impl("eq.Tensor", &eq_Tensor);
  m.impl("eq.Tensor_out", &eq_Tensor_out);
  m.impl("eq.Scalar", &eq_Scalar);
  m.impl("eq.Scalar_out", &eq_Scalar_out);
  m.impl("eq_.Tensor", &eq_Tensor_);
  m.impl("eq_.Scalar", &eq_Scalar_);
  m.impl("ge.Tensor", &ge_Tensor);
  m.impl("ge.Tensor_out", &ge_Tensor_out);
  m.impl("ge.Scalar", &ge_Scalar);
  m.impl("ge.Scalar_out", &ge_Scalar_out);
  m.impl("ge_.Tensor", &ge_Tensor_);
  m.impl("ge_.Scalar", &ge_Scalar_);
  m.impl("gt.Tensor", &gt_Tensor);
  m.impl("gt.Tensor_out", &gt_Tensor_out);
  m.impl("gt.Scalar", &gt_Scalar);
  m.impl("gt.Scalar_out", &gt_Scalar_out);
  m.impl("gt_.Tensor", &gt_Tensor_);
  m.impl("gt_.Scalar", &gt_Scalar_);
  m.impl("le.Tensor", &le_Tensor);
  m.impl("le.Tensor_out", &le_Tensor_out);
  m.impl("le.Scalar", &le_Scalar);
  m.impl("le.Scalar_out", &le_Scalar_out);
  m.impl("le_.Tensor", &le_Tensor_);
  m.impl("le_.Scalar", &le_Scalar_);
  m.impl("less.Tensor", &less_Tensor);
  m.impl("less.Tensor_out", &less_Tensor_out);
  m.impl("less.Scalar", &less_Scalar);
  m.impl("less.Scalar_out", &less_Scalar_out);
  m.impl("less_.Tensor", &less_Tensor_);
  m.impl("less_.Scalar", &less_Scalar_);
  m.impl("less_equal.Tensor", &less_equal_Tensor);
  m.impl("less_equal.Tensor_out", &less_equal_Tensor_out);
  m.impl("less_equal.Scalar", &less_equal_Scalar);
  m.impl("less_equal.Scalar_out", &less_equal_Scalar_out);
  m.impl("less_equal_.Tensor", &less_equal_Tensor_);
  m.impl("less_equal_.Scalar", &less_equal_Scalar_);
  m.impl("lt.Tensor", &lt_Tensor);
  m.impl("lt.Tensor_out", &lt_Tensor_out);
  m.impl("lt.Scalar", &lt_Scalar);
  m.impl("lt.Scalar_out", &lt_Scalar_out);
  m.impl("lt_.Tensor", &lt_Tensor_);
  m.impl("lt_.Scalar", &lt_Scalar_);
  m.impl("mul.Tensor", &mul_Tensor);
  m.impl("mul.out", &mul_out);
  m.impl("mul_.Tensor", &mul_Tensor_);
  m.impl("ne.Tensor", &ne_Tensor);
  m.impl("ne.Tensor_out", &ne_Tensor_out);
  m.impl("ne.Scalar", &ne_Scalar);
  m.impl("ne_.Tensor", &ne_Tensor_);
  m.impl("ne_.Scalar", &ne_Scalar_);
  m.impl("sub.Tensor", &sub_Tensor);
  m.impl("sub.out", &sub_out);
  m.impl("sub_.Tensor", &sub_Tensor_);
  m.impl("sub.Scalar", &sub_Scalar);
  m.impl("sub_.Scalar", &sub_Scalar_);
  m.impl("pow.Scalar_out", &pow_Scalar_out);
  m.impl("remainder.Tensor_out", &remainder_Tensor_out);
}

} // namespace at::mcpu

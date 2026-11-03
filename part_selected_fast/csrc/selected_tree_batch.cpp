#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <exception>
#include "tree_core.h"
#ifdef _OPENMP
#include <omp.h>
#endif
namespace py = pybind11;

py::tuple build_batch(py::array_t<double,py::array::c_style> vectors,
                      py::array_t<double,py::array::c_style> raw,
                      py::array_t<bool,py::array::c_style> mask, int threads) {
  using namespace selected_fast;
  require(vectors.ndim()==3 && vectors.shape(2)==4,"vectors must be [B,N,4]");
  const int b=static_cast<int>(vectors.shape(0)), n=static_cast<int>(vectors.shape(1));
  require(n>=1 && n<=128 && b>=1,"invalid batch dimensions");
  require(raw.ndim()==3 && raw.shape(0)==b && raw.shape(1)==n && raw.shape(2)==14,"raw must be [B,N,14]");
  require(mask.ndim()==2 && mask.shape(0)==b && mask.shape(1)==n,"mask must be [B,N]");
  require(threads>=1 && threads<=64,"threads must be 1..64");
#ifndef _OPENMP
  require(threads==1,"parallel backend requires OpenMP");
#endif
  const int iw=integer_width(n), fw=float_width(n);
  py::array_t<int32_t> ints({b,iw}); py::array_t<float> floats({b,fw});
  auto* ip=ints.mutable_data(); auto* fp=floats.mutable_data();
  const auto* vp=vectors.data(); const auto* rp=raw.data(); const auto* mp=mask.data();
  std::vector<std::exception_ptr> errors(b);
  {
    py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
    for (int i=0;i<b;++i) {
      try { build(vp+i*n*4,rp+i*n*14,mp+i*n,n,ip+i*iw,fp+i*fw); }
      catch (...) { errors[i]=std::current_exception(); }
    }
    // All native workers have joined. Report the first failing jet in input order.
    for (const auto& error: errors) if (error) std::rethrow_exception(error);
  }
  return py::make_tuple(ints,floats);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {
  module.def("build_batch", &build_batch);
#ifdef _OPENMP
  module.attr("openmp")=true;
#else
  module.attr("openmp")=false;
#endif
  module.attr("contract")="selected_tree_batch_v1";
}

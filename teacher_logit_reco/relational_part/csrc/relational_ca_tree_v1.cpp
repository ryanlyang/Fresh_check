#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double kEpsilon = 1.0e-6;

struct FourVector {
  double px, py, pz, energy;
};

struct Node {
  FourVector vector;
  int parent = -1;
  int left = -1;
  int right = -1;
  int multiplicity = 1;
  double merge_delta_r = 0.0;
  double merge_kt = 0.0;
  double merge_z = 0.0;
  std::vector<std::string> leaf_keys;
};

double pt(const FourVector& value) {
  return std::hypot(value.px, value.py);
}

double mass(const FourVector& value) {
  const double mass2 = value.energy * value.energy - value.px * value.px -
                       value.py * value.py - value.pz * value.pz;
  return std::sqrt(std::max(mass2, 0.0));
}

double eta(const FourVector& value) {
  return std::asinh(value.pz / std::max(pt(value), 1.0e-300));
}

double phi(const FourVector& value) {
  return std::atan2(value.py, value.px);
}

double delta_phi(double left, double right) {
  return std::atan2(std::sin(left - right), std::cos(left - right));
}

double delta_r(const FourVector& left, const FourVector& right) {
  return std::hypot(eta(left) - eta(right),
                    delta_phi(phi(left), phi(right)));
}

std::string canonical_double(double value) {
  if (value == 0.0) value = 0.0;
  if (std::isnan(value)) {
    const uint64_t canonical_nan = 0x7ff8000000000000ULL;
    std::string result(8, '\0');
    for (int index = 0; index < 8; ++index) {
      result[index] = static_cast<char>(
          (canonical_nan >> (56 - 8 * index)) & 0xff);
    }
    return result;
  }
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(double));
  std::string result(8, '\0');
  for (int index = 0; index < 8; ++index) {
    result[index] =
        static_cast<char>((bits >> (56 - 8 * index)) & 0xff);
  }
  return result;
}

struct Candidate {
  double distance2;
  int first;
  int second;
  std::vector<std::string> first_keys;
  std::vector<std::string> second_keys;
};

struct CandidateAfter {
  bool operator()(const Candidate& left, const Candidate& right) const {
    if (left.distance2 != right.distance2)
      return left.distance2 > right.distance2;
    if (left.first_keys != right.first_keys)
      return left.first_keys > right.first_keys;
    return left.second_keys > right.second_keys;
  }
};

Candidate make_candidate(
    int first, int second, const std::vector<Node>& nodes) {
  auto first_keys = nodes[first].leaf_keys;
  auto second_keys = nodes[second].leaf_keys;
  if (second_keys < first_keys) {
    // Canonicalize only the tie key; child links preserve active-node order.
    std::swap(first_keys, second_keys);
  }
  const double dr = delta_r(nodes[first].vector, nodes[second].vector);
  return Candidate{dr * dr, first, second, first_keys, second_keys};
}

template <typename T>
torch::Tensor tensor_copy(
    const std::vector<T>& values, torch::ScalarType dtype) {
  if (values.empty()) return torch::empty({0}, torch::dtype(dtype));
  return torch::from_blob(
             const_cast<T*>(values.data()),
             {static_cast<int64_t>(values.size())},
             torch::TensorOptions().dtype(dtype))
      .clone();
}

py::dict build_tree(
    torch::Tensor vectors_input,
    torch::Tensor raw_input,
    torch::Tensor mask_input) {
  TORCH_CHECK(!vectors_input.is_cuda() && !raw_input.is_cuda() &&
                  !mask_input.is_cuda(),
              "relational_ca_tree_v1 accepts CPU tensors only");
  auto vectors = vectors_input.to(torch::kFloat64).contiguous();
  auto raw = raw_input.to(torch::kFloat64).contiguous();
  auto mask = mask_input.to(torch::kBool).contiguous();
  TORCH_CHECK(vectors.dim() == 2 && vectors.size(1) == 4,
              "vectors must have shape [particles,4]");
  TORCH_CHECK(raw.dim() == 2 && raw.size(0) == vectors.size(0) &&
                  raw.size(1) == 14,
              "raw tokens must have shape [particles,14]");
  TORCH_CHECK(mask.dim() == 1 && mask.size(0) == vectors.size(0),
              "mask must have shape [particles]");
  const int length = static_cast<int>(vectors.size(0));
  const auto vector_access = vectors.accessor<double, 2>();
  const auto raw_access = raw.accessor<double, 2>();
  const auto mask_access = mask.accessor<bool, 1>();
  std::vector<std::pair<std::string, int>> leaves;
  for (int original = 0; original < length; ++original) {
    if (!mask_access[original]) continue;
    std::string key;
    for (int field = 0; field < 4; ++field) {
      TORCH_CHECK(std::isfinite(vector_access[original][field]),
                  "valid vector is nonfinite");
      key += canonical_double(vector_access[original][field]);
    }
    for (int field = 0; field < 14; ++field) {
      TORCH_CHECK(std::isfinite(raw_access[original][field]),
                  "valid raw token is nonfinite");
      key += canonical_double(raw_access[original][field]);
    }
    leaves.emplace_back(key, original);
  }
  std::sort(
      leaves.begin(), leaves.end(),
      [](const auto& left, const auto& right) {
        return left.first < right.first;
      });
  const int n_valid = static_cast<int>(leaves.size());
  TORCH_CHECK(n_valid <= 128, "tree exceeds 128 constituents");
  py::dict result;
  result["contract"] = "relational_ca_tree_packed_v1";
  result["n_particles"] = length;
  result["n_valid"] = n_valid;
  if (n_valid == 0) {
    result["n_nodes"] = 0;
    result["root"] = -1;
    result["leaf_to_node"] = torch::full({length}, -1, torch::kInt32);
    for (const char* name : {"parent", "left", "right", "depth",
                             "multiplicity"}) {
      result[name] = torch::empty({0}, torch::kInt32);
    }
    for (const char* name : {"pt", "mass", "merge_delta_r", "merge_kt",
                             "merge_z", "merge_mass"}) {
      result[name] = torch::empty({0}, torch::kFloat32);
    }
    result["vectors"] = torch::empty({0, 4}, torch::kFloat32);
    for (int resolution : {2, 4, 8}) {
      result[("assignment_K" + std::to_string(resolution)).c_str()] =
          torch::full({length}, -1, torch::kInt32);
      result[("actual_count_K" + std::to_string(resolution)).c_str()] = 0;
    }
    return result;
  }

  std::vector<Node> nodes;
  std::vector<int> original_for_leaf;
  for (const auto& leaf : leaves) {
    const int original = leaf.second;
    nodes.push_back(Node{
        FourVector{vector_access[original][0], vector_access[original][1],
                   vector_access[original][2], vector_access[original][3]},
        -1, -1, -1, 1, 0.0, 0.0, 0.0, {leaf.first}});
    original_for_leaf.push_back(original);
  }
  std::vector<bool> active(n_valid, true);
  std::priority_queue<Candidate, std::vector<Candidate>, CandidateAfter> queue;
  for (int first = 0; first < n_valid; ++first)
    for (int second = first + 1; second < n_valid; ++second)
      queue.push(make_candidate(first, second, nodes));
  int active_count = n_valid;
  while (active_count > 1) {
    Candidate candidate;
    do {
      TORCH_CHECK(!queue.empty(), "tree priority queue exhausted");
      candidate = queue.top();
      queue.pop();
    } while (!active[candidate.first] || !active[candidate.second]);
    const int first = candidate.first;
    const int second = candidate.second;
    const double dr = std::sqrt(candidate.distance2);
    const double first_pt = pt(nodes[first].vector);
    const double second_pt = pt(nodes[second].vector);
    Node merged;
    merged.vector = {
        nodes[first].vector.px + nodes[second].vector.px,
        nodes[first].vector.py + nodes[second].vector.py,
        nodes[first].vector.pz + nodes[second].vector.pz,
        nodes[first].vector.energy + nodes[second].vector.energy};
    merged.left = first;
    merged.right = second;
    merged.multiplicity =
        nodes[first].multiplicity + nodes[second].multiplicity;
    merged.merge_delta_r = dr;
    merged.merge_kt = std::min(first_pt, second_pt) * dr;
    merged.merge_z =
        std::min(first_pt, second_pt) / (first_pt + second_pt + kEpsilon);
    merged.leaf_keys = nodes[first].leaf_keys;
    merged.leaf_keys.insert(
        merged.leaf_keys.end(), nodes[second].leaf_keys.begin(),
        nodes[second].leaf_keys.end());
    std::sort(merged.leaf_keys.begin(), merged.leaf_keys.end());
    const int new_index = static_cast<int>(nodes.size());
    nodes[first].parent = new_index;
    nodes[second].parent = new_index;
    nodes.push_back(std::move(merged));
    active[first] = false;
    active[second] = false;
    active.push_back(true);
    --active_count;
    for (int node = 0; node < new_index; ++node)
      if (active[node]) queue.push(make_candidate(node, new_index, nodes));
  }
  const int root = static_cast<int>(nodes.size()) - 1;
  std::vector<int32_t> parent, left, right, depth(nodes.size(), -1);
  std::vector<int32_t> multiplicity;
  std::vector<float> vectors_flat, pts, masses, merge_dr, merge_kt, merge_z;
  parent.reserve(nodes.size());
  left.reserve(nodes.size());
  right.reserve(nodes.size());
  multiplicity.reserve(nodes.size());
  for (const auto& node : nodes) {
    parent.push_back(node.parent);
    left.push_back(node.left);
    right.push_back(node.right);
    multiplicity.push_back(node.multiplicity);
    vectors_flat.insert(vectors_flat.end(),
                        {static_cast<float>(node.vector.px),
                         static_cast<float>(node.vector.py),
                         static_cast<float>(node.vector.pz),
                         static_cast<float>(node.vector.energy)});
    pts.push_back(static_cast<float>(pt(node.vector)));
    masses.push_back(static_cast<float>(mass(node.vector)));
    merge_dr.push_back(static_cast<float>(node.merge_delta_r));
    merge_kt.push_back(static_cast<float>(node.merge_kt));
    merge_z.push_back(static_cast<float>(node.merge_z));
  }
  depth[root] = 0;
  std::vector<int> stack{root};
  while (!stack.empty()) {
    int node = stack.back();
    stack.pop_back();
    for (int child : {nodes[node].left, nodes[node].right}) {
      if (child >= 0) {
        depth[child] = depth[node] + 1;
        stack.push_back(child);
      }
    }
  }
  std::vector<int32_t> leaf_to_node(length, -1);
  for (int leaf = 0; leaf < n_valid; ++leaf)
    leaf_to_node[original_for_leaf[leaf]] = leaf;
  result["n_nodes"] = static_cast<int>(nodes.size());
  result["root"] = root;
  result["leaf_to_node"] = tensor_copy(leaf_to_node, torch::kInt32);
  result["parent"] = tensor_copy(parent, torch::kInt32);
  result["left"] = tensor_copy(left, torch::kInt32);
  result["right"] = tensor_copy(right, torch::kInt32);
  result["depth"] = tensor_copy(depth, torch::kInt32);
  result["multiplicity"] = tensor_copy(multiplicity, torch::kInt32);
  result["vectors"] =
      tensor_copy(vectors_flat, torch::kFloat32).reshape({-1, 4});
  result["pt"] = tensor_copy(pts, torch::kFloat32);
  result["mass"] = tensor_copy(masses, torch::kFloat32);
  result["merge_delta_r"] = tensor_copy(merge_dr, torch::kFloat32);
  result["merge_kt"] = tensor_copy(merge_kt, torch::kFloat32);
  result["merge_z"] = tensor_copy(merge_z, torch::kFloat32);
  result["merge_mass"] = tensor_copy(masses, torch::kFloat32);
  for (int resolution : {2, 4, 8}) {
    const int count = std::min(resolution, n_valid);
    std::vector<int> clusters{root};
    while (static_cast<int>(clusters.size()) < count) {
      auto split_it = std::max_element(
          clusters.begin(), clusters.end(),
          [&](int lhs, int rhs) {
            const bool lhs_internal = nodes[lhs].left >= 0;
            const bool rhs_internal = nodes[rhs].left >= 0;
            if (lhs_internal != rhs_internal) return !lhs_internal;
            return lhs < rhs;
          });
      const int split = *split_it;
      TORCH_CHECK(nodes[split].left >= 0, "exclusive split reached a leaf");
      clusters.erase(split_it);
      clusters.push_back(nodes[split].left);
      clusters.push_back(nodes[split].right);
    }
    std::vector<int32_t> assignment(length, -1);
    for (int leaf = 0; leaf < n_valid; ++leaf) {
      int node = leaf;
      while (std::find(clusters.begin(), clusters.end(), node) ==
             clusters.end())
        node = nodes[node].parent;
      assignment[original_for_leaf[leaf]] = node;
    }
    result[("assignment_K" + std::to_string(resolution)).c_str()] =
        tensor_copy(assignment, torch::kInt32);
    result[("actual_count_K" + std::to_string(resolution)).c_str()] = count;
  }
  return result;
}

py::dict backend_manifest() {
  py::dict result;
  result["contract_id"] = "relational_ca_tree_v1";
  result["schema_version"] = 1;
  result["compiler_flags"] = py::make_tuple(
      "-O3", "-std=c++17", "-fopenmp", "-fno-fast-math",
      "-fno-associative-math", "-ffp-contract=off");
#ifdef _OPENMP
  result["openmp_available"] = true;
#else
  result["openmp_available"] = false;
#endif
  result["pytorch_cxx11_abi"] =
#ifdef _GLIBCXX_USE_CXX11_ABI
      static_cast<bool>(_GLIBCXX_USE_CXX11_ABI);
#else
      false;
#endif
  result["compiler_family"] =
#if defined(__clang__)
      "clang";
  result["compiler_major_version"] = __clang_major__;
#elif defined(__GNUC__)
      "gcc";
  result["compiler_major_version"] = __GNUC__;
#elif defined(_MSC_VER)
      "msvc";
  result["compiler_major_version"] = _MSC_VER / 100;
#else
      "unknown";
  result["compiler_major_version"] = 0;
#endif
  result["compiler_version"] =
#if defined(__VERSION__)
      __VERSION__;
#else
      "unknown";
#endif
  result["platform_architecture"] =
#if defined(__aarch64__)
      "aarch64";
#elif defined(__x86_64__) || defined(_M_X64)
      "x86_64";
#elif defined(__arm__)
      "arm";
#else
      "unknown";
#endif
  return result;
}

py::dict self_test() {
  py::dict result;
  result["contract_id"] = "relational_ca_tree_v1";
  result["vector"] = py::make_tuple(3.0, 4.0, 0.0, 5.0);
  result["pt"] = 5.0;
  result["mass"] = 0.0;
  return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("backend_manifest", &backend_manifest);
  module.def("self_test", &self_test);
  module.def("build_tree", &build_tree);
}

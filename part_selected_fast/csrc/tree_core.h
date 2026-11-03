// Per-jet arithmetic/order copied from relational_ca_tree_v1.cpp. No Python,
// Torch or shared mutable state: independent jets may run concurrently.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace selected_fast {
constexpr double kEpsilon = 1.0e-6;
struct FourVector { double px, py, pz, energy; };
struct Node {
  FourVector vector;
  int parent = -1, left = -1, right = -1, multiplicity = 1;
  double merge_delta_r = 0.0, merge_kt = 0.0, merge_z = 0.0;
  std::vector<std::string> leaf_keys;
};
inline double pt(const FourVector& v) { return std::hypot(v.px, v.py); }
inline double mass(const FourVector& v) {
  const double m2 = v.energy*v.energy-v.px*v.px-v.py*v.py-v.pz*v.pz;
  return std::sqrt(std::max(m2, 0.0));
}
inline double eta(const FourVector& v) { return std::asinh(v.pz/std::max(pt(v), 1.0e-300)); }
inline double phi(const FourVector& v) { return std::atan2(v.py, v.px); }
inline double delta_phi(double a, double b) { return std::atan2(std::sin(a-b), std::cos(a-b)); }
inline double delta_r(const FourVector& a, const FourVector& b) {
  return std::hypot(eta(a)-eta(b), delta_phi(phi(a), phi(b)));
}
inline std::string canonical_double(double value) {
  if (value == 0.0) value = 0.0;
  uint64_t bits;
  if (std::isnan(value)) bits = 0x7ff8000000000000ULL;
  else std::memcpy(&bits, &value, sizeof(double));
  std::string result(8, '\0');
  for (int i=0; i<8; ++i) result[i] = static_cast<char>((bits >> (56-8*i)) & 0xff);
  return result;
}
struct Candidate {
  double distance2;
  int first, second;
  std::vector<std::string> first_keys, second_keys;
};
struct CandidateAfter {
  bool operator()(const Candidate& a, const Candidate& b) const {
    if (a.distance2 != b.distance2) return a.distance2 > b.distance2;
    if (a.first_keys != b.first_keys) return a.first_keys > b.first_keys;
    return a.second_keys > b.second_keys;
  }
};
inline Candidate make_candidate(int a, int b, const std::vector<Node>& nodes) {
  auto ak = nodes[a].leaf_keys, bk = nodes[b].leaf_keys;
  if (bk < ak) std::swap(ak, bk);
  const double dr = delta_r(nodes[a].vector, nodes[b].vector);
  return Candidate{dr*dr, a, b, ak, bk};
}
inline void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

// Fixed-width row layout, including padding, matches region.pack exactly.
// Integers: parent, depth, multiplicity [M each]; leaf [N]; assignments [3N];
// roots, n_valid. Floats: pt,mass,dr,kt,z,merge_mass [M each], vectors [M,4],
// leaf_ranks [3,N]. 32-bit indices suffice for <=128 particles; Python expands
// them to the reference int64 before ancestor/feature calculations.
inline int integer_width(int n) { return 3*(2*n-1)+4*n+2; }
inline int float_width(int n) { return 10*(2*n-1)+3*n; }

inline void build(const double* vectors, const double* raw, const bool* mask,
                  int n, int32_t* integer, float* floating) {
  require(n >= 1 && n <= 128, "particle width must be 1..128");
  const int m = 2*n-1;
  std::fill(integer, integer+integer_width(n), 0);
  std::fill(floating, floating+float_width(n), 0.0f);
  std::vector<std::pair<std::string,int>> leaves;
  for (int i=0; i<n; ++i) {
    if (!mask[i]) continue;
    std::string key;
    for (int f=0; f<4; ++f) {
      require(std::isfinite(vectors[4*i+f]), "valid vector is nonfinite");
      key += canonical_double(vectors[4*i+f]);
    }
    for (int f=0; f<14; ++f) {
      require(std::isfinite(raw[14*i+f]), "valid raw token is nonfinite");
      key += canonical_double(raw[14*i+f]);
    }
    leaves.emplace_back(key, i);
  }
  std::sort(leaves.begin(), leaves.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  const int nv = static_cast<int>(leaves.size());
  integer[3*m+4*n+1] = nv;
  if (!nv) return;
  std::vector<Node> nodes;
  std::vector<int> original;
  for (const auto& leaf: leaves) {
    const int i=leaf.second;
    nodes.push_back(Node{FourVector{vectors[4*i], vectors[4*i+1], vectors[4*i+2], vectors[4*i+3]},
                        -1,-1,-1,1,0.0,0.0,0.0,{leaf.first}});
    original.push_back(i);
  }
  std::vector<bool> active(nv, true);
  std::priority_queue<Candidate,std::vector<Candidate>,CandidateAfter> queue;
  for (int a=0; a<nv; ++a)
    for (int b=a+1; b<nv; ++b) queue.push(make_candidate(a,b,nodes));
  int active_count=nv;
  while (active_count > 1) {
    Candidate candidate;
    do {
      require(!queue.empty(), "tree priority queue exhausted");
      candidate=queue.top(); queue.pop();
    } while (!active[candidate.first] || !active[candidate.second]);
    const int a=candidate.first, b=candidate.second;
    const double dr=std::sqrt(candidate.distance2), apt=pt(nodes[a].vector), bpt=pt(nodes[b].vector);
    Node merged;
    merged.vector={nodes[a].vector.px+nodes[b].vector.px, nodes[a].vector.py+nodes[b].vector.py,
                   nodes[a].vector.pz+nodes[b].vector.pz, nodes[a].vector.energy+nodes[b].vector.energy};
    merged.left=a; merged.right=b;
    merged.multiplicity=nodes[a].multiplicity+nodes[b].multiplicity;
    merged.merge_delta_r=dr; merged.merge_kt=std::min(apt,bpt)*dr;
    merged.merge_z=std::min(apt,bpt)/(apt+bpt+kEpsilon);
    merged.leaf_keys=nodes[a].leaf_keys;
    merged.leaf_keys.insert(merged.leaf_keys.end(),nodes[b].leaf_keys.begin(),nodes[b].leaf_keys.end());
    std::sort(merged.leaf_keys.begin(),merged.leaf_keys.end());
    const int next=static_cast<int>(nodes.size());
    nodes[a].parent=next; nodes[b].parent=next;
    nodes.push_back(std::move(merged));
    active[a]=false; active[b]=false; active.push_back(true); --active_count;
    for (int i=0; i<next; ++i) if (active[i]) queue.push(make_candidate(i,next,nodes));
  }
  const int root=static_cast<int>(nodes.size())-1;
  integer[3*m+4*n]=root;
  std::vector<int> depth(nodes.size(),-1), stack{root}; depth[root]=0;
  while (!stack.empty()) {
    int i=stack.back(); stack.pop_back();
    for (int child: {nodes[i].left,nodes[i].right}) if (child>=0) {
      depth[child]=depth[i]+1; stack.push_back(child);
    }
  }
  for (int i=0; i<=root; ++i) {
    const Node& node=nodes[i];
    integer[i]=i==root ? root : node.parent;
    integer[m+i]=depth[i]; integer[2*m+i]=node.multiplicity;
    const float node_mass=static_cast<float>(mass(node.vector));
    floating[i]=static_cast<float>(pt(node.vector)); floating[m+i]=node_mass;
    floating[2*m+i]=static_cast<float>(node.merge_delta_r);
    floating[3*m+i]=static_cast<float>(node.merge_kt);
    floating[4*m+i]=static_cast<float>(node.merge_z); floating[5*m+i]=node_mass;
    floating[6*m+4*i]=static_cast<float>(node.vector.px);
    floating[6*m+4*i+1]=static_cast<float>(node.vector.py);
    floating[6*m+4*i+2]=static_cast<float>(node.vector.pz);
    floating[6*m+4*i+3]=static_cast<float>(node.vector.energy);
  }
  for (int i=0; i<nv; ++i) integer[3*m+original[i]]=i;
  for (int k=0; k<3; ++k) {
    const int count=std::min(2<<k,nv);
    std::vector<int> clusters{root};
    while (static_cast<int>(clusters.size())<count) {
      auto it=std::max_element(clusters.begin(),clusters.end(),[&](int a,int b) {
        const bool ai=nodes[a].left>=0, bi=nodes[b].left>=0;
        if (ai!=bi) return !ai;
        return a<b;
      });
      const int split=*it;
      require(nodes[split].left>=0,"exclusive split reached a leaf");
      clusters.erase(it); clusters.push_back(nodes[split].left); clusters.push_back(nodes[split].right);
    }
    for (int leaf=0; leaf<nv; ++leaf) {
      int node=leaf;
      while (std::find(clusters.begin(),clusters.end(),node)==clusters.end()) node=nodes[node].parent;
      integer[3*m+n+k*n+original[leaf]]=node;
    }
    // region.pack ranks float32 pt, including its half-rank tie rule.
    for (int cluster: clusters) {
      int greater=0, equal=0;
      for (int other: clusters) {
        greater += floating[other]>floating[cluster]; equal += floating[other]==floating[cluster];
      }
      const float rank=static_cast<float>((greater+0.5*(equal-1))/std::max(1,count-1));
      for (int i=0; i<n; ++i)
        if (integer[3*m+n+k*n+i]==cluster) floating[10*m+k*n+i]=rank;
    }
  }
  for (int i=0; i<float_width(n); ++i) require(std::isfinite(floating[i]),"nonfinite tree output");
}
}  // namespace selected_fast

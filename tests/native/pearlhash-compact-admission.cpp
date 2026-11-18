// Source-derived host admission fixture: no SYCL SDK, compiler runtime or GPU.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "claim.h"
#include "job-boundary.h"

struct CompactProfile {
  static constexpr int k=65536, rank=128, rows=24, cols=8;
#include "shape.inc"
};
namespace compact_private {
#include "selected.inc"
}
struct Buffers { int proof_rows=16,proof_cols=16; };
struct Native {
  bool admitted=false,compact=false,fault=false;
  bool ensure(int&, const Buffers&, int,int,int,int) {
    if(fault) throw std::runtime_error("submitted mock failure");
    return admitted;
  }
};
struct State { Native hip,cuda; };
struct Result { int rows,cols; bool compact,pair; };
static Result choose(State& st, Buffers& b, bool hip_jit, bool cuda_jit,
                     int k=65536, int rank=128) {
  int q=0,m=16320,n=15360;
  (void)m;(void)n;(void)q;
#include "admission.inc"
  return {b.proof_rows,b.proof_cols,compact_mode,pair_dist8};
}
static void check(bool yes,const char* message) {
  if(!yes) throw std::runtime_error(message);
}
static void check_target(const Result& r,uint32_t expected) {
  uint32_t factor=0;
  check(mom_pearlhash::pearlhash_adjustment_factor(r.rows,r.cols,65536,128,&factor)
        &&factor==expected,"wrong effective factor");
  std::array<uint8_t,32> base{},out{};base[31]=1;
  check(mom_pearlhash::pearlhash_search_target_le(nullptr,base.data(),r.rows,r.cols,
        65536,128,out.data()),"target rejected");
  for(unsigned i=0;i<32;++i)
    check(out[i]==(i<4?uint8_t(expected>>(8*i)):0),"wrong effective target");
}
int main() {
  try {
    // The same legal shape must work without a private environment prerequisite.
    unsetenv("MOM_PRIVATE_COMPACT");
    check(compact_private::selected(16320,15360,65536,128),"private flag still required");
    setenv("MOM_PRIVATE_COMPACT","0",1);
    check(compact_private::selected(16320,15360,65536,128),"private flag changes admission");
    for(const auto& x:std::array<std::array<int,4>,5>{{
        {16321,15360,65536,128},{16320,15361,65536,128},{16320,15360,8192,128},
        {16320,15360,65536,256},{0,15360,65536,128}}})
      check(!compact_private::selected(x[0],x[1],x[2],x[3]),"unsupported shape admitted");
    unsigned cases=0;
    for(bool hip_jit:{false,true})for(bool cuda_jit:{false,true})
      for(bool hip_ok:{false,true})for(bool cuda_ok:{false,true})
        for(bool cuda_compact:{false,true}) {
          State st{{hip_ok,false,false},{cuda_ok,cuda_compact,false}};
          // Deliberately stale geometry: every selection must replace it.
          Buffers b{10,16};
          const auto r=choose(st,b,hip_jit,cuda_jit);
          const bool compact=(!hip_jit||!hip_ok)&&cuda_jit&&cuda_ok&&cuda_compact;
          check(r.compact==compact,"wrong compact admission");
          check(r.rows==(compact?24:hip_jit?4:16)&&r.cols==(compact?8:hip_jit?64:16),
                "fallback leaked compact geometry");
          check(r.pair==compact,"pair mode changed");
          check_target(r,uint32_t(r.rows*r.cols*65536));++cases;
        }
    State legacy{{false,false,false},{true,false,false}};Buffers b{10,16};
    const auto r=choose(legacy,b,false,true,8192,128);
    check(!r.compact&&r.pair&&r.rows==16&&r.cols==16,"legacy pair changed");
    State broken{{false,false,false},{true,true,true}};
    bool threw=false;try{(void)choose(broken,b,false,true);}catch(const std::runtime_error&){threw=true;}
    check(threw,"submitted fault was swallowed");
    std::printf("COMPACT_ADMISSION_PASS cases=%u stale_geometry_reset=1 actual_factor_target=1 GPU_calls=0\n",cases);
    return 0;
  }catch(const std::exception& e){std::fprintf(stderr,"COMPACT_ADMISSION_FAIL %s\n",e.what());return 1;}
}

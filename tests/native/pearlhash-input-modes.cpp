// Host-only isolation and proof-cache regression; all generator bodies are
// extracted from actual source, not substitutes for GPU arithmetic.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "../../sycl/pearlhash/blake3.inc"
#include "../../sycl/pearlhash/claim.h"
#include "../../sycl/pearlhash/seed.inc"
#include "../../native/job-boundary.h"
namespace mom_pearlhash {
#include "bounds.inc"
#include "gv.inc"
#include "host.inc"
#include "../../sycl/pearlhash/compact-policy.inc"
}
using namespace mom_pearlhash;
// Independent retained legacy stream and the qualified compact stream. Keeping
// these here lets the regression run without SDKs or archived experiments.
static int8_t old_gv(uint32_t seed,uint32_t idx,bool positive) {
  uint32_t x=seed+idx*0x9e3779b9u;
  x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;
  return positive?int8_t(63+(x&1u)):int8_t((x&127u)-64);
}
static int8_t qualified_gv(uint32_t seed,uint32_t idx,bool) {
  return int8_t(idx<32?63+((seed>>idx)&1u):64);
}
static void require(bool value,const char* why) {if(!value)throw std::runtime_error(why);}
// Check the host shortcut against the independent scalar streams, including
// large wrapped indices without allocating the corresponding GPU matrices.
static void check_leaf_modes() {
  for(const auto& shape:std::array<std::array<int,3>,9>{{
      {1,1,1},{1,1,31},{1,1,32},{3,5,67},{160,256,65536},
      {32640,15360,65536},{65536,65536,65536},{32768,65536,65536},{1,65536,65536}}}) {
    const int m=shape[0],n=shape[1],k=shape[2];
    for(char which:{'A','B'}) {
      const size_t total=size_t(which=='A'?m:n)*k,count=(total+1023)/1024;
      for(size_t leaf:std::set<size_t>{0,count/2,count-1,std::min(size_t{32},count-1)})
        for(uint32_t seed:{0u,1u,17u,0x12345678u,UINT32_MAX})
          for(bool positive:{false,true})for(bool prefix:{false,true}) {
            const size_t offset=leaf*1024,len=std::min(size_t{1024},total-offset);
            std::array<uint8_t,1024> actual{},expected{};actual.fill(0xa5);
            for(size_t t=0;t<len;++t) {
              const size_t byte=offset+t;
              const uint32_t index=static_cast<uint32_t>(which=='A'?byte:
                  size_t(m)*k+(byte%k)*size_t(n)+byte/k);
              expected[t]=static_cast<uint8_t>(positive&&prefix?
                  qualified_gv(seed,index,true):old_gv(seed,index,positive));
            }
            require(gen_leaf(which,seed,leaf,m,n,k,actual.data(),positive,prefix)==len,
                    "compact leaf length");
            require(actual==expected,"compact leaf bytes");
          }
    }
  }
}
static bool select(bool compact_mode,bool positive) {
  struct {bool positive_inputs,seed_prefix_inputs;} b{positive,true};
#include "select.inc"
  return b.seed_prefix_inputs;
}
static std::string hex(const uint8_t* data,size_t count) {
  constexpr char digits[]="0123456789abcdef";std::string out;
  for(size_t i=0;i<count;++i){out+=digits[data[i]>>4];out+=digits[data[i]&15];}return out;
}
static std::vector<CV> leaves(char which,uint32_t seed,int m,int n,const uint8_t* key) {
  const size_t count=checked_chunk_count(checked_elements(which=='A'?m:n,65536,"fixture"),"fixture");
  std::vector<CV> result(count);uint8_t bytes[1024];
  for(size_t i=0;i<count;++i) {
    const size_t size=gen_leaf(which,seed,i,m,n,65536,bytes,true,true);
    pearlhash_b3::chunk_cv(bytes,uint32_t(size),i,key,result[i].data());
  }
  return result;
}
int main() {
 try {
  unsigned long long checks=0;
  for(bool compact:{false,true})for(bool positive:{false,true})
    require(select(compact,positive)==(compact&&positive),"stale seed-prefix mode");
  for(uint32_t seed:{0u,1u,17u,0x12345678u,UINT32_MAX}) {
    for(uint32_t i=0;i<131072;++i)for(bool positive:{false,true}) {
      require(gv(seed,i,positive)==old_gv(seed,i,positive),"legacy generator changed");
      require(gv(seed,i,positive,false)==old_gv(seed,i,positive),"disabled prefix changed bytes");
      require(gv(seed,i,false,true)==old_gv(seed,i,false),"prefix escaped positive gate");
      require(gv(seed,i,true,true)==qualified_gv(seed,i,true),"qualified prefix stream changed");
      ++checks;
    }
    for(uint32_t i:{0u,1u,31u,32u,UINT32_MAX-31u,UINT32_MAX}) {
      require(gv(seed,i,true,true)==qualified_gv(seed,i,true),"prefix boundary changed");++checks;
    }
  }
  check_leaf_modes();
  std::array<uint8_t,76> header{};header[0]=3;
  for(size_t i=4;i<header.size();++i)header[i]=uint8_t(i*29+7);
  std::array<uint8_t,32> key{};CompactProfile::derive_key(header.data(),key.data());
  constexpr int m=192,n=256;
  auto a=leaves('A',0,m,n,key.data()),b=leaves('B',0,m,n,key.data());
  CompactRootPath cache;cache.capture(key,m,n,65536,128,0,a,b);
  unsigned proofs=0;
  for(uint32_t seed:{0u,1u,17u,UINT32_MAX}) {
    cache.update(key,m,n,65536,128,seed);
    uint8_t bytes[1024];const size_t size=gen_leaf('A',seed,0,m,n,65536,bytes,true,true);
    pearlhash_b3::chunk_cv(bytes,uint32_t(size),0,key.data(),a[0].data());
    for(const auto& rc:std::array<std::array<int,2>,2>{{{0,0},{14,55}}}) {
      std::vector<int> rows,cols;require(CompactProfile::lists(rc[0],rc[1],m,n,rows,cols),"invalid fixture origin");
      uint32_t expected_factor=0,actual_factor=0,retry_factor=0;
      const auto expected=build_plain_proof_lists(seed,m,n,65536,128,key.data(),rows,cols,
          &expected_factor,&a,&b,true,nullptr,nullptr,nullptr,true);
      const auto actual=cache.proof(key,m,n,65536,128,seed,rc[0],rc[1],&actual_factor);
      require(!actual.empty()&&actual==expected,"prefix cached proof mismatch");
      require(actual==cache.proof(key,m,n,65536,128,seed,rc[0],rc[1],&retry_factor),"retry proof changed");
      require(expected_factor==12582912&&actual_factor==expected_factor&&retry_factor==actual_factor,"factor changed");
      const char* export_proofs=std::getenv("MOM_TEST_EXPORT_PROOFS");
      if(export_proofs&&std::strcmp(export_proofs,"1")==0) {
        std::ofstream out("claim-"+std::to_string(++proofs)+".json");
        out<<"{\"header\":\""<<hex(header.data(),header.size())<<"\",\"plain_proof\":\""<<actual
           <<"\",\"m\":"<<m<<",\"n\":"<<n<<",\"seed\":"<<seed<<",\"factor\":"<<actual_factor<<"}\n";
        require(bool(out),"proof output failed");
      }
    }
  }
  std::printf("PEARL_INPUT_MODES_PASS comparisons=%llu proofs=8 immutable_retries=1 legacy_unchanged=1 GPU_calls=0\n",checks);
  return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
  catch(const std::string& e){std::fprintf(stderr,"FAIL %s\n",e.c_str());return 1;}
}

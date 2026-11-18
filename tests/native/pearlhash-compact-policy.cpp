// Source-derived CPU key/opening-policy regression. The BLAKE3 spy captures its
// exact input only; full digest/proof correctness is verified independently.
#include <array>
#include <vector>
#include <cstdint>
#include <cstring>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include "native/job-boundary.h"
#include "sycl/pearlhash/claim.h"
static std::array<uint8_t,128> keyed_input;
namespace pearlhash_b3 { static void b3(const uint8_t* p,size_t n,const void*,uint8_t*) { if(n!=128) throw std::runtime_error("key length"); std::copy(p,p+n,keyed_input.begin()); } }
#include "config_pattern.inc"

#include "compact_profile.inc"

static void ck(bool x,const char* why){if(!x)throw std::runtime_error(why);}
int main(){try{
 ck(CompactProfile::supports(32640,15360,65536,128),"shape");
 for(const auto d: {std::array<int,4>{32641,15360,65536,128},{32640,15361,65536,128},{32640,15360,65536,256},{32640,15360,8192,128},{0,15360,65536,128},{2147483520,15360,65536,128}})
  ck(!CompactProfile::supports(d[0],d[1],d[2],d[3]),"invalid shape");
 std::array<uint8_t,76> header{};for(int i=0;i<76;++i)header[i]=uint8_t(i);std::array<uint8_t,32> key{};
 CompactProfile::derive_key(header.data(),key.data());
 ck(std::equal(header.begin(),header.end(),keyed_input.begin()),"header");
 std::array<uint8_t,52> gold{};gold[2]=1;gold[4]=128;
 const uint8_t ap[6]={0,1,7,11,0,0},bp[6]={7,1,3,3,0,0};
 std::copy(ap,ap+6,gold.begin()+8);std::copy(bp,bp+6,gold.begin()+14);
 ck(std::equal(gold.begin(),gold.end(),keyed_input.begin()+76),"canonical config");
 std::vector<int> rows,cols;unsigned origins=0;
 for(int row=0;row<768;++row)for(int col=0;col<512;++col){
  bool valid=row%192<16&&row%2==0&&col%256<56&&col%16<8;
  ck(CompactProfile::lists(row,col,768,512,rows,cols)==valid,"origin");
  if(!valid){ck(rows.empty()&&cols.empty(),"invalid stale lists");continue;}
  ++origins;ck(rows.size()==24&&cols.size()==8,"counts");
  for(int i=0;i<24;++i)ck(rows[i]==row+(i/2)*16+i%2,"row opening");
  for(int i=0;i<8;++i)ck(cols[i]==col+(i/2)*64+(i%2)*8,"column opening");
 }
 for(const auto rc:{std::array<int,2>{-1,0},{0,-1},{32640,0},{0,15360},{32639,15359}})
  ck(!CompactProfile::lists(rc[0],rc[1],32640,15360,rows,cols),"bounds");
 uint32_t factor=0;ck(mom_pearlhash::pearlhash_adjustment_factor(24,8,65536,128,&factor)&&factor==12582912,"factor");
 std::printf("P24_SOURCE_POLICY_PASS origins=%u config=canonical factor=%u stale_lists=cleared invalid_shapes=rejected GPU_calls=0\n",origins,factor);
 return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");
const {spawnSync} = require("node:child_process");
const os = require("node:os");
const {runInNewContext} = require("node:vm");
const {extractCppBlock} = require("../common/cpp-source");

/** The production CUDA kernel text embedded in its raw-string literal. */
function source() {
  const file = fs.readFileSync(path.join(__dirname, "../../sycl/pearlhash/cuda-compact-kernel.inc"), "utf8");
  const raw = /R"(\w*)\(([\s\S]*)\)\1"/u.exec(file);
  assert.ok(raw?.[2], "compact kernel raw-string source");
  return raw[2];
}

/** Drop comments and insignificant whitespace so scalar translation does not depend on layout.
 * @param {string} code */
function normalize(code) {
  return code.replace(/\/\/[^\n]*/gu, "").replace(/\s+/gu, " ").replace(/ ?([^\w\s]) ?/gu, "$1");
}

/** Execute scalar expressions only; collective instructions are independently GPU-qualified.
 * @param {string} code @param {string} declaration */
function scalarBody(code, declaration) {
  const block = normalize(extractCppBlock(code, declaration));
  return block.slice(block.indexOf("{") + 1, block.lastIndexOf("}"))
    .replace(/static_assert\([^;]+;/gu, "")
    .replace(/#pragma unroll/gu, "")
    .replace(/\b(?:constexpr|const) (?:uint32_t|uint64_t|int|uint4) /gu, "const ")
    .replace(/\b(?:uint4|uint32_t) (?=[a-z])/gu, "let ")
    .replace(/for\(int /gu, "for(let ")
    .replace(/\b(?:uint64_t|uint32_t|int)\(/gu, "Number(")
    .replace(/\b(0x[0-9a-f]+|\d+)u\b/giu, "$1")
    .replace(/selector_byte<(\d)>\(following\)/gu, "((following >>> (8*$1)) & 255)")
    .replace(/if constexpr\(/gu, "if(");
}

/** Check native operand/cell indexing and last-use refill with tagged scalar values.
 * This is a mapping/lifetime regression, not an emulation of tensor arithmetic.
 * @param {string} code */
function matrixExercise(code) {
  let calls = 0;
  /** @param {Uint32Array} words @param {number} i */
  const ref = (words, i) => {
    assert.ok(Number.isInteger(i) && i >= 0 && i < words.length);
    return {words, i, valueOf: () => words[i]};
  };
  // Distinct operands contribute independently, making operand/order aliasing observable.
  /** @param {number[]} x @returns {[number,number,number,number]} */
  const deltas = x => {
    assert.equal(x.length,6);
    const [a0,a1,a2,a3,b0,b1]=/** @type {[number,number,number,number,number,number]} */(x);
    return [Math.imul(a0,b0)+Math.imul(a1,b1),Math.imul(a2,b0)+Math.imul(a3,b1),
      Math.imul(a0^a3,b0)+Math.imul(a1^a2,b1),Math.imul(a0,b1)+Math.imul(a3,b0)];
  };
  /** @param {...any} args */
  const fma = (...args) => {
    assert.equal(args.length, 14);++calls;
    const values = deltas(args.slice(4, 10).map(Number));
    const before = args.slice(10, 14).map(Number);
    for (let i = 0; i < 4; ++i) {
      assert.equal(args[i].i, args[10 + i].i);
      const old=before[i],delta=values[i];assert.ok(old!==undefined&&delta!==undefined);
      args[i].words[args[i].i] = old + delta;
    }
  };
  /** @param {string} body */
  const bind = body => body.replace(
    "auto a=recast<uint32_t>(fragment_a);auto c=recast<uint32_t>(accumulator);",
    "const a=(i)=>ref(fragment_a,i);const c=(v,m,n)=>ref(accumulator,v+4*(m+4*n));")
    .replaceAll("SM80_16x8x32_S32U8U8S32_TN::fma", "fma");
  const pair = runInNewContext(`((Pair,bv,fragment_a,accumulator) => {
    ${bind(scalarBody(code, "__device__ __forceinline__ void mma_column_pair("))}})`, {ref, fma});
  const finishText = bind(scalarBody(code,
    "__device__ __forceinline__ void finish_pair_and_load_next_a("))
    .replace("cute::cast_smem_ptr_to_uint(&storage.ebr_rank[owner*8192])", "owner*8192")
    .replace("RawBPair pending{};", "let pending={};")
    .replace(/a\((group\*4(?:\+\d+)?)\)=p\.([xyzw]);/gu, "fragment_a[$1]=p.$2;");
  const finish = runInNewContext(`((bv,fragment_a,accumulator,owner,following,future) => {
    const storage={};${finishText}})`, {ref, fma,kPitchB:256,
    load_private_b_rows: (/** @type {number} */ first, /** @type {number} */ second) => {
      // This tagged-data fixture uses owner0/following123456 (rank bytes1/0).
      // The separate address regression below covers all other rank/owner cases.
      const group=second/2048;
      assert.ok(Number.isInteger(group)&&group>=0&&group<4);
      assert.equal(first,16+2048*group);return {group,following:123456};
    },
    finish_private_b: (/** @type {{group:number,following:number}} */ p) => Object.fromEntries(["x", "y", "z", "w"].map((s,i) =>
      [s, (p.following + 19 * (4 * p.group + i)) >>> 0]))});
  for (const seed of [0, 1, 0x5a17c0de, 0xffffffff]) {
    for (const future of [false, true]) {
      const expected = Uint32Array.from({length: 192}, (_, i) => (seed + 97 * i) >>> 0);
      const actual = expected.slice();
      const old = Uint32Array.from({length: 16}, (_, i) => (seed ^ (197 * i + 53)) >>> 0);
      const input = old.slice();calls = 0;
      for (let p = 0; p < 6; ++p) {
        const v = (seed + Math.imul(0x9e3779b9, p + 1)) >>> 0;
        const bv = {x:v, y:(v ^ 0x7feb352d) >>> 0, z:(v + 0x846ca68b) >>> 0, w:Math.imul(v,17) >>> 0};
        if (p < 5) {pair(p,bv,input,actual);} else {finish(bv,input,actual,0,123456,future);}
        for (let m = 0; m < 4; ++m) {
          for (let half = 0; half < 2; ++half) {
            const d = deltas([...old.subarray(4*m,4*m+4),half?bv.z:bv.x,half?bv.w:bv.y]);
            for (let k = 0; k < 4; ++k) {
              const index=k+4*(m+4*(2*p+half)),old=expected[index],delta=d[k];
              assert.ok(old!==undefined&&delta!==undefined);expected[index]=old+delta;
            }
          }
        }
      }
      assert.equal(calls,48);assert.deepEqual(actual,expected);
      assert.deepEqual(input,future?Uint32Array.from({length:16},(_,i)=>123456+19*i):old);
    }
  }
}

test("PearlHash compact transposed MMA and last-use refill preserve every cell", () => {
  matrixExercise(source());
});

/** Execute the actual common bases and refill offsets; compare to the independent
 * rank-major-to-slab mapping, not a second copy of the optimized expression.
 * @param {string} code */
function addressExercise(code) {
  const helper=normalize(extractCppBlock(code,"__device__ __forceinline__ void finish_pair_and_load_next_a("));
  const owner=helper.match(/&storage\.ebr_rank\[([^\]]+)\]/u)?.[1];
  const first=helper.match(/first_row=(base[^;]+);/u)?.[1];
  const second=helper.match(/second_row=(base[^;]+);/u)?.[1];
  const refill=helper.match(/if\(group<3\)pending=load_private_b_rows\(([^;]+)\);/u)?.[1];
  assert.ok(owner&&first&&second&&refill);
  /** @param {string} expression */
  const js=expression=>expression.replace(/selector_byte<(\d)>\(following\)/gu,"((following >>> (8*$1)) & 255)")
    .replace(/\b(\d+)u\b/gu,"$1");
  const bases=runInNewContext(`((storageBase,owner,following)=>{
    const kPitchB=256,base=storageBase+(${owner});return [${js(first)},${js(second)}];})`);
  const next=runInNewContext(`((first_row,second_row,group)=>[${js(refill)}])`);
  let cases=0;
  for(const storageBase of [0,1024,4096]) {
    for(let owner=0;owner<4;++owner) {
      for(let a=0;a<128;++a) {
        for(let b=0;b<128;++b) {
          const packed=((b<<24)|(a<<16)|(23<<8)|19)>>>0,[x,y]=bases(storageBase,owner,packed);
          for(let group=0;group<4;++group) {
            const actual=group?next(x,y,group-1):[x,y];
            const row=owner*64+group*16;
            const slab=/** @param {number} rank */ rank=>(Math.floor(row/16)*128+rank)*16+row%16;
            assert.deepEqual(Array.from(actual),[storageBase+slab(a),storageBase+slab(b)]);++cases;
          }
        }
      }
    }
  }
  assert.equal(cases,786432);
}

test("PearlHash compact private-B row bases preserve every rank and refill address",()=>{
  addressExercise(source());
});

test("PearlHash compact B upload is a bijective aligned slab layout",()=>{
  const upload=normalize(extractCppBlock(source(),"__device__ __forceinline__ void upload_noise_words("));
  const expression=upload.match(/if constexpr\(Rows==256\)output\[([^\]]+)\]/u)?.[1];
  assert.ok(expression);
  const offset=runInNewContext(`(row,rank,byte)=>${expression.replace("row/16","Math.floor(row/16)")}`);
  const seen=new Set();
  for(let row=0;row<256;row++){
    for(let rank=0;rank<128;rank++){
      const actual=offset(row,rank,0);
      assert.equal(actual,(Math.floor(row/16)*128+rank)*16+row%16);
      assert.ok(actual>=0&&actual<32768&&!seen.has(actual));seen.add(actual);
      if(row%16===0){assert.equal(actual%16,0);}
    }
  }
  assert.equal(seen.size,32768);
});

/** Execute the actual initial/rolling producer ownership and word permutations.
 * @param {string} code */
function stageExercise(code) {
  const initial=scalarBody(code,"__device__ __forceinline__ void produce_fragment_stage(");
  const publish=scalarBody(code,"__device__ __forceinline__ void publish_pending_a(");
  /** @param {number} x @param {number} y @param {number} z @param {number} w */
  const make=(x,y,z,w)=>({x:x>>>0,y:y>>>0,z:z>>>0,w:w>>>0});
  /** @param {number} first @param {number} second */
  const raw=(first,second)=>make(first+1,first+3,second+5,second+7);
  const fill=runInNewContext(`((Initial,storage,slot,warp,lane,packed,global_row_base,seed) => {${initial}})`,
    {kPitchA:208,make_uint4:make,rank_pair_packet:raw,apply_a_prefix:(/** @type {{x:number}} */ p)=>{p.x^=0x12345;}});
  const next=runInNewContext(`((Extra,storage,slot,owner,group,lane,p) => {
    ${publish}})`,{make_uint4:make,kByteBias:0x40404040});
  /** The extracted shader writes these cells through the VM's modeled pointer view.
   * @returns {{eal_rank:number,a_stage:any}} */
  const empty=()=>({eal_rank:0,a_stage:Array.from({length:2},()=>
    Array.from({length:2},()=>Array.from({length:6},()=>Array(32).fill(null))))});
  for(const prefix of [false,true]) {
    for(const first of [0,7,31,63,127]) {
      const second=127-first,storage=empty(),published=empty(),packed=first|(second<<8);
      for(let slot=0;slot<2;++slot) {
        for(let warp=0;warp<8;++warp) {
          for(let lane=0;lane<32;++lane) {
            fill(prefix,storage,slot,warp,lane,packed,0,0);
            const owner=warp>>2,group=warp&3;
            for(const extra of group<2?[false,true]:[false]) {
              const p=group+(extra?4:0),row=owner*96+p*16;
              const v=raw(first*208+row,second*208+row);
              const pending={first:make(v.x,v.y,v.z,v.w),second:make(0,0,0,0)};
              next(extra,published,slot,owner,group,lane,pending);
              assert.deepEqual(published.a_stage[slot][owner][p][lane],
                make(v.x+0x40404040,v.z+0x40404040,v.y+0x40404040,v.w+0x40404040));
            }
          }
        }
      }
      for(let slot=0;slot<2;++slot) {
        for(let owner=0;owner<2;++owner) {
          for(let p=0;p<6;++p) {
            const row=96*owner+16*p,v=raw(first*208+row,second*208+row);
            if(prefix&&owner===0&&p===0) {v.x^=0x12345;}
            for(let lane=0;lane<32;++lane) {
              assert.deepEqual(storage.a_stage[slot][owner][p][lane],make(v.x,v.z,v.y,v.w));
            }
          }
        }
      }
    }
  }
}

test("PearlHash compact initial and rolling producers retain all transposed A packets", () => {
  stageExercise(source());
});

/** Tie the cold physical permutations and emitted origins to public 24x8 openings.
 * @param {string} code */
function publicMapExercise(code) {
  const upload=normalize(extractCppBlock(code,"__device__ __forceinline__ void upload_noise_words("));
  const physical=upload.match(/const int physical_row=([^;]+);/u)?.[1];assert.ok(physical);
  // Preserve the four C++ integer quotients; all other expressions are exact integer JS arithmetic.
  const integerPhysical=physical.replace("(row%96)/8","Math.floor((row%96)/8)")
    .replace("row/96","Math.floor(row/96)").replace("(row%64)/16","Math.floor((row%64)/16)")
    .replace("row/64","Math.floor(row/64)");
  const map=runInNewContext(`((Rows,row)=>(${integerPhysical}))`);
  for(const rows of [192,256]) {
    const seen=new Set();for(let row=0;row<rows;++row) {
      const p=Number(map(rows,row));assert.ok(Number.isInteger(p)&&p>=0&&p<rows);seen.add(p);
    }
    assert.equal(seen.size,rows);
  }
  const flat=normalize(code);
  const rowText=flat.match(/const uint32_t row=([^;]+);/u)?.[1];
  const colText=flat.match(/const uint32_t col=([^;]+);/u)?.[1];assert.ok(rowText&&colText);
  /** @param {string} expression */
  const js=expression=>expression.replace(/\b(?:uint64_t|uint32_t)\(/gu,"Number(").replace(/\b(\d+)u\b/gu,"$1");
  const origin=runInNewContext(`((block_m,block_n,warp_m,warp_n,lane)=>{
    const row_offset=0;return [${js(rowText)},${js(colText)}];})`);
  /** @type {[number,number][]} */
  const blocks=[[0,0],[1,1],[169,59]];
  for(const [blockM,blockN] of blocks) {
    const cells=new Set();
    for(let wa=0;wa<2;++wa) {
      for(let wb=0;wb<4;++wb) {
        for(let lane=0;lane<32;++lane) {
          const [row,col]=origin(blockM,blockN,wa,wb,lane),actual=new Set(),expected=new Set();
          for(let r=0;r<24;++r) {
            for(let c=0;c<8;++c) {
              expected.add(`${row+16*Math.floor(r/2)+r%2}:${col+64*Math.floor(c/2)+8*(c%2)}`);
            }
          }
          // Public SM80 16x8 atom coordinates; CuTe agreement was independently established.
          // Native M is logical B and native N is logical A, not vice versa.
          for(let m=0;m<4;++m) {
            for(let n=0;n<12;++n) {
              for(let v=0;v<4;++v) {
                const nativeRow=16*m+(lane>>2)+8*(v>>1),nativeCol=8*n+2*(lane&3)+(v&1);
                const a=blockM*192+map(192,wa*96+nativeCol),b=blockN*256+map(256,wb*64+nativeRow);
                assert.ok(a>=0&&a<32640&&b>=0&&b<15360);
                const key=`${a}:${b}`;assert.ok(!cells.has(key));cells.add(key);actual.add(key);
              }
            }
          }
          assert.deepEqual(actual,expected);
        }
      }
    }
    assert.equal(cells.size,49152);
  }
}

test("PearlHash compact cold permutations and result origins open the same public cells",()=>{
  publicMapExercise(source());
});

test("PearlHash compact policy binds canonical key, openings and target geometry", {
  skip:process.platform==="win32"?"requires a host C++ compiler":false,
}, () => {
  const root=path.join(__dirname,"../../sycl/pearlhash");
  const policy=extractCppBlock(fs.readFileSync(path.join(root,"compact-policy.inc"),"utf8"),"struct CompactProfile {")+";\n";
  const config=extractCppBlock(fs.readFileSync(path.join(root,"host.inc"),"utf8"),"static void config_bytes_pattern(");
  const fixture=fs.mkdtempSync(path.join(os.tmpdir(),"mom-pearl-compact-policy-"));
  try {
    fs.writeFileSync(path.join(fixture,"compact_profile.inc"),policy);
    fs.writeFileSync(path.join(fixture,"config_pattern.inc"),config);
    const exe=path.join(fixture,"policy");
    const build=spawnSync("c++",["-std=c++17","-O2","-Wall","-Wextra","-Werror",
      "-I"+fixture,"-I"+path.join(__dirname,"../.."),
      path.join(__dirname,"../native/pearlhash-compact-policy.cpp"),"-o",exe],{encoding:"utf8",timeout:30000});
    assert.equal(build.status,0,build.stderr);
    const result=spawnSync(exe,[],{encoding:"utf8",timeout:5000});
    assert.equal(result.error,undefined);assert.equal(result.signal,null);
    assert.equal(result.status,0,result.stdout+result.stderr);
    assert.match(result.stdout,/P24_SOURCE_POLICY_PASS/u);
  } finally {fs.rmSync(fixture,{recursive:true,force:true});}
});

test("compact NVIDIA profiles launch whole waves of the executed kernel CTA", {
  skip:process.platform==="win32"?"requires a host C++ compiler":false,
}, () => {
  const kernel=normalize(source());
  /** @param {string} name */
  const constant=name=>{
    const values=new Set([...kernel.matchAll(new RegExp(`constexpr int ${name}=(\\d+);`,"gu"))].map(m=>m[1]));
    assert.equal(values.size,1,name);return [...values][0];
  };
  const fixture=fs.mkdtempSync(path.join(os.tmpdir(),"mom-pearl-compact-cta-"));
  try {
    const exe=path.join(fixture,"cta");
    const build=spawnSync("c++",["-std=c++17","-O2","-Wall","-Wextra","-Werror",
      `-DCOMPACT_CTA_M=${constant("kCtaM")}`,`-DCOMPACT_CTA_N=${constant("kCtaN")}`,
      path.join(__dirname,"../native/pearlhash-compact-cta.cpp"),"-o",exe],{encoding:"utf8",timeout:30000});
    assert.equal(build.status,0,build.stderr);
    const result=spawnSync(exe,[],{encoding:"utf8",timeout:5000});
    assert.equal(result.status,0,result.stdout+result.stderr);
    // Only unit counts whose wave quantum exceeds the 32-bit index bound may fail closed.
    assert.ok(Number(/checked=(\d+)/u.exec(result.stdout)?.[1])>=200,result.stdout);
  } finally {fs.rmSync(fixture,{recursive:true,force:true});}
});

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");
const {spawnSync} = require("node:child_process");
const os = require("node:os");
const {runInNewContext} = require("node:vm");
const {extractCppBlock} = require("../common/cpp-source");

const source = () => fs.readFileSync(path.join(__dirname,
  "../../sycl/pearlhash/cuda-compact-kernel.inc"), "utf8");

/** Execute scalar expressions only; collective instructions are independently GPU-qualified.
 * @param {string} code @param {string} declaration */
function scalarBody(code, declaration) {
  const block = extractCppBlock(code, declaration);
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
  const code=source();matrixExercise(code);
  /** @type {[string,string][]} */
  const mutations=[["c(0,m,n)","c(0,0,n)"], ["a(4*m+2)","a(4*m+1)"],
    ["half?bv.z:bv.x","bv.x"], ["a(group*4+1)=p.y;","a(group*4)=p.y;"]];
  for(const [from,to] of mutations) {
    assert.ok(code.includes(from));assert.throws(()=>matrixExercise(code.replaceAll(from,to)),{code:"ERR_ASSERTION"});
  }
  const helper=extractCppBlock(code,"__device__ __forceinline__ void finish_pair_and_load_next_a(");
  const refill=extractCppBlock(helper,"    if(future) {");
  const moved=helper.replace(refill,"").replace("    for(int half=0;half<2;++half)",refill+"    for(int half=0;half<2;++half)");
  assert.notEqual(helper,moved);assert.throws(()=>matrixExercise(code.replace(helper,moved)),
    {code:"ERR_ASSERTION"},"refilling native A before its last two consumers must fail");
});

/** Execute the actual common bases and refill offsets; compare to the independent
 * rank-major-to-slab mapping, not a second copy of the optimized expression.
 * @param {string} code */
function addressExercise(code) {
  const helper=extractCppBlock(code,"__device__ __forceinline__ void finish_pair_and_load_next_a(");
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
  const raw=extractCppBlock(code,"__device__ __forceinline__ RawBPair load_private_b_rows(");
  assert.ok(raw.includes(':"=r"(p.first.x),"=r"(p.first.y),"=r"(p.first.z),"=r"(p.first.w):"r"(first)'));
  assert.ok(raw.includes(':"=r"(p.second.x),"=r"(p.second.y),"=r"(p.second.z),"=r"(p.second.w):"r"(second)'));
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
  const code=source();addressExercise(code);
  /** @type {[string,string][]} */
  const mutations=[["selector_byte<2>(following)","selector_byte<0>(following)"],
    ["selector_byte<3>(following)","selector_byte<1>(following)"],
    ["first_row+2048u*(group+1)","first_row+16u*(group+1)"],
    ["&storage.ebr_rank[owner*8192]","&storage.ebr_rank[owner*64]"]];
  for(const [from,to] of mutations) {
    assert.ok(code.includes(from));assert.throws(()=>addressExercise(code.replaceAll(from,to)),{code:"ERR_ASSERTION"});
  }
});

test("PearlHash compact B upload is a bijective aligned slab layout",()=>{
  const code=source(),upload=extractCppBlock(code,"__device__ __forceinline__ void upload_noise_words(");
  const expression=upload.match(/if constexpr\(Rows==256\)\s*output\[([^\]]+)\]/u)?.[1];
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
  const code=source();stageExercise(code);
  /** @type {[string,string][]} */
  const mutations=[["if(group<2) {","if(false) {"],
    ["const int row=owner*96+group*16;","const int row=owner*64+group*16;"],
    ["make_uint4(packet.x,packet.z,packet.y,packet.w)","make_uint4(packet.x,packet.y,packet.z,packet.w)"],
    ["group+(Extra?4:0)","group+(Extra?2:0)"],
    ["if(global_row_base==0&&warp==0)","if(global_row_base==0&&warp==4)"]];
  for(const [from,to] of mutations) {
    assert.ok(code.includes(from));assert.throws(()=>stageExercise(code.replaceAll(from,to)),{code:"ERR_ASSERTION"});
  }
});

/** Tie the cold physical permutations and emitted origins to public 24x8 openings.
 * @param {string} code */
function publicMapExercise(code) {
  const upload=extractCppBlock(code,"__device__ __forceinline__ void upload_noise_words(");
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
  const rowText=code.match(/const uint32_t row = ([^;]+);/u)?.[1];
  const colText=code.match(/const uint32_t col = ([^;]+);/u)?.[1];assert.ok(rowText&&colText);
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
  const code=source();publicMapExercise(code);
  /** @type {[string,string][]} */
  const mutations=[["8*(row/96)","16*(row/96)"],["+row%8:","+row%4:"],
    ["uint64_t(2*(lane&3))","uint64_t(lane&3)"],
    ["uint64_t(warp_n)*16u","uint64_t(warp_n)*8u"]];
  for(const [from,to] of mutations) {
    assert.ok(code.includes(from));assert.throws(()=>publicMapExercise(code.replaceAll(from,to)),{code:"ERR_ASSERTION"});
  }
});

test("PearlHash compact policy binds canonical key, openings and target geometry", {
  skip:process.platform==="win32"?"requires a host C++ compiler":false,
}, async t => {
  const root=path.join(__dirname,"../../sycl/pearlhash");
  const policy=fs.readFileSync(path.join(root,"compact-policy.inc"),"utf8").split("// Only A's")[0];
  assert.ok(policy);
  const host=fs.readFileSync(path.join(root,"host.inc"),"utf8");
  const config=extractCppBlock(host,"static void config_bytes_pattern(");
  const fixture=fs.mkdtempSync(path.join(os.tmpdir(),"mom-pearl-compact-policy-"));
  /** @param {string} name @param {string} body @param {RegExp} [failure] */
  const run=(name,body,failure)=>{
    fs.writeFileSync(path.join(fixture,"compact_profile.inc"),body);
    fs.writeFileSync(path.join(fixture,"config_pattern.inc"),config);
    const exe=path.join(fixture,name);
    const build=spawnSync("c++",["-std=c++17","-O2","-Wall","-Wextra","-Werror",
      "-I"+fixture,"-I"+path.join(__dirname,"../.."),
      path.join(__dirname,"../native/pearlhash-compact-policy.cpp"),"-o",exe],{encoding:"utf8",timeout:30000});
    assert.equal(build.status,0,build.stderr);
    const result=spawnSync(exe,[],{encoding:"utf8",timeout:5000});
    assert.equal(result.error,undefined);assert.equal(result.signal,null);
    assert.equal(result.status,failure?1:0,result.stdout+result.stderr);
    if(failure){assert.match(result.stderr,failure);}else{assert.match(result.stdout,/P24_SOURCE_POLICY_PASS/u);}
  };
  try {
    await t.test("actual policy config, origin bounds, lists and factor",()=>run("healthy",policy));
    /** @type {[string,string,string,RegExp][]} */
    const mutations=[
      ["key-pattern","row_pattern = {0, 1, 7, 11","row_pattern = {0, 1, 7, 9",/canonical config/u],
      ["odd-origin","&& row % 2 == 0","",/origin/u],
      ["column-origin","col % 16 < 8","col % 16 < 16",/origin/u],
      ["row-stride","row + 16 * (i / 2)","row + 8 * (i / 2)",/row opening/u],
      ["column-stride","col + 64 * (i / 2)","col + 32 * (i / 2)",/column opening/u],
    ];
    for(const [name,from,to,failure] of mutations) {
      assert.ok(policy.includes(from));await t.test(name+" negative really fails",()=>run(name,policy.replace(from,to),failure));
    }
    const lateClear=policy.replace("a.clear(); b.clear();","")
      .replace("for (int i = 0; i < rows; ++i)","a.clear(); b.clear();\n    for (int i = 0; i < rows; ++i)");
    assert.notEqual(lateClear,policy);
    await t.test("invalid calls cannot retain old lists",()=>run("stale-lists",lateClear,/invalid stale lists/u));
  } finally {fs.rmSync(fixture,{recursive:true,force:true});}
});

test("compact shape heuristic follows the executed kernel CTA", () => {
  const code = source();
  const rows = [...code.matchAll(/constexpr int kCtaM = (\d+);/gu)].map(m => Number(m[1]));
  const cols = [...code.matchAll(/constexpr int kCtaN = (\d+);/gu)].map(m => Number(m[1]));
  assert.equal(rows.length, 2);
  assert.equal(cols.length, 2);
  assert.equal(rows[0], rows[1]);
  assert.equal(cols[0], cols[1]);
  const policy = fs.readFileSync(path.join(__dirname, "../../sycl/pow-intensity.h"), "utf8");
  const quantum = policy.match(/const uint64_t quantum = uint64_t\((\d+)\) \* device\.compute_units/gu);
  assert.equal(quantum?.length, 1);
  assert.ok(quantum?.[0].includes("uint64_t(" + rows[0] + ")"));
  assert.ok(policy.includes("unsigned(n / " + cols[0] + ")"));
});

test("PearlHash compact unrolls only the fixed full-proof checkpoint loop", () => {
  /** @param {string} code */
  const check = code => {
    const expression = code.match(/#pragma unroll ([^\n]+)\n\s*for \(int checkpoint = 0; checkpoint < checkpoint_count; \+\+checkpoint\)/u)?.[1];
    assert.ok(expression, "checkpoint loop and its bounded unroll policy");
    // CPU policy check, not a CUDA control-flow emulator. GPU vectors also
    // cover generic K tails; widening their loop caused real register spills.
    const factor = runInNewContext(`(FixedK) => (${expression})`);
    assert.equal(factor(65536), 4);
    for (const k of [0, 128, 384, 8192]) {
      assert.equal(factor(k), 1);
    }
    assert.match(code, /if \(k == 65536\)\s*compact_gemm_impl<65536>/u);
    assert.match(code, /else\s*compact_gemm_impl<0>/u);
  };
  const code = source();
  check(code);
  const policy = "FixedK == 65536 ? 4 : 1";
  assert.ok(code.includes(policy));
  for (const wrong of ["4", "1", "FixedK == 65536 ? 1 : 4"]) {
    assert.throws(() => check(code.replace(policy, wrong)), {code: "ERR_ASSERTION"});
  }
});

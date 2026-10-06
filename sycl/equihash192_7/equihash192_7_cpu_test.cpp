// Host-only regression for the authoritative Zclassic block-3209920 proof.

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "../zhash/equihash_core.hpp"

namespace {

template <std::size_t N, std::size_t Characters>
void from_hex(const char (&text)[Characters], std::uint8_t (&out)[N]) {
  static_assert(Characters == 2 * N + 1, "fixture hex length mismatch");
  for (std::size_t i = 0; i < N; ++i) {
    const auto digit = [](const char c) -> unsigned {
      return c >= '0' && c <= '9'   ? static_cast<unsigned>(c - '0')
             : c >= 'a' && c <= 'f' ? static_cast<unsigned>(c - 'a' + 10)
             : c >= 'A' && c <= 'F' ? static_cast<unsigned>(c - 'A' + 10)
                                    : 0xffu;
    };
    const unsigned high = digit(text[2 * i]), low = digit(text[2 * i + 1]);
    if (high > 15 || low > 15)
      std::abort();
    out[i] = static_cast<std::uint8_t>((high << 4) | low);
  }
}

template <unsigned FieldCount, unsigned LocalBits, unsigned ParentBucketBits,
          unsigned ParentSlots>
bool test_bucket_collision_round_trip() {
  using Spec = mom_equihash::Equihash192_7;
  using Record = mom_equihash::PackedBucketCollisionRecord<
      Spec, FieldCount, LocalBits, ParentBucketBits, ParentSlots>;
  for (unsigned high = 0; high < 2; ++high) {
    const std::uint32_t output_bucket = high ? (1u << (24 - LocalBits)) - 1u : 0u;
    const std::uint32_t parent_bucket = high ? (1u << ParentBucketBits) - 1u : 0u;
    const std::uint32_t left_slot = high ? ParentSlots - 2u : 0u;
    const std::uint32_t right_slot = high ? ParentSlots - 1u : 0u;
    std::uint32_t fields[FieldCount] = {};
    fields[0] = (output_bucket << LocalBits) | (high ? (1u << LocalBits) - 1u : 0u);
    for (unsigned i = 1; i < FieldCount; ++i)
      fields[i] = ((0x13579bu * i) ^ (high ? 0xffffffu : 0u)) & 0xffffffu;
    Record record{};
    mom_equihash::store_bucket_collision(record, fields, parent_bucket, left_slot, right_slot);
    for (unsigned i = 0; i < FieldCount; ++i)
      if (mom_equihash::load_bucket_collision_field(record, output_bucket, i) != fields[i])
        return false;
    if (mom_equihash::load_bucket_collision_parent(record, false) !=
            parent_bucket * ParentSlots + left_slot ||
        mom_equihash::load_bucket_collision_parent(record, true) !=
            parent_bucket * ParentSlots + right_slot)
      return false;
  }
  return true;
}

} // namespace

int main() {
  using Spec = mom_equihash::Equihash192_7;
  static_assert(Spec::row_count == (1u << 25) && Spec::hash_count == (1u << 24));
  // The capacities have padded tails; exercise the last two valid slots so their rounded
  // parent-slot fields are checked rather than only power-of-two-looking offsets.
  if (!test_bucket_collision_round_trip<5, 12, 12, 9216>() ||
      !test_bucket_collision_round_trip<5, 12, 12, 9232>() ||
      !test_bucket_collision_round_trip<5, 11, 13, 4608>() ||
      !test_bucket_collision_round_trip<5, 11, 13, 4616>() ||
      !test_bucket_collision_round_trip<5, 12, 14, 2304>() ||
      !test_bucket_collision_round_trip<5, 12, 14, 2308>() ||
      !test_bucket_collision_round_trip<7, 10, 14, 2308>() ||
      !test_bucket_collision_round_trip<6, 10, 14, 2308>() ||
      !test_bucket_collision_round_trip<2, 10, 12, 9232>())
    return 7;
  std::uint8_t header[Spec::header_length], solution[Spec::solution_length];
  from_hex("04000000ecf888bb9e8440dff1eca5ff69c277e85462f306ec785719a76e4dd20f0b0000c450f3fd2a66b46"
           "2f4133c48cc636655ac055de95072bd693514c9c7156dfa8de2004d086a6929b60cb4e4efbbfcf41d3fda50"
           "ad985fc421c990217a1daef400c58d776ad03c141e8001fde00f6dcbadb169c9131b3a07c44e9b11ca00000"
           "000000000005b15db75",
           header);
  from_hex(
      "000870397e46deb5da1b012d60132a670f44b56bcc3d62efab0f5fe274a4d7c74b5ac57ba1f80d89873459c67a1d"
      "bd5c11b10af944f4a507b01d143e4c10fa3f99975a00fb4f2b9dc5c9db0c9ce7fd89d8aeab27b9ab4b65974673cf"
      "d566c56e63c90e4911756fab9a579ed290934b91545505ecb0796eaed74172eac24232aec77064e87335363a5e67"
      "a282a69cfd3e15abe16cabcb28bbb9b268e90a9785c80be303e2713b47b43188f7e3361c8045200feefe0368b5a1"
      "fdd5527a45c3365f2ccebaf29576a7e801eae51de4694c3a678da0f0b04ce00a654cfc20fe0eb132ae0d295a07af"
      "b7c51514124bdd8180dd614363f5bca625f9dd2d02cdf0d40a0066ad121de47c736e894bf5de9b8bd29ff09a7228"
      "4877514d695635121f1a4b63c2859da3279a6ad375fba1e7022f9428b76950fea7140ab892e5c64adb33f90ed025"
      "ec12a417c13551ab551408d72d1111a4b14ea4efa729bf64db883f38131bdb8aa6324f1fd4bfe2f99781bc554e5d"
      "6635d4e15b4335301b0cc3272d1f18c13226aef428fe049fe36241d98fafc610",
      solution);
  std::uint8_t digest[Spec::hash_length], expected[Spec::hash_length];
  from_hex("d22c1ef2a4fcd68dbc24ef7b7ac6df11c264375afae2010eba986ec05717f17ef9617c716cd961115eff3f2"
           "1fd401b73",
           expected);
  mom_equihash::hash_index<Spec>(header, 0, digest);
  for (unsigned i = 0; i < Spec::hash_length; ++i)
    if (digest[i] != expected[i])
      return 3;
  if (!mom_equihash::verify_solution<Spec>(header, solution))
    return 1;
  std::uint32_t indices[Spec::proof_indices];
  mom_equihash::decode_indices<Spec>(solution, indices);
  const std::uint32_t first = indices[0];
  indices[0] = indices[1];
  indices[1] = first;
  mom_equihash::encode_indices<Spec>(indices, solution);
  if (mom_equihash::verify_solution<Spec>(header, solution))
    return 4;
  const std::uint32_t second = indices[0];
  indices[0] = indices[1];
  indices[1] = second;
  mom_equihash::encode_indices<Spec>(indices, solution);
  if (!mom_equihash::verify_solution<Spec>(header, solution))
    return 5;
  solution[0] ^= 0x80;
  if (mom_equihash::verify_solution<Spec>(header, solution))
    return 6;
  return 0;
}

pragma circom 2.0.0;
include "./clampTensor.circom";


template IsZero() {
  signal input in;
  signal output out;

  signal inv <-- in != 0 ? 1 / in : 0;
  out <== 1 - (in * inv);

  in * out === 0;
}

template IntDiv(n) {
  signal input in[2];
  signal output out;

  signal is_non_zero <== IsZero()(in[1]);
  0 === is_non_zero;

  var quot_hint = in[0] \ in[1];
  var rem_hint = in[0] % in[1];
  signal quot <-- quot_hint;
  signal rem <-- rem_hint;
  in[0] === quot * in[1] + rem;

  signal rem_is_valid <== LessThan(n)([rem, in[1]]);
  1 === rem_is_valid;

  out <-- quot;
}

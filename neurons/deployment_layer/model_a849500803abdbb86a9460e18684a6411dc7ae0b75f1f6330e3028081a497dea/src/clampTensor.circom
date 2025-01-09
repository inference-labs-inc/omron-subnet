pragma circom 2.0.0;

template Num2Bits(n) {
  assert(n < 254);
  signal input in;
  signal output out[n];

  var lc = 0;
  var bit_value = 1;

  for (var i = 0; i < n; i++) {
    out[i] <-- (in >> i) & 1;
    AssertBit()(out[i]);

    lc += out[i] * bit_value;
    bit_value <<= 1;
  }

  lc === in;
}
template AssertBit() {
  signal input in;

  in * (in - 1) === 0;
}

template LessThan(n) {
  assert(n <= 252);
  signal input in[2];
  signal output out;

  component toBits = Num2Bits(n+1);
  toBits.in <== ((1 << n) + in[0]) - in[1];

  out <== 1 - toBits.out[n];
}

template Clamp(b){
    signal input val;
    signal input min;
    signal input max;
    signal output out;
    signal temp_max;
    signal temp_min;
    signal temp_1[2];
    signal temp_2[2];

    component LessThan[2];

    LessThan[0] = LessThan(b);
    LessThan[1] = LessThan(b);

    LessThan[0].in[0] <== val;
    LessThan[0].in[1] <== min;
    temp_1[0] <== val * (1 - LessThan[0].out);
    temp_2[0] <== min * LessThan[0].out;
    temp_max <==  temp_1[0] + temp_2[0];

    LessThan[1].in[0] <== max;
    LessThan[1].in[1] <== temp_max;

    temp_1[1] <== temp_max * (1 - LessThan[1].out);
    temp_2[1] <== max * LessThan[1].out;
    temp_min <==  temp_1[1] + temp_2[1];

    out <== temp_min*1;
}

template clampTensor (n, b) {
    var i;

    signal input val[n];
    signal input min[n];
    signal input max[n];
    signal output out[n];
    component comp[n];

    for (i=0; i<n; i++) {
        comp[i] = Clamp(b);
        comp[i].val <== val[i];
        comp[i].min <== min[i];
        comp[i].max <== max[i];
        out[i] <== comp[i].out;
    }
}

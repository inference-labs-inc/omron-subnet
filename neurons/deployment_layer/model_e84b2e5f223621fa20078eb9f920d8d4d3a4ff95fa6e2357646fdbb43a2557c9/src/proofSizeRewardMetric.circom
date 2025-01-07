pragma circom 2.0.0;
include "./subtractTensor.circom";
include "./clampTensor.circom";
include "./integerDivision.circom";

template ProofSizeRewardMetric(b){
    signal input PROOF_SIZE_WEIGHT;
    signal input proof_size;
    signal input PROOF_SIZE_THRESHOLD;
    signal input scaling;
    signal output proof_size_reward_metric;
    signal division;
    signal remainder;
    signal pos;
    signal temp_scale;
    signal temp_mul;
    signal temp_max1;
    signal temp_max2;
    signal temp_clamp;
    component clamp;
    component int_div;
    component int_div_2;
    component less_than;

    temp_scale <== proof_size * scaling;

    int_div = IntDiv(b);
    int_div.in[0] <== temp_scale;
    int_div.in[1] <== PROOF_SIZE_THRESHOLD;
    division <== int_div.out;

    clamp = Clamp(b);
    clamp.val <== division;
    clamp.min <== 0;
    clamp.max <== scaling;

    temp_mul <== PROOF_SIZE_WEIGHT * clamp.out;

    int_div_2 = IntDiv(b);
    int_div_2.in[0] <== temp_mul;
    int_div_2.in[1] <== scaling;
    proof_size_reward_metric <== int_div_2.out;

}

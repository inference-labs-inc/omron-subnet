pragma circom 2.0.0;
include "./subtractTensor.circom";
include "./clampTensor.circom";
include "./integerDivision.circom";

template CalculatedScoreFraction(b){
    signal input response_time_reward_metric;
    signal input proof_size_reward_metric;
    signal input accuracy_weighted;
    signal input scaling;
    signal output out;

    signal total;
    total <== response_time_reward_metric + proof_size_reward_metric + accuracy_weighted;

    component divider = IntDiv(b);
    divider.in[0] <== total;
    divider.in[1] <== 3;

    component clamp = Clamp(b);
    clamp.val <== divider.out;
    clamp.min <== 0;
    clamp.max <== scaling;

    out <== clamp.out;
}

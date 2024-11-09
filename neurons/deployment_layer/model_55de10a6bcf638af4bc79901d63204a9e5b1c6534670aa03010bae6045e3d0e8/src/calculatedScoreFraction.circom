pragma circom 2.0.0;
include "./subtractTensor.circom";
include "./clampTensor.circom";

template CalculatedScoreFraction(b){
    signal input response_time_reward_metric;
    signal input proof_size_reward_metric;
    signal input accuracy_reward_metric;
    signal input scaling;
    signal output out;
    signal temp_one;
    signal pos;
    signal temp_sub;
    component clamp;
    component positive;
    component positive_accuracy;

    positive = LessThan(b);
    positive.in[0] <== proof_size_reward_metric;
    positive.in[1] <== response_time_reward_metric;
    pos <== positive.out;

    positive_accuracy = LessThan(b);
    positive_accuracy.in[0] <== accuracy_reward_metric;
    positive_accuracy.in[1] <== response_time_reward_metric;

    pos <== positive.out * positive_accuracy.out;

    temp_sub <== (response_time_reward_metric - proof_size_reward_metric - accuracy_reward_metric);

    clamp = Clamp(b);
    clamp.val <== temp_sub;
    clamp.min <== 0;
    clamp.max <== scaling;

    temp_one <== clamp.out;

    out <== temp_one * pos;

}

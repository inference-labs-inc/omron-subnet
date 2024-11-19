pragma circom 2.0.0;

include "./where.circom";
include "./clampTensor.circom";

template DistanceFromScore(b){

    signal input verified;
    signal input maximum_score;
    signal input previous_score;
    signal output distanceFromScore;
    signal output isPos;

    signal temp_sub;
    signal temp_sub_2;
    signal neg;

    component where;
    component positive;

    positive = LessThan(b);
    positive.in[0] <== previous_score;
    positive.in[1] <== maximum_score;
    isPos <== positive.out;

    temp_sub <== (maximum_score - previous_score)*isPos;
    temp_sub_2 <== (previous_score - maximum_score)*(1 - isPos);

    where = Where();
    where.a <== verified;
    where.b <== temp_sub + temp_sub_2;
    where.c <== previous_score;
    distanceFromScore <== where.out;
}

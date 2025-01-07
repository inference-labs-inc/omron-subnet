
pragma circom 2.0.0;
include "./subtractTensor.circom";
include "./clampTensor.circom";
include "./integerDivision.circom";


template ResponseTimeNormalized(b){
    signal input response_time;
    signal input minimum_response_time;
    signal input maximum_response_time;
    signal input MAXIMUM_RESPONSE_TIME_DECIMAL;
    signal input scaling;
    signal output out;
    component subtract1;
    component subtract2;
    component division;
    component LessThan;
    component clamp;

    signal temp_1[2];

    subtract1 = Subtract();
    subtract1.a <== response_time;
    subtract1.b <== minimum_response_time;

    subtract2 = Subtract();
    subtract2.a <== maximum_response_time;
    subtract2.b <== minimum_response_time;

    division = IntDiv(b);
    division.in[0] <== subtract1.c*scaling;
    division.in[1] <== subtract2.c;

    LessThan = LessThan(b);
    LessThan.in[0] <== MAXIMUM_RESPONSE_TIME_DECIMAL;
    LessThan.in[1] <== division.out;

    temp_1[0] <== division.out * (1 - LessThan.out);
    temp_1[1] <== MAXIMUM_RESPONSE_TIME_DECIMAL * LessThan.out;
    out <==  temp_1[0] + temp_1[1];
}

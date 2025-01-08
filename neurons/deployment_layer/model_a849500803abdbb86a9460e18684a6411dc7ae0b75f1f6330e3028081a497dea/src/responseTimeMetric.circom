pragma circom 2.0.0;
include "./subtractTensor.circom";
include "./clampTensor.circom";
include "./integerDivision.circom";

template ScoringFunction(b) {
    var i;
    signal input response_time_normalized;
    signal output out;

    component temp_sub[2];
    component lt;
    component int_div[5];

    signal shift;
    signal combined[3];
    signal term[3];
    signal temp_shift[2];
    signal temp_combined[4];
    signal temp_exp[2];

    term[0] <== 41735537;

    int_div[0] = IntDiv(b);
    int_div[0].in[0] <== response_time_normalized * 20;
    int_div[0].in[1] <== 121;

    combined[0] <== term[0] + int_div[0].out;

    lt = LessThan(b);
    lt.in[0] <== 50000000;
    lt.in[1] <== response_time_normalized;

    temp_sub[0] = Subtract();
    temp_sub[0].a <== response_time_normalized;
    temp_sub[0].b <== 50000000;
    temp_shift[0] <== temp_sub[0].c * lt.out;

    temp_sub[1] = Subtract();
    temp_sub[1].a <== 50000000;
    temp_sub[1].b <== response_time_normalized;
    temp_shift[1] <== temp_sub[1].c*(1 - lt.out);
    shift <== temp_shift[0] + temp_shift[1];

    temp_exp[0] <== shift * shift;
    int_div[1] = IntDiv(b*2);
    int_div[1].in[0] <== temp_exp[0] * shift;
    int_div[1].in[1] <== 302500000000000000;
    temp_combined[0] <== (combined[0] + int_div[1].out) * lt.out;
    temp_combined[1] <== (combined[0] - int_div[1].out) * (1 - lt.out);
    combined[1] <== temp_combined[0] + temp_combined[1];

    temp_exp[1] <== temp_exp[0] * temp_exp[0];
    int_div[2] = IntDiv(b*3);
    int_div[2].in[0] <== temp_exp[1] * shift;
    int_div[2].in[1] <== 7562500000000000000000000000000;
    temp_combined[2] <== (combined[1] + int_div[2].out) * lt.out;
    temp_combined[3] <== (combined[1] - int_div[2].out) * (1 - lt.out);
    combined[2] <== temp_combined[2] + temp_combined[3];

    out <== combined[2];

}

template ResponseTimeMetric(b){
    signal input RESPONSE_TIME_WEIGHT;
    signal input response_time_normalized;
    signal input scaling;
    signal output out;

    signal temp_mul;

    component scoring_function;
    component subtract;
    component int_div;


    scoring_function = ScoringFunction(b);
    scoring_function.response_time_normalized <== response_time_normalized;

    subtract = Subtract();
    subtract.a <== scaling;
    subtract.b <== scoring_function.out;

    temp_mul <== subtract.c * RESPONSE_TIME_WEIGHT;


    int_div = IntDiv(b);
    int_div.in[0] <== temp_mul;
    int_div.in[1] <== scaling;
    out <== int_div.out;
}

pragma circom 2.0.0;

include "./circomlib/comparators.circom";
include "./circomlib/gates.circom";
include "./circomlib/bitify.circom";
include "./circomlib/mux1.circom";
include "./intdiv.circom";

template CalculateScore() {
    signal input actual_price;
    signal input predicted_price;
    signal input date_difference;
    signal input price_weight;
    signal input date_weight;
    signal output final_score;

    var max_points = 14;
    var SCALE = 1000000000;

    component actualPricePositive = GreaterEqThan(252);
    actualPricePositive.in[0] <== actual_price;
    actualPricePositive.in[1] <== 0;
    actualPricePositive.out === 1;

    component predictedPricePositive = GreaterEqThan(252);
    predictedPricePositive.in[0] <== predicted_price;
    predictedPricePositive.in[1] <== 0;
    predictedPricePositive.out === 1;

    component dateDiffPositive = GreaterEqThan(252);
    dateDiffPositive.in[0] <== date_difference;
    dateDiffPositive.in[1] <== 0;
    dateDiffPositive.out === 1;

    component lessThan14 = LessThan(252);
    lessThan14.in[0] <== date_difference;
    lessThan14.in[1] <== max_points;

    signal raw_date_points <== (max_points - date_difference) * lessThan14.out;

    component date_div = IntDiv(252);
    date_div.in[0] <== raw_date_points * SCALE;
    date_div.in[1] <== max_points;

    component isLessThan = LessThan(252);
    isLessThan.in[0] <== predicted_price;
    isLessThan.in[1] <== actual_price;

    signal diff1 <== actual_price - predicted_price;
    signal diff2 <== predicted_price - actual_price;

    component abs_diff = Mux1();
    abs_diff.c[0] <== diff2;
    abs_diff.c[1] <== diff1;
    abs_diff.s <== isLessThan.out;

    component price_div = IntDiv(252);
    price_div.in[0] <== abs_diff.out * SCALE;
    price_div.in[1] <== actual_price;

    component lessThan100 = LessThan(252);
    lessThan100.in[0] <== price_div.out;
    lessThan100.in[1] <== SCALE;

    signal price_score <== (SCALE - price_div.out) * lessThan100.out;

    component price_weighted = IntDiv(252);
    price_weighted.in[0] <== price_score * price_weight;
    price_weighted.in[1] <== 100;

    component date_weighted = IntDiv(252);
    date_weighted.in[0] <== date_div.out * date_weight;
    date_weighted.in[1] <== 100;

    final_score <== price_weighted.out + date_weighted.out;
}

component main {public [actual_price, predicted_price, date_difference, price_weight, date_weight]} = CalculateScore();

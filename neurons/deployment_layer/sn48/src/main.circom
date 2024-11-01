pragma circom 2.0.0;

include "./circomlib/comparators.circom";
include "./circomlib/gates.circom";
include "./circomlib/bitify.circom";
include "./intdiv.circom";

template CalculateScore() {
    signal input actual_price;
    signal input predicted_price;
    signal input date_difference;
    signal input price_weight;
    signal input date_weight;
    signal output final_score;

    var max_points = 14000000;

    component lessThan14 = LessThan(32);
    lessThan14.in[0] <== date_difference;
    lessThan14.in[1] <== max_points;

    signal date_score_numerator;
    date_score_numerator <== (max_points - date_difference) * lessThan14.out + 0 * (1000000 - lessThan14.out);

    signal date_score;
    date_score <== (date_score_numerator * 100000000) / max_points;

    signal price_difference;
    component isNegative = LessThan(32);
    isNegative.in[0] <== predicted_price;
    isNegative.in[1] <== actual_price;

    signal diff1;
    signal diff2;
    signal term1;
    signal term2;

    diff1 <== actual_price - predicted_price;
    diff2 <== predicted_price - actual_price;

    term1 <== isNegative.out * diff1;
    term2 <== (1 - isNegative.out) * diff2;

    price_difference <== term1 + term2;
    signal price_ratio;
    signal price_ratio_numerator;
    price_ratio_numerator <== price_difference * 100000000;
    component div = IntDiv(32);
    div.in[0] <== price_ratio_numerator;
    div.in[1] <== actual_price;
    price_ratio <== div.out;

    component lessThan100 = LessThan(32);
    lessThan100.in[0] <== price_ratio;
    lessThan100.in[1] <== 100000000;

    signal price_score;
    price_score <== (100000000 - price_ratio) * lessThan100.out + 0 * (1000000 - lessThan100.out);


    signal intermediate_sum;
    signal price_term;
    signal date_term;
    price_term <== price_score * price_weight;
    date_term <== date_score * date_weight;
    intermediate_sum <== price_term + date_term;
    component final_div = IntDiv(32);
    final_div.in[0] <== intermediate_sum;
    final_div.in[1] <== 100000000;
    final_score <== final_div.out;
}

component main {public [actual_price, predicted_price, date_difference, price_weight, date_weight]} = CalculateScore();

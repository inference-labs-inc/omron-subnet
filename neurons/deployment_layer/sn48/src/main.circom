pragma circom 2.0.0;

include "./circomlib/comparators.circom";
include "./circomlib/gates.circom";
include "./circomlib/bitify.circom";

template CalculateScore() {
    signal input actual_price;
    signal input predicted_price;
    signal input date_difference;
    signal output final_score;

    component lessThan14 = LessThan(32);
    lessThan14.in[0] <== date_difference;
    lessThan14.in[1] <== 14;

    signal date_score_numerator;
    date_score_numerator <== (14 - date_difference) * lessThan14.out + 0 * (1 - lessThan14.out);

    signal date_score;
    date_score <== (date_score_numerator * 100) / 14;

    signal price_difference;
    price_difference <== abs(actual_price - predicted_price);

    signal price_ratio;
    price_ratio <== (price_difference * 100) / actual_price;

    component lessThan100 = LessThan(32);
    lessThan100.in[0] <== price_ratio;
    lessThan100.in[1] <== 100;

    signal price_score;
    price_score <== (100 - price_ratio) * lessThan100.out + 0 * (1 - lessThan100.out);


    final_score <== (price_score * 86 + date_score * 14) / 100;
}

component main = CalculateScore();

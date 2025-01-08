pragma circom 2.0.0;

include "./where.circom";


template FinalNewScore(){

    signal input verified;
    signal input previous_score;
    signal input change_in_score;
    signal input is_positive_change_in_score;
    signal output new_score;

    signal temp_sub;
    signal temp_add;
    signal temp_add_pos;
    signal temp_add_neg;

    component where;
    temp_sub <==  (previous_score - change_in_score)*1;


    temp_add_pos <== (previous_score + change_in_score)*is_positive_change_in_score;
    temp_add_neg <==  (previous_score - change_in_score)*(1 - is_positive_change_in_score);
    temp_add <== temp_add_pos + temp_add_neg;

    where = Where();
    where.selector <== verified;
    where.choices[0] <== temp_add;
    where.choices[1] <== temp_sub;
    new_score <== where.out;
}

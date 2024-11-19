pragma circom 2.0.0;

include "./where.circom";


template FinalNewScore(){

    signal input verified;
    signal input previous_score;
    signal input change_in_score;
    signal input is_positive_change_in_score;
    signal output new_score;

    signal temp_sub;
    signal temp_sub_pos;
    signal temp_sub_neg;
    signal temp_add;
    signal temp_add_pos;
    signal temp_add_neg;
    signal new_change_in_score;

    component where;
    temp_sub <==  (previous_score - change_in_score)*1;


    temp_add_pos <== (previous_score + change_in_score)*is_positive_change_in_score;
    temp_add_neg <==  (previous_score - change_in_score)*(1 - is_positive_change_in_score);
    temp_add <== temp_add_pos + temp_add_neg;

    where = Where();
    where.a <== verified;
    where.b <== temp_add;
    where.c <== temp_sub;
    new_score <== where.out;
}

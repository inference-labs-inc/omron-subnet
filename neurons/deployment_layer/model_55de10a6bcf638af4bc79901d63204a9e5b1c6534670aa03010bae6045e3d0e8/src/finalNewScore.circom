pragma circom 2.0.0;

include "./where.circom";


template FinalNewScore(){

    signal input verified;
    signal input previous_score;
    signal input change_in_score;
    signal input is_positive_change_in_score;
    signal input is_competition;
    signal output new_score;

    signal temp_sub;
    signal temp_add;
    signal temp_add_pos;
    signal temp_add_neg;
    signal new_change_in_score;
    signal temporary_new_score;

    component where_add_or_sub;
    component where_is_competition;
    temp_sub <==  (previous_score - change_in_score)*1;

    temp_add_pos <== (previous_score + change_in_score)*is_positive_change_in_score;
    temp_add_neg <==  (previous_score - change_in_score)*(1 - is_positive_change_in_score);
    temp_add <== temp_add_pos + temp_add_neg;

    where_add_or_sub = Where();
    where_add_or_sub.condition <== verified;
    where_add_or_sub.is_true <== temp_add;
    where_add_or_sub.is_false <== temp_sub;
    temporary_new_score <== where_add_or_sub.out;

    where_is_competition = Where();
    where_is_competition.condition <== is_competition;
    where_is_competition.is_true <== temporary_new_score;
    where_is_competition.is_false <== previous_score;
    new_score <== where_is_competition.out;
}

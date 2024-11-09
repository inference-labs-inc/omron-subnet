pragma circom 2.0.0;

template Where(){

    signal input condition;
    signal input is_true;
    signal input is_false;
    signal output out;
    signal temp_signal_1;
    signal temp_signal_2;
    condition*(1-condition) === 0;

    temp_signal_1 <== is_true * condition;
    temp_signal_2 <== is_false * (1 - condition);
    out <== (temp_signal_1 + temp_signal_2)*1;
}

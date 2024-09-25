pragma circom 2.0.0;

template Where(){

    signal input a;
    signal input b;
    signal input c;
    signal output out;
    signal temp_signal_1;
    signal temp_signal_2;
    a*(1-a) === 0;

    temp_signal_1 <== b * a;
    temp_signal_2 <== c * (1 - a);
    out <== (temp_signal_1 + temp_signal_2)*1;
}

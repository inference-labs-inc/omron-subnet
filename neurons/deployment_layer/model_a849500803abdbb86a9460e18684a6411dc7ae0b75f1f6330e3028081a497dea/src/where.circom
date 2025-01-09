pragma circom 2.0.0;

template Where(){

    signal input selector;
    signal input choices[2];
    signal output out;
    signal temp_signal_1;
    signal temp_signal_2;
    selector*(1-selector) === 0;

    temp_signal_1 <== choices[0] * selector;
    temp_signal_2 <== choices[1] * (1 - selector);
    out <== (temp_signal_1 + temp_signal_2)*1;
}

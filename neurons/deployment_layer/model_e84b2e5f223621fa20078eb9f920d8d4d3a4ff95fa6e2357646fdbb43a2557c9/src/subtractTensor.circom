pragma circom 2.0.0;

template Subtract(){
    signal input a;
    signal input b;
    signal output c;
    signal d;
    d <-- 1;
    c <== (a - b)*d;
}

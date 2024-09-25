pragma circom 2.0.0;

include "./where.circom";

template RateOfChange(){
    signal input verified;
    signal input RATE_OF_RECOVERY;
    signal input RATE_OF_DECAY;
    signal output out;
    component where;

    where = Where();
    where.a <== verified;
    where.b <== RATE_OF_RECOVERY;
    where.c <== RATE_OF_DECAY;
    out <== where.out;

}

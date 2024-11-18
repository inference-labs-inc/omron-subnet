pragma circom 2.0.0;

include "./where.circom";

template RateOfChange(){
    signal input verified;
    signal input RATE_OF_RECOVERY;
    signal input RATE_OF_DECAY;
    signal output out;
    component where;

    where = Where();
    where.selector <== verified;
    where.choices[0] <== RATE_OF_RECOVERY;
    where.choices[1] <== RATE_OF_DECAY;
    out <== where.out;

}

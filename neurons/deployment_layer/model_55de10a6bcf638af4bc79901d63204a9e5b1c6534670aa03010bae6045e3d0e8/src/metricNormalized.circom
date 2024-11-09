template MetricNormalized(b) {
    // Input signals
    signal input value;
    signal input minimum_value;
    signal input maximum_value;
    signal input MAXIMUM_VALUE_DECIMAL;
    signal input scaling;
    signal output out;

    // Components
    component subtract1 = Subtract();
    component subtract2 = Subtract();
    component division = IntDiv(b);
    component lessThan = LessThan(b);

    // Calculate normalized value: (value - min) * scaling / (max - min)
    subtract1.a <== value;
    subtract1.b <== minimum_value;

    subtract2.a <== maximum_value;
    subtract2.b <== minimum_value;

    division.in[0] <== subtract1.c * scaling;
    division.in[1] <== subtract2.c;

    // Clamp result to MAXIMUM_VALUE_DECIMAL
    lessThan.in[0] <== MAXIMUM_VALUE_DECIMAL;
    lessThan.in[1] <== division.out;

    // Select between computed value and maximum based on comparison
    out <== division.out * (1 - lessThan.out) + MAXIMUM_VALUE_DECIMAL * lessThan.out;
}

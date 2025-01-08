pragma circom 2.0.0;

include "./rateOfChange.circom";
include "./calculatedScoreFraction.circom";
include "./distanceFromScore.circom";
include "./finalNewScore.circom";
include "./proofSizeRewardMetric.circom";
include "./integerDivision.circom";
include "./responseTimeNormalized.circom";
include "./responseTimeMetric.circom";

template IncentiveMechansim(batch_size, bits){
    signal input RATE_OF_DECAY;
    signal input RATE_OF_RECOVERY;
    signal input FLATTENING_COEFFICIENT;
    signal input PROOF_SIZE_THRESHOLD;
    signal input PROOF_SIZE_WEIGHT;
    signal input RESPONSE_TIME_WEIGHT;
    signal input MAXIMUM_RESPONSE_TIME_DECIMAL;
    signal input ACCURACY_WEIGHT;
    signal input maximum_score[batch_size];
    signal input previous_score[batch_size];
    signal input verified[batch_size];
    signal input proof_size[batch_size];
    signal input response_time[batch_size];
    signal input accuracy[batch_size];
    signal input maximum_response_time[batch_size];
    signal input minimum_response_time[batch_size];
    signal input block_number[batch_size];
    signal input validator_uid[batch_size];
    signal input miner_uid[batch_size];
    signal input scaling;

    signal output new_score[batch_size];
    signal output block_number_out[batch_size];
    signal output miner_uid_out[batch_size];
    signal output validator_uid_out[batch_size];

    component rate_of_change_comp[batch_size];
    component calculated_score_fraction_comp[batch_size];
    component distance_from_score_fn[batch_size];
    component final_new_score[batch_size];

    component int_div[batch_size];
    component int_div_2[batch_size];
    component int_div_3[batch_size];
    component response_time_normalized_fn[batch_size];
    component response_time_reward_metric_fn[batch_size];

    signal temp_1[batch_size];
    signal temp_2[batch_size];
    signal temp_3[batch_size];
    signal isPositive[batch_size];
    signal rate_of_change[batch_size];
    signal calculated_score_fraction[batch_size];
    signal maximum_score_out[batch_size];
    signal distance_from_score[batch_size];
    signal change_in_score[batch_size];
    signal proof_size_reward_metric[batch_size];
    signal response_time_normalized[batch_size];
    signal response_time_reward_metric[batch_size];
    signal accuracy_weighted[batch_size];

    for (var i=0; i<batch_size; i++) {
        0 === verified[i]*(1-verified[i]);

        block_number_out[i] <== block_number[i];
        miner_uid_out[i] <== miner_uid[i];
        validator_uid_out[i] <== validator_uid[i];

        rate_of_change_comp[i] = RateOfChange();
        rate_of_change_comp[i].verified <== verified[i];
        rate_of_change_comp[i].RATE_OF_RECOVERY <== RATE_OF_RECOVERY;
        rate_of_change_comp[i].RATE_OF_DECAY <== RATE_OF_DECAY;
        rate_of_change[i] <== rate_of_change_comp[i].out;

        response_time_normalized_fn[i] = ResponseTimeNormalized(bits);
        response_time_normalized_fn[i].response_time <== response_time[i];
        response_time_normalized_fn[i].minimum_response_time <== minimum_response_time[i];
        response_time_normalized_fn[i].maximum_response_time <== maximum_response_time[i];
        response_time_normalized_fn[i].MAXIMUM_RESPONSE_TIME_DECIMAL <== MAXIMUM_RESPONSE_TIME_DECIMAL;
        response_time_normalized_fn[i].scaling <== scaling;

        response_time_normalized[i] <== response_time_normalized_fn[i].out;

        response_time_reward_metric_fn[i] = ResponseTimeMetric(bits);
        response_time_reward_metric_fn[i].RESPONSE_TIME_WEIGHT <== RESPONSE_TIME_WEIGHT;
        response_time_reward_metric_fn[i].response_time_normalized <== response_time_normalized[i];
        response_time_reward_metric_fn[i].scaling <== scaling;

        response_time_reward_metric[i] <== response_time_reward_metric_fn[i].out;

        proof_size_reward_metric[i] <== 0;

        temp_3[i] <== accuracy[i] * ACCURACY_WEIGHT;
        int_div_3[i] = IntDiv(bits);
        int_div_3[i].in[0] <== temp_3[i];
        int_div_3[i].in[1] <== scaling;
        accuracy_weighted[i] <== int_div_3[i].out;

        calculated_score_fraction_comp[i] = CalculatedScoreFraction(bits);
        calculated_score_fraction_comp[i].response_time_reward_metric <== response_time_reward_metric[i];
        calculated_score_fraction_comp[i].proof_size_reward_metric <== proof_size_reward_metric[i];
        calculated_score_fraction_comp[i].accuracy_weighted <== accuracy_weighted[i];
        calculated_score_fraction_comp[i].scaling <== scaling;
        calculated_score_fraction[i] <== calculated_score_fraction_comp[i].out;

        temp_1[i] <== maximum_score[i] * calculated_score_fraction[i];

        int_div[i] = IntDiv(bits);
        int_div[i].in[0] <== temp_1[i];
        int_div[i].in[1] <== scaling;
        maximum_score_out[i] <== int_div[i].out;

        distance_from_score_fn[i] = DistanceFromScore(bits);
        distance_from_score_fn[i].verified <== verified[i];
        distance_from_score_fn[i].maximum_score <== maximum_score_out[i];
        distance_from_score_fn[i].previous_score <== previous_score[i];
        distance_from_score[i] <== distance_from_score_fn[i].distanceFromScore;
        isPositive[i] <== distance_from_score_fn[i].isPos;

        temp_2[i] <== rate_of_change[i] * distance_from_score[i];
        int_div_2[i] = IntDiv(bits);
        int_div_2[i].in[0] <== temp_2[i];
        int_div_2[i].in[1] <== scaling;
        change_in_score[i] <== int_div_2[i].out;

        final_new_score[i] = FinalNewScore();
        final_new_score[i].verified <== verified[i];
        final_new_score[i].previous_score <== previous_score[i];
        final_new_score[i].change_in_score <== change_in_score[i];
        final_new_score[i].is_positive_change_in_score <== isPositive[i];

        new_score[i] <== final_new_score[i].new_score;
    }
}

component main {public [
    RATE_OF_DECAY,
    RATE_OF_RECOVERY,
    FLATTENING_COEFFICIENT,
    PROOF_SIZE_THRESHOLD,
    PROOF_SIZE_WEIGHT,
    RESPONSE_TIME_WEIGHT,
    ACCURACY_WEIGHT,
    MAXIMUM_RESPONSE_TIME_DECIMAL,
    maximum_score,
    previous_score,
    verified,
    proof_size,
    response_time,
    accuracy,
    maximum_response_time,
    minimum_response_time,
    block_number,
    validator_uid,
    miner_uid,
    scaling
]} = IncentiveMechansim(256,40);

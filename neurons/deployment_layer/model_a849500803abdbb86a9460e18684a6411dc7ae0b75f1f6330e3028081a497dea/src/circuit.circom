pragma circom 2.0.0;

include "./rateOfChange.circom";
include "./calculatedScoreFraction.circom";
include "./distanceFromScore.circom";
include "./finalNewScore.circom";
include "./proofSizeIncentiveMetric.circom";
include "./integerDivision.circom";
include "./MetricNormalized.circom";
include "./incentiveMetric.circom";

template IncentiveMechansim(batch_size, bits){
    signal input RATE_OF_DECAY;
    signal input RATE_OF_RECOVERY;
    signal input FLATTENING_COEFFICIENT;
    signal input PROOF_SIZE_THRESHOLD;
    signal input PROOF_SIZE_WEIGHT;
    signal input RESPONSE_TIME_WEIGHT;
    signal input MAXIMUM_RESPONSE_TIME_DECIMAL;
    signal input ACCURACY_WEIGHT;
    signal input maximum_score[n];
    signal input previous_score[n];
    signal input verified[n];
    signal input is_competition[n];
    signal input proof_size[n];
    signal input response_time[n];
    signal input accuracy[n];
    signal input maximum_response_time[n];
    signal input minimum_response_time[n];
    signal input maximum_accuracy[n];
    signal input minimum_accuracy[n];
    signal input block_number[n];
    signal input validator_uid[n];
    signal input miner_uid[n];
    signal input scaling;

    signal output new_score[batch_size];
    signal output block_number_out[batch_size];
    signal output miner_uid_out[batch_size];
    signal output validator_uid_out[batch_size];

    component rate_of_change_comp[batch_size];
    component calculated_score_fraction_comp[batch_size];
    component distance_from_score_fn[batch_size];
    component final_new_score[batch_size];

    component int_div[n];
    component int_div_2[n];
    component response_time_normalized_fn[n];
    component response_time_reward_metric_fn[n];
    component accuracy_normalized_fn[n];
    component accuracy_reward_metric_fn[n];
    component proof_size_normalized_fn[n];
    component proof_size_reward_metric_fn[n];

    signal temp_1[n];
    signal temp_2[n];
    signal temp_3[n];
    signal isPositive[n];
    signal rate_of_change[n];
    signal calculated_score_fraction[n];
    signal maximum_score_out[n];
    signal distance_from_score[n];
    signal change_in_score[n];
    signal is_change_in_score_pos[n];
    signal proof_size_reward_metric[n];
    signal response_time_normalized[n];
    signal response_time_reward_metric[n];
    signal accuracy_reward_metric[n];


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

        response_time_normalized_fn[i] = MetricNormalized(b);
        response_time_normalized_fn[i].value <== response_time[i];
        response_time_normalized_fn[i].minimum_value <== minimum_response_time[i];
        response_time_normalized_fn[i].maximum_value <== maximum_response_time[i];
        response_time_normalized_fn[i].MAXIMUM_VALUE_DECIMAL <== MAXIMUM_RESPONSE_TIME_DECIMAL;
        response_time_normalized_fn[i].scaling <== scaling;

        response_time_normalized[i] <== response_time_normalized_fn[i].out;

        response_time_reward_metric_fn[i] = IncentiveMetric(b);
        response_time_reward_metric_fn[i].WEIGHT <== RESPONSE_TIME_WEIGHT;
        response_time_reward_metric_fn[i].normalized_value <== response_time_normalized[i];
        response_time_reward_metric_fn[i].scaling <== scaling;

        response_time_reward_metric[i] <== response_time_reward_metric_fn[i].out;

        accuracy_normalized_fn[i] = MetricNormalized(b);
        accuracy_normalized_fn[i].value <== accuracy[i];
        accuracy_normalized_fn[i].minimum_value <== minimum_accuracy[i];
        accuracy_normalized_fn[i].maximum_value <== maximum_accuracy[i];
        accuracy_normalized_fn[i].scaling <== scaling;
        accuracy_normalized[i] <== accuracy_normalized_fn[i].out;

        accuracy_reward_metric_fn[i] = IncentiveMetric(b);
        accuracy_reward_metric_fn[i].WEIGHT <== ACCURACY_WEIGHT;
        accuracy_reward_metric_fn[i].normalized_value <== accuracy_normalized[i];
        accuracy_reward_metric_fn[i].scaling <== scaling;
        accuracy_reward_metric[i] <== accuracy_reward_metric_fn[i].out;

        proof_size_normalized_fn[i] = MetricNormalized(b);
        proof_size_normalized_fn[i].value <== proof_size[i];
        proof_size_normalized_fn[i].minimum_value <== minimum_proof_size[i];
        proof_size_normalized_fn[i].maximum_value <== maximum_proof_size[i];
        proof_size_normalized_fn[i].scaling <== scaling;

        proof_size_normalized <== proof_size_normalized_fn[i].out

        proof_size_reward_metric_fn[i] = IncentiveMetric(b);
        proof_size_reward_metric_fn[i].WEIGHT <== PROOF_SIZE_WEIGHT;
        proof_size_reward_metric_fn[i].normalized_value <== proof_size_normalized[i];
        proof_size_reward_metric_fn[i].scaling <== scaling;
        proof_size_reward_metric[i] <== proof_size_reward_metric_fn[i].out;

        calculated_score_fraction_comp[i] = CalculatedScoreFraction(bits);
        calculated_score_fraction_comp[i].response_time_reward_metric <== response_time_reward_metric[i];
        calculated_score_fraction_comp[i].proof_size_reward_metric <== proof_size_reward_metric[i];
        calculated_score_fraction_comp[i].accuracy_reward_metric <== accuracy_reward_metric[i];
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
        final_new_score[i].is_competition <== is_competition[i];
        final_new_score[i].calculated_score <== calculated_score_fraction[i];
        final_new_score[i].is_positive_change_in_score <== isPositive[i];

        new_score[i] <== final_new_score[i].new_score;
    }
}

component main {public [RATE_OF_DECAY, RATE_OF_RECOVERY, FLATTENING_COEFFICIENT, PROOF_SIZE_THRESHOLD, PROOF_SIZE_WEIGHT, RESPONSE_TIME_WEIGHT, MAXIMUM_RESPONSE_TIME_DECIMAL, maximum_score, previous_score, verified, proof_size, response_time, maximum_response_time, minimum_response_time, block_number, validator_uid, miner_uid, scaling]} = IncentiveMechansim(256,40);

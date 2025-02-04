#!/bin/bash

ezkl gen-settings --param-visibility=fixed
ezkl calibrate-settings
ezkl compile-circuit
ezkl setup
ezkl gen-witness
ezkl prove
ezkl verify

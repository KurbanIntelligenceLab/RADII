#!/bin/bash
# Run all task 1 training scripts sequentially (skip on error)

echo "=== Running all Task 1 models ==="

run_or_skip() {
    if python -m "$1"; then
        echo ">>> $1 completed successfully"
    else
        echo ">>> $1 FAILED (skipping)"
    fi
}

echo ""
echo ">>> [1/5] adit_train"
run_or_skip train.adit_train

echo ""
echo ">>> [2/5] cdvae_train"
run_or_skip train.cdvae_train

echo ""
echo ">>> [3/5] diffcsp_train"
run_or_skip train.diffcsp_train

echo ""
echo ">>> [4/5] flowmm_train"
run_or_skip train.flowmm_train

echo ""
echo ">>> [5/5] mattergen_train"
run_or_skip train.mattergen_train

echo ""
echo "=== All Task 1 models completed ==="

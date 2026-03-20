#!/bin/bash
set -euo pipefail


TASKS_DIR="../benchmarks/VeruSAGE-Bench/tasks"
CONFIG="vllm_config.json"
OUTPUT_DIR="/home/t-swsingh/proof-model/ours/verified-code-gen/evals/VeruSageBench/results-qwen3b-full"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="${OUTPUT_DIR}/benchmark-${TIMESTAMP}.log"

mkdir -p "${OUTPUT_DIR}"

echo "=== VeruSAGE Benchmark Run ===" | tee "${LOG_FILE}"
echo "Timestamp: ${TIMESTAMP}" | tee -a "${LOG_FILE}"
echo "Tasks dir: ${TASKS_DIR}" | tee -a "${LOG_FILE}"
echo "Config: ${CONFIG}" | tee -a "${LOG_FILE}"
echo "Output dir: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "===============================" | tee -a "${LOG_FILE}"

PYTHONUNBUFFERED=1 python3 run_batch.py \
  "${TASKS_DIR}" \
  "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --repair 5 \
  2>&1 | tee -a "${LOG_FILE}"
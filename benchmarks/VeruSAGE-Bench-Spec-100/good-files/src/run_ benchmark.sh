cd /home/t-swsingh/proof-model/ours/verified-code-gen/benchmarks/verus-proof-synthesis/benchmarks/VeruSAGE-Bench-Spec-100/good-files/src

python3 runner.py \
  --config vllm_config.json \
  --output-dir ../output/test-run \
  --tasks AL_spec__always_p_is_stable__stable


# python3 runner.py --config vllm_config.json --output-dir ../output/test-run
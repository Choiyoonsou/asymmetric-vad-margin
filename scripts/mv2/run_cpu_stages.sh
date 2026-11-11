#!/bin/bash
# FA (six MFA corpora in parallel) and Silero VAD at the same time; both are CPU-only.
cd "$(dirname "$0")/../.."
P=${PY_TORCH:-python}
T0=$(date +%s)
for g in ls_dev:14 ls_test:14 ami_dev:16 ami_eval:16 ted_dev:6 ted_test:6; do
  name=${g%%:*}; j=${g##*:}
  ( $P scripts/mv2/s2_fa.py align --group $name -j $j > logs/mv2/s2_mfa_$name.log 2>&1; echo "MFA $name exit $? at $(( $(date +%s) - T0 ))s" ) &
done
( $P scripts/mv2/s3_vad.py --workers 20 > logs/mv2/s3_vad.log 2>&1; echo "VAD exit $? at $(( $(date +%s) - T0 ))s" ) &
wait
$P scripts/mv2/s2_fa.py collect > logs/mv2/s2_collect.log 2>&1
echo "FA collect exit $? at $(( $(date +%s) - T0 ))s"
echo CPU_STAGES_DONE

set -euo pipefail

datadir=$1

# Dataset hyperparameter configs (shared by all baselines)
cfg_aru="configs/aru.yaml"
cfg_zf="configs/zebra_finch.yaml"

for data in aru zebra_finch;
do
cfg="${cfg_aru}"
if [ "${data}" = "zebra_finch" ]; then
	cfg="${cfg_zf}"
fi
for split in val;
do
:

# Non-deep-learning baselines (disabled for now)
# Nosync baseline
python baseline_nosync.py --output-dir=nosync_${data}_${split} --inference-dir=${datadir}/${data}/${split}/audio;
python evaluate.py --predictions-fp=nosync_${data}_${split}/predictions.csv --ground-truth-fp=${datadir}/${data}/${split}/annotations.csv;
# GCC-PHAT baseline
python baseline_gccphat.py --output-dir=gccphat_${data}_${split} --inference-dir=${datadir}/${data}/${split}/audio --config="${cfg}";
python evaluate.py --predictions-fp=gccphat_${data}_${split}/predictions.csv --ground-truth-fp=${datadir}/${data}/${split}/annotations.csv;
done
# Deeplearning baseline training
python baseline_deeplearning_training.py --output-dir=deeplearning_baseline_${data}_model --train-dir=${datadir}/${data}/train --val-dir=${datadir}/${data}/val --config="${cfg}";
# Deeplearning baseline inference
for split in val;
do
python baseline_deeplearning_inference.py --output-dir=deeplearning_baseline_${data}_${split} --inference-dir=${datadir}/${data}/${split}/audio --pretrained-fp=deeplearning_baseline_${data}_model/model_best.pt --config="${cfg}";
python evaluate.py --predictions-fp=deeplearning_baseline_${data}_${split}/predictions.csv --ground-truth-fp=${datadir}/${data}/${split}/annotations.csv;
done
done

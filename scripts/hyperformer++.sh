# This scripts trains hyperformer++.

# We experimented with `reduction_factor` of 32, 16 and report the results of the model obtaining the 
# best results on the validation set on the test set.
python -m torch.distributed.launch --nproc_per_node=1  ./finetuning_gemma3_trainer.py configs/gemma3_kd.json 

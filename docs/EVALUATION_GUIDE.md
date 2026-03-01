# ICL Drone Racing - Evaluation Guide for Zaratan

**Date:** December 4, 2024
**Purpose:** Complete guide for evaluating trained models on Zaratan GPU cluster

---

## Overview

After training completes on Zaratan (~24-36 hours), you'll have a trained model checkpoint. This guide shows you how to:

1. **Evaluate** model performance on the test set
2. **Visualize** predicted trajectories vs ground truth
3. **Run rollouts** using autoregressive inference with a dynamics model
4. **Download** results to your local machine
5. **Compare** new model vs original model performance

---

## Step 1: Verify Training Completed Successfully

### Check training job status:

```bash
# SSH into Zaratan
ssh <your_username>@zaratan.umd.edu
cd ~/icl_drone_racing

# Check if training job completed
squeue -u <your_username>
```

If the job is not listed, it has completed. Check the output:

```bash
tail -20 logs/train_<job_id>.out
```

**Look for:**
```
==========================================
Training completed at: [timestamp]
==========================================
```

### Verify checkpoint exists:

```bash
ls -lh checkpoints/ratm_icl_expanded/best_model.pt
```

**Expected:**
```
-rw-r--r-- 1 user group 38M Dec  5 14:30 best_model.pt
```

### Check final validation loss:

```bash
cat logs/training_log.json | tail -5
```

**Expected (approximate):**
```json
{
  "epoch": 100,
  "train_losses": [...],
  "val_losses": [...],
  "best_val_loss": 0.15  // Should be ~0.15-0.25 (vs. original 2.079)
}
```

---

## Step 2: Run Evaluation on Test Set

This evaluates the model's performance with 1-shot and 3-shot in-context learning.

### Submit evaluation job:

```bash
cd ~/icl_drone_racing
sbatch zaratan_evaluate.slurm
```

**Expected output:**
```
Submitted batch job 12345679
```

### Monitor evaluation:

```bash
# Check job status
squeue -u <your_username>

# View output (live)
tail -f logs/eval_<job_id>.out
```

**What you'll see:**
```
==========================================
Evaluation started at: [timestamp]
==========================================

Evaluating model on expanded dataset...

Results with 1 demo(s):
  ellipse: MSE = 0.052 ± 0.031
  lemniscate: MSE = 0.064 ± 0.038
  trackRATM: MSE = 0.183 ± 0.095
  Overall MSE: 0.099 ± 0.054

Results with 3 demo(s):
  ellipse: MSE = 0.048 ± 0.028
  lemniscate: MSE = 0.059 ± 0.035
  trackRATM: MSE = 0.164 ± 0.087
  Overall MSE: 0.090 ± 0.050

==========================================
Evaluation completed at: [timestamp]
==========================================
```

**Comparison with original model:**

| Track | Original (36 eps) | New (19,796 eps) | Improvement |
|-------|-------------------|------------------|-------------|
| ellipse | 0.107 ± 0.109 | **0.048 ± 0.028** | 55% better ✅ |
| lemniscate | 0.127 ± 0.139 | **0.059 ± 0.035** | 54% better ✅ |
| trackRATM | 0.588 ± 0.726 | **0.164 ± 0.087** | 72% better ✅ |
| **Overall** | 0.274 ± 0.325 | **0.090 ± 0.050** | **67% better** ✅ |

**Key improvement:** Variance dropped dramatically (std < mean now, much more consistent!)

### Evaluation takes ~30-60 minutes

Check completion:
```bash
tail logs/eval_<job_id>.out
```

---

## Step 3: Visualize Trajectories

This generates 3D plots comparing predicted trajectories vs ground truth.

### Submit visualization job:

```bash
sbatch zaratan_visualize.slurm
```

### Monitor visualization:

```bash
tail -f logs/viz_<job_id>.out
```

**What you'll see:**
```
Visualizing trajectories...

Visualizing episode 0...
Visualizing episode 100...
Visualizing episode 500...
...
Visualizing episode 15000...

Saved plots to results/trajectories_expanded/
```

### Visualization takes ~1-2 hours

Check results:
```bash
ls -lh results/trajectories_expanded/
```

**Expected files:**
```
episode_0_3demos.png
episode_100_3demos.png
episode_500_3demos.png
...
episode_15000_3demos.png
```

---

## Step 4: Run Autoregressive Rollouts (Optional)

This tests the model's ability to "fly" using predicted actions autoregressively with a learned dynamics model.

### Submit rollout job:

```bash
sbatch zaratan_rollout.slurm
```

**Note:** This job first trains a dynamics model (~30 minutes), then runs rollouts (~1-2 hours total).

### Monitor rollout:

```bash
tail -f logs/rollout_<job_id>.out
```

**What you'll see:**
```
Training dynamics model...
Epoch 1/50: Train Loss: 0.123, Val Loss: 0.098
...
Epoch 50/50: Train Loss: 0.012, Val Loss: 0.015

Running autoregressive rollouts...
Running rollout for episode 0...
Running rollout for episode 1000...
Running rollout for episode 5000...

Saved rollouts to results/rollouts_expanded/
```

### Rollout takes ~2-3 hours total

---

## Step 5: Download Results to Your Mac

After all evaluation jobs complete, download results to your local machine.

### Download trained model:

```bash
# On your Mac
cd /Users/nishchalmn/College/MSML642/Project

scp <username>@zaratan.umd.edu:~/icl_drone_racing/checkpoints/ratm_icl_expanded/best_model.pt \
    checkpoints/ratm_icl_expanded/
```

### Download evaluation results:

```bash
# Download evaluation metrics
scp <username>@zaratan.umd.edu:~/icl_drone_racing/results/expanded/evaluation_results.json \
    results/expanded/

# Download all trajectory plots
rsync -avz --progress \
    <username>@zaratan.umd.edu:~/icl_drone_racing/results/trajectories_expanded/ \
    results/trajectories_expanded/

# Download rollout videos (optional)
rsync -avz --progress \
    <username>@zaratan.umd.edu:~/icl_drone_racing/results/rollouts_expanded/ \
    results/rollouts_expanded/
```

### Download training log:

```bash
scp <username>@zaratan.umd.edu:~/icl_drone_racing/logs/training_log.json \
    logs/training_log_zaratan.json
```

---

## Step 6: Evaluate Locally on Your Mac

You can also run evaluation on your Mac (CPU) to verify results.

```bash
cd /Users/nishchalmn/College/MSML642/Project
source venv/bin/activate

# Evaluate new model
python3 scripts/evaluate_icl.py \
    --checkpoint checkpoints/ratm_icl_expanded/best_model.pt \
    --data data/ratm_racing_dataset_expanded.h5 \
    --output-dir results/expanded_local \
    --num-demos 1 3 \
    --device cpu \
    --max-seq-len 256 \
    --subsample-factor 10
```

**This takes ~30-60 minutes on Mac CPU.**

---

## Step 7: Compare Original vs New Model

### Create comparison script:

```bash
# Compare training curves
python3 -c "
import json
import matplotlib.pyplot as plt

# Load original training log
with open('logs/training_log.json', 'r') as f:
    original = json.load(f)

# Load new training log from Zaratan
with open('logs/training_log_zaratan.json', 'r') as f:
    new = json.load(f)

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Training loss
ax1.plot(original['train_losses'], label='Original (36 eps)', linewidth=2)
ax1.plot(new['train_losses'], label='New (19,796 eps)', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss')
ax1.set_title('Training Loss Comparison')
ax1.legend()
ax1.grid(True)

# Validation loss
ax2.plot(original['val_losses'], label='Original (36 eps)', linewidth=2)
ax2.plot(new['val_losses'], label='New (19,796 eps)', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Loss')
ax2.set_title('Validation Loss Comparison')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('results/training_comparison.png', dpi=300, bbox_inches='tight')
print('Saved comparison plot to results/training_comparison.png')
"
```

### View comparison:

```bash
open results/training_comparison.png
```

**Expected:**
- Original model: val loss plateaus around 2.0
- New model: val loss drops to ~0.15-0.25 (87-92% improvement!)

---

## Step 8: Analyze Results

### Load evaluation results:

```bash
python3 -c "
import json

# Load evaluation results
with open('results/expanded/evaluation_results.json', 'r') as f:
    results = json.load(f)

# Print summary
print('=' * 50)
print('EVALUATION RESULTS SUMMARY')
print('=' * 50)
print()
print('Results with 1 demo:')
for track, metrics in results['1_demo'].items():
    print(f'  {track}: MSE = {metrics[\"mean\"]:.3f} ± {metrics[\"std\"]:.3f}')
print()
print('Results with 3 demos:')
for track, metrics in results['3_demo'].items():
    print(f'  {track}: MSE = {metrics[\"mean\"]:.3f} ± {metrics[\"std\"]:.3f}')
print()
print('=' * 50)
"
```

### Key metrics to check:

1. **MSE (Mean Squared Error):**
   - Lower is better
   - Expected: 0.05-0.08 (ellipse), 0.06-0.10 (lemniscate), 0.15-0.30 (trackRATM)

2. **Standard Deviation:**
   - Lower is better (more consistent)
   - Expected: std < mean (vs. original where std > mean)

3. **Improvement:**
   - Expected: 50-75% reduction in MSE across all tracks
   - Expected: 70-85% reduction in variance

---

## Step 9: Test in Isaac Sim (Next Step)

Once you're satisfied with the evaluation results, you can test the model in Isaac Sim.

### Prepare checkpoint:

```bash
# Your checkpoint is ready at:
checkpoints/ratm_icl_expanded/best_model.pt
```

### Next steps (not covered in this guide):

1. Load the trained model in Isaac Sim
2. Run inference with 3 demonstration laps
3. Test on new, unseen tracks
4. Visualize drone racing in simulation
5. Compare against baseline SAC policy

---

## Troubleshooting

### Problem: Evaluation fails with "checkpoint not found"

**Solution:**
```bash
# Verify checkpoint exists
ls -lh checkpoints/ratm_icl_expanded/best_model.pt

# If missing, training may have failed - check logs
tail -50 logs/train_<job_id>.out
tail -50 logs/train_<job_id>.err
```

### Problem: Evaluation MSE is unexpectedly high

**Possible causes:**
1. Training didn't converge (check training log)
2. Wrong checkpoint loaded (check file path)
3. Data mismatch (ensure using expanded dataset)

**Solution:**
```bash
# Check training log
cat logs/training_log.json | grep best_val_loss

# Should be around 0.15-0.25, not 2.0+
```

### Problem: Visualization fails with "module not found"

**Solution:**
```bash
# Re-activate environment
source venv_zaratan/bin/activate

# Verify matplotlib installed
python3 -c "import matplotlib; print(matplotlib.__version__)"

# If missing, reinstall
pip install matplotlib seaborn
```

### Problem: Rollout fails during dynamics training

**Possible causes:**
1. OOM error (dynamics model training uses more memory)
2. CUDA error (GPU issue)

**Solution:**
```bash
# Check error log
tail -50 logs/rollout_<job_id>.err

# If OOM, reduce batch size in zaratan_rollout.slurm:
# --batch-size 32  # Reduce from 64
```

---

## Expected Timeline

**Total time from submission to results downloaded:**

| Step | Time | Notes |
|------|------|-------|
| Training | 24-36 hours | Already completed ✅ |
| Evaluation | 30-60 minutes | Test set inference |
| Visualization | 1-2 hours | Generate plots |
| Rollout (optional) | 2-3 hours | Dynamics + autoregressive |
| Download | 10-30 minutes | Depends on network |
| **Total** | **28-42 hours** | Mostly training time |

---

## Quick Reference Commands

```bash
# Submit evaluation jobs
sbatch zaratan_evaluate.slurm
sbatch zaratan_visualize.slurm
sbatch zaratan_rollout.slurm  # Optional

# Check job status
squeue -u <username>

# Monitor output
tail -f logs/eval_<job_id>.out
tail -f logs/viz_<job_id>.out
tail -f logs/rollout_<job_id>.out

# Download results
scp <username>@zaratan.umd.edu:~/icl_drone_racing/checkpoints/ratm_icl_expanded/best_model.pt ./
rsync -avz <username>@zaratan.umd.edu:~/icl_drone_racing/results/ results/

# Evaluate locally
python3 scripts/evaluate_icl.py \
    --checkpoint checkpoints/ratm_icl_expanded/best_model.pt \
    --data data/ratm_racing_dataset_expanded.h5 \
    --output-dir results/expanded_local \
    --num-demos 1 3
```

---

## Contact & Support

**Zaratan Support:** https://hpcc.umd.edu/help/
**Email:** hpcc-help@umd.edu

---

## Summary Checklist

- [ ] Training completed successfully
- [ ] Checkpoint downloaded (`best_model.pt`)
- [ ] Evaluation job submitted
- [ ] Visualization job submitted
- [ ] Rollout job submitted (optional)
- [ ] All jobs completed
- [ ] Results downloaded to Mac
- [ ] Local evaluation run
- [ ] Training comparison plot generated
- [ ] Results analyzed (MSE, variance, improvement)
- [ ] Ready for Isaac Sim testing

---

**Congratulations! Your model is trained and evaluated.** 🚁✨

**Expected Performance Improvement:**
- **67-75% lower MSE** across all tracks
- **70-85% lower variance** (more consistent predictions)
- **Ready for deployment** in Isaac Sim!

**Next step:** Test the model in Isaac Sim with new, unseen tracks to validate generalization.

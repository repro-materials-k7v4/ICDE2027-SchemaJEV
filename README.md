# SchemaJEV — ICDE 2027 reproducibility artifact

This repository accompanies the SchemaJEV manuscript. It contains the inputs and code for rebuilding the experiment figures, the exact training and evaluation data scope, and the core implementation of the 929,984-parameter PCA-T2L student.

## Rebuild the reported figures

Use Python 3.13. The figures are generated from the committed aggregate results; this step does not require a GPU, raw data, or model weights.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
MPLBACKEND=Agg python figures/code/build.py
```

Open `figures/paper_figures.pdf`. Individual figures and tables are written under `figures/figures/` and `figures/appendix/`. The [committed CSV inputs](figures/data/) contain the current per-seed results for the fixed evaluation descriptions. The same command also rebuilds the three additional experiments under `figures/new_experiments/figures/`.

### Final evaluation descriptions

The [evaluation catalog](data/eval_views.json) contains the final 18 descriptions used for the released results, together with their per-description SHA-256 fingerprints. [Evaluation-input provenance](data/eval_conditions_provenance.json) specifies the canonical JSON hashing rule and the ordered view/mask contract. Verify these inputs without a GPU:

```sh
python -m schemajev.evaluation_inputs
```

Use the final catalog when integrating an evaluation runner:

```python
from schemajev.evaluation_inputs import load_evaluation_conditions

conditions = load_evaluation_conditions()
condition = conditions[view_id]
# Pass this context/instruction object to the native conditioning API.
text = model.conditioning_text(condition)
```

View IDs index the catalog; they are not concatenated into the model's description. Training descriptions remain in `data/train_conditions.json`. The evaluation description map is fixed across the compared description-conditioned methods; record masks, option order and score bins are unchanged. The bare-plus-description control prepends its catalog description to the query. The description-effect figures use paired seeds 43 and 44. The wrong-type map changes five views and leaves the other 13 unchanged; the other controlled corruptions cover all 18 views. Only aggregate/per-seed intervention results are included alongside the final evaluation catalog.

### Three additional experiments

The [additional experiment directory](figures/new_experiments/) contains the plotting code, per-seed scalar CSVs, and runtime measurements. Rebuild it separately with:

```sh
python figures/new_experiments/plot_new_mechanisms.py \
  --data figures/new_experiments/data \
  --output figures/new_experiments/figures
```

- **Adaptation strength:** scale all decoded B factors by `s`, retaining A, so the effective update is multiplied by `s`. The grid is 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5 and 2. Nonzero settings use seeds 42–44; the independently measured zero-B control has n=1. This is a post-training strength sweep, separate from the training balance coefficient and KD mixing ratio.
- **Injection-site ablation:** retain the 88 encoder sites, the eight Transformer decision-layer sites, or all 96 sites at the same trained checkpoint. The frozen description encoder is unchanged. The final scalar scorer has no LoRA. The trained conditions have n=3; the bare control has n=1.
- **Runtime and resources:** the fixed seed-43 checkpoint is measured on one A800. Three separate process starts measure cold response including checkpoint SHA/I/O, restore, and numerical validation. Resident measurements use all 18 tasks, 10 warmups and 50 timing repetitions per cache mode. The native 4,096-token packing budget permits an oversized singleton: SCOTUS uses one 6,105-padded-token record; the other 17 tasks use four records and at most 1,800 padded tokens. Timing repetitions are not model seeds; standalone stage times are not summed. File sizes distinguish the student from the shared basis, frozen base and teacher library. Teacher-training compute is separate.

[Adapter transformation helpers](schemajev/mechanisms.py) and [runtime measurement helpers](schemajev/runtime_benchmark.py) expose the integration APIs. They require the native model/runtime to execute an evaluation; the figure commands above use only committed scalars.

## Download and prepare data

- [Training data](data/train_views.csv): the 60 task views used to train the default teacher library, 20 each for choice, binary, and score. Open the `source_url` for each view and follow the upstream download instructions and terms. Hugging Face dataset repositories can be downloaded with `hf download OWNER/NAME --repo-type dataset --local-dir work/raw/NAME`; for GitHub, Zenodo and dataset homepages use their published release downloads. Where multiple views share a source, download it once and prepare each typed view separately.
- [Evaluation data](data/eval_views.json): the 18 task views in Table I (12 ID held-out views and six OOD task-transfer views, 30,046 records total). Each entry gives the source, fixed description, cohort, record count, and mask hash. Amazon Reviews binary and score are separate views. OOD indicates transfer between task views and does not establish that underlying corpora are disjoint.
- [Training descriptions](data/train_conditions.json) contain the actual B_NO_ID generator inputs used by the main run. Dataset IDs index the JSON entries, while the model receives only their context and instruction values. Those values omit the explicit dataset-ID prefix. The FED score description retains the sentence “FED automatic-model scores are not labels.” [Description provenance](data/train_conditions_provenance.json) records the fixed source-map hash. [Expanded-basis teachers](data/basis_teachers.json) lists the additional archived adapters used for the N64/N128 appendix conditions, including their member training views and factor-file hashes.

Download the released backbone and its description encoder from the same pinned revision:

```sh
hf download convaiinnovations/laya-multilingual \
  --revision e4e9ddf21a7b1903b7acffd8814ad4307bf63a67 \
  --local-dir work/laya
```

Use [backbone.json](data/backbone.json) to verify the downloaded files. The scorer and frozen description encoder use the same checkpoint. Description features are masked means of the encoder's final token states, with adapters and memory tokens disabled.

### Select records and verify the split

1. Prepare the canonical LAYA training index for the 60 listed typed views, retaining the original TRAIN occurrences, source order, and canonical input digests. The source has 159,146 occurrences. Keep identical inputs in the same partition, including when they appear in more than one view.
2. Run the split below. The seed is 514; assign a digest to DEV if the integer value of SHA-256 over `514|H2-global-exact-input-FIT80-DEV20-v1|<canonical_input_sha256>` is below `floor(2^256/5)`; otherwise assign it to FIT. The expected result is 127,241 FIT and 31,905 DEV occurrences. The script checks per-view counts and preserves the original course order within each partition.
3. Use the fixed held-out and task-transfer masks from the paper for evaluation. Preserve the original option order and score bins, with the same masks for every method. Verify the prepared evaluation index before scoring.

```sh
python -m schemajev.data split work/canonical_train.jsonl --output work/split
python -m schemajev.data check-panel work/canonical_eval.jsonl
```

The training index uses one JSON object per record (or original windows with an `entries` array). Required fields: `dataset_id`, zero-based `row_index`, `native_record_id`, `task_type`, `canonical_input_sha256`. The evaluation index uses one object per record in panel order, with `dataset_id`, `row_index`, `canonical_record_id`, `canonical_input_sha256`, `input_ids_sha256`, and `option_mapping`. These are prepared LAYA indices, not the raw downloads. Keep the original digests and option metadata: computing a hash of arbitrary raw JSON will not reproduce the canonical identity.

The data catalog points to upstream downloads and does not redistribute records. Exact row recovery from current upstream releases is not guaranteed: this repository does not include the original dataset-specific LAYA adapters, raw snapshot manifests, held-out index bodies, or teacher/student weight files. The mask hashes verify identities; they cannot regenerate a mask. As packaged, the artifact rebuilds figures and documents/verifies the training protocol; running the original end-to-end GPU experiments requires the native LAYA runtime and canonical inputs.

## Training and checkpoints

- **LoRA:** rank 64, unit scale, 96 targets. In the 22 encoder layers inject into `attn.Wqkv`, `attn.Wo`, `mlp.Wi`, and `mlp.Wo`. In each of two decision-head layers inject into `self_attn.in_proj_weight`, `self_attn.out_proj`, `linear1`, and `linear2`. The native implementation stores factors as `A[d_in,r]`, `B[r,d_out]`.
- **Teachers:** one rank-64 adapter per training view, 200 updates, batch 80, seed 42, Gaussian A (SD 0.02), zero B. AdamW with betas (0.9, 0.999), epsilon 1e-8, no weight decay and gradient clipping at 1. Warm up to 3e-4 over 10 updates; after that, update `t` uses `3e-4 * (201 - t) / 190`. Select from steps 25, 50, …, 200 using that teacher's DEV utility (choice accuracy, binary composite, or 1−NMAE), then lower NLL, then earlier step. The selected teacher must improve over the bare backbone. The training catalog records the chosen steps and weight fingerprints. [teacher_selection.py](schemajev/teacher_selection.py) implements the fixed-mask selection.
- **Default PCA basis:** [default_basis.json](data/default_basis.json) records the original teacher order, prior seed, and scaling. The reference code fits the uncentered native A/B factor rows per layer, scales directions by `S/sqrt(60)`, and uses a separate fixed teacher-mixture prior. See [basis_math.py](schemajev/basis_math.py) and [released_pca_model.py](schemajev/released_pca_model.py).
- **Main student:** K=60, no KD and no balance penalty; 1,000 reconstruction updates, then 200 task CE updates. The frozen backbone and PCA decoder are kept fixed. Each phase starts a fresh AdamW optimizer with 10 warmup updates and gradient clipping at 1. Reconstruction head/trunk LR: 1e-2 / 3e-4; CE head/trunk LR: 3e-4 / 3e-5. The main row uses fixed `CE200.pt` for seeds 42, 43, and 44; no test-based checkpoint selection. `schemajev/` contains the native generator, frozen feature encoding, factor loss, CE course and training loops. The generator and loops are integration modules with runtime arguments, not standalone command-line trainers.
- **Reporting:** report choice accuracy, binary (accuracy + Yes-F1 + balanced accuracy)/3, score NMAE, and native NLL as specified. Macro means give each task view within a cohort/type equal weight. Three-run error bars are sample SD across seeds, not significance tests. Training time separates reconstruction and CE, and excludes teacher training.

Dataset and backbone terms are determined by their upstream releases.

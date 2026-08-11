# Prompt, Retrieval, and Multimodal Cues for Emotion Recognition in Conversation

This repository extends the PRC-Emo framework for Emotion Recognition in Conversation (ERC) with multimodal cue verbalization, multimodal LLM training, and embedding-based classification experiments. The codebase keeps the original text-centric PRC-Emo pipeline, then studies whether deterministic acoustic and visual cues, active-speaker face crops, and direct multimodal inputs improve utterance-level emotion classification.

The main ingredients used across experiments are:

- conversation context
- speaker description
- explicit emotion interpretation
- implicit emotion interpretation
- retrieval of similar emotional examples
- OpenSMILE-based audio cue descriptions
- OpenFace-based visual cue descriptions
- Qwen2.5-Omni multimodal inputs
- embedding-based classifiers over prompt-constructed inputs

In the command examples below, use `<HPC_ROOT>` as the machine-specific absolute prefix for your environment. Keep the relative tail of each path unchanged, for example `<HPC_ROOT>/PRC-Emo/models/qwen_3_8b` or `<HPC_ROOT>/dataset/MELD_audio/train`.

## Overview

The repository supports four main experiment families:

1. PRC-Emo-style prompt-based LLM fine-tuning.
2. Prompt-based LLM fine-tuning with deterministic OpenSMILE and OpenFace cue descriptions.
3. Qwen2.5-Omni multimodal training with video, audio, and active-speaker face-crop inputs.
4. Embedding-based classifiers using Qwen3-Embedding or VideoMAE representations.

The prompt assembly logic is centered in `src/reformat_data_ft_llm_combine.py`. Prompt sections are switched on and off dynamically with training flags such as:

- `--include_conversation_context`
- `--include_speaker_description`
- `--include_explicit_emotion`
- `--include_implicit_emotion`
- `--include_reference_similar_emotions`
- `--include_audio_description`
- `--include_visual_description`

## Repository Layout

The most relevant top-level directories are:

- `data/`: dataset JSONs, generated cue files, and preprocessed JSONL prompt files.
- `src/`: training code, prompt construction, cue extraction, and utility modules.
- `scripts/`: runnable SLURM/bash entrypoints for the main experiments.
- `models/`: local base model checkpoints used by the experiments.
- `logs/`: SLURM stdout/stderr logs from the bash scripts.
- `finetuned_llm/`: text or text+cue LoRA fine-tuning outputs from `src/ft_llm_cl_original.py`.
- `finetuned_llm_video_image/`: Qwen2.5-Omni multimodal outputs from `src/ft_llm_qwen25_omni_modalities.py`.
- `mlp/`: Qwen3-Embedding + MLP checkpoints and optional LoRA adapters.
- `videomae/`: VideoMAE experiment checkpoints and saved encoder weights.
- `wavlm/`: WavLM experiment checkpoints.
- `analysis/embed_analysis/`: saved PCA/t-SNE/UMAP embedding visualizations.

## Environment

The original project setup is:

```bash
conda create --prefix=./env_py38 python=3.9
conda activate ./env_py38
pip install -r requirements.txt
```

In practice, the provided scripts activate one of the repository-local environments:

```bash
source prc-emo-env/bin/activate
source prc-emo-omni-env/bin/activate
```

Use `prc-emo-env` for the standard prompt, embedding, and VideoMAE experiments, and `prc-emo-omni-env` for Qwen2.5-Omni.

## Data Preparation

### 1. Inherited PRC-Emo Components

These steps are reused by the newer experiments.

Generate explicit and implicit emotion interpretations:

```bash
python src/llm_emotion_extract_v2.py
```

Generate speaker descriptions:

```bash
python src/llm_bio_extract_v2.py
```

Build the retrieval library:

```bash
python src/get_rag_final.py
```

Typical generated artifacts are one-liners:

- implicit/explicit interpretations: `data/{meld,iemocap}.{train,valid,test}_ImplicitEmotion_V3_qwen_3_14b.json`
- speaker descriptions: `data/{meld,iemocap}.{train,valid,test}_spdescV6_qwen_3_14b.json`
- retrieval library: `data/Emotion_Retrieval_Library.json`

### 2. OpenSMILE Audio Cue Preparation

For MELD, extract utterance-level eGeMAPSv02 functionals:

```bash
python opensmile_ext.py --audio_dir /path/to/MELD_audio/train --output_csv opensmile_train_features.csv
python opensmile_ext.py --audio_dir /path/to/MELD_audio/dev --output_csv opensmile_dev_features.csv
python opensmile_ext.py --audio_dir /path/to/MELD_audio/test --output_csv opensmile_test_features.csv
```

Convert those functionals into text descriptions:

```bash
python opensmile_feat_to_text.py --input <HPC_ROOT>/PRC-Emo/opensmile_train_features.csv --split train
python opensmile_feat_to_text.py --input <HPC_ROOT>/PRC-Emo/opensmile_dev_features.csv --split dev
python opensmile_feat_to_text.py --input <HPC_ROOT>/PRC-Emo/opensmile_test_features.csv --split test
```

The converter writes root-level files, which should be kept under `data/` for training:

- `data/opensmile_{train,dev,test}_features_audio_descriptions.json`

For IEMOCAP, run utterance-aligned extraction first:

```bash
python iemocap_opensmile_ext.py \
  --iemocap_root <HPC_ROOT>/dataset/IEMOCAP_full_release \
  --valid_json <HPC_ROOT>/PRC-Emo/data/iemocap.valid.json \
  --output_dir <HPC_ROOT>/PRC-Emo \
  --save_log
```

Then verbalize those features:

```bash
python iemocap_opensmile_feat_to_text.py --input <HPC_ROOT>/PRC-Emo/iemocap_train_opensmile_utterance_wise.csv --split train
python iemocap_opensmile_feat_to_text.py --input <HPC_ROOT>/PRC-Emo/iemocap_valid_opensmile_utterance_wise.csv --split valid
python iemocap_opensmile_feat_to_text.py --input <HPC_ROOT>/PRC-Emo/iemocap_test_opensmile_utterance_wise.csv --split test
```

Expected training-side files:

- `data/iemocap_{train,valid,test}_opensmile_utterance_wise_audio_descriptions.json`

### 3. OpenFace Visual Cue Preparation

For IEMOCAP, convert per-utterance OpenFace AU CSVs into symbolic facial descriptions:

```bash
python iemocap_convert_aus.py --input <HPC_ROOT>/PRC-Emo/iemocap_train_opensmile_utterance_wise.csv --split train
python iemocap_convert_aus.py --input <HPC_ROOT>/PRC-Emo/iemocap_valid_opensmile_utterance_wise.csv --split valid
python iemocap_convert_aus.py --input <HPC_ROOT>/PRC-Emo/iemocap_test_opensmile_utterance_wise.csv --split test
```

The prompt loader expects:

- `data/iemocap_{train,valid,test}_openface_utterance_wise_visual_descriptions.json`

For MELD, the code directly loads precomputed split files:

- `data/train_dialogue_visual_expressions.json`
- `data/dev_dialogue_visual_expressions.json`
- `data/test_dialogue_visual_expressions.json`

## Dynamic Prompt Construction

Prompt construction is driven by `src/reformat_data_ft_llm_combine.py`. The default naming pattern for generated JSONL files is:

```text
data/{dataset}.{split}.{kshot}shot_w{window}_{prompt_type}_{extract_llm_id}{suffix}.jsonl
```

Examples:

- `data/meld.train.0shot_w5_ImplicitEmotion_V3_qwen_3_14b.jsonl`
- `data/meld.train.0shot_w5_ImplicitEmotion_V3_qwen_3_14b_Aud_Vis.jsonl`
- `data/meld.train.0shot_w5_ImplicitEmotion_V3_qwen_3_14b_image_Aud_Vis_Omni.jsonl`
- `data/meld.train.0shot_w5_ImplicitEmotion_V3_qwen_3_14b_video_image_Aud_Vis_Omni.jsonl`

If you want training to regenerate JSONL data, pass `--re_gen_data` together with the desired cue flags.

Example: text + retrieval + speaker + OpenSMILE + OpenFace prompt generation through the training entrypoint:

```bash
python src/ft_llm_cl_original.py \
  --do_train \
  --re_gen_data \
  --base_model_id <HPC_ROOT>/PRC-Emo/models/qwen_3_8b \
  --ft_model_id debug_run \
  --data_name meld \
  --data_folder ./data \
  --output_folder ./finetuned_llm \
  --prompting_type ImplicitEmotion_V3 \
  --extract_prompting_llm_id qwen_3_14b \
  --window 5 \
  --kshot 0 \
  --epoch 1 \
  --max_seq_len 2048 \
  --include_conversation_context \
  --include_speaker_description \
  --include_explicit_emotion \
  --include_implicit_emotion \
  --include_reference_similar_emotions \
  --include_audio_description \
  --include_visual_description \
  --suffix "_Aud_Vis"
```

## Main Experiments

### 1. Prompt-Based LLM Fine-Tuning

The primary bash entrypoint is:

```bash
bash scripts/train_llm.sh
```

This script:

- activates `prc-emo-env`
- trains `src/ft_llm_cl_original.py`
- uses MELD by default
- enables curriculum learning
- includes context, speaker description, explicit/implicit interpretations, retrieval, OpenSMILE audio descriptions, and OpenFace visual descriptions
- saves outputs to `./finetuned_llm/`

Equivalent core CLI:

```bash
python src/ft_llm_cl_original.py \
  --do_train --do_eval_dev --do_eval_test \
  --curriculum --bucket_number 2 --curriculum_update_epochs 1 \
  --base_model_id <HPC_ROOT>/PRC-Emo/models/qwen_3_8b \
  --ft_model_id meld_qwen3_prompt \
  --data_name meld \
  --prompting_type ImplicitEmotion_V3 \
  --extract_prompting_llm_id qwen_3_14b \
  --window 5 --kshot 0 --epoch 4 --max_seq_len 2048 \
  --lr_scheduler linear --lr 3e-4 --lora_r 32 \
  --data_folder ./data/ --output_folder ./finetuned_llm/ \
  --include_conversation_context \
  --include_speaker_description \
  --include_explicit_emotion \
  --include_implicit_emotion \
  --include_reference_similar_emotions \
  --include_audio_description \
  --include_visual_description \
  --suffix "_Aud_Vis"
```

Model outputs are saved as curriculum phases plus a final full-finetune directory:

- `finetuned_llm/<ft_model_id>_phase_*`
- `finetuned_llm/<ft_model_id>_final_full_finetune`
- `finetuned_llm/<ft_model_id>/all_phase_test_results.json` at the base output prefix

### 2. Qwen3-Embedding + MLP

The main embedding-based script is:

```bash
bash scripts/train_llm_embed.sh
```

This launches `src/qwen3_embedding_mlp_emotion.py` on JSONL prompt files, with the current example using:

- Qwen3-Embedding-8B
- MELD
- context + audio description + visual description
- LoRA contrastive fine-tuning
- prototype learning
- embedding analysis

Outputs are saved to:

- `mlp/*.pt` for classifier checkpoints
- `mlp/*_lora_adapter/` when LoRA contrastive fine-tuning is enabled
- `analysis/embed_analysis/*.png` when `--embedding_analysis` is enabled

### 3. Qwen2.5-Omni Multimodal Training

The multimodal Qwen2.5-Omni entrypoint is:

```bash
bash scripts/train_qwen25_omni.sh
```

This script:

- activates `prc-emo-omni-env`
- trains `src/ft_llm_qwen25_omni_modalities.py`
- uses MELD
- enables curriculum learning
- loads `qwen_2_5_omni_7b`
- uses multimodal JSONL with `--video_input --image_input`
- uses `--preprocessed_data_suffix "_video_image_Aud_Vis_Omni.jsonl"`
- saves outputs to `./finetuned_llm_video_image/`

In the code, the multimodal inputs are interpreted as:

- `video`: video clip of the current target utterance
- `image`: face-cropped image of the active speaker
- `audio`: separately extracted audio clip of the current target utterance

The default MELD media directories are:

- video: `<HPC_ROOT>/dataset/MELD/MELD.Raw/{train_splits,dev_splits_complete,output_repeated_splits_test}`
- audio: `<HPC_ROOT>/dataset/MELD_audio/{train,dev,test}`
- face crops: `<HPC_ROOT>/dataset/MELD_face_crops/{train,dev,test}`

Equivalent multimodal training command:

```bash
python src/ft_llm_qwen25_omni_modalities.py \
  --do_train --do_eval_dev --do_eval_test \
  --curriculum --bucket_number 2 --curriculum_update_epochs 1 \
  --base_model_id <HPC_ROOT>/PRC-Emo/models/qwen_2_5_omni_7b \
  --ft_model_id meld_qwen25_omni_video_image \
  --data_name meld \
  --data_folder ./data/ \
  --output_folder ./finetuned_llm_video_image/ \
  --prompting_type ImplicitEmotion_V3 \
  --extract_prompting_llm_id qwen_3_14b \
  --window 5 --kshot 0 --epoch 4 --max_seq_len 2048 \
  --lr_scheduler linear --lr 5e-5 --lora_r 32 \
  --video_input --image_input \
  --skip_invalid_media \
  --preprocessed_data_suffix "_video_image_Aud_Vis_Omni.jsonl"
```

Outputs are saved as:

- `finetuned_llm_video_image/<ft_model_id>_phase_*`
- `finetuned_llm_video_image/<ft_model_id>_final_full_finetune`
- `finetuned_llm_video_image/<ft_model_id>/all_phase_test_results.json` at the base output prefix

### 4. Active-Speaker Face-Crop Experiment

The face-crop experiment is the image-input branch of the Qwen2.5-Omni pipeline. It uses the active speaker crop instead of the full video clip.

To train the image-only variant, run the same Omni training script but switch the modality flags and suffix:

```bash
python src/ft_llm_qwen25_omni_modalities.py \
  --do_train --do_eval_dev --do_eval_test \
  --curriculum --bucket_number 2 --curriculum_update_epochs 1 \
  --base_model_id <HPC_ROOT>/PRC-Emo/models/qwen_2_5_omni_7b \
  --ft_model_id meld_qwen25_omni_image \
  --data_name meld \
  --data_folder ./data/ \
  --output_folder ./finetuned_llm_image/ \
  --prompting_type ImplicitEmotion_V3 \
  --extract_prompting_llm_id qwen_3_14b \
  --window 5 --kshot 0 --epoch 4 --max_seq_len 2048 \
  --lr_scheduler linear --lr 5e-5 --lora_r 32 \
  --image_input \
  --skip_invalid_media \
  --preprocessed_data_suffix "_image_Aud_Vis_Omni.jsonl"
```

If you want image+audio instead, use:

- `--image_input --audio_input`
- `--preprocessed_data_suffix "_image_audio_Aud_Vis_Omni.jsonl"`

The face-crop directories used by the prompt generator are:

- `--meld_train_image_dir <HPC_ROOT>/dataset/MELD_face_crops/train`
- `--meld_valid_image_dir <HPC_ROOT>/dataset/MELD_face_crops/dev`
- `--meld_test_image_dir <HPC_ROOT>/dataset/MELD_face_crops/test`

### 5. VideoMAE Experiment

The VideoMAE-based visual experiment is launched with:

```bash
bash scripts/train_videomae.sh
```

This script runs `src/videomaev2_base_mlp_emotion.py` and currently uses:

- MELD raw video clips
- VideoMAEv2-Base embeddings
- prototype learning
- embedding analysis

Outputs are saved to:

- `videomae/*.pt` for classifier checkpoints
- `videomae/*_videomae_model/` when the encoder weights are saved
- `videomae/*_lora_adapter/` if LoRA contrastive fine-tuning is enabled later
- `analysis/embed_analysis/*.png` for visualization outputs

### 6. WavLM Experiment

The WavLM-based audio experiment is launched with:

```bash
bash scripts/train_wavlm.sh
```

This script runs `src/wavlm_large_mlp_emotion.py` and currently uses:

- IEMOCAP by default
- WavLM-Large embeddings
- utterance-level audio paths from the IEMOCAP metadata CSVs
- embedding analysis

The default base model path used by the trainer is:

- `<HPC_ROOT>/PRC-Emo/models/wavlm_large`

Equivalent direct command:

```bash
python src/wavlm_large_mlp_emotion.py \
  --mode train_eval \
  --dataset iemocap \
  --model_path <HPC_ROOT>/PRC-Emo/models/wavlm_large \
  --seed 42 \
  --epochs 50 \
  --embedding_analysis
```

Outputs are saved to:

- `wavlm/*.pt` for classifier checkpoints
- `wavlm/*_lora_adapter/` if LoRA contrastive fine-tuning is enabled
- `analysis/embed_analysis/*.png` for visualization outputs

## Bash Scripts

The scripts in `scripts/` are the easiest way to reproduce the main settings because they already bundle:

- environment activation
- SLURM resource requests
- default model paths
- dataset paths
- output folders
- experiment-specific flags

Most scripts follow the same pattern:

1. create `logs/`
2. activate the required environment
3. define experiment hyperparameters
4. launch one Python entrypoint

The most relevant ones are:

- `scripts/train_llm.sh`: prompt-based LLM fine-tuning with deterministic audio and visual cue descriptions
- `scripts/train_llm_embed.sh`: Qwen3-Embedding + MLP
- `scripts/train_qwen25_omni.sh`: Qwen2.5-Omni multimodal training with video + active-speaker face crops
- `scripts/train_videomae.sh`: VideoMAE experiment over utterance video clips
- `scripts/train_wavlm.sh`: WavLM experiment over utterance audio clips
- `scripts/eval_llm.sh`: evaluation examples for multimodal Qwen2.5-Omni checkpoints

If you want to start a new variant, copying the closest existing script and only changing:

- `MODEL_ID`
- `output_folder`
- media flags such as `--video_input`, `--image_input`, `--audio_input`
- `preprocessed_data_suffix`

is usually safer than building the command from scratch.

## Where Outputs Go

This is the short reference version:

- prompt JSONL files: `data/*.jsonl`
- SLURM logs: `logs/`
- text/prompt LLM checkpoints: `finetuned_llm/` or the `--output_folder` you choose
- Omni multimodal checkpoints: `finetuned_llm_video_image/` or the `--output_folder` you choose
- Qwen3-Embedding classifiers and adapters: `mlp/`
- VideoMAE classifiers and saved encoders: `videomae/`
- WavLM classifiers and saved encoders: `wavlm/`
- embedding visualizations: `analysis/embed_analysis/`

## Notes

- The Qwen2.5-Omni modality-selection script is currently wired for MELD.
- The face-crop experiment is implemented through the Omni `image_input` pathway rather than a separate standalone trainer.
- `--include_llm_aud_vis_desc` is a different branch from the deterministic OpenSMILE/OpenFace cue setup and should not be mixed with `--include_audio_description` or `--include_visual_description`.
- Several preprocessing utilities write outputs to the repository root first; the training loaders typically expect the final cue JSON files under `data/`.

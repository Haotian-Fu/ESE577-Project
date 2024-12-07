# ESE577 Course Porject
> Name: Haotian Fu \
> Date: Dec. 6th, 2024

## Code Strcture

### Data Preprocessing
Dataset is `QA_pair.xlsx` and the code to preprocess it is `data_preprocess.py`. The output of the python script is `QA_dataset.json`.

### Finetune
Finetuning codes are in `finetune.py` where:
```
 # Quantization configuration for memory efficiency with CPU offloading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                     # Enable 4-bit precision
        bnb_4bit_quant_type="nf4",             # Use normal floating-point 4-bit quantization
        bnb_4bit_use_double_quant=True,        # Use double quantization
        bnb_4bit_compute_dtype=torch.float16,  # Set compute dtype
        llm_int8_enable_fp32_cpu_offload=True  # Enable FP32 CPU offloading
    )

 # ----------------- Configure LoRA -------------
    lora_config = LoraConfig(
        r=8,  # Reduced LoRA rank from 16 to 8
        lora_alpha=16,  # Adjusted scaling factor accordingly
        target_modules=["self_attn.q_proj", "self_attn.v_proj"],  # Adjust based on model architecture
        lora_dropout=0.1,  # Slightly increased dropout for better generalization
        bias="none",  # Do not modify biases
        task_type="CAUSAL_LM",  # Task type for causal language modeling
    )

# ------------------ Define Training Arguments --------------------
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model_epoch3",
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        logging_steps=1,              # Log every step to record loss at each step
        save_steps=500,               # Save checkpoint every 500 steps
        learning_rate=2e-4,           # Reduced learning rate for stability
        per_device_train_batch_size=2,  # Reduced batch size for GPU memory constraints
        gradient_accumulation_steps=2,  # Effective batch size = 2 * 2 = 4
        num_train_epochs=3,           # Increased epochs from 3 to 6 for better performance
        fp16=True,                    # Enable mixed-precision training
        save_total_limit=2,           # Keep only the last 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Specify the metric for best model
        optim="adamw_torch",
        weight_decay=0.01,
        report_to=["tensorboard"],           # Disable reporting to external services
        save_strategy="epoch",
        save_on_each_node=True,
        dataloader_num_workers=2,     # Reduce number of workers
        run_name="fine_tuned_mistral_lora",
        seed=42,                      # For reproducibility
    )
```
Notice that `num_train_epochs=3` here is to define #epoch. In this implementation, #epoch = 3, 4, 6, 8.

### Evaluation
Base model evaluation is in `base_model.py` and finetuned model evaluation is in `finetuned_model.py`.

File I/O uses relative paths here but are still subject to change in different file structures.

The comparison of all model inferences is introduced in `compare_models.py`.

## Visualizations

All necessary visualizations are attached in the codes in place. Several pfigures are renamed manually and stored in `./exp_result` directory.

No separate and individual visualization codes available here.

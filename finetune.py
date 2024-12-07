# fine_tuned_model.py

import os
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.trainer_callback import TrainerCallback
import json
import matplotlib.pyplot as plt
import nlpaug.augmenter.word as naw
import logging

# ---------------------------- Setup Logging ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------- Preprocessing Functions ----------------------------
def preprocess_function(examples, tokenizer, max_length=256):
    """
    Tokenizes the input question and answer pairs and sets up labels for loss computation.
    """
    questions = examples["question"]
    answers = examples["answer"]
    # Format: <s>[INST] Question [/INST] Answer
    inputs = [
        f"<s>[INST] {q} [/INST] {a}" for q, a in zip(questions, answers)
    ]
    tokenized = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=max_length,  # Reduced from 512 to 256 for faster training
    )
    
    # Set labels equal to input_ids for loss computation
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def augment_data(df, augmenter, augmentation_factor=1):
    """
    Augments the dataset by applying synonym replacement.
    Reduced augmentation_factor to limit training time.
    """
    augmented_data = []
    for _, row in df.iterrows():
        augmented_question = augmenter.augment(row['question'])
        augmented_answer = augmenter.augment(row['answer'])
        
        # Ensure augmented data is string
        augmented_question = ' '.join(augmented_question) if isinstance(augmented_question, list) else str(augmented_question)
        augmented_answer = ' '.join(augmented_answer) if isinstance(augmented_answer, list) else str(augmented_answer)
        
        augmented_data.append({
            'question': augmented_question,
            'answer': augmented_answer
        })
    augmented_df = pd.DataFrame(augmented_data)
    return pd.concat([df, augmented_df], ignore_index=True)

def plot_loss(training_history, output_dir):
    """
    Plots training and validation loss curves.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot Training Loss
    if training_history.get('training_loss') and training_history.get('training_steps'):
        plt.plot(training_history['training_steps'], training_history['training_loss'], label='Training Loss')
    
    # Plot Validation Loss
    if training_history.get('validation_loss') and training_history.get('validation_steps'):
        plt.plot(training_history['validation_steps'], training_history['validation_loss'], label='Validation Loss')
    
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

# ---------------------------- Custom Callback ----------------------------
class LossRecorderCallback(TrainerCallback):
    """
    Custom callback to record training and validation loss along with step numbers.
    """
    def __init__(self):
        self.training_steps = []
        self.training_loss = []
        self.validation_steps = []
        self.validation_loss = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Debugging: Print the logs received
            print(f"Log at step {state.global_step}: {logs}")  # Debugging line

            # Record Training Loss
            if 'loss' in logs:
                self.training_steps.append(state.global_step)
                self.training_loss.append(logs['loss'])
            
            # Record Validation Loss
            if 'eval_loss' in logs:
                self.validation_steps.append(state.global_step)
                self.validation_loss.append(logs['eval_loss'])

    def on_train_end(self, args, state, control, **kwargs):
        """
        Optionally, you can perform actions when training ends.
        """
        pass

# ---------------------------- Main Training Function ----------------------------
def main():
    # ---------------------------- Load and Prepare Data ----------------------------
    logger.info("Loading JSON data...")
    with open("QA_dataset.json", "r", encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    df = pd.DataFrame(qa_pairs)
    
    # Validate DataFrame columns
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("The JSON file must contain 'question' and 'answer' keys.")
    
    logger.info(f"Original dataset size: {len(df)}")
    
    # Initialize the augmenter (Synonym Replacement)
    augmenter = naw.SynonymAug(aug_src='wordnet')
    
    # Apply data augmentation
    logger.info("Augmenting data...")
    df = augment_data(df, augmenter, augmentation_factor=1)  # Reduced augmentation
    logger.info(f"Augmented dataset size: {len(df)}")
    
    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Ensure all entries are strings
    df['question'] = df['question'].astype(str)
    df['answer'] = df['answer'].astype(str)
    
    # ---------------------------- Create Hugging Face Datasets ----------------------------
    split_ratio = 0.8
    train_size = int(len(df) * split_ratio)
    dataset_train = Dataset.from_pandas(df.iloc[:train_size].reset_index(drop=True))
    dataset_val = Dataset.from_pandas(df.iloc[train_size:].reset_index(drop=True))
    dataset = DatasetDict({"train": dataset_train, "test": dataset_val})
    
    logger.info(f"Training dataset size: {len(dataset_train)}")
    logger.info(f"Validation dataset size: {len(dataset_val)}")
    
    # ---------------------------- Load Model and Tokenizer ----------------------------
    base_model = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Quantization configuration for memory efficiency with CPU offloading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                     # Enable 4-bit precision
        bnb_4bit_quant_type="nf4",             # Use normal floating-point 4-bit quantization
        bnb_4bit_use_double_quant=True,        # Use double quantization
        bnb_4bit_compute_dtype=torch.float16,  # Set compute dtype
        llm_int8_enable_fp32_cpu_offload=True  # Enable FP32 CPU offloading
    )
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning("Pad token was not set. Using eos_token as pad_token.")
    
    logger.info("Loading model with quantization...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        logger.info("Model loaded successfully.\n")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Disable cache due to gradient checkpointing compatibility
    model.config.use_cache = False
    
    # Prepare the model for k-bit training
    logger.info("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    # ---------------------------- Configure LoRA ----------------------------
    lora_config = LoraConfig(
        r=8,  # Reduced LoRA rank from 16 to 8
        lora_alpha=16,  # Adjusted scaling factor accordingly
        target_modules=["self_attn.q_proj", "self_attn.v_proj"],  # Adjust based on model architecture
        lora_dropout=0.1,  # Slightly increased dropout for better generalization
        bias="none",  # Do not modify biases
        task_type="CAUSAL_LM",  # Task type for causal language modeling
    )
    
    logger.info("Applying LoRA configuration...")
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters to verify LoRA is applied
    model.print_trainable_parameters()
    
    # ---------------------------- Tokenize Datasets ----------------------------
    logger.info("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=["question", "answer"],
    )
    
    # ---------------------------- Define Training Arguments ----------------------------
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
    
    # ---------------------------- Initialize Trainer ----------------------------
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # Early stopping
    )
    
    # Initialize loss recorder callback
    loss_recorder = LossRecorderCallback()
    trainer.add_callback(loss_recorder)
    
    # ---------------------------- Start Training ----------------------------
    logger.info("Starting training...")
    trainer.train()
    
    # ---------------------------- Save the Fine-Tuned Model ----------------------------
    logger.info("Saving the fine-tuned model...")
    model.save_pretrained("./fine_tuned_model_epoch3")
    tokenizer.save_pretrained("./fine_tuned_model_epoch3")
    
    # ---------------------------- Plot Loss Curves ----------------------------
    logger.info("Plotting loss curves...")
    plot_loss(
        {
            'training_steps': loss_recorder.training_steps,
            'training_loss': loss_recorder.training_loss,
            'validation_steps': loss_recorder.validation_steps,
            'validation_loss': loss_recorder.validation_loss
        },
        "./fine_tuned_model_epoch3"
    )
    
    # ---------------------------- Save Loss Data to CSV ----------------------------
    logger.info("Saving loss data to CSV...")
    # Create separate DataFrames for training and validation loss
    training_loss_df = pd.DataFrame({
        'step': loss_recorder.training_steps,
        'training_loss': loss_recorder.training_loss
    })
    
    validation_loss_df = pd.DataFrame({
        'step': loss_recorder.validation_steps,
        'validation_loss': loss_recorder.validation_loss
    })
    
    # Merge the two DataFrames on the 'step' column
    loss_data = pd.merge(training_loss_df, validation_loss_df, on='step', how='outer').sort_values('step')
    
    # Define the output file path
    loss_csv_path = os.path.join("./fine_tuned_model_epoch3", "loss_data.csv")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(loss_csv_path), exist_ok=True)
    
    # Save to CSV
    loss_data.to_csv(loss_csv_path, index=False)
    
    logger.info(f"Loss data saved to {loss_csv_path}")
    logger.info("Fine-tuning complete!")

if __name__ == "__main__":
    main()

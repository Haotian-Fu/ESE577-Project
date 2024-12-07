# evaluate_base_model.py

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import matplotlib.pyplot as plt
import os

def main():
    # -------------------------------
    # Step 1: Check CUDA Availability
    # -------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Use GPU 0
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detected and available: {gpu_name}\n")
    else:
        device = torch.device("cpu")
        print("No GPU detected. Using CPU.\n")

    # -------------------------------
    # Step 2: Define Comparison Data
    # -------------------------------
    # Define questions and corresponding reference answers
    comparison_data = [
        {
            "question": "How does the course handle late submissions differently from regular grading?",
            "reference_answer": "A 20% penalty per day is applied, but students have three one-day extensions."
        },
        {
            "question": "Is 'Markov decision processes (MDPs)' part of this course?",
            "reference_answer": "Yes, they are included."
        },
        {
            "question": "What is the focus of ESE 577?",
            "reference_answer": "The focus of ESE 577 is on deep learning algorithms and their applications."
        },
        {
            "question": "What percentage of the final grade is determined by the midterm?",
            "reference_answer": "The midterm will account for 20% of the final grade."
        },
        {
            "question": "How will the quiz grades be calculated?",
            "reference_answer": "Only the best 10 quiz grades will be counted for each student."
        },
    ]

    # -------------------------------
    # Step 3: Initialize ROUGE Scorer
    # -------------------------------
    # Initialize ROUGE scorer with desired metrics
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smooth_fn = SmoothingFunction()

    # -------------------------------
    # Step 4: Load Base Model and Tokenizer with Quantization
    # -------------------------------
    # Define base model name (Hugging Face Hub) or local path
    base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # Replace with your base model name if different

    # Quantization configuration for memory efficiency with CPU offloading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                     # Enable 4-bit precision
        bnb_4bit_quant_type="nf4",             # Use normal floating-point 4-bit quantization
        bnb_4bit_use_double_quant=True,        # Use double quantization
        bnb_4bit_compute_dtype=torch.float16,  # Set compute dtype
        llm_int8_enable_fp32_cpu_offload=True  # Enable FP32 CPU offloading
    )

    print("Loading Base Model...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",  # Automatic device mapping with CPU offloading
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True  # Required if the model uses custom code
        )
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        base_model.eval()  # Set model to evaluation mode
        print("Base Model loaded successfully.\n")
    except Exception as e:
        print(f"Error loading Base Model: {e}")
        return

    # -------------------------------
    # Step 5: Adjust Tokenizer and Model for pad_token
    # -------------------------------
    # Ensure the tokenizer has a distinct pad token by using eos_token if necessary
    if base_tokenizer.pad_token is None or base_tokenizer.pad_token == base_tokenizer.eos_token:
        # Use eos_token as pad_token
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_model.config.pad_token_id = base_tokenizer.pad_token_id
        print(f"Set pad token to eos_token: {base_tokenizer.pad_token}\n")
    else:
        print(f"Pad token already set to: {base_tokenizer.pad_token}\n")

    # -------------------------------
    # Step 6: Define Helper Functions
    # -------------------------------
    # Helper Function to Format Prompt
    def build_prompt(question):
        return f"<s>[INST] {question} [/INST]"

    # Function to Generate and Compare Answers
    def evaluate_model(model, tokenizer, comparison_data, model_name="Model"):
        results = []
        bleu_scores = []
        bert_scores = []
        for idx, data in enumerate(comparison_data, 1):
            question = data["question"]
            reference_answer = data["reference_answer"]
            prompt = build_prompt(question)

            try:
                # Tokenize input and generate answer
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],  # Pass attention_mask
                        max_length=100,          # Adjust max_length as needed
                        num_beams=5,             # Beam search for better quality
                        early_stopping=True
                    )
                generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            except Exception as e:
                print(f"Error generating answer for question '{question}': {e}")
                generated_answer = ""

            # Calculate ROUGE scores
            scores = scorer.score(reference_answer, generated_answer)
            
            # Calculate BLEU score
            reference_tokens = [reference_answer.split()]
            candidate_tokens = generated_answer.split()
            bleu = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smooth_fn.method4)
            bleu_scores.append(bleu)

            # Calculate BERTScore
            bert_precision, bert_recall, bert_f1 = bert_score(
                [generated_answer], [reference_answer], lang="en", device=device
            )
            bert_scores.append(bert_f1.mean().item())


            # Store results
            results.append({
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "rouge_scores": {
                    metric: {"precision": scores[metric].precision,
                             "recall": scores[metric].recall,
                             "fmeasure": scores[metric].fmeasure}
                    for metric in scores
                },
                "bleu_score": bleu,
                "bertscore_f1": bert_f1.mean().item()
            })

            # Print results
            print(f"\n--- {model_name} ---")
            print(f"Question {idx}: {question}")
            print(f"Reference Answer: {reference_answer}")
            print(f"Generated Answer: {generated_answer}")
            print("ROUGE Scores:")
            for rouge_metric, score in scores.items():
                print(f"  {rouge_metric}: Precision: {score.precision:.4f}, Recall: {score.recall:.4f}, F1: {score.fmeasure:.4f}")
            print("--------------------------\n")

        return results

    # Function to Plot ROUGE Scores for a Model
    def plot_model_rouge_scores(model_results, model_name="Model"):
        # Collect F1 scores
        rouge1_f1 = [r['rouge_scores']['rouge1']['fmeasure'] for r in model_results]
        rouge2_f1 = [r['rouge_scores']['rouge2']['fmeasure'] for r in model_results]
        rougel_f1 = [r['rouge_scores']['rougeL']['fmeasure'] for r in model_results]

        # Calculate average F1 scores
        avg_rouge1 = sum(rouge1_f1) / len(rouge1_f1) if rouge1_f1 else 0
        avg_rouge2 = sum(rouge2_f1) / len(rouge2_f1) if rouge2_f1 else 0
        avg_rougel = sum(rougel_f1) / len(rougel_f1) if rougel_f1 else 0

        scores = [avg_rouge1, avg_rouge2, avg_rougel]
        metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']

        plt.figure(figsize=(8, 5))
        bars = plt.bar(metrics, scores, color=['skyblue', 'lightgreen', 'salmon'])
        plt.xlabel('ROUGE Metrics')
        plt.ylabel('Average F1 Scores')
        plt.title(f'{model_name} Average ROUGE Scores')
        plt.ylim([0, 1])

        # Annotate bars with score values
        for bar, score in zip(bars, scores):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{score:.2f}", ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        # Ensure the output directory exists
        os.makedirs("./base_model", exist_ok=True)
        plt.savefig(os.path.join("./base_model", f"{model_name}_rouge_scores.png"))
        plt.close()
        print(f"ROUGE scores plot saved to './base_model/{model_name}_rouge_scores.png'\n")
        
    # Function to Plot BLEU and BERTScore
    def plot_bleu_and_bertscore(model_results, model_name="Model"):
        # Collect BLEU and BERTScore F1 scores from model results
        bleu_scores = [r['bleu_score'] for r in model_results]
        bert_scores = [r['bertscore_f1'] for r in model_results]
        
        # Calculate average scores
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        avg_bert_f1 = sum(bert_scores) / len(bert_scores) if bert_scores else 0

        # Plot BLEU and BERTScore
        plt.figure(figsize=(8, 5))
        metrics = ["BLEU", "BERTScore (F1)"]
        scores = [avg_bleu, avg_bert_f1]
        bars = plt.bar(metrics, scores, color=["skyblue", "salmon"])
        plt.xlabel("Metrics")
        plt.ylabel("Scores")
        plt.title(f"{model_name} Average BLEU and BERTScore")
        plt.ylim([0, 1])

        # Annotate bars with score values
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01, f"{score:.2f}", ha="center", va="bottom")

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join("./base_model", f"{model_name}_bleu_bertscore.png"))
        plt.close()
        print(f"BLEU and BERTScore plot saved to './base_model/{model_name}_bleu_bertscore.png'\n")


    # -------------------------------
    # Step 7: Evaluate Base Model
    # -------------------------------
    print("Evaluating Base Model...")
    base_results = evaluate_model(base_model, base_tokenizer, comparison_data, model_name="Base Model")

    # -------------------------------
    # Step 8: Save Results to JSON
    # -------------------------------
    final_results = {
        "base_model": base_results
    }

    output_path = "./base_model/base_model_comparison_with_rouge.json"
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(final_results, f, indent=4, ensure_ascii=False)
        print(f"Comparison results saved to {output_path}\n")
    except Exception as e:
        print(f"Error saving results to JSON: {e}\n")

    # -------------------------------
    # Step 9: Plot ROUGE Scores
    # -------------------------------
    print("Plotting ROUGE Scores for Base Model...")
    plot_model_rouge_scores(base_results, model_name="Base Model")
    print("Plotting BLEU and BERTScore for Base Model...")
    plot_bleu_and_bertscore(base_results, model_name="Fine-Tuned Model")

if __name__ == "__main__":
    main()

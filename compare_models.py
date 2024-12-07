import json
import matplotlib.pyplot as plt
import os

def load_results(file_path):
    """
    Load JSON results from a file.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_comparison(base_scores, fine_tuned_scores_list, fine_tuned_labels):
    """
    Plot comparison between base model and multiple fine-tuned models.
    """
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    
    # Extract base model F1 scores
    base_f1 = [
        base_scores['base_model'][0]['rouge_scores']['rouge1']['fmeasure'],
        base_scores['base_model'][1]['rouge_scores']['rouge2']['fmeasure'],
        base_scores['base_model'][2]['rouge_scores']['rougeL']['fmeasure']
    ]
    
    # Extract fine-tuned model F1 scores
    fine_tuned_f1_list = []
    for fine_tuned_scores in fine_tuned_scores_list:
        fine_tuned_f1 = [
            fine_tuned_scores['fine_tuned_model'][0]['rouge_scores']['rouge1']['fmeasure'],
            fine_tuned_scores['fine_tuned_model'][1]['rouge_scores']['rouge2']['fmeasure'],
            fine_tuned_scores['fine_tuned_model'][2]['rouge_scores']['rougeL']['fmeasure']
        ]
        fine_tuned_f1_list.append(fine_tuned_f1)
    
    x = range(len(metrics))
    width = 0.15  # Adjusted width for multiple models

    plt.figure(figsize=(12, 8))
    
    # Plot Base Model
    plt.bar([p - width * (len(fine_tuned_f1_list) / 2) for p in x], base_f1, width, label='Base Model', color='skyblue')

    # Plot Fine-Tuned Models
    for i, fine_tuned_f1 in enumerate(fine_tuned_f1_list):
        plt.bar([p - width * (len(fine_tuned_f1_list) / 2 - i - 1) for p in x], fine_tuned_f1, width, label=fine_tuned_labels[i], color=f'C{i}')
    
    plt.xlabel('ROUGE Metrics')
    plt.ylabel('F1 Scores')
    plt.title('ROUGE Score Comparison Between Base and Fine-Tuned Models')
    plt.xticks(x, metrics)
    plt.ylim([0, 1])
    plt.legend()

    # Annotate bars with score values
    for i in x:
        plt.text(i - width * (len(fine_tuned_f1_list) / 2), base_f1[i] + 0.01, f"{base_f1[i]:.2f}", ha='center', va='bottom', fontweight='bold')
        for j, fine_tuned_f1 in enumerate(fine_tuned_f1_list):
            plt.text(i - width * (len(fine_tuned_f1_list) / 2 - j - 1), fine_tuned_f1[i] + 0.01, f"{fine_tuned_f1[i]:.2f}", ha='center', va='bottom', fontweight='bold')

    # Show the plot
    plt.tight_layout()
    plt.show()

def save_responses_as_json(base_scores, fine_tuned_scores_list, fine_tuned_labels, output_file):
    """
    Save all responses from the base model and fine-tuned models into a JSON file.
    """
    combined_results = []
    for idx, base_item in enumerate(base_scores['base_model']):
        combined_entry = {
            "question": base_item['question'],
            "reference_answer": base_item['reference_answer'],
            "base_model_response": base_item['generated_answer'],
            "fine_tuned_model_responses": {
                fine_tuned_labels[model_idx]: fine_tuned_scores_list[model_idx]['fine_tuned_model'][idx]['generated_answer']
                for model_idx in range(len(fine_tuned_scores_list))
            }
        }
        combined_results.append(combined_entry)
    
    # Save to JSON file
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(combined_results, f, indent=4, ensure_ascii=False)
        print(f"Responses saved to {output_file}")
    except Exception as e:
        print(f"Error saving responses to JSON: {e}")

def main():
    # Path to the base model JSON results
    base_result_path = "./exp_result/base_model.json"
    
    # Paths to multiple fine-tuned model JSON results
    fine_tuned_result_paths = [
        "./exp_result/fine_tuned_model_epoch3.json",
        "./exp_result/fine_tuned_model_epoch4.json",
        "./exp_result/fine_tuned_model_epoch6.json",
        "./exp_result/fine_tuned_model_epoch8.json",
        # "./exp_result/fine_tuned_model_epoch10.json"
    ]
    
    # Labels for the fine-tuned models
    fine_tuned_labels = [
        "Fine-Tuned Model (3 Epochs)",
        "Fine-Tuned Model (4 Epochs)",
        "Fine-Tuned Model (6 Epochs)",
        "Fine-Tuned Model (8 Epochs)",
        # "Fine-Tuned Model (10 Epochs)"
    ]
    
    # Load results
    base_results = load_results(base_result_path)
    fine_tuned_results = [load_results(path) for path in fine_tuned_result_paths]

    # Check if all files are loaded
    if not base_results or any(result is None for result in fine_tuned_results):
        print("Failed to load one or more result files.")
        return

    # Save responses to JSON
    output_file = "./exp_result/combined_responses.json"
    save_responses_as_json(base_results, fine_tuned_results, fine_tuned_labels, output_file)

    # Plot comparison
    print("\n--- Plotting Comparison ---")
    plot_comparison(base_results, fine_tuned_results, fine_tuned_labels)

if __name__ == "__main__":
    main()

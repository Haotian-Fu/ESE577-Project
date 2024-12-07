import pandas as pd
import nlpaug.augmenter.word as naw
import random
import json

# ------------------- Load the QA Dataset -------------------
file_path = "QA_pair.xlsx"  # Update with the correct path to your file
qa_data = pd.read_excel(file_path)

# Check the structure of the dataset
if "Question" not in qa_data.columns or "Answer" not in qa_data.columns:
    raise ValueError("Dataset must contain 'Question' and 'Answer' columns.")

# Convert the dataset into a list of QA pairs
qa_pairs = []
for _, row in qa_data.iterrows():
    question = row["Question"]
    answer = row["Answer"]
    qa_pairs.append({"question": question, "answer": answer})

# Print the first few QA pairs to verify
print("Sample QA Pairs (Original):")
for pair in qa_pairs[:5]:
    print(f"Q: {pair['question']}")
    print(f"A: {pair['answer']}")
    print("---")

# ------------------- Data Augmentation with WordNet -------------------
# Initialize the augmenter
augmenter = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=2)

# Create augmented data
augmented_qa_pairs = []
for pair in qa_pairs:
    # Augment the question
    augmented_question = augmenter.augment(pair["question"])
    # Augment the answer
    augmented_answer = augmenter.augment(pair["answer"])
    
    # Ensure augmentation returns strings
    if isinstance(augmented_question, list):
        augmented_question = " ".join(augmented_question)
    if isinstance(augmented_answer, list):
        augmented_answer = " ".join(augmented_answer)
    
    # Add the augmented pair to the dataset
    augmented_qa_pairs.append({"question": augmented_question, "answer": augmented_answer})

# Combine original and augmented pairs
qa_pairs.extend(augmented_qa_pairs)

# ------------------- Shuffle the Dataset -------------------
random.shuffle(qa_pairs)

# ------------------- Save as JSON -------------------
output_path = "test.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, indent=4, ensure_ascii=False)

print(f"QA dataset saved as JSON at {output_path}")
print(f"Total QA pairs: {len(qa_pairs)}")

# ------------------- Verify Final Output -------------------
print("Sample QA Pairs (Shuffled):")
for pair in qa_pairs[:5]:
    print(f"Q: {pair['question']}")
    print(f"A: {pair['answer']}")
    print("---")

import os
import json
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch.nn.functional as F


def load_prompts(json_file):
    """Load prompts from a JSON file."""
    prompts = []
    with open(json_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            # Extract the student_prompt
            if "parent_prompt" in data:
                prompts.append(data["student_prompt"])
    return prompts


def load_image(image_path):
    """Load an image."""
    image = Image.open(image_path).convert("RGB")
    return image


def compute_clip_similarity_hf(prompts, image_dir, model, processor, device):
    """Compute CLIP similarity scores using Hugging Face's CLIP model."""
    scores = []
    
    for i in range(len(prompts)):
        image_path = os.path.join(image_dir, f"{i}.jpg")
        if os.path.exists(image_path):
            image = load_image(image_path)
            inputs = processor(text=[prompts[i]], images=image, return_tensors="pt", padding=True).to(device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image  # Image-to-text similarity
                similarity = logits_per_image.item()

                scores.append(similarity)
            
            scores.append(similarity)
    
    return scores


def aggregate_scores(scores):
    """Aggregate scores for a single directory."""
    if not scores:
        return None
    return sum(scores) / len(scores)  # Average score


def main(prompt_image_pairs):
    """Main function to evaluate and aggregate CLIP similarity per tuple."""
    # Load CLIP model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    results = {}
    
    for json_path, image_dir in prompt_image_pairs:
        # Load prompts
        prompts = load_prompts(json_path)
        # Compute similarity
        print(f"Evaluating prompts from {json_path} against images in {image_dir}...")
        scores = compute_clip_similarity_hf(prompts, image_dir, model, processor, device)
        
        # Aggregate scores for this tuple
        aggregated_score = aggregate_scores(scores)
        results[(json_path, image_dir)] = aggregated_score
        print(f"Aggregated Score for {json_path} and {image_dir}: {aggregated_score:.4f}")
    
    return results


if __name__ == "__main__":
    # Example: Provide a list of tuples with JSON file paths and image directories
    HPDv2 = '/usr3/hcontant/Datasets/Filtered/HPDv2.jsonl'
    VADER = '/usr3/hcontant/Datasets/Filtered/VADER.jsonl'
    SPO = '/usr3/hcontant/Datasets/Filtered/SPO.jsonl'
    train_eval_dirs = [f'/usr3/hcontant/pixart-project-recent/outputs-SPO/00000-gpus2-batch16-bf16-PixArt-alpha-bs16x32-fp16-textenc-uniform-skip5-5step-stu-data-2step-lr5e-6-wt=snr-sample-2step-t=4.0/train-000{i}01' for i in range (1,8)]
    
    VADER_eval_dirs = [f'/usr3/hcontant/pixart-project-recent/outputs-VADER/00000-gpus2-batch16-bf16-PixArt-alpha-bs16x32-fp16-textenc-uniform-skip5-5step-stu-data-2step-lr5e-6-wt=snr-sample-2step-t=4.0/eval-000{i}01' for i in range (1,8)]
    SPO_eval_dirs = [f'/usr3/hcontant/pixart-project-recent/outputs-SPO/00000-gpus2-batch16-bf16-PixArt-alpha-bs16x32-fp16-textenc-uniform-skip5-5step-stu-data-2step-lr5e-6-wt=snr-sample-2step-t=4.0/eval-000{i}01' for i in range (1,8)]

    prompt_image_pairs = []

    for train_eval_dir in train_eval_dirs:
        prompt_image_pairs.append((HPDv2, train_eval_dir))

    for VADER_eval_dir in VADER_eval_dirs:
        prompt_image_pairs.append((VADER, VADER_eval_dir))
    
    for SPO_eval_dir in SPO_eval_dirs:
        prompt_image_pairs.append((SPO, SPO_eval_dir))
    

    results = main(prompt_image_pairs)
    print("\nFinal Results:")
    for (json_path, image_dir), score in results.items():
        print(f"{json_path} & {image_dir}: {score:.4f}")

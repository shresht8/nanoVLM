# python upload_to_hf.py --repo_id shresht8/small-vlm1 --checkpoint_path ./checkpoints/nanoVLM-222M --private
import argparse
from huggingface_hub import HfApi, create_repo
from models.vision_language_model import VisionLanguageModel
import models.config as config

def push_model_to_hub(
    checkpoint_path: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload VLM model"
):
    """
    Push a trained VLM model to Hugging Face Hub
    
    Args:
        checkpoint_path: Path to the saved model checkpoint
        repo_id: Repository ID on HuggingFace Hub (format: username/repo-name)
        private: Whether to create a private repository
        commit_message: Commit message for the upload
    """
    
    print(f"Loading model from {checkpoint_path}...")
    model = VisionLanguageModel.from_pretrained(checkpoint_path)
    
    print(f"Pushing model to {repo_id}...")
    
    # This assumes your VisionLanguageModel has a push_to_hub method
    # If it inherits from PreTrainedModel, this should work
    try:
        model.push_to_hub(
            repo_id=repo_id,
            private=private
        )
        print(f"✓ Model successfully pushed to https://huggingface.co/{repo_id}")
    except AttributeError:
        # Fallback: manually upload files using HfApi
        print("Model doesn't have push_to_hub method, using manual upload...")
        api = HfApi()
        
        # Create repo if it doesn't exist
        try:
            create_repo(repo_id, private=private, exist_ok=True)
            print(f"✓ Repository {repo_id} created/verified")
        except Exception as e:
            print(f"Error creating repo: {e}")
            return
        
        # Upload the entire checkpoint directory
        api.upload_folder(
            folder_path=checkpoint_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message
        )
        print(f"✓ Files successfully uploaded to https://huggingface.co/{repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Push trained VLM model to Hugging Face Hub")
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./checkpoints',
        help='Path to the model checkpoint directory'
    )
    parser.add_argument(
        '--repo_id',
        type=str,
        required=True,
        help='Repository ID on HuggingFace Hub (format: username/repo-name)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Create a private repository'
    )
    parser.add_argument(
        '--commit_message',
        type=str,
        default='Upload VLM model',
        help='Commit message for the upload'
    )
    
    args = parser.parse_args()
    
    push_model_to_hub(
        checkpoint_path=args.checkpoint_path,
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message
    )

if __name__ == "__main__":
    main()
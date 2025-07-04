"""
QA Simple Dataset Loader for Dataset-Agnostic RLHF Pipeline
Provides data in RawBatchData format for the new pipeline
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from typing import List, Dict, Any

from .dataset_processor import RawBatchData

logger = logging.getLogger(__name__)

class QASimpleDataset(Dataset):
    def __init__(self, data_path: str, split: str = "train"):
        """
        Load QA Simple dataset for RLHF training
        
        Args:
            data_path: Path to qa_simple data directory
            split: 'train' or 'val'
        """
        self.data_path = data_path
        self.split = split
        
        # Load data
        file_path = f"{data_path}/{split}.json"
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'story': item['paragraph'],
            'question': item['question'],
            'answers': item['answers'],
            'correct_answer_id': item['correctAnswerId'],
            'story_title': item.get('title', ''),
            'index': idx
        }
        
    def get_stats(self):
        """Get dataset statistics"""
        if not self.data:
            return {}
            
        total_samples = len(self.data)
        avg_story_length = sum(len(item.get('paragraph', '').split()) for item in self.data) / total_samples
        avg_question_length = sum(len(item.get('question', '').split()) for item in self.data) / total_samples
        
        # Answer distribution
        correct_answers = [item.get('correctAnswerId', 0) for item in self.data if 'correctAnswerId' in item]
        answer_0_count = sum(1 for ans in correct_answers if ans == 0)
        answer_1_count = sum(1 for ans in correct_answers if ans == 1)
        
        stats = {
            'total_samples': total_samples,
            'avg_story_length_words': round(avg_story_length, 1),
            'avg_question_length_words': round(avg_question_length, 1),
            'answer_0_count': answer_0_count,
            'answer_1_count': answer_1_count,
            'answer_balance': round(answer_0_count / (answer_0_count + answer_1_count) * 100, 1) if correct_answers else 0
        }
        
        return stats

def create_qa_simple_dataloader(data_path: str, split: str = "train", batch_size: int = 32, 
                               shuffle: bool = True, dataset_processor=None) -> DataLoader:
    """
    Create DataLoader that outputs RawBatchData for the new pipeline
    
    Args:
        data_path: Path to qa_simple data directory
        split: 'train' or 'val'
        batch_size: Batch size
        shuffle: Whether to shuffle data
        dataset_processor: DatasetProcessor for story truncation (optional)
        
    Returns:
        DataLoader that yields RawBatchData objects
    """
    dataset = QASimpleDataset(data_path, split)
    
    def collate_fn(batch) -> RawBatchData:
        """Collate function that creates RawBatchData with truncated stories"""
        stories = [item['story'] for item in batch]
        
        # Truncate stories if processor is provided
        if dataset_processor and hasattr(dataset_processor.config, 'max_story_tokens'):
            if dataset_processor.config.max_story_tokens:
                stories = dataset_processor.truncate_stories_batch(
                    stories, dataset_processor.config.max_story_tokens
                )
        
        return RawBatchData(
            stories=stories,
            questions=[item['question'] for item in batch], 
            answer_choices=[item['answers'] for item in batch],
            correct_answer_ids=[item['correct_answer_id'] for item in batch],
            indices=[item['index'] for item in batch]
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

def load_qa_simple_for_rlhf(data_dir: str, split: str = "train") -> QASimpleDataset:
    """
    Convenience function to load QA Simple dataset
    """
    return QASimpleDataset(data_dir, split)

# Test the data loader
if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directories to path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Test data loading
    data_dir = "/data/lhx/minmax_monitor/dataset/qa_simple/data"
    
    print("Testing QA Simple Dataset Loader...")
    
    # Test train split
    train_dataset = load_qa_simple_for_rlhf(data_dir, "train")
    print(f"Train dataset: {len(train_dataset)} samples")
    print("Train stats:", train_dataset.get_stats())
    
    # Test val split  
    val_dataset = load_qa_simple_for_rlhf(data_dir, "val")
    print(f"Val dataset: {len(val_dataset)} samples")
    print("Val stats:", val_dataset.get_stats())
    
    # Test sample
    sample = train_dataset[0]
    print(f"\nSample data structure:")
    for key, value in sample.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"{key}: {value[:100]}...")
        else:
            print(f"{key}: {value}")
            
    # Test new dataloader
    dataloader = create_qa_simple_dataloader(data_dir, "train", batch_size=2)
    
    batch = next(iter(dataloader))
    print(f"\nRawBatchData structure:")
    print(f"stories: {len(batch.stories)} items")
    print(f"questions: {len(batch.questions)} items") 
    print(f"answer_choices: {len(batch.answer_choices)} items")
    print(f"correct_answer_ids: {batch.correct_answer_ids}")
    print(f"indices: {batch.indices}")
        
    print("\nData loader test completed successfully!")

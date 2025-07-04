"""
QA Simple Dataset Loader

This module provides utilities for loading and processing the QA Simple dataset,
a reading comprehension benchmark with binary choice questions and reasoning arguments.

Dataset Structure:
- Each entry contains a paragraph/article, a question, and two answer choices
- Training data includes rich metadata: argument, judge rating, confidence, etc.
- Validation data has a simpler structure with just correct answer ID
- Both files are JSON arrays containing question-answer objects

Usage:
    from dataset.load_qa_simple import QASimpleDataLoader
    
    # Load training data
    loader = QASimpleDataLoader()
    train_data = loader.load_split('train')
    
    # Iterate through samples
    for sample in train_data:
        print(f"Question: {sample.question}")
        print(f"Answers: {sample.answers}")
        print(f"Correct: {sample.correct_answer_id}")
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator, Any
import logging
from dataclasses import dataclass
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QASimpleSample:
    """Represents a single sample from the QA Simple dataset."""
    # Core fields present in both train and validation
    question: str
    answers: List[str]  # Always 2 choices
    correct_answer_id: int  # 0 or 1
    paragraph: str
    
    # Training-specific fields (may be None for validation data)
    story_title: Optional[str] = None
    setting: Optional[str] = None
    correct: Optional[bool] = None  # Legacy field from training data
    confidence: Optional[str] = None
    argument: Optional[str] = None
    judge: Optional[str] = None  # "agree" or "disagree"
    chose_answer_id: Optional[int] = None
    
    # Validation-specific fields (may be None for training data)
    title: Optional[str] = None  # Different from story_title


class QASimpleDataLoader:
    """Data loader for the QA Simple reading comprehension dataset."""
    
    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the QA Simple data loader.
        
        Args:
            data_dir: Path to the QA Simple dataset directory. If None, uses the default location.
        """
        if data_dir is None:
            data_dir = Path(__file__).parent / "qa_simple" / "data"
        else:
            data_dir = Path(data_dir)
            
        self.data_dir = data_dir
        
        # Validate data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
        # Define available splits
        self.splits = ["train", "val"]
        
        logger.info(f"Initialized QA Simple loader with data_dir: {self.data_dir}")
    
    def _get_file_path(self, split: str) -> Path:
        """Get the file path for a given split."""
        if split not in self.splits:
            raise ValueError(f"Invalid split: {split}. Must be one of {self.splits}")
        
        filename = f"{split}.json"
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        return file_path
    
    def _parse_sample(self, sample_data: Dict[str, Any]) -> QASimpleSample:
        """Parse a sample dictionary into a QASimpleSample object."""
        
        # Handle different field names between train and val
        title = sample_data.get("story_title") or sample_data.get("title")
        
        return QASimpleSample(
            question=sample_data["question"],
            answers=sample_data["answers"],
            correct_answer_id=sample_data["correctAnswerId"],
            paragraph=sample_data["paragraph"],
            
            # Training-specific fields
            story_title=sample_data.get("story_title"),
            setting=sample_data.get("setting"),
            correct=sample_data.get("correct"),
            confidence=sample_data.get("confidence"),
            argument=sample_data.get("argument"),
            judge=sample_data.get("judge"),
            chose_answer_id=sample_data.get("choseAnswerId"),
            
            # Validation-specific fields
            title=sample_data.get("title")
        )
    
    def load_split(self, split: str) -> List[QASimpleSample]:
        """
        Load all samples from a specific split.
        
        Args:
            split: The split to load ('train' or 'val')
            
        Returns:
            List of QASimpleSample objects
        """
        file_path = self._get_file_path(split)
        
        logger.info(f"Loading {split} split from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for i, sample_data in enumerate(data):
            try:
                sample = self._parse_sample(sample_data)
                samples.append(sample)
            except Exception as e:
                logger.error(f"Error parsing sample {i} in {split}: {e}")
                raise
        
        logger.info(f"Loaded {len(samples)} samples from {split} split")
        return samples
    
    def load_split_generator(self, split: str) -> Iterator[QASimpleSample]:
        """
        Load samples from a specific split as a generator (memory efficient).
        
        Args:
            split: The split to load ('train' or 'val')
            
        Yields:
            QASimpleSample objects one at a time
        """
        file_path = self._get_file_path(split)
        
        logger.info(f"Loading {split} split from {file_path} (generator)")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for i, sample_data in enumerate(data):
            try:
                yield self._parse_sample(sample_data)
            except Exception as e:
                logger.error(f"Error parsing sample {i} in {split}: {e}")
                raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            "splits": {},
            "overall": {
                "total_samples": 0,
                "avg_paragraph_length": 0,
                "avg_question_length": 0,
                "avg_answer_length": 0,
                "settings": defaultdict(int),
                "story_titles": set(),
                "confidence_distribution": defaultdict(int),
                "judge_distribution": defaultdict(int)
            }
        }
        
        paragraph_lengths = []
        question_lengths = []
        answer_lengths = []
        
        for split in self.splits:
            try:
                samples = self.load_split(split)
                
                split_stats = {
                    "num_samples": len(samples),
                    "avg_paragraph_length": 0,
                    "avg_question_length": 0,
                    "avg_answer_length": 0,
                    "has_training_metadata": split == "train"
                }
                
                split_paragraph_lengths = []
                split_question_lengths = []
                split_answer_lengths = []
                
                for sample in samples:
                    para_len = len(sample.paragraph.split())
                    q_len = len(sample.question.split())
                    ans_len = sum(len(ans.split()) for ans in sample.answers) / len(sample.answers)
                    
                    split_paragraph_lengths.append(para_len)
                    split_question_lengths.append(q_len)
                    split_answer_lengths.append(ans_len)
                    
                    paragraph_lengths.append(para_len)
                    question_lengths.append(q_len)
                    answer_lengths.append(ans_len)
                    
                    # Collect metadata
                    if sample.setting:
                        stats["overall"]["settings"][sample.setting] += 1
                    if sample.story_title:
                        stats["overall"]["story_titles"].add(sample.story_title)
                    if sample.title:
                        stats["overall"]["story_titles"].add(sample.title)
                    if sample.confidence:
                        stats["overall"]["confidence_distribution"][sample.confidence] += 1
                    if sample.judge:
                        stats["overall"]["judge_distribution"][sample.judge] += 1
                
                if split_paragraph_lengths:
                    split_stats["avg_paragraph_length"] = sum(split_paragraph_lengths) / len(split_paragraph_lengths)
                    split_stats["avg_question_length"] = sum(split_question_lengths) / len(split_question_lengths)
                    split_stats["avg_answer_length"] = sum(split_answer_lengths) / len(split_answer_lengths)
                
                stats["splits"][split] = split_stats
                stats["overall"]["total_samples"] += len(samples)
                
            except FileNotFoundError:
                logger.warning(f"Split {split} not found, skipping...")
                continue
        
        # Calculate overall averages
        if paragraph_lengths:
            stats["overall"]["avg_paragraph_length"] = sum(paragraph_lengths) / len(paragraph_lengths)
            stats["overall"]["avg_question_length"] = sum(question_lengths) / len(question_lengths)
            stats["overall"]["avg_answer_length"] = sum(answer_lengths) / len(answer_lengths)
        
        # Convert set to count for JSON serialization
        stats["overall"]["num_unique_stories"] = len(stats["overall"]["story_titles"])
        stats["overall"]["story_titles"] = list(stats["overall"]["story_titles"])[:10]  # Show only first 10
        
        return stats
    
    def get_sample_by_question(self, question_text: str, split: Optional[str] = None) -> Optional[QASimpleSample]:
        """
        Retrieve a specific sample by question text (partial matching).
        
        Args:
            question_text: The question text to search for
            split: Specific split to search in, or None to search all splits
            
        Returns:
            QASimpleSample if found, None otherwise
        """
        splits_to_search = [split] if split else self.splits
        
        for split_name in splits_to_search:
            try:
                for sample in self.load_split_generator(split_name):
                    if question_text.lower() in sample.question.lower():
                        return sample
            except FileNotFoundError:
                continue
                
        return None
    
    def filter_by_setting(self, setting: str, split: str = "train") -> List[QASimpleSample]:
        """
        Filter samples by setting (only available in training data).
        
        Args:
            setting: The setting to filter by (e.g., "consultancy", "debate")
            split: The split to filter from (default: "train")
            
        Returns:
            List of samples matching the setting
        """
        samples = self.load_split(split)
        return [sample for sample in samples if sample.setting == setting]
    
    def filter_by_confidence(self, min_confidence: float, split: str = "train") -> List[QASimpleSample]:
        """
        Filter samples by minimum confidence score (only available in training data).
        
        Args:
            min_confidence: Minimum confidence threshold
            split: The split to filter from (default: "train")
            
        Returns:
            List of samples with confidence >= min_confidence
        """
        samples = self.load_split(split)
        filtered = []
        
        for sample in samples:
            if sample.confidence:
                try:
                    conf = float(sample.confidence)
                    if conf >= min_confidence:
                        filtered.append(sample)
                except ValueError:
                    continue
        
        return filtered
    
    def export_to_format(self, split: str, output_path: Union[str, Path], 
                        format_type: str = "jsonl", include_metadata: bool = True) -> None:
        """
        Export a split to a different format.
        
        Args:
            split: The split to export
            output_path: Path to save the exported data
            format_type: Export format ('jsonl', 'json')
            include_metadata: Whether to include training metadata
        """
        output_path = Path(output_path)
        samples = self.load_split(split)
        
        if format_type == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    # Create minimal export dict
                    sample_dict = {
                        "question": sample.question,
                        "answers": sample.answers,
                        "correct_answer_id": sample.correct_answer_id,
                        "paragraph": sample.paragraph
                    }
                    
                    if include_metadata and split == "train":
                        sample_dict.update({
                            "story_title": sample.story_title,
                            "setting": sample.setting,
                            "confidence": sample.confidence,
                            "argument": sample.argument,
                            "judge": sample.judge
                        })
                    
                    f.write(json.dumps(sample_dict, ensure_ascii=False) + '\n')
                    
        elif format_type == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                data = []
                for sample in samples:
                    sample_dict = {
                        "question": sample.question,
                        "answers": sample.answers,
                        "correct_answer_id": sample.correct_answer_id,
                        "paragraph": sample.paragraph
                    }
                    
                    if include_metadata and split == "train":
                        sample_dict.update({
                            "story_title": sample.story_title,
                            "setting": sample.setting,
                            "confidence": sample.confidence,
                            "argument": sample.argument,
                            "judge": sample.judge
                        })
                    
                    data.append(sample_dict)
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        logger.info(f"Exported {len(samples)} samples to {output_path}")


def main():
    """Example usage and testing of the QA Simple data loader."""
    # Initialize the loader
    loader = QASimpleDataLoader()
    
    # Get dataset statistics
    print("Dataset Statistics:")
    print("=" * 50)
    stats = loader.get_statistics()
    
    for split, split_stats in stats["splits"].items():
        print(f"\n{split.upper()} Split:")
        print(f"  Samples: {split_stats['num_samples']}")
        print(f"  Avg paragraph length: {split_stats['avg_paragraph_length']:.1f} words")
        print(f"  Avg question length: {split_stats['avg_question_length']:.1f} words")
        print(f"  Avg answer length: {split_stats['avg_answer_length']:.1f} words")
        print(f"  Has training metadata: {split_stats['has_training_metadata']}")
    
    print(f"\nOverall Statistics:")
    print(f"  Total samples: {stats['overall']['total_samples']}")
    print(f"  Unique stories: {stats['overall']['num_unique_stories']}")
    print(f"  Avg paragraph length: {stats['overall']['avg_paragraph_length']:.1f} words")
    print(f"  Settings: {dict(stats['overall']['settings'])}")
    print(f"  Judge distribution: {dict(stats['overall']['judge_distribution'])}")
    
    # Load and examine a sample
    print("\n" + "=" * 50)
    print("Sample Data:")
    
    # Load first sample from train split
    train_samples = loader.load_split('train')
    if train_samples:
        sample = train_samples[0]
        print(f"\nStory Title: {sample.story_title}")
        print(f"Setting: {sample.setting}")
        print(f"Question: {sample.question}")
        print(f"Answers: {sample.answers}")
        print(f"Correct Answer ID: {sample.correct_answer_id}")
        print(f"Correct Answer: {sample.answers[sample.correct_answer_id]}")
        print(f"Confidence: {sample.confidence}")
        print(f"Judge: {sample.judge}")
        print(f"Paragraph length: {len(sample.paragraph.split())} words")
        
        # Show argument if available
        if sample.argument:
            print(f"\nArgument (first 200 chars):")
            print(f"  {sample.argument[:200]}...")
    
    # Test filtering functions
    print("\n" + "=" * 50)
    print("Filtering Examples:")
    
    # Filter by setting
    consultancy_samples = loader.filter_by_setting("consultancy")
    print(f"Consultancy samples: {len(consultancy_samples)}")
    
    # Filter by confidence
    high_conf_samples = loader.filter_by_confidence(90.0)
    print(f"High confidence samples (>=90%): {len(high_conf_samples)}")


if __name__ == "__main__":
    main()

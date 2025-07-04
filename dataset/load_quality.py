"""
QuALITY v1.0.1 Dataset Loader

This module provides utilities for loading and processing the QuALITY v1.0.1 dataset,
a reading comprehension benchmark with long articles and multiple-choice questions.

Dataset Structure:
- Each line is a JSON object representing one article with multiple questions
- Fields include: article_id, article, questions, metadata
- Available splits: train (300 samples), dev (230 samples), test (232 samples)
- Both regular and HTML-stripped versions available

Usage:
    from dataset.load_quality import QualityDataLoader
    
    # Load training data
    loader = QualityDataLoader()
    train_data = loader.load_split('train')
    
    # Iterate through samples
    for sample in train_data:
        print(f"Article ID: {sample['article_id']}")
        print(f"Title: {sample['title']}")
        print(f"Questions: {len(sample['questions'])}")
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator, Any
import logging
from dataclasses import dataclass, field
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QualityQuestion:
    """Represents a single question from the QuALITY dataset."""
    question: str
    question_unique_id: str
    options: List[str]
    writer_label: Optional[int] = None  # May not be available in test set
    gold_label: Optional[int] = None    # May not be available in test set
    validation: List[Dict[str, Any]] = field(default_factory=list)
    speed_validation: List[Dict[str, Any]] = field(default_factory=list)
    difficult: int = 0


@dataclass
class QualitySample:
    """Represents a complete sample from the QuALITY dataset."""
    article_id: str
    set_unique_id: str
    batch_num: str
    writer_id: str
    source: str
    title: str
    year: int
    author: str
    topic: str
    article: str
    questions: List[QualityQuestion]
    url: str
    license: str


class QualityDataLoader:
    """Data loader for the QuALITY v1.0.1 reading comprehension dataset."""
    
    def __init__(self, data_dir: Optional[Union[str, Path]] = None, use_html_stripped: bool = True):
        """
        Initialize the QuALITY data loader.
        
        Args:
            data_dir: Path to the QuALITY dataset directory. If None, uses the default location.
            use_html_stripped: Whether to use HTML-stripped versions of the articles (recommended).
        """
        if data_dir is None:
            data_dir = Path(__file__).parent / "quality" / "data" / "v1.0.1"
        else:
            data_dir = Path(data_dir)
            
        self.data_dir = data_dir
        self.use_html_stripped = use_html_stripped
        
        # Validate data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
        # Define file patterns
        self.file_suffix = ".htmlstripped" if use_html_stripped else ""
        self.splits = ["train", "dev", "test"]
        
        logger.info(f"Initialized QuALITY loader with data_dir: {self.data_dir}")
        logger.info(f"Using HTML-stripped files: {use_html_stripped}")
    
    def _get_file_path(self, split: str) -> Path:
        """Get the file path for a given split."""
        if split not in self.splits:
            raise ValueError(f"Invalid split: {split}. Must be one of {self.splits}")
        
        filename = f"QuALITY.v1.0.1{self.file_suffix}.{split}"
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        return file_path
    
    def _parse_question(self, question_data: Dict[str, Any]) -> QualityQuestion:
        """Parse a question dictionary into a QualityQuestion object."""
        return QualityQuestion(
            question=question_data["question"],
            question_unique_id=question_data["question_unique_id"],
            options=question_data["options"],
            writer_label=question_data.get("writer_label"),  # Optional, not in test set
            gold_label=question_data.get("gold_label"),      # Optional, not in test set
            validation=question_data.get("validation", []),
            speed_validation=question_data.get("speed_validation", []),
            difficult=question_data.get("difficult", 0)
        )
    
    def _parse_sample(self, line: str) -> QualitySample:
        """Parse a JSON line into a QualitySample object."""
        try:
            data = json.loads(line.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON line: {e}")
        
        # Parse questions
        questions = [self._parse_question(q) for q in data["questions"]]
        
        return QualitySample(
            article_id=data["article_id"],
            set_unique_id=data["set_unique_id"],
            batch_num=data["batch_num"],
            writer_id=data["writer_id"],
            source=data["source"],
            title=data["title"],
            year=data["year"],
            author=data["author"],
            topic=data["topic"],
            article=data["article"],
            questions=questions,
            url=data["url"],
            license=data["license"]
        )
    
    def load_split(self, split: str) -> List[QualitySample]:
        """
        Load all samples from a specific split.
        
        Args:
            split: The split to load ('train', 'dev', or 'test')
            
        Returns:
            List of QualitySample objects
        """
        file_path = self._get_file_path(split)
        samples = []
        
        logger.info(f"Loading {split} split from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():  # Skip empty lines
                    try:
                        sample = self._parse_sample(line)
                        samples.append(sample)
                    except Exception as e:
                        logger.error(f"Error parsing line {line_num} in {split}: {e}")
                        raise
        
        logger.info(f"Loaded {len(samples)} samples from {split} split")
        return samples
    
    def load_split_generator(self, split: str) -> Iterator[QualitySample]:
        """
        Load samples from a specific split as a generator (memory efficient).
        
        Args:
            split: The split to load ('train', 'dev', or 'test')
            
        Yields:
            QualitySample objects one at a time
        """
        file_path = self._get_file_path(split)
        
        logger.info(f"Loading {split} split from {file_path} (generator)")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():  # Skip empty lines
                    try:
                        yield self._parse_sample(line)
                    except Exception as e:
                        logger.error(f"Error parsing line {line_num} in {split}: {e}")
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
                "total_articles": 0,
                "total_questions": 0,
                "sources": defaultdict(int),
                "years": defaultdict(int),
                "difficulty_distribution": defaultdict(int),
                "avg_article_length": 0,
                "avg_questions_per_article": 0
            }
        }
        
        article_lengths = []
        questions_per_article = []
        
        for split in self.splits:
            try:
                samples = self.load_split(split)
                
                split_stats = {
                    "num_articles": len(samples),
                    "num_questions": 0,
                    "avg_article_length": 0,
                    "avg_questions_per_article": 0
                }
                
                split_article_lengths = []
                split_questions = []
                
                for sample in samples:
                    article_len = len(sample.article.split())
                    num_questions = len(sample.questions)
                    
                    split_article_lengths.append(article_len)
                    split_questions.append(num_questions)
                    article_lengths.append(article_len)
                    questions_per_article.append(num_questions)
                    
                    # Update overall stats
                    stats["overall"]["sources"][sample.source] += 1
                    stats["overall"]["years"][sample.year] += 1
                    
                    for question in sample.questions:
                        stats["overall"]["difficulty_distribution"][question.difficult] += 1
                        split_stats["num_questions"] += 1
                
                if split_article_lengths:
                    split_stats["avg_article_length"] = sum(split_article_lengths) / len(split_article_lengths)
                    split_stats["avg_questions_per_article"] = sum(split_questions) / len(split_questions)
                
                stats["splits"][split] = split_stats
                stats["overall"]["total_articles"] += len(samples)
                stats["overall"]["total_questions"] += split_stats["num_questions"]
                
            except FileNotFoundError:
                logger.warning(f"Split {split} not found, skipping...")
                continue
        
        # Calculate overall averages
        if article_lengths:
            stats["overall"]["avg_article_length"] = sum(article_lengths) / len(article_lengths)
            stats["overall"]["avg_questions_per_article"] = sum(questions_per_article) / len(questions_per_article)
        
        return stats
    
    def get_sample_by_id(self, article_id: str, split: Optional[str] = None) -> Optional[QualitySample]:
        """
        Retrieve a specific sample by article ID.
        
        Args:
            article_id: The article ID to search for
            split: Specific split to search in, or None to search all splits
            
        Returns:
            QualitySample if found, None otherwise
        """
        splits_to_search = [split] if split else self.splits
        
        for split_name in splits_to_search:
            try:
                for sample in self.load_split_generator(split_name):
                    if sample.article_id == article_id:
                        return sample
            except FileNotFoundError:
                continue
                
        return None
    
    def export_to_format(self, split: str, output_path: Union[str, Path], 
                        format_type: str = "jsonl") -> None:
        """
        Export a split to a different format.
        
        Args:
            split: The split to export
            output_path: Path to save the exported data
            format_type: Export format ('jsonl', 'json', 'csv')
        """
        output_path = Path(output_path)
        samples = self.load_split(split)
        
        if format_type == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    # Convert back to dict for JSON serialization
                    sample_dict = {
                        "article_id": sample.article_id,
                        "title": sample.title,
                        "article": sample.article,
                        "questions": [
                            {
                                "question": q.question,
                                "options": q.options,
                                "gold_label": q.gold_label
                            }
                            for q in sample.questions
                        ]
                    }
                    f.write(json.dumps(sample_dict, ensure_ascii=False) + '\n')
                    
        elif format_type == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                data = []
                for sample in samples:
                    sample_dict = {
                        "article_id": sample.article_id,
                        "title": sample.title,
                        "article": sample.article,
                        "questions": [
                            {
                                "question": q.question,
                                "options": q.options,
                                "gold_label": q.gold_label
                            }
                            for q in sample.questions
                        ]
                    }
                    data.append(sample_dict)
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        logger.info(f"Exported {len(samples)} samples to {output_path}")





def main():
    """Example usage and testing of the QuALITY data loader."""
    # Initialize the loader
    loader = QualityDataLoader()
    
    # Get dataset statistics
    print("Dataset Statistics:")
    print("=" * 50)
    stats = loader.get_statistics()
    
    for split, split_stats in stats["splits"].items():
        print(f"\n{split.upper()} Split:")
        print(f"  Articles: {split_stats['num_articles']}")
        print(f"  Questions: {split_stats['num_questions']}")
        print(f"  Avg article length: {split_stats['avg_article_length']:.1f} words")
        print(f"  Avg questions per article: {split_stats['avg_questions_per_article']:.1f}")
    
    print(f"\nOverall Statistics:")
    print(f"  Total articles: {stats['overall']['total_articles']}")
    print(f"  Total questions: {stats['overall']['total_questions']}")
    print(f"  Avg article length: {stats['overall']['avg_article_length']:.1f} words")
    
    # Load and examine a sample
    print("\n" + "=" * 50)
    print("Sample Data:")
    
    # Load first sample from train split
    train_samples = loader.load_split('train')
    if train_samples:
        sample = train_samples[0]
        print(f"\nTitle: {sample.title}")
        print(f"Author: {sample.author}")
        print(f"Year: {sample.year}")
        print(f"Article length: {len(sample.article.split())} words")
        print(f"Number of questions: {len(sample.questions)}")
        
        # Show first question
        if sample.questions:
            q = sample.questions[0]
            print(f"\nFirst Question:")
            print(f"  Question: {q.question}")
            print(f"  Options: {q.options}")
            print(f"  Correct answer (1-indexed): {q.gold_label}")


if __name__ == "__main__":
    main()

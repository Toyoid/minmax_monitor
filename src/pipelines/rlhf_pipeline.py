"""
Unified RLHF Pipeline for Dataset-Agnostic Training
Orchestrates the entire RLHF forward pass with policy, reward, and judge models
"""
import torch
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..data.dataset_processor import (
    DatasetProcessor, RawBatchData, PolicyOutput, RLHFBatch
)
from ..models.policy_model import PolicyModel
from ..models.reward_model import RewardModel
from ..models.judge_model import JudgeModel
from ..config.dataset_config import DatasetConfig

logger = logging.getLogger(__name__)

class RLHFPipeline:
    """
    Unified pipeline for RLHF training that orchestrates all models with device coordination
    """
    
    def __init__(self, policy_model: PolicyModel, reward_model: RewardModel, 
                 judge_model: JudgeModel, dataset_processor: DatasetProcessor,
                 device_manager=None):
        """
        Initialize RLHF pipeline with models and processor
        
        Args:
            policy_model: Dataset-agnostic policy model
            reward_model: Dataset-agnostic reward model  
            judge_model: Dataset-agnostic judge model
            dataset_processor: Dataset-specific processor
            device_manager: Device manager for coordinating multi-GPU operations
        """
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.judge_model = judge_model
        self.processor = dataset_processor
        self.device_manager = device_manager
        
        # Connect tokenizers to processor
        self.processor.set_tokenizers(
            policy_model.get_tokenizer(),
            reward_model.tokenizer,
            judge_model.tokenizer
        )
        
        logger.info("RLHF Pipeline initialized with dataset-agnostic models")
        
    def forward_pass(self, batch_data: RawBatchData, max_new_tokens: Optional[int] = None) -> RLHFBatch:
        """
        Complete forward pass through all models
        
        Args:
            batch_data: Raw batch data (stories, questions, answer_choices, etc.)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            RLHFBatch with all model outputs and scores
        """
        logger.debug(f"Starting RLHF forward pass for batch size: {len(batch_data.stories)}")
        
        # 1. Policy generation
        logger.debug("Step 1: Policy model generation")
        policy_prompts = self.processor.create_policy_prompts_batch(
            batch_data.stories, batch_data.questions, batch_data.answer_choices
        )
        
        policy_inputs = self.processor.tokenize_policy_batch(policy_prompts)
        
        generated_tokens = self.policy_model.generate_batch(
            policy_inputs['input_ids'], 
            policy_inputs['attention_mask'],
            max_new_tokens=max_new_tokens
        )
        
        # 2. Extract and parse policy responses
        logger.debug("Step 2: Response extraction and parsing")
        responses = self.processor.extract_responses_batch(
            generated_tokens, policy_inputs['input_lengths']
        )
        parsed_answers = self.processor.parse_answers_batch(responses, batch_data.answer_choices)
        # justifications = self.processor.extract_justifications_batch(responses)
        justifications = responses  # For now, just use responses as justifications
        
        print("Parsed responses:")
        print(60 * "=")
        for i, response in enumerate(responses):
            print(f"Response {i+1}: ")
            print(f"Parsed answer: {parsed_answers[i]}")
            print(f"Justification: {justifications[i]}")
            print(60 * "=")

        # Create policy output object
        policy_output = PolicyOutput(
            generated_ids=generated_tokens,
            parsed_answers=parsed_answers,
            justifications=justifications,
        )
        
        # 3. Reward evaluation
        logger.debug("Step 3: Reward model evaluation")
        reward_prompts = self.processor.create_reward_inputs_batch(
            batch_data.questions, batch_data.answer_choices, parsed_answers, justifications
        )
        reward_inputs = self.processor.tokenize_reward_batch(reward_prompts)
        reward_scores = self.reward_model.score_batch(
            reward_inputs['input_ids'], reward_inputs['attention_mask']
        )
        print("Reward scores:")
        print(60 * "=")
        print(reward_scores)
        print(60 * "=")
        
        # 4. Judge evaluation  
        logger.debug("Step 4: Judge model evaluation")
        judge_prompts = self.processor.create_judge_inputs_batch(
            batch_data.questions, batch_data.answer_choices, parsed_answers, justifications
        )
        judge_inputs = self.processor.tokenize_judge_batch(judge_prompts)
        judge_scores = self.judge_model.judge_batch(
            judge_inputs['input_ids'], judge_inputs['attention_mask']
        )
        print("Judge scores:")
        print(60 * "=")
        print(judge_scores)
        print(60 * "=")
        
        # 5. Compute ground truth correctness
        logger.debug("Step 5: Computing ground truth correctness")
        ground_truth_correct = self.processor.compute_correctness_batch(
            parsed_answers, batch_data.correct_answer_ids
        )
        
        # 6. Synchronize all tensors to policy device for PPO training
        if self.device_manager:
            logger.debug("Step 6: Synchronizing tensors to policy device")
            reward_scores = self.device_manager.move_to_policy_device(reward_scores)
            judge_scores = self.device_manager.move_to_policy_device(judge_scores) 
            ground_truth_correct = self.device_manager.move_to_policy_device(ground_truth_correct)
            # Note: generated_tokens should already be on policy device
        
        # Create final RLHF batch
        rlhf_batch = RLHFBatch(
            generated_tokens=generated_tokens,
            parsed_answers=parsed_answers,
            justifications=justifications,
            reward_scores=reward_scores,
            judge_scores=judge_scores,
            ground_truth_correct=ground_truth_correct,
            original_data=batch_data,
            # Store policy inputs for PPO training
            policy_input_ids=policy_inputs['input_ids'],
            policy_attention_mask=policy_inputs['attention_mask'],
            policy_input_lengths=policy_inputs['input_lengths']
        )
        
        logger.debug("RLHF forward pass completed successfully")
        return rlhf_batch
    
    def evaluate_batch(self, batch_data: RawBatchData, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate a batch and return metrics
        
        Args:
            batch_data: Raw batch data
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with evaluation metrics
        """
        rlhf_batch = self.forward_pass(batch_data, max_new_tokens)
        
        # Compute metrics
        batch_size = len(batch_data.stories)
        
        # Accuracy
        accuracy = rlhf_batch.ground_truth_correct.mean().item()
        
        # Average scores
        avg_reward = rlhf_batch.reward_scores.mean().item()
        avg_judge = rlhf_batch.judge_scores.mean().item()
        
        # Parsing success rate
        valid_answers = sum(1 for ans in rlhf_batch.parsed_answers if ans != "")
        parsing_success_rate = valid_answers / batch_size
        
        # Agreement between judge and ground truth
        judge_binary = (rlhf_batch.judge_scores > 0.0).float()
        judge_accuracy = (judge_binary == rlhf_batch.ground_truth_correct).float().mean().item()
        judge_positive_cnt = (judge_binary == 1).sum().item()  # Count of positive judge scores
        # Judge says correct, but ground truth is incorrect (False Positive)
        judge_false_pos_cnt = ((judge_binary == 1) & (rlhf_batch.ground_truth_correct == 0)).sum().item()
        # Judge says incorrect, but ground truth is correct (False Negative)
        judge_false_neg_cnt = ((judge_binary == 0) & (rlhf_batch.ground_truth_correct == 1)).sum().item()
        
        metrics = {
            'batch_size': batch_size,
            'accuracy': accuracy,
            'avg_reward_score': avg_reward,
            'avg_judge_score': avg_judge,
            'parsing_success_rate': parsing_success_rate,
            'judge_accuracy': judge_accuracy,
            'judge_positive_cnt': judge_positive_cnt,  # Count of positive judge scores
            'judge_false_pos_cnt': judge_false_pos_cnt,  # deceived by the policy model
            'judge_false_neg_cnt': judge_false_neg_cnt,
            'reward_scores': rlhf_batch.reward_scores.cpu().numpy().tolist(),
            'judge_scores': rlhf_batch.judge_scores.cpu().numpy().tolist(),
            'ground_truth_correct': rlhf_batch.ground_truth_correct.cpu().numpy().tolist(),
            'parsed_answers': rlhf_batch.parsed_answers,
        }
        
        return metrics
    
    def get_models_for_training(self) -> Dict[str, Any]:
        """
        Get model objects needed for PPO training
        
        Returns:
            Dictionary with policy model and tokenizer for training
        """
        return {
            'policy_model': self.policy_model.get_model_for_training(),
            'policy_tokenizer': self.policy_model.get_tokenizer(),
            'reward_model': self.reward_model.get_model(),
            'judge_model': self.judge_model.get_model(),
        }
    
    def create_ppo_inputs(self, rlhf_batch: RLHFBatch, 
                         reward_weight: float = 0.8, judge_weight: float = 0.2) -> Dict[str, torch.Tensor]:
        """
        Create inputs for PPO training from RLHF batch
        
        Args:
            rlhf_batch: Output from forward_pass
            reward_weight: Weight for reward model scores
            judge_weight: Weight for judge model scores
            
        Returns:
            Dictionary with tensors needed for PPO training
        """
        # Combine reward and judge scores
        combined_rewards = (
            reward_weight * rlhf_batch.reward_scores + 
            judge_weight * rlhf_batch.judge_scores
        )
        
        # Extract input sequences (prompts only, without generated responses)
        input_lengths = []
        for i, response in enumerate(rlhf_batch.parsed_answers):
            # This should be computed during forward pass and stored
            # For now, estimate based on original data
            story = rlhf_batch.original_data.stories[i]
            question = rlhf_batch.original_data.questions[i]
            choices = rlhf_batch.original_data.answer_choices[i]
            
            # Create prompt and tokenize to get length
            prompt = self.processor.create_policy_prompts_batch([story], [question], [choices])[0]
            tokenized = self.policy_model.get_tokenizer()(prompt, return_tensors="pt")
            input_lengths.append(tokenized['input_ids'].size(1))
        
        # Extract query and response tokens
        queries = []
        responses = []
        
        for i, input_len in enumerate(input_lengths):
            full_sequence = rlhf_batch.generated_tokens[i]
            query = full_sequence[:input_len]
            response = full_sequence[input_len:]
            
            queries.append(query)
            responses.append(response)
        
        # Pad sequences to same length
        max_query_len = max(len(q) for q in queries)
        max_response_len = max(len(r) for r in responses)
        
        padded_queries = torch.zeros(len(queries), max_query_len, dtype=torch.long)
        padded_responses = torch.zeros(len(responses), max_response_len, dtype=torch.long)
        
        for i, (query, response) in enumerate(zip(queries, responses)):
            padded_queries[i, :len(query)] = query
            padded_responses[i, :len(response)] = response
        
        return {
            'queries': padded_queries,
            'responses': padded_responses,
            'rewards': combined_rewards,
            'ground_truth_correct': rlhf_batch.ground_truth_correct,
            'reward_scores': rlhf_batch.reward_scores,
            'judge_scores': rlhf_batch.judge_scores,
        }
    
    def compute_combined_reward(self, rlhf_output) -> torch.Tensor:
        """
        Compute combined reward from multiple sources using config weights
        
        Args:
            rlhf_output: RLHFBatch containing reward and judge scores
            
        Returns:
            Combined reward tensor
        """
        # Import here to avoid circular imports
        from ..train.rlhf_trainer import RLHFTrainer
        
        # Get weights from the trainer's config if available
        # For now, use default values - this will be passed from trainer
        reward_weight = getattr(self, '_reward_weight', 0.8)
        judge_weight = getattr(self, '_judge_weight', 0.2)
        
        # Normalize scores to [-1, 1] range
        # Reward scores: use tanh to map raw logits to [-1, 1]
        normalized_reward = torch.tanh(rlhf_output.reward_scores)
        
        # Judge scores: already in [-1, 1] range
        normalized_judge = rlhf_output.judge_scores
        
        # Combine rewards
        combined_reward = (reward_weight * normalized_reward + 
                          judge_weight * normalized_judge)
        
        return combined_reward
    
    def set_reward_weights(self, reward_weight: float, judge_weight: float):
        """Set reward combination weights"""
        self._reward_weight = reward_weight
        self._judge_weight = judge_weight


def create_rlhf_pipeline(policy_model: PolicyModel, reward_model: RewardModel, 
                        judge_model: JudgeModel, dataset_config: DatasetConfig,
                        device_manager=None, policy_model_config=None) -> RLHFPipeline:
    """
    Factory function to create RLHF pipeline with device coordination
    
    Args:
        policy_model: Initialized policy model
        reward_model: Initialized reward model
        judge_model: Initialized judge model
        dataset_config: Dataset configuration
        device_manager: Device manager for multi-GPU coordination
        policy_model_config: Policy model configuration
        
    Returns:
        Configured RLHF pipeline
    """
    processor = DatasetProcessor(dataset_config, policy_model_config, None, "rlhf")
    return RLHFPipeline(policy_model, reward_model, judge_model, processor, device_manager)

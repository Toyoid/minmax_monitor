"""
MinMax Pipeline for LLM-vs-Monitor Training
Independent pipeline that orchestrates policy, monitor, reward, and judge models
"""
import torch
import torch.nn.functional as F
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..data.dataset_processor import (
    DatasetProcessor, RawBatchData, MinMaxOutput
)
from ..models.policy_model import PolicyModel
from ..models.monitor_model import MonitorModel
from ..models.reward_model import RewardModel
from ..models.judge_model import JudgeModel
from ..config.dataset_config import DatasetConfig

logger = logging.getLogger(__name__)

class MinMaxPipeline:
    """
    Independent MinMax pipeline for LLM-vs-Monitor training
    Does not inherit from RLHFPipeline to avoid redundancy
    """
    
    def __init__(self, policy_model: PolicyModel, monitor_model: MonitorModel,
                 reward_model: RewardModel, judge_model: JudgeModel,
                 dataset_processor: DatasetProcessor, device_manager=None):
        """
        Initialize MinMax pipeline with all models
        
        Args:
            policy_model: Dataset-agnostic policy model
            monitor_model: Dataset-agnostic monitor model
            reward_model: Dataset-agnostic reward model  
            judge_model: Dataset-agnostic judge model
            dataset_processor: Dataset-specific processor
            device_manager: Device manager for coordinating multi-GPU operations
        """
        self.policy_model = policy_model
        self.monitor_model = monitor_model
        self.reward_model = reward_model
        self.judge_model = judge_model
        self.processor = dataset_processor
        self.device_manager = device_manager
        
        # Connect tokenizers to processor
        self.processor.set_tokenizers(
            policy_model.get_tokenizer(),
            reward_model.tokenizer,
            judge_model.tokenizer,
            monitor_model.get_tokenizer()
        )
        
        logger.info("MinMax Pipeline initialized with all models")
        
    def forward_pass(self, batch_data: RawBatchData, max_new_tokens: Optional[int] = None) -> MinMaxOutput:
        """
        Complete forward pass through all models for minmax training
        
        Args:
            batch_data: Raw batch data (stories, questions, answer_choices, etc.)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            MinMaxOutput with all model outputs and scores
        """
        logger.debug(f"Starting MinMax forward pass for batch size: {len(batch_data.stories)}")
        
        try:
            # 1. Policy generation: S → O
            logger.debug("Step 1: Policy model generation")
            policy_outputs = self._generate_policy_responses(batch_data, max_new_tokens)
            logger.debug(f"Policy outputs generated: {len(policy_outputs['parsed_answers'])} answers")
            
            # 2. Monitor generation: (S, O) → O'
            logger.debug("Step 2: Monitor critique generation")
            monitor_critiques = self._generate_monitor_critiques(batch_data, policy_outputs)
            logger.debug(f"Monitor critiques generated: {len(monitor_critiques['critiques'])} critiques")
            
            # Log first critique for debugging
            if monitor_critiques['critiques']:
                logger.debug(f"Sample monitor critique: {monitor_critiques['critiques'][0][:200]}...")
            
            # 3. Reward evaluation: O → reward_scores
            logger.debug("Step 3: Reward model evaluation")
            reward_scores = self._evaluate_reward(batch_data, policy_outputs)
            logger.debug(f"Reward scores shape: {reward_scores.shape}")
            
            # 4. Judge evaluation: (O, O') → judge_scores
            logger.debug("Step 4: Judge model evaluation with critiques")
            judge_scores = self._evaluate_judge_with_critiques(batch_data, policy_outputs, monitor_critiques['critiques'])
            logger.debug(f"Judge scores shape: {judge_scores.shape}")
            
            # 5. Combined reward: R(O,O')
            logger.debug("Step 5: Computing combined reward")
            combined_scores = self._compute_combined_reward(reward_scores, judge_scores)
            logger.debug(f"Combined scores shape: {combined_scores.shape}")
            
            # 6. Truthfulness constraint: F(π_M)
            logger.debug("Step 6: Computing truthfulness constraint")
            ground_truth_correct = self.processor.compute_correctness_batch(
                policy_outputs['parsed_answers'], batch_data.correct_answer_ids
            )
            logger.debug(f"Ground truth shape: {ground_truth_correct.shape}")
            
            truthfulness_penalty = self._compute_truthfulness_constraint(
                monitor_critiques['critiques'], ground_truth_correct
            )
            logger.debug(f"Truthfulness penalty shape: {truthfulness_penalty.shape}")
            logger.debug(f"Truthfulness penalty mean: {truthfulness_penalty.mean().item():.4f}")
            
            # Log constraint analysis for first few samples
            if len(monitor_critiques['critiques']) > 0:
                sample_critique = monitor_critiques['critiques'][0]
                sample_gt = ground_truth_correct[0].item() if ground_truth_correct.numel() > 0 else "Unknown"
                sample_penalty = truthfulness_penalty[0].item() if truthfulness_penalty.numel() > 0 else "Unknown"
                logger.debug(f"Sample constraint analysis - GT: {sample_gt}, Penalty: {sample_penalty:.4f}")
                logger.debug(f"Sample critique snippet: {sample_critique[:150]}...")
            
            # Create MinMaxOutput (tensors stay on their native devices)
            minmax_output = MinMaxOutput(
                generated_tokens=policy_outputs['generated_tokens'],
                parsed_answers=policy_outputs['parsed_answers'],
                justifications=policy_outputs['justifications'],
                monitor_critiques=monitor_critiques['critiques'],
                monitor_generated_tokens=monitor_critiques['generated_tokens'],  # Add monitor generated tokens
                reward_scores=reward_scores,
                judge_scores=judge_scores,
                combined_scores=combined_scores,
                truthfulness_penalty=truthfulness_penalty,
                policy_input_lengths=policy_outputs['input_lengths'],
                monitor_input_lengths=monitor_critiques['input_lengths'],
                # Legacy fields for backward compatibility
                policy_input_ids=policy_outputs['input_ids'],
                policy_attention_mask=policy_outputs['attention_mask'],
                monitor_input_ids=monitor_critiques['input_ids'],
                monitor_attention_mask=monitor_critiques['attention_mask'],
                ground_truth_correct=ground_truth_correct,
                original_data=batch_data
            )
            
            logger.debug("MinMax forward pass completed successfully")
            return minmax_output
            
        except Exception as e:
            logger.error(f"Error in MinMax forward pass: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _generate_policy_responses(self, batch_data: RawBatchData, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Generate policy model responses"""
        # Create policy prompts
        policy_prompts = self.processor.create_policy_prompts_batch(
            batch_data.stories, batch_data.questions, batch_data.answer_choices
        )
        
        # Tokenize prompts
        policy_inputs = self.processor.tokenize_policy_batch(policy_prompts)
        
        # Generate responses
        generated_tokens = self.policy_model.generate_batch(
            policy_inputs['input_ids'], 
            policy_inputs['attention_mask'],
            max_new_tokens=max_new_tokens
        )
        
        # Extract and parse responses
        responses = self.processor.extract_responses_batch(
            generated_tokens, policy_inputs['input_lengths']
        )
        parsed_answers = self.processor.parse_answers_batch(responses, batch_data.answer_choices)
        justifications = responses  # For now, use full responses as justifications
        
        return {
            'generated_tokens': generated_tokens,
            'parsed_answers': parsed_answers,
            'justifications': justifications,
            'input_ids': policy_inputs['input_ids'],
            'attention_mask': policy_inputs['attention_mask'],
            'input_lengths': policy_inputs['input_lengths']
        }
    
    def _generate_monitor_critiques(self, batch_data: RawBatchData, policy_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate monitor critiques of policy outputs (efficient version)"""
        # Create monitor input prompts using dataset processor
        monitor_prompts = self.processor.create_monitor_critique_inputs_batch(
            batch_data.stories,
            batch_data.questions,
            batch_data.answer_choices,
            policy_outputs['parsed_answers'],
            policy_outputs['justifications']
        )
        
        # Tokenize monitor inputs
        monitor_inputs = self.processor.tokenize_monitor_batch(monitor_prompts)
        logger.debug(f"Monitor input_ids shape: {monitor_inputs['input_ids'].shape}")
        logger.debug(f"Monitor input lengths: {monitor_inputs['input_lengths'][:3]}...")  # First 3 lengths
        
        # Generate critiques using monitor model
        logger.debug(f"Calling monitor model generate_batch with max_new_tokens: {self.monitor_model.config.max_new_tokens}")
        generated_tokens = self.monitor_model.generate_batch(
            monitor_inputs['input_ids'],
            monitor_inputs['attention_mask'],
            max_new_tokens=self.monitor_model.config.max_new_tokens
        )
        
        logger.debug(f"Generated tokens shape: {generated_tokens.shape}")
        
        # Extract critiques from generated tokens
        critiques = self.processor.extract_monitor_responses_batch(
            generated_tokens, monitor_inputs['input_lengths']
        )
    
        return {
            'critiques': critiques,
            'generated_tokens': generated_tokens,  # Store the full generated sequence
            'input_ids': monitor_inputs['input_ids'],
            'attention_mask': monitor_inputs['attention_mask'],
            'input_lengths': monitor_inputs['input_lengths']
        }
    
    def _evaluate_reward(self, batch_data: RawBatchData, policy_outputs: Dict[str, Any]) -> torch.Tensor:
        """Evaluate policy outputs with reward model"""
        reward_prompts = self.processor.create_reward_inputs_batch(
            batch_data.questions, 
            batch_data.answer_choices, 
            policy_outputs['parsed_answers'], 
            policy_outputs['justifications']
        )
        
        reward_inputs = self.processor.tokenize_reward_batch(reward_prompts)
        reward_scores = self.reward_model.score_batch(
            reward_inputs['input_ids'], reward_inputs['attention_mask']
        )
        
        return reward_scores
    
    def _evaluate_judge_with_critiques(self, batch_data: RawBatchData, 
                                     policy_outputs: Dict[str, Any],
                                     monitor_critiques: List[str]) -> torch.Tensor:
        """Evaluate policy outputs + monitor critiques with judge model"""
        # Create judge prompts using dataset processor
        judge_prompts = self.processor.create_judge_with_critique_inputs_batch(
            batch_data.questions,
            batch_data.answer_choices,
            policy_outputs['justifications'],
            monitor_critiques
        )
        
        judge_inputs = self.processor.tokenize_judge_batch(judge_prompts)
        judge_scores = self.judge_model.judge_batch(
            judge_inputs['input_ids'], judge_inputs['attention_mask']
        )
        
        return judge_scores
    
    def _compute_combined_reward(self, reward_scores: torch.Tensor, judge_scores: torch.Tensor,
                                reward_weight: float = 0.5, judge_weight: float = 0.5) -> torch.Tensor:
        """Compute combined reward R(O,O')"""
        # Normalize scores
        normalized_reward = torch.tanh(reward_scores)
        normalized_judge = judge_scores  # Assuming judge scores are already normalized
        
        # Combine rewards
        combined_scores = (reward_weight * normalized_reward + 
                          judge_weight * normalized_judge)
        
        return combined_scores
    
    def _compute_truthfulness_constraint(self, monitor_critiques: List[str], 
                                       ground_truth_correct: torch.Tensor) -> torch.Tensor:
        """
        Compute truthfulness constraint F(π_M) using Ground Truth Alignment
        Monitor should critique wrong answers more harshly than correct ones
        """
        try:
            # Parse the new VERDICT/ARGUMENT format
            critique_verdicts = []
            
            for critique in monitor_critiques:
                critique_upper = critique.upper()
                
                # Look for verdict in the structured response
                if 'VERDICT:' in critique_upper:
                    # Extract verdict
                    verdict_start = critique_upper.find('VERDICT:') + 8
                    verdict_section = critique_upper[verdict_start:verdict_start+50]  # Look ahead 50 chars
                    
                    if 'CORRECT' in verdict_section:
                        verdict = 1.0  # Monitor thinks answer is correct
                    elif 'INCORRECT' in verdict_section:
                        verdict = 0.0  # Monitor thinks answer is incorrect  
                    else:
                        verdict = 0.5  # Uncertain/unclear verdict
                else:
                    # Fallback to sentiment analysis if no verdict format found
                    critique_lower = critique.lower()
                    positive_words = ['correct', 'accurate', 'good', 'right', 'valid']
                    negative_words = ['wrong', 'incorrect', 'error', 'mistake', 'false', 'flawed']
                    
                    pos_count = sum(1 for word in positive_words if word in critique_lower)
                    neg_count = sum(1 for word in negative_words if word in critique_lower)
                    
                    if pos_count > neg_count:
                        verdict = 1.0
                    elif neg_count > pos_count:
                        verdict = 0.0
                    else:
                        verdict = 0.5
                
                critique_verdicts.append(verdict)
            
            monitor_verdicts = torch.tensor(critique_verdicts, dtype=torch.float32, device=ground_truth_correct.device)
            
            # Truthfulness constraint: monitor's verdict should align with ground truth
            # Perfect alignment = 0 penalty, complete misalignment = 4 penalty
            constraint_violation = F.mse_loss(monitor_verdicts, ground_truth_correct)
            constraint_violation = torch.clamp(constraint_violation * 4.0, 0.0, 4.0)
            
            # Return as a tensor with the same batch dimension as other scores
            batch_size = len(monitor_critiques)
            return constraint_violation.expand(batch_size)
            
        except Exception as e:
            logger.error(f"Error in truthfulness constraint computation: {e}")
            logger.error(f"Monitor critiques sample: {monitor_critiques[:2] if monitor_critiques else 'None'}")
            # Return a moderate penalty as fallback
            batch_size = len(monitor_critiques)
            return torch.ones(batch_size, dtype=torch.float32, device=ground_truth_correct.device) * 2.0
    
    def evaluate_batch(self, batch_data: RawBatchData, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate a batch and return minmax-specific metrics
        
        Args:
            batch_data: Raw batch data
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with evaluation metrics
        """
        minmax_output = self.forward_pass(batch_data, max_new_tokens)
        
        # Compute metrics
        batch_size = len(batch_data.stories)
        
        # Basic accuracy metrics
        accuracy = minmax_output.ground_truth_correct.mean().item()
        avg_reward = minmax_output.reward_scores.mean().item()
        avg_judge = minmax_output.judge_scores.mean().item()
        avg_combined = minmax_output.combined_scores.mean().item()
        avg_truthfulness_penalty = minmax_output.truthfulness_penalty.mean().item()
        
        # Parsing success rate
        valid_answers = sum(1 for ans in minmax_output.parsed_answers if ans != "")
        parsing_success_rate = valid_answers / batch_size
        
        # Judge-ground truth agreement
        judge_binary = (minmax_output.judge_scores > 0.0).float()
        judge_accuracy = (judge_binary == minmax_output.ground_truth_correct).float().mean().item()
        
        # MinMax specific metrics
        monitor_critique_lengths = [len(critique.split()) for critique in minmax_output.monitor_critiques]
        avg_critique_length = sum(monitor_critique_lengths) / len(monitor_critique_lengths)
        
        metrics = {
            'batch_size': batch_size,
            'accuracy': accuracy,
            'avg_reward_score': avg_reward,
            'avg_judge_score': avg_judge,
            'avg_combined_score': avg_combined,
            'avg_truthfulness_penalty': avg_truthfulness_penalty,
            'parsing_success_rate': parsing_success_rate,
            'judge_accuracy': judge_accuracy,
            'avg_critique_length': avg_critique_length,
            'reward_scores': minmax_output.reward_scores.cpu().numpy().tolist(),
            'judge_scores': minmax_output.judge_scores.cpu().numpy().tolist(),
            'combined_scores': minmax_output.combined_scores.cpu().numpy().tolist(),
            'truthfulness_penalties': minmax_output.truthfulness_penalty.cpu().numpy().tolist(),
            'ground_truth_correct': minmax_output.ground_truth_correct.cpu().numpy().tolist(),
            'parsed_answers': minmax_output.parsed_answers,
            'monitor_critiques': minmax_output.monitor_critiques,
        }
        
        return metrics


def create_minmax_pipeline(policy_model: PolicyModel, monitor_model: MonitorModel,
                          reward_model: RewardModel, judge_model: JudgeModel, 
                          dataset_config: DatasetConfig, device_manager=None,
                          policy_model_config=None, monitor_model_config=None) -> MinMaxPipeline:
    """
    Factory function to create MinMax pipeline
    
    Args:
        policy_model: Initialized policy model
        monitor_model: Initialized monitor model
        reward_model: Initialized reward model
        judge_model: Initialized judge model
        dataset_config: Dataset configuration
        device_manager: Device manager for multi-GPU coordination
        policy_model_config: Policy model configuration
        monitor_model_config: Monitor model configuration
        
    Returns:
        Configured MinMax pipeline
    """
    processor = DatasetProcessor(dataset_config, policy_model_config, monitor_model_config, "minmax")
    return MinMaxPipeline(policy_model, monitor_model, reward_model, judge_model, processor, device_manager)

"""
Dataset-Agnostic Data Processor for RLHF Training
Handles all dataset-specific logic including prompt creation, tokenization, and response parsing
"""
import torch
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from string import Template

from ..config.dataset_config import DatasetConfig

logger = logging.getLogger(__name__)

@dataclass
class RawBatchData:
    """Raw input data before any processing"""
    stories: List[str]
    questions: List[str]
    answer_choices: List[List[str]]
    correct_answer_ids: List[int]
    indices: List[int]

@dataclass  
class ProcessedBatchData:
    """After dataset-specific processing"""
    policy_input_ids: torch.Tensor
    policy_attention_mask: torch.Tensor
    input_lengths: List[int]  # For response extraction
    original_data: RawBatchData

@dataclass
class PolicyOutput:
    """Policy model generation results"""
    generated_ids: torch.Tensor      # Raw generation tokens
    parsed_answers: List[str]        # A/B or 0/1 etc
    justifications: List[str]        # Extracted reasoning

@dataclass
class RLHFBatch:
    """Complete RLHF forward pass results"""
    generated_tokens: torch.Tensor
    parsed_answers: List[str] 
    justifications: List[str]
    reward_scores: torch.Tensor
    judge_scores: torch.Tensor
    ground_truth_correct: torch.Tensor
    original_data: RawBatchData
    # Store policy inputs for PPO training
    policy_input_ids: torch.Tensor
    policy_attention_mask: torch.Tensor
    policy_input_lengths: List[int]

@dataclass
class MinMaxOutput:
    """Complete output from MinMax pipeline with all necessary data"""
    # Core generation data
    generated_tokens: torch.Tensor       # Policy: input + generated response
    parsed_answers: List[str]
    justifications: List[str]
    monitor_critiques: List[str]
    monitor_generated_tokens: torch.Tensor  # Monitor: input + generated critique
    
    # Scores
    reward_scores: torch.Tensor          # R(O) - reward model scores
    judge_scores: torch.Tensor           # R(O') - judge scores with critiques
    combined_scores: torch.Tensor        # R(O,O') - final combined reward
    truthfulness_penalty: torch.Tensor  # F(π_M) - constraint violation
    
    # Training data for policy (generated_tokens + input_lengths is sufficient)
    policy_input_lengths: List[int]
    
    # Training data for monitor (monitor_generated_tokens + input_lengths is sufficient)
    monitor_input_lengths: List[int]
    
    # Ground truth and metadata
    ground_truth_correct: torch.Tensor
    original_data: RawBatchData

    # Legacy fields (kept for backward compatibility, but can be removed)
    policy_input_ids: Optional[torch.Tensor] = None
    policy_attention_mask: Optional[torch.Tensor] = None
    monitor_input_ids: Optional[torch.Tensor] = None
    monitor_attention_mask: Optional[torch.Tensor] = None


class DatasetProcessor:
    """
    Dataset-agnostic processor that handles all dataset-specific logic
    """
    
    def __init__(self, dataset_config: DatasetConfig, policy_model_config=None, 
                 monitor_model_config=None, pipeline_type: str = "rlhf"):
        self.config = dataset_config
        self.policy_model_config = policy_model_config
        self.monitor_model_config = monitor_model_config  
        self.pipeline_type = pipeline_type
        
        self.policy_tokenizer = None  # Set externally
        self.reward_tokenizer = None
        self.judge_tokenizer = None
        
        # Store template strings (not Template objects since we use .format())
        self._policy_template_str = None
        self._reward_template_str = self.config.reward_input_template
        self._judge_template_str = self.config.judge_input_template
        self._monitor_template_str = self.config.monitor_critique_template
        self._judge_with_critique_template_str = self.config.judge_with_critique_template
        
    def set_tokenizers(self, policy_tokenizer, reward_tokenizer, judge_tokenizer, monitor_tokenizer=None):
        """Set tokenizers for all models"""
        self.policy_tokenizer = policy_tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.judge_tokenizer = judge_tokenizer
        self.monitor_tokenizer = monitor_tokenizer or policy_tokenizer  # Use policy tokenizer as fallback
        
        # Ensure left padding for consistent batch processing (critical for tensor optimization)
        for tokenizer in [self.policy_tokenizer, self.monitor_tokenizer, self.judge_tokenizer]:
            if tokenizer is not None:
                tokenizer.padding_side = 'left'
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Tokenizers configured with left padding for optimized batch processing")
        
        # Update max_story_tokens based on pipeline type and model constraints
        if hasattr(self.config, 'get_policy_template'):
            policy_template = self.config.get_policy_template(getattr(policy_tokenizer, 'name_or_path', ''))
        else:
            policy_template = self.config.policy_prompt_template
            
        self._policy_template_str = policy_template
        
        # Calculate template overhead for the appropriate pipeline
        if self.pipeline_type == "minmax":
            # For MinMax: use monitor template overhead (which includes policy output)
            monitor_template = self.config.monitor_critique_template
            template_overhead_sample = monitor_template.replace('{story}', '')
            sample_overhead = template_overhead_sample.format(
                question="Sample question for token counting?",
                choice_a="Sample choice A for overhead calculation",
                choice_b="Sample choice B for overhead calculation", 
                selected_answer="A",
                ai_response="Sample AI response for overhead calculation that represents typical policy output length"
            )
            overhead_tokens = len(self.monitor_tokenizer.encode(sample_overhead, add_special_tokens=False))
            
            # Compute max story tokens using monitor model constraints
            self.config.compute_max_story_tokens(
                self.pipeline_type, 
                self.policy_model_config, 
                self.monitor_model_config, 
                overhead_tokens
            )
            logger.info(f"Computed max_story_tokens for MinMax pipeline: {self.config.max_story_tokens}")
            logger.info(f"Monitor template overhead: {overhead_tokens} tokens")
        else:
            # For RLHF: use policy template overhead
            policy_overhead_sample = policy_template.replace('{story}', '')
            sample_overhead = policy_overhead_sample.format(
                question="Sample question for token counting?",
                choice_a="Sample choice A for overhead calculation",
                choice_b="Sample choice B for overhead calculation"
            )
            overhead_tokens = len(policy_tokenizer.encode(sample_overhead, add_special_tokens=False))
            
            # Compute max story tokens using policy model constraints
            self.config.compute_max_story_tokens(
                self.pipeline_type, 
                self.policy_model_config, 
                None,  # No monitor needed for RLHF
                overhead_tokens
            )
            logger.info(f"Computed max_story_tokens for RLHF pipeline: {self.config.max_story_tokens}")
            logger.info(f"Policy template overhead: {overhead_tokens} tokens")
        
    # =====================================
    # Policy Model Processing
    # =====================================
    
    def create_policy_prompts_batch(self, stories: List[str], questions: List[str], 
                                  answer_choices_batch: List[List[str]]) -> List[str]:
        """Create prompts using dataset-specific template (stories should already be truncated)"""
        if not self._policy_template_str:
            raise RuntimeError("Policy tokenizer not set. Call set_tokenizers() first.")
        
        prompts = []
        for i, (story, question, choices) in enumerate(zip(stories, questions, answer_choices_batch)):
            if len(choices) != self.config.num_choices:
                raise ValueError(f"Expected {self.config.num_choices} choices, got {len(choices)}")
                
            prompt = self._policy_template_str.format(
                story=story,
                question=question,
                choice_a=choices[0],
                choice_b=choices[1]
            )

            prompts.append(prompt)
            
        return prompts
    
    def tokenize_policy_batch(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize prompts with appropriate truncation"""
        if not self.policy_tokenizer:
            raise RuntimeError("Policy tokenizer not set. Call set_tokenizers() first.")
        if not self.policy_model_config:
            raise RuntimeError("Policy model config not set. Pass it to DatasetProcessor constructor.")
            
        # Use config-based max_length calculation  
        max_input_length = (self.policy_model_config.max_length - 
                           self.policy_model_config.max_new_tokens - 
                           self.config.safety_margin)
            
        # Tokenize batch
        inputs = self.policy_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length
        )
        
        # Store input lengths for response extraction
        # Since we use left padding and truncation, all inputs have the same length
        input_length = inputs['input_ids'].size(1)
        input_lengths = [input_length] * inputs['input_ids'].size(0)
        
        # DEBUG: Log tokenization details
        logger.debug(f"Policy tokenization - max_input_length: {max_input_length}, actual input length: {input_length} (uniform for all {len(input_lengths)} samples)")
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'input_lengths': input_lengths
        }
    
    def truncate_stories_batch(self, stories: List[str], max_tokens: int) -> List[str]:
        """Apply dataset-specific truncation strategy to batch of stories"""
        if self.config.story_truncation_strategy == "intelligent":
            return [self._truncate_story_intelligently(story, max_tokens) for story in stories]
        elif self.config.story_truncation_strategy == "head":
            return [self._truncate_story_head(story, max_tokens) for story in stories]
        elif self.config.story_truncation_strategy == "tail":
            return [self._truncate_story_tail(story, max_tokens) for story in stories]
        elif self.config.story_truncation_strategy == "middle":
            return [self._truncate_story_middle(story, max_tokens) for story in stories]
        else:
            raise ValueError(f"Unknown truncation strategy: {self.config.story_truncation_strategy}")
    
    def _truncate_story_intelligently(self, story: str, max_tokens: int) -> str:
        """Intelligently truncate story preserving paragraph/sentence boundaries"""
        if not self.policy_tokenizer:
            return story[:max_tokens * 4]  # Rough estimate if no tokenizer
            
        # Try to keep complete paragraphs from the beginning
        paragraphs = story.split('\n\n')
        
        truncated_paragraphs = []
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            paragraph_tokens = len(self.policy_tokenizer.encode(paragraph, add_special_tokens=False))
            
            if current_tokens + paragraph_tokens <= max_tokens:
                truncated_paragraphs.append(paragraph)
                current_tokens += paragraph_tokens
            else:
                # Try to fit part of this paragraph
                remaining_tokens = max_tokens - current_tokens
                
                if remaining_tokens > 50:  # Only if we have reasonable space left
                    # Try to truncate at sentence boundaries
                    sentences = paragraph.split('. ')
                    for i, sentence in enumerate(sentences):
                        sentence_text = sentence + ('.' if i < len(sentences) - 1 else '')
                        sentence_tokens = len(self.policy_tokenizer.encode(sentence_text, add_special_tokens=False))
                        
                        if current_tokens + sentence_tokens <= max_tokens:
                            truncated_paragraphs.append(sentence_text)
                            current_tokens += sentence_tokens
                        else:
                            break
                break
        
        truncated_story = '\n\n'.join(truncated_paragraphs)
        
        # Final check: if still too long, do hard truncation
        final_tokens = self.policy_tokenizer.encode(truncated_story, add_special_tokens=False)
        if len(final_tokens) > max_tokens:
            truncated_tokens = final_tokens[:max_tokens]
            truncated_story = self.policy_tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            
        return truncated_story
    
    def _truncate_story_head(self, story: str, max_tokens: int) -> str:
        """Take first max_tokens from story"""
        if not self.policy_tokenizer:
            return story[:max_tokens * 4]
            
        tokens = self.policy_tokenizer.encode(story, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return story
            
        truncated_tokens = tokens[:max_tokens]
        return self.policy_tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    def _truncate_story_tail(self, story: str, max_tokens: int) -> str:
        """Take last max_tokens from story"""
        if not self.policy_tokenizer:
            return story[-max_tokens * 4:]
            
        tokens = self.policy_tokenizer.encode(story, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return story
            
        truncated_tokens = tokens[-max_tokens:]
        return self.policy_tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    def _truncate_story_middle(self, story: str, max_tokens: int) -> str:
        """Take tokens from middle of story"""
        if not self.policy_tokenizer:
            start_idx = max(0, len(story) // 2 - max_tokens * 2)
            return story[start_idx:start_idx + max_tokens * 4]
            
        tokens = self.policy_tokenizer.encode(story, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return story
            
        start_idx = max(0, len(tokens) // 2 - max_tokens // 2)
        truncated_tokens = tokens[start_idx:start_idx + max_tokens]
        return self.policy_tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    # =====================================
    # Response Processing
    # =====================================
    
    def extract_responses_batch(self, generated_tokens: torch.Tensor, 
                              input_lengths: List[int]) -> List[str]:
        """Extract responses from generated tokens using dataset strategy"""
        if not self.policy_tokenizer:
            raise RuntimeError("Policy tokenizer not set. Call set_tokenizers() first.")
        
        # Since we use left padding, all inputs have the same length
        input_length = input_lengths[0] if input_lengths else generated_tokens.size(1)
        
        # Extract generated tokens for all sequences at once
        generated_only_batch = generated_tokens[:, input_length:]
        
        responses = []
        for i, tokens in enumerate(generated_only_batch):
            # Decode the generated response
            response = self.policy_tokenizer.decode(tokens, skip_special_tokens=True)
            responses.append(response.strip())
            
        return responses
    
    def parse_answers_batch(self, responses: List[str], 
                          answer_choices_batch: List[List[str]]) -> List[str]:
        """Parse answers using dataset-specific strategy"""
        if self.config.answer_parsing_strategy == "ab_pattern":
            return [self._parse_ab_response(resp) for resp in responses]
        elif self.config.answer_parsing_strategy == "numeric":
            return [self._parse_numeric_response(resp) for resp in responses]
        elif self.config.answer_parsing_strategy == "content_match":
            return [self._parse_content_match_response(resp, choices) 
                   for resp, choices in zip(responses, answer_choices_batch)]
        else:
            raise ValueError(f"Unknown parsing strategy: {self.config.answer_parsing_strategy}")
    
    def _parse_ab_response(self, response: str) -> str:
        """Parse A/B pattern responses"""
        response = response.strip()
        
        if not response:
            return ""
        
        # Look for patterns like "A:" or "B:" at the beginning
        if re.match(r'^A\s*:', response, re.IGNORECASE):
            return "A"
        elif re.match(r'^B\s*:', response, re.IGNORECASE):
            return "B"
        elif re.match(r'^A\b', response, re.IGNORECASE):
            return "A"
        elif re.match(r'^B\b', response, re.IGNORECASE):
            return "B"
        else:
            # Look for patterns like "Answer A" or "Answer B"
            match = re.search(r'(?:Answer\s+|answer\s+)([AB])\b', response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
            else:
                # Look for A: or B: anywhere in the first 100 characters
                first_part = response[:100]
                a_match = re.search(r'\bA\s*:', first_part, re.IGNORECASE)
                b_match = re.search(r'\bB\s*:', first_part, re.IGNORECASE)
                
                if a_match and not b_match:
                    return "A"
                elif b_match and not a_match:
                    return "B"
                elif a_match and b_match:
                    return "A" if a_match.start() < b_match.start() else "B"
                else:
                    # Final fallback: look for standalone A or B in first few lines
                    lines = response.split('\n')[:3]
                    for line in lines:
                        if re.match(r'^\s*A\b', line, re.IGNORECASE):
                            return "A"
                        elif re.match(r'^\s*B\b', line, re.IGNORECASE):
                            return "B"
        
        return ""
    
    def _parse_numeric_response(self, response: str) -> str:
        """Parse numeric pattern responses (0/1)"""
        response = response.strip()
        
        # Look for numeric patterns
        patterns = [
            r'(?:Answer|answer):\s*(\d+)',
            r'^(\d+)(?:\.|:)',
            r'(?:choose|select|pick)\s*(?:answer\s*)?(\d+)',
            r'(?:option|choice)\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    num = int(match.group(1))
                    if num in [0, 1]:
                        return str(num)
                except ValueError:
                    continue
                    
        return ""
    
    def _parse_content_match_response(self, response: str, answer_choices: List[str]) -> str:
        """Parse responses by matching content with answer choices"""
        if len(answer_choices) != 2:
            return ""
            
        response_words = set(response.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                     'in', 'on', 'at', 'to', 'of', 'for', 'with', 'by'}
        response_words -= stop_words
        
        # Calculate overlap with each choice
        overlaps = []
        for choice in answer_choices:
            choice_words = set(choice.lower().split()) - stop_words
            overlap = len(choice_words.intersection(response_words)) / max(len(choice_words), 1)
            overlaps.append(overlap)
        
        # Return the choice with highest overlap if significant
        if max(overlaps) > 0.3:
            best_idx = overlaps.index(max(overlaps))
            return "A" if best_idx == 0 else "B"
            
        return ""
    
    def extract_justifications_batch(self, responses: List[str]) -> List[str]:
        """Extract reasoning/justification from responses"""
        justifications = []
        
        for response in responses:
            # Remove the answer choice part and keep the justification
            cleaned = response.strip()
            
            # Try to find where the answer ends and justification begins
            patterns = [
                r'^[AB]\s*[:\.]?\s*(.*)',  # "A: justification" or "B. justification"
                r'^(?:Answer\s+)?[AB]\s+(.*)',  # "Answer A justification"
            ]
            
            justification = ""
            for pattern in patterns:
                match = re.match(pattern, cleaned, re.IGNORECASE | re.DOTALL)
                if match:
                    justification = match.group(1).strip()
                    break
            
            # If no pattern matched, use the full response as justification
            if not justification:
                justification = cleaned
                
            justifications.append(justification)
            
        return justifications
    
    # =====================================
    # Reward/Judge Model Processing
    # =====================================
    
    def create_reward_inputs_batch(self, questions: List[str], answer_choices_batch: List[List[str]],
                                 selected_answers: List[str], justifications: List[str]) -> List[str]:
        """Create reward model inputs using dataset template"""
        inputs = []
        
        for question, choices, answer, justification in zip(questions, answer_choices_batch, selected_answers, justifications):
            input_text = self._reward_template_str.format(
                question=question,
                choice_a=choices[0],
                choice_b=choices[1],
                selected_answer=answer,
                justification=justification
            )
            inputs.append(input_text)
            
        return inputs
    
    def create_judge_inputs_batch(self, questions: List[str], answer_choices_batch: List[List[str]],
                                selected_answers: List[str], justifications: List[str]) -> List[str]:
        """Create judge model inputs using dataset template"""
        inputs = []
        
        for question, choices, answer, justification in zip(questions, answer_choices_batch, selected_answers, justifications):
            input_text = self._judge_template_str.format(
                question=question,
                choice_a=choices[0],
                choice_b=choices[1],
                selected_answer=answer,
                justification=justification
            )
            inputs.append(input_text)
            
        return inputs
    
    def create_monitor_critique_inputs_batch(self, stories: List[str], questions: List[str], 
                                           answer_choices_batch: List[List[str]],
                                           selected_answers: List[str], 
                                           ai_responses: List[str]) -> List[str]:
        """Create monitor critique inputs using dataset template"""
        inputs = []
        
        for i, (story, question, choices, answer, ai_response) in enumerate(zip(stories, questions, answer_choices_batch, selected_answers, ai_responses)):
            input_text = self._monitor_template_str.format(
                story=story,
                question=question,
                choice_a=choices[0],
                choice_b=choices[1],
                selected_answer=answer,
                ai_response=ai_response
            )
            inputs.append(input_text)
            
            # DEBUG: Log monitor input details for first sample
            if i == 0:
                story_tokens = len(self.monitor_tokenizer.encode(story)) if self.monitor_tokenizer else len(story.split())
                full_prompt_tokens = len(self.monitor_tokenizer.encode(input_text)) if self.monitor_tokenizer else len(input_text.split())
                logger.debug(f"Monitor input {i} - Story tokens: {story_tokens}, Full prompt tokens: {full_prompt_tokens}")
                logger.debug(f"Monitor input {i} - Selected answer: {answer}")
                logger.debug(f"Monitor input {i} - AI response preview: {ai_response}")
                # logger.debug(f"Monitor input {i} - Full prompt preview: {input_text}")
            
        return inputs
    
    def create_judge_with_critique_inputs_batch(self, questions: List[str], answer_choices_batch: List[List[str]],
                                              policy_outputs: List[str], monitor_critiques: List[str]) -> List[str]:
        """Create judge model inputs that include monitor critiques using dataset template"""
        inputs = []
        
        for question, choices, policy_output, monitor_critique in zip(questions, answer_choices_batch, policy_outputs, monitor_critiques):
            # Format choices
            choices_text = "\n".join([f"{chr(65+j)}) {choice}" for j, choice in enumerate(choices)])
            
            input_text = self._judge_with_critique_template_str.format(
                question=question,
                choices_text=choices_text,
                policy_output=policy_output,
                monitor_critique=monitor_critique
            )
            inputs.append(input_text)
            
        return inputs
    
    def tokenize_reward_batch(self, input_texts: List[str], reward_model_config=None) -> Dict[str, torch.Tensor]:
        """Tokenize reward model inputs"""
        if not self.reward_tokenizer:
            raise RuntimeError("Reward tokenizer not set. Call set_tokenizers() first.")
        
        # Use reward model config max_length if provided, otherwise use a reasonable default
        max_length = reward_model_config.max_length if reward_model_config else 512
            
        return self.reward_tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
    
    def tokenize_judge_batch(self, input_texts: List[str], judge_model_config=None) -> Dict[str, torch.Tensor]:
        """Tokenize judge model inputs"""
        if not self.judge_tokenizer:
            raise RuntimeError("Judge tokenizer not set. Call set_tokenizers() first.")
        
        # Use judge model config max_length if provided, otherwise use a reasonable default
        max_length = judge_model_config.max_length if judge_model_config else 1024
            
        return self.judge_tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
    
    def tokenize_monitor_batch(self, input_texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize monitor model inputs"""
        if not hasattr(self, 'monitor_tokenizer') or not self.monitor_tokenizer:
            # Use policy tokenizer as fallback
            tokenizer = self.policy_tokenizer
        else:
            tokenizer = self.monitor_tokenizer
            
        if not tokenizer:
            raise RuntimeError("Monitor/Policy tokenizer not set. Call set_tokenizers() first.")
        if not self.monitor_model_config:
            raise RuntimeError("Monitor model config not set. Pass it to DatasetProcessor constructor.")
            
        # Use config-based max_length calculation
        max_input_length = (self.monitor_model_config.max_length - 
                           self.monitor_model_config.max_new_tokens - 
                           self.config.safety_margin)
            
        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length
        )
        
        # Store input lengths for response extraction
        # Since we use left padding and truncation, all inputs have the same length
        input_length = inputs['input_ids'].size(1)
        input_lengths = [input_length] * inputs['input_ids'].size(0)
        inputs['input_lengths'] = input_lengths
        
        # DEBUG: Log tokenization details
        logger.debug(f"Monitor tokenization - max_input_length: {max_input_length}, actual input length: {input_length} (uniform for all {len(input_lengths)} samples)")
        
        return inputs
    
    # =====================================
    # Utility Methods
    # =====================================
    
    def compute_correctness_batch(self, parsed_answers: List[str], 
                                correct_answer_ids: List[int]) -> torch.Tensor:
        """Compute correctness for batch of answers"""
        correct = []
        
        for parsed, correct_id in zip(parsed_answers, correct_answer_ids):
            if self.config.answer_choices_format == "AB":
                # Convert A/B to 0/1
                if parsed == "A":
                    predicted_id = 0
                elif parsed == "B":
                    predicted_id = 1
                else:
                    predicted_id = -1  # Invalid/unparseable
            elif self.config.answer_choices_format == "01":
                # Already numeric
                try:
                    predicted_id = int(parsed)
                except (ValueError, TypeError):
                    predicted_id = -1
            else:
                raise ValueError(f"Unsupported answer format: {self.config.answer_choices_format}")
                
            is_correct = 1.0 if predicted_id == correct_id else 0.0
            correct.append(is_correct)
            
        return torch.tensor(correct, dtype=torch.float32)
    
    def evaluate_ground_truth_batch(self, parsed_answers: List[str], 
                                   correct_answer_ids: List[int]) -> torch.Tensor:
        """Alias for compute_correctness_batch for backward compatibility"""
        return self.compute_correctness_batch(parsed_answers, correct_answer_ids)
    
    def extract_monitor_responses_batch(self, generated_tokens: torch.Tensor, input_lengths: List[int]) -> List[str]:
        """Extract monitor responses from generated tokens"""
        if not hasattr(self, 'monitor_tokenizer') or not self.monitor_tokenizer:
            # Use policy tokenizer as fallback
            tokenizer = self.policy_tokenizer
        else:
            tokenizer = self.monitor_tokenizer
            
        if not tokenizer:
            raise RuntimeError("Monitor/Policy tokenizer not set. Call set_tokenizers() first.")
        
        # Since we use left padding, all inputs have the same length
        input_length = input_lengths[0] if input_lengths else generated_tokens.size(1)
        
        # Extract generated tokens for all sequences at once
        generated_only_batch = generated_tokens[:, input_length:]
        
        responses = []
        for i, tokens in enumerate(generated_only_batch):
            # Decode and clean response
            response = tokenizer.decode(tokens, skip_special_tokens=True)
            responses.append(response.strip())
            
            # DEBUG: Log monitor generation details for first sample
            if i == 0:
                full_sequence = tokenizer.decode(generated_tokens[i], skip_special_tokens=True)
                logger.debug(f"Monitor response {i} - Input length: {input_length}, Generated tokens: {len(tokens)}")
                logger.debug(f"Monitor response {i} - Full sequence length: {len(generated_tokens[i])}")
                logger.debug(f"Monitor response {i} - Generated response: '{response}'")
                # logger.debug(f"Monitor response {i} - Full sequence preview: {full_sequence}")
                if len(response) == 0:
                    logger.warning(f"Monitor response {i} - WARNING: Empty response generated!")
                    logger.debug(f"Monitor response {i} - Generated token IDs: {tokens.tolist()}")
        
        return responses
    
    def parse_monitor_judgments_batch(self, monitor_critiques: List[str]) -> List[str]:
        """
        Parse monitor critiques to extract CORRECT/INCORRECT judgments
        
        Args:
            monitor_critiques: List of monitor critique responses
            
        Returns:
            List of parsed judgments ("CORRECT", "INCORRECT", or "")
        """
        judgments = []
        
        for critique in monitor_critiques:
            judgment = self._parse_monitor_correctness_assessment(critique)
            judgments.append(judgment)
            
        return judgments
    
    def _parse_monitor_correctness_assessment(self, critique: str) -> str:
        """
        Parse individual monitor critique for correctness judgment
        
        Args:
            critique: Monitor critique text
            
        Returns:
            "CORRECT", "INCORRECT", or "" if unparseable
        """
        critique = critique.strip()
        
        if not critique:
            return ""
        
        # Convert to lowercase for pattern matching
        critique_lower = critique.lower()
        
        # Look for explicit correctness assessments
        correct_patterns = [
            r'\bcorrect\b',
            r'\bright\b',
            r'\btrue\b',
            r'\baccurate\b',
            r'\bvalid\b',
            r'answer is correct',
            r'response is correct',
            r'this is correct',
            r'the answer.*correct',
            r'correctly answered',
        ]
        
        incorrect_patterns = [
            r'\bincorrect\b',
            r'\bwrong\b',
            r'\bfalse\b',
            r'\binaccurate\b',
            r'\binvalid\b',
            r'\bmistaken\b',
            r'answer is wrong',
            r'response is wrong',
            r'this is wrong',
            r'the answer.*wrong',
            r'incorrectly answered',
            r'not correct',
            r'is not right',
        ]
        
        # Check for correct patterns
        for pattern in correct_patterns:
            if re.search(pattern, critique_lower):
                # Make sure it's not negated
                negation_patterns = [r'not\s+' + pattern, r'isn\'t\s+' + pattern, r'is\s+not\s+' + pattern]
                is_negated = any(re.search(neg_pattern, critique_lower) for neg_pattern in negation_patterns)
                if not is_negated:
                    return "CORRECT"
        
        # Check for incorrect patterns
        for pattern in incorrect_patterns:
            if re.search(pattern, critique_lower):
                # Make sure it's not negated (double negative = positive)
                negation_patterns = [r'not\s+' + pattern, r'isn\'t\s+' + pattern, r'is\s+not\s+' + pattern]
                is_negated = any(re.search(neg_pattern, critique_lower) for neg_pattern in negation_patterns)
                if not is_negated:
                    return "INCORRECT"
        
        # Look for judgment indicators with sentiment
        if re.search(r'\bgood\b|\bwell\b|\bsound\b|\bappropriate\b', critique_lower):
            return "CORRECT"
        elif re.search(r'\bbad\b|\bpoor\b|\bflawed\b|\binappropriate\b', critique_lower):
            return "INCORRECT"
        
        # Final fallback: look for positive/negative sentiment at the beginning
        first_sentences = '. '.join(critique.split('. ')[:2])  # First two sentences
        first_sentences_lower = first_sentences.lower()
        
        if re.search(r'^(yes|indeed|certainly|absolutely)', first_sentences_lower):
            return "CORRECT"
        elif re.search(r'^(no|never|definitely not|absolutely not)', first_sentences_lower):
            return "INCORRECT"
        
        # If no clear pattern found, return empty string
        return ""
    
    def compute_monitor_correctness_batch(self, monitor_judgments: List[str], 
                                        policy_correctness: torch.Tensor) -> torch.Tensor:
        """
        Compute monitor correctness scores (how often monitor correctly assesses policy)
        
        Args:
            monitor_judgments: List of monitor judgments ("CORRECT", "INCORRECT", "")
            policy_correctness: Ground truth correctness of policy answers (0/1)
            
        Returns:
            Tensor of monitor correctness scores (1 if monitor judgment matches ground truth, 0 otherwise)
        """
        monitor_correct = []
        
        for judgment, ground_truth in zip(monitor_judgments, policy_correctness.tolist()):
            if judgment == "":
                # Unparseable judgment = incorrect
                monitor_correct.append(0.0)
            elif judgment == "CORRECT" and ground_truth == 1.0:
                # Monitor says correct, and it is correct
                monitor_correct.append(1.0)
            elif judgment == "INCORRECT" and ground_truth == 0.0:
                # Monitor says incorrect, and it is incorrect
                monitor_correct.append(1.0)
            else:
                # Monitor judgment doesn't match ground truth
                monitor_correct.append(0.0)
                
        return torch.tensor(monitor_correct, dtype=torch.float32)

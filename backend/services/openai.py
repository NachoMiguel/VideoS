import os
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import re
import openai

from core.config import settings
from core.exceptions import AIGenerationError, APIError, APILimitError, OpenAIError
from core.parallel import parallel_processor, parallel_task
from core.credit_manager import credit_manager, ServiceType
from core.parallel_error_handler import OperationType
from core.logger import logger
from .text_cleaner import text_cleaner
# from .topic_driven_generator import TopicDrivenScriptGenerator # REMOVED - using lazy import

# logger = logging.getLogger(__name__)  # COMMENTED OUT - using core.logger instead

class OpenAIService:
    """Enhanced OpenAI service with parallel processing and context-aware script modifications."""
    
    def __init__(self):
        # Initialize basic configuration
        self.model = "gpt-4o"  # Latest and most creative
        self.max_tokens = 8000  # GPT-4o supports much higher limits
        self.temperature = 0.7
        self.timeout = 60.0  # INCREASED from 30.0 for longer responses
        
        # Initialize OpenAI client using credit manager
        try:
            account = credit_manager.get_available_account(ServiceType.OPENAI)
            self.client = AsyncOpenAI(
                api_key=account.api_key,
                timeout=self.timeout
            )
            logger.info(f"OpenAI service initialized with account: {account.account_id}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI service: {str(e)}")
            raise AIGenerationError(f"OpenAI service initialization failed: {str(e)}")
        
        # Load prompts from prompts.md
        self.prompts = self._load_prompts()
        
        # 🎯 NEW: Initialize ScriptQualityAnalyzer instance
        self.quality_analyzer = ScriptQualityAnalyzer(self)
        
        # Add topic-driven generator
        self.topic_generator = None  # Will be initialized when needed
        
        # Script modification actions
        self.modification_actions = {
            'shorten': 'Shorten',
            'expand': 'Expand', 
            'rewrite': 'Rewrite',
            'make_engaging': 'Make Engaging',
            'delete': 'Delete'
        }
        
        logger.info("OpenAI service initialized with credit management and quality analyzer")
    
    def _get_topic_generator(self):
        """Lazy initialization of topic generator to avoid circular import."""
        logger.info("🔍 _get_topic_generator called")
        
        if self.topic_generator is None:
            logger.info(" Topic generator is None, initializing...")
            try:
                from .topic_driven_generator import TopicDrivenScriptGenerator
                logger.info("✅ Successfully imported TopicDrivenScriptGenerator")
                
                self.topic_generator = TopicDrivenScriptGenerator(self)
                logger.info("✅ TopicDrivenScriptGenerator initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize TopicDrivenScriptGenerator: {str(e)}")
                logger.error(f"❌ Import error details:", exc_info=True)
                raise
        else:
            logger.info("✅ Topic generator already initialized")
        
        return self.topic_generator
    
    async def _generate_with_semantic_continuation_fallback(self, transcript: str, custom_prompt: Optional[str] = None) -> str:
        """Fallback to original semantic continuation method."""
        # This is your existing method as fallback
        if custom_prompt:
            prompt = custom_prompt.format(transcript=transcript)
        else:
            prompt_template = self.prompts.get('basic_youtube_content_analysis', 
                                                self._get_default_prompts()['basic_youtube_content_analysis'])
            prompt = prompt_template.format(transcript=transcript)
        
        optimal_tokens = self._calculate_optimal_tokens(transcript)
        return await self.quality_analyzer._generate_with_semantic_continuation(prompt, optimal_tokens, target_length=25000)
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompts from prompts.md file."""
        try:
            prompts_file = Path(__file__).parent.parent / "prompts.md"
            if not prompts_file.exists():
                logger.warning("prompts.md file not found, using default prompts")
                return self._get_default_prompts()
            
            with open(prompts_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            prompts = {}
            current_section = None
            current_prompt = []
            in_code_block = False
            
            for line in content.split('\n'):
                if line.startswith('### ') and not in_code_block:
                    if current_section and current_prompt:
                        prompts[current_section.lower().replace(' ', '_')] = '\n'.join(current_prompt).strip()
                    current_section = line[4:].strip()
                    current_prompt = []
                elif line.strip() == '```' and current_section:
                    in_code_block = not in_code_block
                    if not in_code_block and current_prompt:
                        prompts[current_section.lower().replace(' ', '_')] = '\n'.join(current_prompt).strip()
                        current_prompt = []
                elif in_code_block and current_section:
                    current_prompt.append(line)
            
            # Add the last prompt if exists
            if current_section and current_prompt:
                prompts[current_section.lower().replace(' ', '_')] = '\n'.join(current_prompt).strip()
            
            logger.info(f"Loaded {len(prompts)} prompts from prompts.md")
            return prompts
            
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            return self._get_default_prompts()
    
    def _get_default_prompts(self) -> Dict[str, str]:
        """Get default prompts if prompts.md is not available."""
        return {
            'basic_youtube_content_analysis': """
Please rewrite the following YouTube transcript into an engaging video script that will be used to create a new video by mixing scenes from multiple source videos.

Requirements:
1. Maintain the core message and key points from the original content
2. Make it more engaging and conversational for video format
3. Break it into clear segments that can be matched with video scenes
4. Add natural transitions between topics
5. Keep it concise and impactful for video consumption
6. Structure it in a way that allows for easy scene matching and video assembly
7. Preserve the educational or entertainment value of the original content

Original Transcript:
{transcript}

Please provide the rewritten script in a clear format with natural segments and smooth transitions that will work well for video assembly.
            """,
            'shorten': """
Shorten the following text while preserving its core meaning and maintaining natural flow with the surrounding context.

Context before: "{context_before}"
Text to shorten: "{selected_text}"
Context after: "{context_after}"

Provide only the shortened version that flows naturally with the context.
            """,
            'expand': """
Expand the following text with more detail, examples, or engaging content while maintaining the same tone and style.

Context before: "{context_before}"
Text to expand: "{selected_text}"
Context after: "{context_after}"

Provide only the expanded version that flows naturally with the context.
            """,
            'rewrite': """
Rewrite the following text to be more engaging and compelling while preserving the core message and maintaining flow with surrounding context.

Context before: "{context_before}"
Text to rewrite: "{selected_text}"
Context after: "{context_after}"

Provide only the rewritten version that flows naturally with the context.
            """,
            'make_engaging': """
Make the following text more engaging, dynamic, and compelling while preserving its meaning and maintaining natural flow.

Context before: "{context_before}"
Text to make engaging: "{selected_text}"
Context after: "{context_after}"

Provide only the more engaging version that flows naturally with the context.
            """,
            'delete': """
The user wants to delete this text: "{selected_text}"

Provide a smooth transition that connects the before and after context naturally:
Context before: "{context_before}"
Context after: "{context_after}"

Provide only the connecting text (can be empty if contexts connect naturally).
            """
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_script(self, transcript: str, custom_prompt: Optional[str] = None, video_id: str = None) -> str:
        """Generate a new script from the transcript using topic-driven approach."""
        try:
            # Check if transcript is too large for single processing
            if len(transcript) > 60000:  # >60k characters (≈15k tokens)
                logger.info(f"Large transcript detected ({len(transcript)} chars), using chunking approach")
                return await self._chunk_and_process_transcript(transcript, custom_prompt)
            
            # 🎯 NEW: Use topic-driven generation instead of semantic continuation
            logger.info("🎯 Using Topic-Driven Script Generation")
            logger.info("🎯 About to get topic generator...")
            
            topic_generator = self._get_topic_generator()  # Lazy initialization
            logger.info("✅ Topic generator obtained successfully")
            
            logger.info("🔍 About to call topic_generator.generate_script...")
            # 🎯 NEW: Pass video_id to topic generator
            result = await topic_generator.generate_script(transcript, video_id)
            logger.info(f"✅ Topic-driven generation successful: {len(result)} characters")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Topic-driven script generation failed: {str(e)}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            logger.error(f"❌ Full error details:", exc_info=True)
            
            # Fallback to original method
            logger.info("🔄 Falling back to original semantic continuation method")
            try:
                fallback_result = await self._generate_with_semantic_continuation_fallback(transcript, custom_prompt)
                logger.info(f"✅ Fallback generation successful: {len(fallback_result)} characters")
                return fallback_result
            except Exception as fallback_error:
                logger.error(f"❌ Fallback generation also failed: {str(fallback_error)}")
                logger.error(f"❌ Fallback error details:", exc_info=True)
                raise  # Re-raise the original error
    
    def _calculate_optimal_tokens(self, transcript: str) -> int:
        """Calculate optimal max_tokens based on transcript length and GPT-4o limits."""
        # Rough estimation: 1 token ≈ 4 characters
        input_tokens = len(transcript) // 4
        
        # GPT-4o limits: 16,384 completion tokens max, 128,000 total context
        MAX_COMPLETION_TOKENS = 16000  # Conservative (16,384 - buffer)
        MAX_CONTEXT_TOKENS = 125000    # Conservative (128,000 - buffer)
        
        # Estimate prompt tokens (system message + user prompt template)
        prompt_overhead = 800  # Larger prompt from prompts.md
        available_context = MAX_CONTEXT_TOKENS - prompt_overhead
        
        if input_tokens > available_context:
            # Transcript too long - will need chunking (unlikely with 125k context)
            logger.warning(f"Transcript ({input_tokens} tokens) exceeds context limit ({available_context}). Consider chunking.")
            optimal_tokens = MAX_COMPLETION_TOKENS
        else:
            # For 20,000-30,000 character target (5,000-7,500 tokens)
            target_chars = 30000  # Upper end of range for longer output
            target_tokens = target_chars // 4  # 7,500 tokens
            
            optimal_tokens = min(target_tokens, MAX_COMPLETION_TOKENS)
            optimal_tokens = max(2000, optimal_tokens)  # Minimum 2000 tokens
        
        logger.info(f"Calculated optimal tokens: {optimal_tokens} for transcript length: {len(transcript)} chars ({input_tokens} tokens)")
        return optimal_tokens
    
    async def _generate_with_continuation(self, initial_prompt: str, max_tokens: int, target_length: int = 25000) -> str:
        """Generate script with overlap-aware continuation system to prevent repetition."""
        MIN_LENGTH = 20000  # Minimum acceptable length
        MAX_CONTINUATIONS = 3  # Prevent infinite loops
        logger.info(f"INSIDE GENERATE WITH CONTINUATION---->BEFORE CHAT COMPLETION")

        # Step 1: Generate initial script
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert video script writer specializing in creating engaging content for video assembly."},
            {"role": "user", "content": initial_prompt}
            ],
            temperature=self.temperature,
        max_tokens=max_tokens,
            timeout=self.timeout
        )
        logger.info(f"INSIDE GENERATE WITH CONTINUATION---->AFTER CHAT COMPLETION")

        current_script = response.choices[0].message.content.strip()
        
        # DEBUG: Log initial response
        logger.info(f"DEBUG: Initial generation - finish_reason: {response.choices[0].finish_reason}")
        logger.info(f"DEBUG: Initial length: {len(current_script)} characters")
        logger.info(f"DEBUG: Initial tokens used: {response.usage.completion_tokens if hasattr(response, 'usage') else 'unknown'}")
        
        # Step 2: Continue if script is too short
        continuation_count = 0
        
        while len(current_script) < MIN_LENGTH and continuation_count < MAX_CONTINUATIONS:
            continuation_count += 1
            remaining_chars = target_length - len(current_script)
            
            logger.info(f"Script too short ({len(current_script)} chars). Continuation {continuation_count}/{MAX_CONTINUATIONS}")
            
            # Create overlap-aware continuation
            continuation_text = await self._generate_continuation_with_overlap_detection(
                current_script, remaining_chars, target_length, continuation_count
            )
            
            if continuation_text:
                # Append continuation with proper spacing
                if not current_script.endswith(('\n', ' ')):
                    current_script += '\n\n'
                current_script += continuation_text
                
                logger.info(f"DEBUG: Continuation {continuation_count} - Added {len(continuation_text)} chars")
                logger.info(f"DEBUG: Total length now: {len(current_script)} characters")
            else:
                logger.warning(f"Continuation {continuation_count} produced no new content")
                break
            
            # Small delay to avoid rate limits
            await asyncio.sleep(1)
        
        # Final result - Clean text for voice generation
        current_script = text_cleaner.clean_for_voice(current_script)
        
        if len(current_script) >= MIN_LENGTH:
            logger.info(f"SUCCESS: Reached target length {len(current_script)} characters after {continuation_count} continuations")
        else:
            logger.warning(f"WARNING: Script still short at {len(current_script)} characters after {MAX_CONTINUATIONS} continuations")
        
        return current_script
    
    async def _generate_continuation_with_overlap_detection(self, current_script: str, needed_chars: int, target_length: int) -> str:
        """Generate continuation with enhanced quality maintenance."""
        
        # Extract the last 2-3 paragraphs for context
        paragraphs = current_script.split('\n\n')
        context = '\n\n'.join(paragraphs[-3:]) if len(paragraphs) >= 3 else current_script[-2000:]
        
        # Enhanced continuation prompt with stronger quality focus
        continuation_prompt = f"""You are continuing a video script. Your task is to write NEW content that extends the narrative while maintaining HIGH QUALITY.

TARGET: Add approximately {needed_chars} characters to reach {target_length} total characters.

CONTEXT (last few paragraphs of existing script):
{context}

CRITICAL INSTRUCTIONS:
1. Write ONLY new content - do NOT repeat or rephrase any part of the existing script
2. Continue the narrative naturally from where it left off
3. Expand with new details, examples, anecdotes, or perspectives
4. Maintain the same engaging tone and writing style
5. Do NOT start by restating what has already been covered
6. FOCUS ON QUALITY: Each sentence must add value and advance the story
7. AVOID REPETITION: Do not repeat concepts already covered
8. MAINTAIN COHERENCE: Ensure logical flow and smooth transitions
9. ADD DEPTH: Provide new insights, analysis, or perspectives
10. KEEP ENGAGING: Maintain audience interest throughout

Write the continuation:"""

        try:
            continuation_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a script writer focused on creating new, non-repetitive content. Never repeat existing content. Maintain high quality and engagement."},
                    {"role": "user", "content": continuation_prompt}
                ],
                temperature=self.temperature,
                max_tokens=min(8000, needed_chars // 3),  # Conservative token limit
                timeout=self.timeout
            )
            
            continuation = continuation_response.choices[0].message.content.strip()
            return continuation
            
        except Exception as e:
            logger.error(f"Error generating continuation: {str(e)}")
            return ""

    def _extract_continuation_context(self, script: str, max_context_chars: int = 2000) -> str:
        """Extract the last few paragraphs as context for continuation."""
        paragraphs = script.split('\n\n')
        
        context = ""
        for paragraph in reversed(paragraphs):
            if len(context + paragraph) <= max_context_chars:
                context = paragraph + '\n\n' + context if context else paragraph
            else:
                break
        
        return context.strip()
    
    def _remove_overlap_with_existing_script(self, existing_script: str, new_content: str) -> str:
        """Advanced overlap detection and removal using multiple strategies."""
        
        # Strategy 1: Remove sentence-level overlaps
        clean_content = self._remove_sentence_overlaps(existing_script, new_content)
        
        # Strategy 2: Remove paragraph-level overlaps
        clean_content = self._remove_paragraph_overlaps(existing_script, clean_content)
        
        # Strategy 3: Remove phrase-level overlaps (for partial duplications)
        clean_content = self._remove_phrase_overlaps(existing_script, clean_content, min_phrase_length=50)
        
        return clean_content.strip()

    def _remove_sentence_overlaps(self, existing_script: str, new_content: str) -> str:
        """Remove sentences that already exist in the script."""
        import re
        
        # Split into sentences
        existing_sentences = set(re.split(r'[.!?]+', existing_script.lower()))
        new_sentences = re.split(r'[.!?]+', new_content)
        
        filtered_sentences = []
        for sentence in new_sentences:
            sentence_clean = sentence.strip().lower()
            if sentence_clean and sentence_clean not in existing_sentences and len(sentence_clean) > 10:
                filtered_sentences.append(sentence.strip())
        
        return '. '.join(filtered_sentences) + '.' if filtered_sentences else ""

    def _remove_paragraph_overlaps(self, existing_script: str, new_content: str) -> str:
        """Remove paragraphs that are too similar to existing ones."""
        existing_paragraphs = [p.strip().lower() for p in existing_script.split('\n\n') if p.strip()]
        new_paragraphs = [p.strip() for p in new_content.split('\n\n') if p.strip()]
        
        filtered_paragraphs = []
        for paragraph in new_paragraphs:
            paragraph_lower = paragraph.lower()
            
            # Check for high similarity with existing paragraphs
            is_duplicate = False
            for existing in existing_paragraphs:
                similarity = self._calculate_similarity(existing, paragraph_lower)
                if similarity > 0.7:  # 70% similarity threshold
                    logger.info(f"Removing duplicate paragraph (similarity: {similarity:.2f})")
                    is_duplicate = True
                    break
            
            if not is_duplicate and len(paragraph) > 50:  # Minimum paragraph length
                filtered_paragraphs.append(paragraph)
        
        return '\n\n'.join(filtered_paragraphs)

    def _remove_phrase_overlaps(self, existing_script: str, new_content: str, min_phrase_length: int = 50) -> str:
        """Remove phrases that appear in the existing script."""
        words = new_content.split()
        filtered_words = []
        i = 0
        
        while i < len(words):
            # Check for overlapping phrases of various lengths
            found_overlap = False
            
            for phrase_length in range(min(20, len(words) - i), 5, -1):  # Check 20 words down to 6 words
                phrase = ' '.join(words[i:i+phrase_length])
                
                if len(phrase) >= min_phrase_length and phrase.lower() in existing_script.lower():
                    logger.info(f"Removing overlapping phrase: '{phrase[:60]}...'")
                    i += phrase_length  # Skip the overlapping phrase
                    found_overlap = True
                    break
            
            if not found_overlap:
                filtered_words.append(words[i])
                i += 1
        
        return ' '.join(filtered_words)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using simple word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

class ScriptQualityAnalyzer:
    """Advanced semantic analysis for script quality validation."""
    
    def __init__(self, openai_service):
        self.openai_service = openai_service
        
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from response, handling markdown formatting."""
        try:
            # Remove markdown code blocks if present
            if response.strip().startswith('```'):
                lines = response.strip().split('\n')
                # Remove first line (```json or ```)
                if lines[0].startswith('```'):
                    lines = lines[1:]
                # Remove last line (```)
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                response = '\n'.join(lines)
            
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Raw response: {response}")
            return self._get_fallback_analysis()

    async def analyze_content_quality(self, script: str) -> Dict[str, Any]:
        """Comprehensive quality analysis using GPT-4 as semantic analyzer."""
        
        analysis_prompt = f"""Analyze this script for content quality and repetition issues:

SCRIPT TO ANALYZE:
{script}

Provide analysis in JSON format (NO MARKDOWN FORMATTING):
{{
    "narrative_progression": {{
        "score": 0.0-1.0,
        "issues": ["list of progression problems"],
        "repetitive_concepts": ["concepts that appear too often"]
    }},
    "content_density": {{
        "score": 0.0-1.0,
        "padding_detected": true/false,
        "low_value_sections": ["sections that don't add value"]
    }},
    "coherence": {{
        "score": 0.0-1.0,
        "transition_quality": 0.0-1.0,
        "logical_flow_breaks": ["places where flow breaks"]
    }},
    "repetition_analysis": {{
        "conceptual_duplicates": [
            {{"concept": "repeated concept", "occurrences": ["locations"], "severity": "low/medium/high"}}
        ],
        "redundant_paragraphs": ["paragraph numbers that are redundant"]
    }},
    "overall_quality": {{
        "score": 0.0-1.0,
        "is_production_ready": true/false,
        "improvement_needed": ["specific improvements needed"]
    }}
}}

Return ONLY the JSON object, no markdown formatting."""

        try:
            response = await self.openai_service.client.chat.completions.create(
                model=self.openai_service.model,
                messages=[
                    {"role": "system", "content": "You are a script quality analyst. Provide detailed analysis in JSON format only."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
                timeout=self.openai_service.timeout
            )
            
            analysis_text = response.choices[0].message.content.strip()
            return self._extract_json_from_response(analysis_text)
            
        except Exception as e:
            logger.error(f"Quality analysis failed: {str(e)}")
            return self._get_fallback_analysis()

    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when GPT analysis fails."""
        return {
            "narrative_progression": {"score": 0.8, "issues": ["Unable to analyze"], "repetitive_concepts": []},
            "content_density": {"score": 0.8, "padding_detected": False, "low_value_sections": []},
            "coherence": {"score": 0.8, "transition_quality": 0.8, "logical_flow_breaks": []},
            "repetition_analysis": {"conceptual_duplicates": [], "redundant_paragraphs": []},
            "overall_quality": {"score": 0.8, "is_production_ready": True, "improvement_needed": ["Manual review needed"]}
        }

    async def _generate_with_semantic_continuation(self, initial_prompt: str, max_tokens: int, target_length: int = 25000) -> str:
        """Generate script with semantic quality validation at each step."""
        
        MIN_QUALITY_SCORE = 0.7
        MIN_LENGTH = target_length * 0.8  # 80% of target
        MAX_CONTINUATIONS = 3
        
        # Step 1: Generate initial script
        logger.info("🎯 Generating initial script with quality focus")
        
        enhanced_prompt = f"""{initial_prompt}

    QUALITY REQUIREMENTS:
    - Each paragraph must introduce NEW information or perspectives
    - Avoid repeating the same concepts with different wording
    - Maintain clear narrative progression throughout
    - Focus on information density over padding
    - Ensure each section adds value to the overall story"""

        response = await self.openai_service.client.chat.completions.create(
            model=self.openai_service.model,
            messages=[
                {"role": "system", "content": "You are an expert video script writer focused on high-quality, non-repetitive content."},
                {"role": "user", "content": enhanced_prompt}
            ],
            temperature=self.openai_service.temperature,
            max_tokens=max_tokens,
            timeout=self.openai_service.timeout
        )

        current_script = response.choices[0].message.content.strip()
        
        # 🎯 DIAGNOSTIC: Log initial script length
        logger.info(f"🔍 DIAGNOSTIC: Initial script length: {len(current_script)} characters")
        logger.info(f"🔍 DIAGNOSTIC: Target length: {MIN_LENGTH} characters")
        logger.info(f"🔍 DIAGNOSTIC: Needs continuation: {len(current_script) < MIN_LENGTH}")
        
        # Step 2: Analyze initial quality
        quality_analysis = await self.analyze_content_quality(current_script)
        
        logger.info(f"📊 Initial Quality Analysis:")
        logger.info(f"   📈 Overall Score: {quality_analysis['overall_quality']['score']:.2f}")
        logger.info(f"   📚 Narrative Progression: {quality_analysis['narrative_progression']['score']:.2f}")
        logger.info(f"   🎯 Content Density: {quality_analysis['content_density']['score']:.2f}")
        logger.info(f"   🔗 Coherence: {quality_analysis['coherence']['score']:.2f}")
        
        # 🎯 DIAGNOSTIC: Log quality threshold check
        logger.info(f"🔍 DIAGNOSTIC: Quality threshold: {MIN_QUALITY_SCORE}")
        logger.info(f"🔍 DIAGNOSTIC: Quality meets threshold: {quality_analysis['overall_quality']['score'] >= MIN_QUALITY_SCORE}")
        
        # Step 3: Quality-aware continuation
        continuation_count = 0
        
        while (len(current_script) < MIN_LENGTH and 
            quality_analysis['overall_quality']['score'] >= MIN_QUALITY_SCORE and 
            continuation_count < MAX_CONTINUATIONS):
            
            continuation_count += 1
            logger.info(f"🔄 Quality-aware continuation {continuation_count}/{MAX_CONTINUATIONS}")
            
            # Generate continuation based on quality analysis
            continuation_text = await self._generate_quality_guided_continuation(
                current_script, quality_analysis, target_length, continuation_count
            )
            
            # 🎯 DIAGNOSTIC: Log continuation results
            logger.info(f"🔍 DIAGNOSTIC: Continuation {continuation_count} length: {len(continuation_text)} characters")
            logger.info(f"🔍 DIAGNOSTIC: Continuation {continuation_count} empty: {not continuation_text}")
            
            if continuation_text:
                # Test combined script quality before committing
                test_script = current_script + '\n\n' + continuation_text
                test_quality = await self.analyze_content_quality(test_script)
                
                # 🎯 DIAGNOSTIC: Log quality test results
                logger.info(f"🔍 DIAGNOSTIC: Test quality score: {test_quality['overall_quality']['score']:.2f}")
                logger.info(f"🔍 DIAGNOSTIC: Quality acceptable: {test_quality['overall_quality']['score'] >= MIN_QUALITY_SCORE}")
                
                # Only accept if quality doesn't degrade
                if test_quality['overall_quality']['score'] >= MIN_QUALITY_SCORE:
                    current_script = test_script
                    quality_analysis = test_quality
                    
                    logger.info(f"✅ Continuation accepted - Quality: {test_quality['overall_quality']['score']:.2f}")
                    logger.info(f"🔍 DIAGNOSTIC: Total length after continuation {continuation_count}: {len(current_script)} characters")
                else:
                    logger.warning(f"❌ Continuation rejected - Quality dropped to {test_quality['overall_quality']['score']:.2f}")
                    break
            else:
                logger.warning(f"🔍 DIAGNOSTIC: Continuation {continuation_count} produced no content")
                break
            
            await asyncio.sleep(1)
        
        # 🎯 DIAGNOSTIC: Log final state
        logger.info(f"🔍 DIAGNOSTIC: Final script length: {len(current_script)} characters")
        logger.info(f"🔍 DIAGNOSTIC: Continuations attempted: {continuation_count}")
        logger.info(f"🔍 DIAGNOSTIC: Length target met: {len(current_script) >= MIN_LENGTH}")
        
        # Final quality check
        final_quality = quality_analysis['overall_quality']['score']
        if final_quality < MIN_QUALITY_SCORE:
            logger.warning(f"⚠️ Final script quality below threshold: {final_quality}")
            
            # Try to improve quality, but ONLY if it doesn't make the script shorter
            original_length = len(current_script)
            improved_script = await self._improve_script_quality(current_script, quality_analysis)
            
            if len(improved_script) >= original_length:
                # Only use improvement if it doesn't make script shorter
                current_script = improved_script
                logger.info(f"✅ Quality improvement successful: {final_quality}")
            else:
                # Keep original script if improvement makes it shorter
                logger.warning(f"⚠️ Quality improvement rejected - would reduce length from {original_length} to {len(improved_script)} chars")
        
        # Clean and return
        current_script = text_cleaner.clean_for_voice(current_script)
        
        # Final logging
        logger.info(f"🏆 SEMANTIC CONTINUATION COMPLETE:")
        logger.info(f"   📊 Length: {len(current_script)} characters")
        logger.info(f"   📈 Quality Score: {final_quality:.2f}")
        logger.info(f"   ✅ Production Ready: {quality_analysis['overall_quality']['is_production_ready']}")
        
        return current_script

    async def _generate_quality_guided_continuation(self, current_script: str, quality_analysis: Dict, target_length: int, continuation_num: int) -> str:
        """Generate continuation specifically addressing quality analysis findings."""
        
        remaining_chars = target_length - len(current_script)
        
        # Extract specific guidance from quality analysis
        issues = quality_analysis['overall_quality'].get('improvement_needed', [])
        repetitive_concepts = quality_analysis['narrative_progression'].get('repetitive_concepts', [])
        
        quality_guidance = "Based on content analysis:\n"
        if issues:
            quality_guidance += f"- Address these issues: {', '.join(issues)}\n"
        if repetitive_concepts:
            quality_guidance += f"- Avoid repeating these concepts: {', '.join(repetitive_concepts)}\n"
        
        # Get fresh context (last paragraph only)
        context = self._extract_fresh_context(current_script)
        
        continuation_prompt = f"""Continue this script with HIGH QUALITY, non-repetitive content.

    TARGET: Add ~{remaining_chars} characters with VALUABLE new information.

    {quality_guidance}

    LAST PARAGRAPH FOR CONTEXT:
    {context}

    CONTINUATION REQUIREMENTS:
    1. Introduce COMPLETELY NEW aspects of the story
    2. Provide fresh insights, examples, or perspectives  
    3. Advance the narrative meaningfully
    4. NO repetition of existing concepts or information
    5. Focus on information density, not padding

    Write the continuation (NEW valuable content only):"""

        response = await self.openai_service.client.chat.completions.create(
            model=self.openai_service.model,
            messages=[
                {"role": "system", "content": "You are a script writer focused on high-quality, information-dense content. Never repeat existing information."},
                {"role": "user", "content": continuation_prompt}
            ],
            temperature=0.8,  # Higher creativity for fresh content
            max_tokens=min(6000, remaining_chars // 3),
            timeout=self.openai_service.timeout
        )
        
        return response.choices[0].message.content.strip()

    async def _improve_script_quality(self, script: str, quality_analysis: Dict) -> str:
        """Improve script quality based on analysis findings."""
        
        improvement_prompt = f"""Improve this script's quality by addressing the identified issues:

    ISSUES TO FIX:
    {json.dumps(quality_analysis['overall_quality']['improvement_needed'], indent=2)}

    REPETITIVE CONCEPTS TO REMOVE:
    {json.dumps(quality_analysis['narrative_progression']['repetitive_concepts'], indent=2)}

    SCRIPT TO IMPROVE:
    {script}

    INSTRUCTIONS:
    1. Remove or consolidate repetitive content
    2. Strengthen narrative progression
    3. Improve transitions and coherence
    4. Increase information density
    5. Maintain or increase length through QUALITY content, not padding

    Return the improved script:"""

        response = await self.openai_service.client.chat.completions.create(
            model=self.openai_service.model,
            messages=[
                {"role": "system", "content": "You are an expert script editor focused on quality improvement without repetition."},
                {"role": "user", "content": improvement_prompt}
            ],
            temperature=0.6,
            max_tokens=12000,
            timeout=self.openai_service.timeout
        )
        
        improved_script = response.choices[0].message.content.strip()
        logger.info(f"📝 Script quality improvement completed")
        
        return improved_script

    def _extract_fresh_context(self, script: str, max_chars: int = 500) -> str:
        """Extract minimal context to avoid feeding repetitive content back."""
        paragraphs = script.split('\n\n')
        return paragraphs[-1][-max_chars:] if paragraphs else ""

    

    
    
    
    async def _chunk_and_process_transcript(self, transcript: str, custom_prompt: Optional[str] = None) -> str:
        """Process very large transcripts by chunking them."""
        MAX_CHUNK_SIZE = 12000  # Characters per chunk (≈3000 tokens)
        
        # Split transcript into chunks
        chunks = []
        words = transcript.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > MAX_CHUNK_SIZE and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.info(f"Split transcript into {len(chunks)} chunks for processing")
        
        # Process each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Use a simpler prompt for chunks
            chunk_prompt = f"""
            Rewrite this part of a YouTube transcript into an engaging script segment.
            Make it conversational and compelling while preserving all key information.
            
            Transcript part {i+1}/{len(chunks)}:
            {chunk}
            
            Rewritten script segment:
            """
            
            account = credit_manager.get_available_account(ServiceType.OPENAI)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert script writer. Rewrite transcript segments into engaging scripts."},
                    {"role": "user", "content": chunk_prompt}
                ],
                temperature=self.temperature,
                max_tokens=3000,  # Safe limit for chunks
                timeout=self.timeout
            )
            
            chunk_result = response.choices[0].message.content.strip()
            chunk_results.append(chunk_result)
            
            # Small delay between chunks to avoid rate limits
            await asyncio.sleep(1)
        
        # Combine all chunk results
        final_script = "\n\n".join(chunk_results)
        
        logger.info(f"Combined {len(chunks)} chunks into final script: {len(final_script)} characters")
        return final_script
    
    @parallel_task('api_call')
    async def modify_script_context_aware(
        self,
        action: str,
        selected_text: str,
        context_before: str = "",
        context_after: str = ""
    ) -> str:
        """Modify script text with context awareness for natural flow."""
        try:
            if action not in self.modification_actions:
                raise ValueError(f"Invalid modification action: {action}")
            
            prompt_template = self.prompts.get(action, self._get_default_prompts()[action])
            prompt = prompt_template.format(
                context_before=context_before,
                selected_text=selected_text,
                context_after=context_after
            )
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert script editor. Provide only the modified text that flows naturally with the context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )
            
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Script modification error: {str(e)}")
            raise AIGenerationError(f"Failed to modify script: {str(e)}")
    
    async def bulk_modify_script(
        self,
        modifications: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Process multiple script modifications in parallel."""
        try:
            # Prepare parallel tasks
            api_calls = []
            for i, mod in enumerate(modifications):
                api_calls.append({
                    'func': self.modify_script_context_aware,
                    'args': [
                        mod['action'],
                        mod['selected_text'],
                        mod.get('context_before', ''),
                        mod.get('context_after', '')
                    ],
                    'kwargs': {},
                    'modification_id': mod.get('id', f'mod_{i}')
                })
            
            # Execute in parallel
            results = await parallel_processor.parallel_api_calls(api_calls, progress_callback)
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results):
                mod_data = modifications[i]
                formatted_results.append({
                    'id': mod_data.get('id', f'mod_{i}'),
                    'action': mod_data['action'],
                    'original_text': mod_data['selected_text'],
                    'modified_text': result.result if result.success else mod_data['selected_text'],
                    'success': result.success,
                    'error': str(result.error) if result.error else None,
                    'execution_time': result.execution_time
                })
            
            return formatted_results

        except Exception as e:
            logger.error(f"Bulk script modification error: {str(e)}")
            raise AIGenerationError(f"Failed to process bulk modifications: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def extract_characters(self, script: str) -> List[str]:
        """Extract character names from the script for face recognition."""
        try:
            prompt = f"""
            Analyze the following script and extract all character names mentioned. 
            Focus on real people, celebrities, public figures, or specific individuals.
            Ignore generic terms like "man", "woman", "person", etc.
            
            Script:
            {script}
            
            Return only a JSON array of character names, for example:
            ["John Doe", "Jane Smith", "Celebrity Name"]
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at identifying characters and people mentioned in scripts. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for consistency
                max_tokens=500,
                timeout=self.timeout
            )
            
            try:
                characters = json.loads(response.choices[0].message.content.strip())
                return characters if isinstance(characters, list) else []
            except json.JSONDecodeError:
                logger.warning("Failed to parse character extraction response as JSON")
                return []

        except Exception as e:
            logger.error(f"Character extraction error: {str(e)}")
            raise AIGenerationError(f"Failed to extract characters: {str(e)}")
    
    @parallel_task('api_call')
    async def analyze_script_segment(self, segment: str, segment_id: str) -> Dict[str, Any]:
        """Analyze a script segment for scene matching."""
        try:
            prompt = f"""
            Analyze this script segment for video scene matching:
            
            Segment:
            {segment}
            
            Provide analysis in JSON format:
            {{
                "characters": ["list of characters mentioned"],
                "emotions": ["list of emotions/moods"],
                "actions": ["list of actions described"],
                "context": "brief description of what's happening",
                "visual_cues": ["list of visual elements mentioned"],
                "duration_estimate": "estimated duration in seconds"
            }}
            """
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing scripts for video production. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800,
                timeout=self.timeout
            )
            
            try:
                analysis = json.loads(response.choices[0].message.content.strip())
                analysis['segment_id'] = segment_id
                return analysis
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse segment analysis response for {segment_id}")
                return {
                    "segment_id": segment_id,
                    "characters": [],
                    "emotions": [],
                    "actions": [],
                    "context": "Unknown",
                    "visual_cues": [],
                    "duration_estimate": "30"
                }

        except Exception as e:
            logger.error(f"Script segment analysis error: {str(e)}")
            raise AIGenerationError(f"Failed to analyze script segment: {str(e)}")
    
    async def parallel_script_analysis(
        self,
        script_segments: List[str],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Analyze multiple script segments in parallel."""
        try:
            # Prepare parallel tasks
            api_calls = []
            for i, segment in enumerate(script_segments):
                api_calls.append({
                    'func': self.analyze_script_segment,
                    'args': [segment, f'segment_{i}'],
                    'kwargs': {}
                })
            
            # Execute in parallel
            results = await parallel_processor.parallel_api_calls(api_calls, progress_callback)
            
            # Return successful results
            analyses = []
            for result in results:
                if result.success:
                    analyses.append(result.result)
                else:
                    logger.error(f"Segment analysis failed: {result.error}")
            
            return analyses

        except Exception as e:
            logger.error(f"Parallel script analysis error: {str(e)}")
            raise AIGenerationError(f"Failed to analyze script segments: {str(e)}")
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get OpenAI API usage statistics."""
        # This would typically integrate with OpenAI's usage API
        # For now, return basic stats
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "available_actions": list(self.modification_actions.keys()),
            "prompts_loaded": len(self.prompts)
        }
    
    async def extract_characters_from_script(self, script_text: str, account_info=None) -> List[str]:
        """Extract character names from script text using OpenAI with credit management."""
        try:
            # Get account info from credit manager if not provided
            if account_info is None:
                account_info = credit_manager.get_available_account(ServiceType.OPENAI)
            
            # Set the API key for this request
            openai.api_key = account_info.api_key
            
            prompt = f"""
            Extract all character names mentioned in this script. Return only the names, one per line.
            Focus on real people, celebrities, historical figures, or fictional characters.
            Ignore generic terms like "narrator", "voice", "speaker".
            
            Script:
            {script_text}
            
            Character names:
            """
            
            logger.info(f"Extracting characters using account: {account_info.account_id}")
            
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a script analyst. Extract character names accurately."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            characters = [name.strip() for name in content.split('\n') if name.strip()]
            
            # Remove duplicates and filter
            unique_characters = list(set(characters))
            filtered_characters = [char for char in unique_characters if len(char) > 1 and char.lower() not in ['narrator', 'voice', 'speaker']]
            
            logger.info(f"✅ Extracted {len(filtered_characters)} characters: {filtered_characters}")
            
            # Record usage will be handled by the parallel error handler
            return filtered_characters
            
        except openai.error.RateLimitError as e:
            logger.warning(f"OpenAI rate limit reached: {str(e)}")
            raise APILimitError(f"OpenAI rate limit exceeded: {str(e)}")
            
        except openai.error.InvalidRequestError as e:
            logger.error(f"Invalid OpenAI request: {str(e)}")
            raise OpenAIError(f"Invalid request: {str(e)}")
            
        except openai.error.AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: {str(e)}")
            raise OpenAIError(f"Authentication failed: {str(e)}")
            
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise OpenAIError(f"API error: {str(e)}")
            
        except Exception as e:
            logger.error(f"Unexpected error in character extraction: {str(e)}")
            # Fallback to regex-based extraction
            return self._extract_characters_fallback(script_text)
    
    def _extract_entities_from_text(self, text: str) -> list[str]:
        """Extract ONLY main character names from text."""
        import re
        
        entities = []
        
        # 🎯 PRECISE: Only extract complete, proper names
        name_patterns = [
            # Full names with hyphens: Jean-Claude Van Damme
            r'\b[A-Z][a-z]+(?:\-[A-Z][a-z]+)*\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            # Full names without hyphens: Jean Claude Van Damme
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            # Standard two-part names: Steven Seagal
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        # 🎯 FILTER: Remove common words and phrases
        common_words = {
            'The', 'This', 'That', 'With', 'And', 'But', 'For', 'You', 'All', 'New', 'Now', 'Old', 
            'How', 'What', 'Why', 'When', 'Where', 'Confirms', 'Truth', 'On', 'At', 'In', 'To'
        }
        
        # 🎯 VALIDATE: Only keep complete names (not partial phrases)
        filtered_entities = []
        for entity in entities:
            # Skip if contains common words
            if any(word in entity for word in common_words):
                continue
            # Skip if too short (likely not a full name)
            if len(entity) < 8:
                continue
            # Skip if contains "On" at the beginning (malformed)
            if entity.startswith('On '):
                continue
            filtered_entities.append(entity)
        
        # 🎯 DEBUG: Print what we found
        print(f"🔍 DEBUG: Extracted character entities: {filtered_entities}")
        
        return filtered_entities
    
    def _extract_characters_fallback(self, script_text: str) -> List[str]:
        """Fallback method to extract character names using regex."""
        try:
            # Look for capitalized names (likely characters)
            pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
            matches = re.findall(pattern, script_text)
            
            # Remove duplicates and filter common non-names
            characters = []
            seen = set()
            common_words = {'The', 'This', 'That', 'They', 'We', 'You', 'He', 'She', 'It'}
            
            for match in matches:
                if match not in seen and not any(word in match for word in common_words):
                    characters.append(match)
                    seen.add(match)
            
            # Limit to reasonable number
            return characters[:10]
            
        except Exception as e:
            logger.error(f"Fallback character extraction error: {str(e)}")
            return ["Unknown Character"]
    
    async def analyze_script_segments(self, script_text: str, account_info=None) -> List[Dict[str, Any]]:
        """Analyze script and break it into segments with OpenAI and credit management."""
        try:
            # Get account info from credit manager if not provided
            if account_info is None:
                account_info = credit_manager.get_available_account(ServiceType.OPENAI)
            
            # Set the API key for this request
            openai.api_key = account_info.api_key
            
            prompt = f"""
            Analyze this script and break it into segments for video compilation.
            For each segment, identify:
            1. Main characters mentioned
            2. Key actions or scenes described
            3. Emotional tone (exciting, calm, dramatic, etc.)
            4. Estimated duration in seconds
            
            Return as JSON array with this structure:
            [
              {{
                "text": "segment text",
                "characters": ["character1", "character2"],
                "actions": ["action1", "action2"],
                "tone": "tone description",
                "duration_estimate": 15
              }}
            ]
            
            Script:
            {script_text}
            """
            
            logger.info(f"Analyzing script segments using account: {account_info.account_id}")
            
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a video script analyst. Provide detailed segment analysis in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                import json
                segments = json.loads(content)
                logger.info(f"✅ Analyzed {len(segments)} script segments")
                return segments
            except json.JSONDecodeError:
                logger.warning("Failed to parse OpenAI JSON response, using fallback")
                return self._create_basic_segments(script_text)
                
        except Exception as e:
            logger.error(f"Error in script analysis: {str(e)}")
            return self._create_basic_segments(script_text)
    
    def _create_basic_segments(self, script_text: str) -> List[Dict[str, Any]]:
        """Create basic segments by splitting script into sentences."""
        try:
            sentences = re.split(r'[.!?]+', script_text)
            segments = []
            
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if len(sentence) > 10:  # Skip very short segments
                    segments.append({
                        "text": sentence,
                        "characters": [],
                        "actions": [],
                        "tone": "neutral",
                        "duration_estimate": max(3, len(sentence.split()) * 0.5)  # ~0.5 seconds per word
                    })
            
            return segments[:20]  # Limit to 20 segments
            
        except Exception as e:
            logger.error(f"Basic segment creation error: {str(e)}")
            return [{"text": script_text[:500], "characters": [], "actions": [], "tone": "neutral", "duration_estimate": 30}]

    

# REMOVE OR COMMENT OUT the global instance line:
# openai_service = OpenAIService()

# REPLACE with lazy initialization function:
_openai_service_instance = None

def get_openai_service():
    """Get or create OpenAI service instance (lazy initialization)"""
    global _openai_service_instance
    if _openai_service_instance is None:
        _openai_service_instance = OpenAIService()
    return _openai_service_instance

# For backward compatibility, create a lazy property-like object
class LazyOpenAIService:
    def __getattr__(self, name):
        return getattr(get_openai_service(), name)

openai_service = LazyOpenAIService() 
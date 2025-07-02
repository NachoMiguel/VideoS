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

from ..core.config import settings
from ..core.exceptions import AIGenerationError, APIError, APILimitError, OpenAIError
from ..core.parallel import parallel_processor, parallel_task
from ..core.credit_manager import credit_manager, ServiceType
from ..core.parallel_error_handler import OperationType

logger = logging.getLogger(__name__)

class OpenAIService:
    """Enhanced OpenAI service with parallel processing and context-aware script modifications."""
    
    def __init__(self):
        # Remove the direct API key setting - will be managed by credit manager
        self.model = "gpt-3.5-turbo"
        self.max_tokens = 4000
        logger.info("OpenAI service initialized with credit management")
        
        # Load prompts from prompts.md
        self.prompts = self._load_prompts()
        
        # Script modification actions
        self.modification_actions = {
            'shorten': 'Shorten',
            'expand': 'Expand', 
            'rewrite': 'Rewrite',
            'make_engaging': 'Make Engaging',
            'delete': 'Delete'
        }
    
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
    async def generate_script(self, transcript: str, custom_prompt: Optional[str] = None) -> str:
        """Generate a new script from the transcript using the default or custom prompt."""
        try:
            if custom_prompt:
                prompt = custom_prompt.format(transcript=transcript)
            else:
                prompt_template = self.prompts.get('basic_youtube_content_analysis', 
                                                 self._get_default_prompts()['basic_youtube_content_analysis'])
                prompt = prompt_template.format(transcript=transcript)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert video script writer specializing in creating engaging content for video assembly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )
            
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Script generation error: {str(e)}")
            raise AIGenerationError(f"Failed to generate script: {str(e)}")
    
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

# Global service instance
openai_service = OpenAIService() 
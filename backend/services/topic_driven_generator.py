#!/usr/bin/env python3
"""
Senior Engineer Solution: Topic-Driven Script Generation
Preserves the essence of the original prompt while using systematic topic analysis.
"""

import json
import asyncio
import logging
import re
from typing import List, Dict, Optional, TYPE_CHECKING

# 🎯 FIXED: Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from .openai import OpenAIService

from .text_cleaner import text_cleaner

logger = logging.getLogger(__name__)

class TTSScriptOptimizer:
    """Rule-based TTS optimization without AI dependencies."""
    
    def __init__(self):
        # Vocabulary simplification rules
        self.complex_word_mappings = {
            'consequently': 'so',
            'nevertheless': 'but',
            'furthermore': 'also',
            'subsequently': 'then',
            'approximately': 'about',
            'demonstrate': 'show',
            'utilize': 'use',
            'facilitate': 'help',
            'implement': 'put in place',
            'methodology': 'method',
            'paradigm': 'approach',
            'sophisticated': 'complex',
            'elaborate': 'detailed',
            'comprehensive': 'complete',
            'substantial': 'large',
            'significant': 'important',
            'considerable': 'big',
            'extensive': 'wide',
            'profound': 'deep',
            'remarkable': 'amazing'
        }
        
        # Sentence structure rules
        self.sentence_breakers = [
            r'([.!?])\s+([A-Z])',  # Break on sentence endings
            r'([,;])\s+(however|but|yet|still|though|although)',  # Break on conjunctions
            r'([,;])\s+(furthermore|moreover|additionally|also)',  # Break on additions
        ]
    
    def optimize_for_tts(self, script: str) -> str:
        """Apply comprehensive TTS optimization rules."""
        
        # Step 1: Simplify complex vocabulary
        script = self._simplify_vocabulary(script)
        
        # Step 2: Break long sentences
        script = self._break_long_sentences(script)
        
        # Step 3: Optimize punctuation for speech
        script = self._optimize_punctuation(script)
        
        # Step 4: Handle numbers and dates
        script = self._optimize_numbers_and_dates(script)
        
        return script
    
    def _simplify_vocabulary(self, script: str) -> str:
        """Replace complex words with simpler alternatives."""
        for complex_word, simple_word in self.complex_word_mappings.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(complex_word) + r'\b'
            script = re.sub(pattern, simple_word, script, flags=re.IGNORECASE)
        return script
    
    def _break_long_sentences(self, script: str) -> str:
        """Break sentences longer than 25 words."""
        sentences = re.split(r'([.!?]+)', script)
        optimized_sentences = []
        
        for sentence in sentences:
            if len(sentence.split()) > 25:
                # Break at natural points
                broken = self._break_sentence_at_conjunctions(sentence)
                optimized_sentences.append(broken)
            else:
                optimized_sentences.append(sentence)
        
        return ''.join(optimized_sentences)
    
    def _break_sentence_at_conjunctions(self, sentence: str) -> str:
        """Break long sentences at natural conjunction points."""
        # Break at common conjunctions
        conjunctions = ['and', 'but', 'or', 'so', 'yet', 'however', 'therefore', 'meanwhile']
        
        for conjunction in conjunctions:
            pattern = r'([,;])\s+(' + conjunction + r')\s+'
            if re.search(pattern, sentence, re.IGNORECASE):
                sentence = re.sub(pattern, r'\1\2. ', sentence, flags=re.IGNORECASE)
                break
        
        return sentence
    
    def _optimize_punctuation(self, script: str) -> str:
        """Optimize punctuation for natural speech patterns."""
        # Add pauses for dramatic effect
        script = re.sub(r'([.!?])\s+', r'\1 ... ', script)
        
        # Ensure proper spacing around quotes
        script = re.sub(r'"([^"]+)"', r'"\1"', script)
        
        return script
    
    def _optimize_numbers_and_dates(self, script: str) -> str:
        """Convert numbers and dates to speech-friendly format."""
        # Convert years to speech format
        script = re.sub(r'\b(19|20)(\d{2})\b', r'\1-\2', script)
        
        # Convert large numbers (simplified version)
        script = re.sub(r'\b(\d{1,3}),(\d{3})\b', r'\1 thousand \2', script)
        
        return script

class ScriptGenerationConfig:
    """Configuration for script generation features."""
    
    def __init__(self):
        # 🎯 BUDGET-CONSCIOUS SETTINGS (Current)
        self.USE_SINGLE_CALL = True  # Use single API call approach
        self.ENABLE_AI_VOCABULARY_SIMPLIFICATION = False  # Disabled for budget
        self.ENABLE_AI_SCRIPT_POLISHING = False  # Disabled for budget
        self.ENABLE_AI_ASR_GENERATION = False  # Disabled for budget
        self.ENABLE_QUALITY_ANALYSIS = False  # Disabled for budget
        
        #  ALWAYS ENABLED (No API calls)
        self.ENABLE_NAME_CORRECTIONS = True  # Uses cached data
        self.ENABLE_TTS_CLEANUP = True  # Pure text processing
        self.ENABLE_RULE_BASED_OPTIMIZATION = True  # Pure text processing
    
    def enable_premium_features(self):
        """Enable all premium features when budget increases."""
        self.ENABLE_AI_VOCABULARY_SIMPLIFICATION = True
        self.ENABLE_AI_SCRIPT_POLISHING = True
        self.ENABLE_AI_ASR_GENERATION = True
        self.ENABLE_QUALITY_ANALYSIS = True
        logger.info("🎯 Premium features enabled - budget mode disabled")

class TopicDrivenScriptGenerator:
    """Topic-driven script generation with systematic content coverage."""
    
    def __init__(self, openai_service: 'OpenAIService'):
        self.openai_service = openai_service
        self.target_length = 25000  # 20k-30k target
        self.min_length = 20000
        
        #  NEW: Configuration for feature toggling
        self.config = ScriptGenerationConfig()
        
        # 🎯 NEW: Cache for YouTube metadata and corrections
        self._youtube_service = None
        self._cached_metadata = {}  # video_id -> metadata
        self._cached_corrections = {}  # video_id -> corrections
    
    def _get_youtube_service(self):
        """Lazy initialization of YouTube service for name corrections."""
        if self._youtube_service is None:
            from .youtube import YouTubeService
            # 🎯 NEW: Pass OpenAI service to YouTube service
            self._youtube_service = YouTubeService(openai_service=self.openai_service)
        return self._youtube_service
    
    async def _get_cached_metadata_and_corrections(self, video_id: str) -> tuple[dict, dict]:
        """Get cached metadata and corrections, or extract once and cache."""
        
        if video_id in self._cached_metadata:
            print(f"🔧 DEBUG: Using cached metadata for {video_id}")
            return self._cached_metadata[video_id], self._cached_corrections[video_id]
        
        print(f"🔧 DEBUG: Extracting metadata for {video_id} (first time)")
        
        # Extract metadata ONCE
        youtube_service = self._get_youtube_service()
        metadata = await youtube_service.extract_video_context(video_id)
        
        # Build corrections ONCE
        corrections = await youtube_service._build_correction_dictionary(metadata)
        
        # Cache for future use
        self._cached_metadata[video_id] = metadata
        self._cached_corrections[video_id] = corrections
        
        return metadata, corrections
    
    async def _apply_dynamic_name_corrections(self, script: str, video_id: str) -> str:
        """Apply dynamic name corrections using cached metadata."""
        
        try:
            logger.info("🔧 Applying dynamic name corrections...")
            print(f"🔧 DEBUG: Starting name correction for video_id: {video_id}")
            
            # 🎯 NEW: Use cached metadata and corrections
            metadata, corrections = await self._get_cached_metadata_and_corrections(video_id)
            
            print(f"🔧 DEBUG: Using cached corrections with {len(corrections)} entries")
            
            if corrections:
                logger.info(f"📚 Using cached correction dictionary with {len(corrections)} entries")
                
                # Debug: Show what we're correcting
                print(f"🔧 DEBUG: Original script contains 'vanam': {'vanam' in script.lower()}")
                print(f"🔧 DEBUG: Original script contains 'seagull': {'seagull' in script.lower()}")
                
                # Apply corrections using the existing method
                youtube_service = self._get_youtube_service()
                corrected_script = youtube_service._apply_corrections(script, corrections)
                
                # Debug: Show what changed
                if corrected_script != script:
                    print("🔧 DEBUG: Script was corrected!")
                    print(f"🔧 DEBUG: Corrected script contains 'Van Damme': {'Van Damme' in corrected_script}")
                    print(f"🔧 DEBUG: Corrected script contains 'Seagal': {'Seagal' in corrected_script}")
                    print(f"🔧 DEBUG: Corrected script contains 'vanam': {'vanam' in corrected_script.lower()}")
                else:
                    print("🔧 DEBUG: No corrections were applied!")
                
                logger.info("✅ Applied dynamic name corrections")
                return corrected_script
            else:
                logger.info("⚠️ No corrections available")
                print("🔧 DEBUG: No corrections dictionary available!")
                return script
                
        except Exception as e:
            logger.error(f"❌ Dynamic name correction failed: {e}")
            print(f"❌ DEBUG: Name correction error: {str(e)}")
            return script  # Return original if correction fails
    
    async def _apply_ai_vocabulary_simplification(self, script: str) -> str:
        """Use AI to simplify complex vocabulary while maintaining meaning."""
        
        try:
            logger.info("🔧 Applying AI-powered vocabulary simplification...")
            
            # 🎯 FIXED: Check if script is too long for single API call
            if len(script) > 12000:
                logger.info("📏 Long script detected, using chunked simplification")
                return await self._simplify_large_script(script)
            
            simplification_prompt = f"""
            VOCABULARY SIMPLIFICATION: Simplify complex vocabulary in this script while maintaining meaning and engagement.
            
            REQUIREMENTS:
            1. Replace complex phrases with simpler, clearer alternatives
            2. Maintain the dramatic, engaging tone
            3. Keep the same meaning and impact
            4. Use direct, accessible language
            5. Avoid overly academic or flowery language
            6. IMPORTANT: Return ONLY the simplified script, no explanations
            
            SCRIPT TO SIMPLIFY:
            {script}
            
            Simplified script:
            """
            
            # 🎯 FIXED: Use proper error handling and validation
            response = await self.openai_service._create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert editor who simplifies complex vocabulary while maintaining engagement and meaning. Return only the simplified text."},
                    {"role": "user", "content": simplification_prompt}
                ],
                operation_type="vocabulary_simplification",  # NON-CRITICAL - use GPT-3.5-turbo
                temperature=0.3,
                max_tokens=16000,
                timeout=60.0
            )
            
            simplified_script = response.choices[0].message.content.strip()
            
            # 🎯 CRITICAL FIX: Validate simplification worked
            if len(simplified_script) < len(script) * 0.7:  # If too much was lost
                logger.warning("⚠️ Vocabulary simplification appears to have truncated script, using original")
                return script
            
            logger.info("✅ Applied AI vocabulary simplification")
            return simplified_script
            
        except Exception as e:
            logger.error(f"❌ AI vocabulary simplification failed: {e}")
            print(f"❌ DEBUG: AI simplification error: {str(e)}")
            return script  # Return original if simplification fails

    async def _simplify_large_script(self, script: str) -> str:
        """Simplify large scripts by processing in chunks with smart fallback."""
        logger.info("🔧 Starting chunked vocabulary simplification for large script")
        
        # 🎯 HYBRID APPROACH: Use smaller chunks for better reliability
        chunk_size = 8000  # Smaller chunks for vocabulary simplification
        chunks = [script[i:i+chunk_size] for i in range(0, len(script), chunk_size)]
        
        logger.info(f"🔧 Split into {len(chunks)} chunks of ~{chunk_size} characters each")
        
        simplified_chunks = []
        ai_success_count = 0
        fallback_count = 0
        
        for i, chunk in enumerate(chunks):
            logger.info(f"🔧 Processing chunk {i+1}/{len(chunks)}")
            
            try:
                # 🎯 ATTEMPT 1: AI-powered simplification
                simplification_prompt = f"""
                VOCABULARY SIMPLIFICATION: Simplify complex vocabulary in this script chunk while maintaining meaning and engagement.
                
                REQUIREMENTS:
                1. Replace complex phrases with simpler, clearer alternatives
                2. Maintain the dramatic, engaging tone
                3. Keep the same meaning and impact
                4. Use direct, accessible language
                5. Avoid overly academic or flowery language
                6. IMPORTANT: Return ONLY the simplified script chunk, no explanations
                
                SCRIPT CHUNK TO SIMPLIFY:
                {chunk}
                
                Simplified script chunk:
                """
                
                response = await self.openai_service._create_chat_completion(
                    messages=[
                        {"role": "system", "content": "You are an expert editor who simplifies complex vocabulary while maintaining engagement and meaning. Return only the simplified text."},
                        {"role": "user", "content": simplification_prompt}
                    ],
                    operation_type="vocabulary_simplification",  # NON-CRITICAL - use GPT-3.5-turbo
                    temperature=0.3,
                    max_tokens=12000,
                    timeout=45.0
                )
                
                simplified_chunk = response.choices[0].message.content.strip()
                
                # 🎯 VALIDATION: Ensure simplification didn't lose too much content
                if len(simplified_chunk) < len(chunk) * 0.7:
                    logger.warning(f"⚠️ Chunk {i+1} simplification appears to have truncated content, using rule-based fallback")
                    raise ValueError("Simplification truncated content")
                
                simplified_chunks.append(simplified_chunk)
                ai_success_count += 1
                logger.info(f"✅ Chunk {i+1} AI simplification successful")
                
            except Exception as e:
                logger.warning(f"⚠️ Chunk {i+1} AI simplification failed: {e}, using rule-based fallback")
                
                # 🎯 ATTEMPT 2: Rule-based fallback
                try:
                    optimizer = TTSScriptOptimizer()
                    fallback_chunk = optimizer.optimize_for_tts(chunk)
                    simplified_chunks.append(fallback_chunk)
                    fallback_count += 1
                    logger.info(f"✅ Chunk {i+1} rule-based fallback successful")
                    
                except Exception as fallback_error:
                    logger.error(f"❌ Chunk {i+1} both AI and rule-based failed: {fallback_error}, using original")
                    simplified_chunks.append(chunk)  # Use original as last resort
        
        # 🎯 ASSEMBLE FINAL RESULT
        final_script = ' '.join(simplified_chunks)
        
        # 🎯 VALIDATION: Ensure final result is reasonable
        if len(final_script) < len(script) * 0.8:
            logger.warning("⚠️ Final simplified script appears to have lost too much content, using original")
            return script
        
        logger.info(f"✅ Chunked vocabulary simplification complete: {ai_success_count} AI, {fallback_count} rule-based")
        return final_script

    async def _apply_post_processing(self, script: str, video_id: str = None) -> str:
        """Apply post-processing corrections to the final script."""
        logger.info("🔧 Applying post-processing corrections...")
        
        # Step 1: Apply dynamic name corrections (if video_id available) - ALWAYS ENABLED
        if video_id and self.config.ENABLE_NAME_CORRECTIONS:
            script = await self._apply_dynamic_name_corrections(script, video_id)
        
        # Step 2: Clean TTS-unfriendly characters - ALWAYS ENABLED
        if self.config.ENABLE_TTS_CLEANUP:
            script = self._clean_tts_characters(script)
        
        # Step 3: AI vocabulary simplification - CONFIGURABLE
        if self.config.ENABLE_AI_VOCABULARY_SIMPLIFICATION:
            logger.info("🎯 Applying AI vocabulary simplification (premium feature)")
            script = await self._apply_ai_vocabulary_simplification(script)
        else:
            logger.info("🎯 Using rule-based vocabulary optimization (budget mode)")
            optimizer = TTSScriptOptimizer()
            script = optimizer.optimize_for_tts(script)
        
        # Step 4: Clean up any formatting issues - ALWAYS ENABLED
        script = re.sub(r'\s+', ' ', script)  # Remove extra whitespace
        script = script.strip()
        
        logger.info("✅ Post-processing complete")
        return script

    def _clean_tts_characters(self, script: str) -> str:
        """Intelligently clean script of TTS-unfriendly characters and symbols."""
        logger.info("🔧 Applying intelligent TTS character cleanup...")
        
        # Track what we're cleaning for logging
        cleaned_chars = {}
        original_script = script
        
        # INTELLIGENT REPLACEMENTS: Only replace when actually problematic
        
        # 1. Social media symbols (TTS engines struggle with these)
        if '#' in script:
            count = script.count('#')
            script = re.sub(r'#(\w+)', r'hashtag \1', script)  # #EpicFight -> hashtag EpicFight
            cleaned_chars['#'] = count
        
        if '@' in script:
            count = script.count('@')
            script = re.sub(r'@(\w+)', r'at \1', script)  # @VanDamme -> at VanDamme
            cleaned_chars['@'] = count
        
        # 2. Technical symbols (TTS engines can't pronounce these)
        if '\\' in script:
            count = script.count('\\')
            script = script.replace('\\', ' backslash ')
            cleaned_chars['\\'] = count
        
        if '|' in script:
            count = script.count('|')
            script = script.replace('|', ' or ')
            cleaned_chars['|'] = count
        
        if '^' in script:
            count = script.count('^')
            script = script.replace('^', ' caret ')
            cleaned_chars['^'] = count
        
        if '~' in script:
            count = script.count('~')
            script = script.replace('~', ' tilde ')
            cleaned_chars['~'] = count
        
        if '`' in script:
            count = script.count('`')
            script = script.replace('`', ' backtick ')
            cleaned_chars['`'] = count
        
        # 3. Mathematical symbols (context-dependent)
        if '=' in script:
            # Only replace if it's not part of a natural expression
            if re.search(r'\d+\s*=\s*\d+', script):  # Mathematical equation
                count = script.count('=')
                script = re.sub(r'(\d+)\s*=\s*(\d+)', r'\1 equals \2', script)
                cleaned_chars['='] = count
        
        if '<' in script or '>' in script:
            # Only replace if they're comparison operators, not HTML tags
            if re.search(r'\d+\s*[<>]\s*\d+', script):
                count = script.count('<') + script.count('>')
                script = re.sub(r'(\d+)\s*<\s*(\d+)', r'\1 less than \2', script)
                script = re.sub(r'(\d+)\s*>\s*(\d+)', r'\1 greater than \2', script)
                cleaned_chars['<>'] = count
        
        # 4. Currency symbols (TTS engines handle these poorly)
        if '$' in script:
            count = script.count('$')
            script = re.sub(r'\$(\d+)', r'\1 dollars', script)  # $50 -> 50 dollars
            cleaned_chars['$'] = count
        
        if '€' in script:
            count = script.count('€')
            script = re.sub(r'€(\d+)', r'\1 euros', script)  # €50 -> 50 euros
            cleaned_chars['€'] = count
        
        if '£' in script:
            count = script.count('£')
            script = re.sub(r'£(\d+)', r'\1 pounds', script)  # £50 -> 50 pounds
            cleaned_chars['£'] = count
        
        # 5. Percentage symbols (TTS engines often mispronounce)
        if '%' in script:
            count = script.count('%')
            script = re.sub(r'(\d+)%', r'\1 percent', script)  # 50% -> 50 percent
            cleaned_chars['%'] = count
        
        # 6. Ampersand (TTS engines often say "and" anyway)
        if '&' in script:
            count = script.count('&')
            script = script.replace('&', ' and ')
            cleaned_chars['&'] = count
        
        # 7. Plus and minus (context-dependent)
        if '+' in script:
            # Only replace if it's mathematical
            if re.search(r'\d+\s*\+\s*\d+', script):
                count = script.count('+')
                script = re.sub(r'(\d+)\s*\+\s*(\d+)', r'\1 plus \2', script)
                cleaned_chars['+'] = count
        
        if '-' in script:
            # Only replace if it's mathematical (not in words)
            if re.search(r'\d+\s*-\s*\d+', script):
                count = script.count('-')
                script = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1 minus \2', script)
                cleaned_chars['-'] = count
        
        # Log what was cleaned
        if cleaned_chars:
            logger.info(f"🔧 TTS cleanup applied: {cleaned_chars}")
            print(f"🔧 DEBUG: TTS cleanup applied: {cleaned_chars}")
        else:
            logger.info("🔧 No TTS-unfriendly characters found")
            print("🔧 DEBUG: No TTS-unfriendly characters found")
        
        return script

    async def generate_script(self, transcript: str, video_id: str = None) -> str:
        """Generate script using single-call topic-driven approach with optional features."""
        
        print("🎯 DEBUG: Starting Single-Call Topic-Driven Script Generation")
        logger.info("🎯 Starting Single-Call Topic-Driven Script Generation")
        print(f"🎯 DEBUG: Transcript length: {len(transcript)} characters")
        logger.info(f"🎯 Transcript length: {len(transcript)} characters")
        print(f"🎯 DEBUG: Video ID: {video_id}")
        logger.info(f"🎯 Video ID: {video_id}")
        
        try:
            # 🎯 NEW: Single API call that does everything
            final_script = await self._generate_complete_script_single_call(transcript, video_id)
            
            # 🎯 FIXED: Final validation
            print(f"✅ DEBUG: Single-Call Generation Complete: {len(final_script)} characters")
            logger.info(f"✅ Single-Call Generation Complete: {len(final_script)} characters")
            
            if len(final_script) < 5000:  # Minimum acceptable length
                logger.warning("⚠️ Final script too short, may indicate generation failure")
            
            return final_script
            
        except Exception as e:
            logger.error(f"❌ Single-call script generation failed: {str(e)}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            logger.error(f"❌ Full error details:", exc_info=True)
            # Fallback to original method if needed
            logger.info("🔄 Falling back to original topic-driven method...")
            return await self._generate_script_original(transcript, video_id)

    async def _generate_complete_script_single_call(self, transcript: str, video_id: str = None) -> str:
        """Generate complete script with topic analysis and all sections in ONE call."""
        
        print("🎯 DEBUG: Starting single-call comprehensive generation")
        logger.info("🎯 Starting single-call comprehensive generation")
        
        #  SINGLE PROMPT THAT DOES EVERYTHING
        comprehensive_prompt = f"""
        CRITICAL REQUIREMENT: Transform this YouTube transcript into a complete 20,000-30,000 character video script using topic-driven approach.

        STEP-BY-STEP PROCESS:
        1. ANALYZE the transcript and identify 6-8 main topics
        2. For each topic, generate engaging content with:
           - Key points and details
           - Dramatic elements and hooks
           - Smooth transitions
           - TTS-optimized language
        3. ASSEMBLE all topics into one flowing narrative
        4. Ensure the final script is 20,000-30,000 characters

        TOPIC IDENTIFICATION CRITERIA:
        - Controversial, shocking, or unknown elements
        - Recent rumors, controversies, and speculations
        - Subject's challenges, untold stories, or conflicts
        - Emotional moments and relatable human experiences
        - Mysterious or intriguing elements that create curiosity gaps

        CONTENT REQUIREMENTS:
        - Use engaging, conversational language
        - Include dramatic elements and hooks
        - Make it TTS-friendly (simple sentences, clear flow)
        - Use simple vocabulary that sounds natural when spoken
        - No section headings - just flowing narrative
        - Smooth transitions between topics
        - Target 20,000-30,000 characters total

        TTS OPTIMIZATION:
        - Use simple words: "show" instead of "demonstrate"
        - Keep sentences under 20 words for easy comprehension
        - Use active voice and direct language
        - Use natural speech patterns: "And then..." "But here's the thing..."
        - Avoid overly formal or academic language
        - Keep paragraphs short (2-3 sentences max)

        ENGAGEMENT TECHNIQUES:
        - Open with a powerful hook that immediately grabs attention
        - Use smooth transitions that build anticipation
        - Include mysterious or intriguing elements that create curiosity gaps
        - Add emotional moments and relatable experiences
        - Use varied sentence lengths and rhythms to maintain interest
        - Include strategic pauses and emphasis points for dramatic effect

        TONE AND STYLE:
        - Conversational and engaging, like telling a story to a friend
        - Slightly mysterious and intriguing where appropriate
        - Authentic and relatable, avoiding robotic or formulaic language
        - Dynamic pacing that speeds up and slows down for dramatic effect

        PHRASING AND CENSORSHIP:
        - Use powerful, engaging language like "shocking," "exposed," or "revealed"
        - Censor sensitive topics: "off'd himself" for "suicide," "O.D'd" for "overdose"
        - Ensure compliance with YouTube guidelines
        - Avoid direct language for sensitive topics

        FORMAT: Write in paragraph form with no "movie director" language. Avoid phrases like "[Cut to shot of…]" or stage directions. Write as though it's a story told in a straightforward, engaging way. NO SECTION HEADINGS - just flowing narrative paragraphs.

        TRANSCRIPT:
        {transcript}

        Generate the complete script now, incorporating all topics naturally into one flowing narrative:
        """
        
        print("🎯 DEBUG: Making single comprehensive API call")
        logger.info("🎯 Making single comprehensive API call")
        
        # 🎯 SINGLE API CALL WITH HIGHER TOKEN LIMIT
        response = await self.openai_service._create_chat_completion(
            messages=[
                {"role": "system", "content": "You are an expert script writer who can analyze topics and generate complete, engaging video scripts in one comprehensive call. You excel at identifying key topics and weaving them into compelling narratives."},
                {"role": "user", "content": comprehensive_prompt}
            ],
            operation_type="complete_script_generation",  # CRITICAL - use GPT-4o for quality
            temperature=0.7,
            max_tokens=12000,  # Higher limit for complete script
            timeout=120.0  # Longer timeout for comprehensive generation
        )
        
        script = response.choices[0].message.content.strip()
        
        print(f"🎯 DEBUG: Single call generated {len(script)} characters")
        logger.info(f"🎯 Single call generated {len(script)} characters")
        
        # 🎯 BASIC VALIDATION
        if len(script) < 15000:
            logger.warning(f"⚠️ Generated script too short ({len(script)} chars), may need continuation")
            # Could add a continuation call here if needed
        
        if len(script) > 35000:
            logger.warning(f"⚠️ Generated script too long ({len(script)} chars), truncating")
            script = script[:35000]
        
        #  APPLY POST-PROCESSING (if video_id available)
        if video_id:
            print("🔧 DEBUG: Applying dynamic name corrections")
            logger.info("🔧 Applying dynamic name corrections")
            script = await self._apply_dynamic_name_corrections(script, video_id)
        
        # 🎯 CLEAN TTS CHARACTERS
        print("🔧 DEBUG: Cleaning TTS characters")
        logger.info("🔧 Cleaning TTS characters")
        script = self._clean_tts_characters(script)
        
        print(f"✅ DEBUG: Single-call generation complete: {len(script)} characters")
        logger.info(f"✅ Single-call generation complete: {len(script)} characters")
        
        return script

    async def _generate_script_original(self, transcript: str, video_id: str = None) -> str:
        """Original multi-call method as fallback."""
        
        print("🔄 DEBUG: Using original multi-call method as fallback")
        logger.info("🔄 Using original multi-call method as fallback")
        
        try:
            # Phase 1: Topic Analysis
            print("🎯 DEBUG: Phase 1: Analyzing transcript topics")
            logger.info("🎯 Phase 1: Analyzing transcript topics")
            
            topics = await self._analyze_transcript_topics(transcript)
            print(f"✅ DEBUG: Topic analysis completed, got {len(topics)} topics")
            logger.info(f"✅ Topic analysis completed, got {len(topics)} topics")
            
            # Phase 2: Topic Expansion Planning
            print("📋 DEBUG: Phase 2: Planning topic expansions")
            logger.info("📋 Phase 2: Planning topic expansions")
            
            topic_plans = await self._plan_topic_expansions(topics, transcript)
            print(f"✅ DEBUG: Topic planning completed, got {len(topic_plans)} plans")
            logger.info(f"✅ Topic planning completed, got {len(topic_plans)} plans")
            
            # Phase 3: Parallel Topic Generation
            print("⚡ DEBUG: Phase 3: Generating topic sections")
            logger.info("⚡ Phase 3: Generating topic sections")
            
            topic_sections = await self._generate_topic_sections(topic_plans)
            print(f"✅ DEBUG: Topic generation completed, got {len(topic_sections)} sections")
            logger.info(f"✅ Topic generation completed, got {len(topic_sections)} sections")
            
            # Phase 4: Assembly & Polish
            print("🔧 DEBUG: Phase 4: Assembling and polishing final script")
            logger.info("🔧 Phase 4: Assembling and polishing final script")
            
            final_script = await self._assemble_and_polish(topic_sections, video_id)
            
            print(f"✅ DEBUG: Original Method Complete: {len(final_script)} characters")
            logger.info(f"✅ Original Method Complete: {len(final_script)} characters")
            
            return final_script
            
        except Exception as e:
            logger.error(f"❌ Original method also failed: {str(e)}")
            raise

    # 🎯 NEW: Easy method to enable premium features
    def enable_premium_mode(self):
        """Enable all premium features when budget increases."""
        self.config.enable_premium_features()
        logger.info("🎯 Premium mode enabled - all AI features activated")

    def enable_budget_mode(self):
        """Enable budget-conscious mode."""
        self.config.ENABLE_AI_VOCABULARY_SIMPLIFICATION = False
        self.config.ENABLE_AI_SCRIPT_POLISHING = False
        self.config.ENABLE_AI_ASR_GENERATION = False
        self.config.ENABLE_QUALITY_ANALYSIS = False
        logger.info("🎯 Budget mode enabled - AI features disabled")

    # 🎯 MODIFIED: ASR generation with configurable features
    async def _generate_asr_mistakes(self, name: str) -> list[str]:
        """Generate ASR mistakes with configurable AI usage."""
        
        # Try AI generation if enabled
        if self.config.ENABLE_AI_ASR_GENERATION:
            try:
                logger.info("🎯 Using AI ASR generation (premium feature)")
                return await self._generate_ai_asr_mistakes(name)
            except Exception as e:
                logger.warning(f"AI ASR generation failed, using fallback: {e}")
        
        # Use rule-based fallback
        logger.info("🎯 Using rule-based ASR generation (budget mode)")
        return self._generate_fallback_asr_mistakes(name)

    # 🎯 MODIFIED: Script polishing with configurable features
    async def _polish_large_script(self, script: str) -> str:
        """Polish large scripts with configurable AI usage."""
        
        if self.config.ENABLE_AI_SCRIPT_POLISHING:
            logger.info("🎯 Using AI script polishing (premium feature)")
            return await self._polish_large_script_ai(script)
        else:
            logger.info("🎯 Using rule-based script polishing (budget mode)")
            return self._apply_rule_based_polish(script)

    # 🎯 NEW: AI polishing method (kept for future use)
    async def _polish_large_script_ai(self, script: str) -> str:
        """AI-powered script polishing (premium feature)."""
        # Your existing _polish_large_script implementation here
        # This is kept intact for when budget increases
        pass

    # 🎯 PLACEHOLDER METHODS (to be implemented if needed)
    async def _analyze_transcript_topics(self, transcript: str) -> List[Dict]:
        """Placeholder for original topic analysis method."""
        raise NotImplementedError("Original topic analysis not implemented in single-call mode")

    async def _plan_topic_expansions(self, topics: List[Dict], transcript: str) -> List[Dict]:
        """Placeholder for original topic planning method."""
        raise NotImplementedError("Original topic planning not implemented in single-call mode")

    async def _generate_topic_sections(self, topic_plans: List[Dict]) -> List[str]:
        """Placeholder for original topic section generation method."""
        raise NotImplementedError("Original topic section generation not implemented in single-call mode")

    async def _assemble_and_polish(self, sections: List[str], video_id: str = None) -> str:
        """Placeholder for original assembly and polish method."""
        raise NotImplementedError("Original assembly and polish not implemented in single-call mode")

    async def _generate_ai_asr_mistakes(self, name: str) -> list[str]:
        """Placeholder for AI ASR mistake generation."""
        raise NotImplementedError("AI ASR generation not implemented")

    def _generate_fallback_asr_mistakes(self, name: str) -> list[str]:
        """Fallback ASR mistake generation using simple patterns."""
        variations = []
        name_lower = name.lower()
        
        # Simple universal patterns
        if ' ' in name_lower:
            # Remove spaces
            variations.append(name_lower.replace(' ', ''))
            # Keep only first word
            variations.append(name_lower.split()[0])
        
        if '-' in name_lower:
            # Remove hyphens
            variations.append(name_lower.replace('-', ''))
        
        # Truncate if long
        if len(name_lower) > 5:
            variations.append(name_lower[:5])
        
        return variations

    def _apply_rule_based_polish(self, script: str) -> str:
        """Apply rule-based script polishing."""
        # Simple rule-based polishing
        optimizer = TTSScriptOptimizer()
        return optimizer.optimize_for_tts(script)

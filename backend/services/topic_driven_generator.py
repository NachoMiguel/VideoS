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

class TopicDrivenScriptGenerator:
    """Topic-driven script generation with systematic content coverage."""
    
    def __init__(self, openai_service: 'OpenAIService'):
        self.openai_service = openai_service
        self.target_length = 25000  # 20k-30k target
        self.min_length = 20000
        
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
            response = await self.openai_service.client.chat.completions.create(
                model=self.openai_service.model,
                messages=[
                    {"role": "system", "content": "You are an expert editor who simplifies complex vocabulary while maintaining engagement and meaning. Return only the simplified text."},
                    {"role": "user", "content": simplification_prompt}
                ],
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
    
    async def _apply_post_processing(self, script: str, video_id: str = None) -> str:
        """Apply all post-processing corrections to the final script."""
        logger.info("🔧 Applying post-processing corrections...")
        
        # Step 1: Apply dynamic name corrections (if video_id available)
        if video_id:
            script = await self._apply_dynamic_name_corrections(script, video_id)
        
        # 🎯 NEW: Step 2: Clean TTS-unfriendly characters
        script = self._clean_tts_characters(script)
        
        # 🎯 TEMPORARILY DISABLED: AI vocabulary simplification (causing truncation)
        # script = await self._apply_ai_vocabulary_simplification(script)
        
        # Step 3: Clean up any formatting issues
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
        
        # 4. Abbreviations that TTS might mispronounce (FIXED regex patterns)
        script = re.sub(r'\bMr\.', 'Mister', script)
        script = re.sub(r'\bDr\.', 'Doctor', script)
        script = re.sub(r'\bProf\.', 'Professor', script)
        script = re.sub(r'\bvs\.', 'versus', script)  # FIXED: Removed \b at end
        script = re.sub(r'\betc\.', 'and so on', script)  # FIXED: Removed \b at end
        
        # 5. Number abbreviations (only when they're not natural)
        # Don't replace 20k if it's in a natural context like "20k followers"
        # But do replace technical abbreviations like "1080p"
        script = re.sub(r'\b(\d+)k\b(?=\s+(?:followers|subscribers|views|fans))', r'\1 thousand', script, flags=re.IGNORECASE)
        script = re.sub(r'\b(\d+)m\b(?=\s+(?:followers|subscribers|views|fans))', r'\1 million', script, flags=re.IGNORECASE)
        
        # 6. Technical specifications (these need cleanup)
        script = re.sub(r'\b(\d+)p\b', r'\1 p', script)  # 1080p -> 1080 p
        script = re.sub(r'\b(\d+)fps\b', r'\1 fps', script)  # 60fps -> 60 fps
        
        # 🎯 PRESERVE NATURAL CHARACTERS: Don't replace these as TTS handles them well
        # $ - TTS reads "dollar" naturally
        # % - TTS reads "percent" naturally  
        # & - TTS reads "and" naturally
        # + - TTS reads "plus" naturally
        # / - TTS reads "slash" naturally
        # * - TTS reads "asterisk" naturally
        
        # Clean up extra spaces created by replacements
        script = re.sub(r'\s+', ' ', script)
        
        # Log cleanup statistics
        if cleaned_chars:
            logger.info(f"🔧 Intelligent TTS cleanup applied: {cleaned_chars}")
            print(f" DEBUG: Intelligent TTS cleanup applied: {cleaned_chars}")
        else:
            logger.info("✅ No problematic TTS characters found")
            print("✅ DEBUG: No problematic TTS characters found")
        
        # Validate no content was lost
        if len(script) < len(original_script) * 0.9:  # If more than 10% was lost
            logger.warning("⚠️ TTS cleanup may have removed too much content, using original")
            return original_script
        
        return script

    async def generate_script(self, transcript: str, video_id: str = None) -> str:
        """Generate script using topic-driven approach with enhanced debugging."""
        
        print("🎯 DEBUG: Starting Topic-Driven Script Generation")
        logger.info("🎯 Starting Topic-Driven Script Generation")
        print(f"🎯 DEBUG: Transcript length: {len(transcript)} characters")
        logger.info(f"🎯 Transcript length: {len(transcript)} characters")
        print(f"🎯 DEBUG: Video ID: {video_id}")
        logger.info(f"🎯 Video ID: {video_id}")
        
        try:
            # Phase 1: Topic Analysis
            print("🎯 DEBUG: Phase 1: Analyzing transcript topics")
            logger.info("🎯 Phase 1: Analyzing transcript topics")
            print("🔍 DEBUG: About to call _analyze_transcript_topics...")
            logger.info("🔍 About to call _analyze_transcript_topics...")
            
            topics = await self._analyze_transcript_topics(transcript)
            print(f"✅ DEBUG: Topic analysis completed, got {len(topics)} topics")
            logger.info(f"✅ Topic analysis completed, got {len(topics)} topics")
            
            # Phase 2: Topic Expansion Planning
            print("📋 DEBUG: Phase 2: Planning topic expansions")
            logger.info("📋 Phase 2: Planning topic expansions")
            print("🔍 DEBUG: About to call _plan_topic_expansions...")
            logger.info("🔍 About to call _plan_topic_expansions...")
            
            topic_plans = await self._plan_topic_expansions(topics, transcript)
            print(f"✅ DEBUG: Topic planning completed, got {len(topic_plans)} plans")
            logger.info(f"✅ Topic planning completed, got {len(topic_plans)} plans")
            
            # Phase 3: Parallel Topic Generation
            print("⚡ DEBUG: Phase 3: Generating topic sections")
            logger.info("⚡ Phase 3: Generating topic sections")
            print("🔍 DEBUG: About to call _generate_topic_sections...")
            logger.info("🔍 About to call _generate_topic_sections...")
            
            topic_sections = await self._generate_topic_sections(topic_plans)
            print(f"✅ DEBUG: Topic generation completed, got {len(topic_sections)} sections")
            logger.info(f"✅ Topic generation completed, got {len(topic_sections)} sections")
            
            # 🎯 FIXED: Validate sections before assembly
            total_section_length = sum(len(section) for section in topic_sections)
            print(f"🔍 DEBUG: Total section length: {total_section_length} characters")
            logger.info(f"🔍 Total section length: {total_section_length} characters")
            
            # Phase 4: Assembly & Polish
            print("🔧 DEBUG: Phase 4: Assembling and polishing final script")
            logger.info("🔧 Phase 4: Assembling and polishing final script")
            print("🔍 DEBUG: About to call _assemble_and_polish...")
            logger.info("🔍 About to call _assemble_and_polish...")
            
            final_script = await self._assemble_and_polish(topic_sections, video_id)
            
            # 🎯 FIXED: Final validation
            print(f"✅ DEBUG: Topic-Driven Generation Complete: {len(final_script)} characters")
            logger.info(f"✅ Topic-Driven Generation Complete: {len(final_script)} characters")
            
            if len(final_script) < 5000:  # Minimum acceptable length
                logger.warning("⚠️ Final script too short, may indicate generation failure")
            
            return final_script
            
        except Exception as e:
            logger.error(f"❌ Topic-driven script generation failed: {str(e)}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            logger.error(f"❌ Full error details:", exc_info=True)
            raise
    
    async def _analyze_transcript_topics(self, transcript: str) -> List[Dict]:
        """Extract main topics from transcript with context."""
        
        # 🎯 NEW: Add retry logic for OpenAI calls
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"🔍 DEBUG: Topic analysis attempt {attempt + 1}/{max_retries}")
                
                # 🎯 NEW: Log the transcript being analyzed
                print("=" * 80)
                print("🎯 TRANSCRIPT ANALYSIS:")
                print("=" * 80)
                print(f"📄 TRANSCRIPT LENGTH: {len(transcript)} characters")
                print(f"📄 TRANSCRIPT PREVIEW: {transcript[:500]}...")
                print("=" * 80)
                
                # 🎯 FORCE DEBUGGING: Use both logging and print
                print("🔍 DEBUG: _analyze_transcript_topics called")
                logger.info("🔍 DEBUG: _analyze_transcript_topics called")
                
                analysis_prompt = f"""
                CRITICAL REQUIREMENT: Extract 6-10 main topics that will create a 20,000-30,000 character script.
                
                CONTENT TRANSFORMATION GOALS:
                1. Identify topics that can be elevated with engaging storytelling techniques
                2. Focus on controversial, shocking, or unknown elements
                3. Include recent rumors, controversies, and speculations
                4. Highlight challenges, untold stories, or conflicts
                
                TOPICS AND THEMES TO LOOK FOR:
                - Controversial, shocking, or unknown elements of the subject's life or career
                - Recent rumors, controversies, and speculations
                - Subject's challenges, untold stories, or conflicts
                - Emotional moments and relatable human experiences
                - Mysterious or intriguing elements that create curiosity gaps
                
                TRANSCRIPT:
                {transcript}
                
                You MUST respond with ONLY valid JSON in this exact format:
                {{
                    "topics": [
                        {{
                            "title": "Topic Title",
                            "key_points": ["point1", "point2", "point3"],
                            "quotes": ["relevant quote 1", "relevant quote 2"],
                            "events": ["event1", "event2"],
                            "context": "brief context about this topic",
                            "controversy_level": "high/medium/low",
                            "storytelling_potential": "high/medium/low",
                            "emotional_hooks": ["hook1", "hook2"]
                        }}
                    ]
                }}
                
                IMPORTANT: Return ONLY the JSON object, no additional text or explanations.
                """
                
                # 🎯 INCREASED timeout for topic analysis
                response = await self.openai_service.client.chat.completions.create(
                    model=self.openai_service.model,
                    messages=[
                        {"role": "system", "content": "You are an expert content analyst. You MUST respond with ONLY valid JSON in the exact format requested. No additional text."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000,
                    timeout=90.0  #  INCREASED timeout
                )
                
                # 🎯 ENHANCED: Debug the actual response
                raw_response = response.choices[0].message.content.strip()
                print(f"🔍 DEBUG: Raw response length: {len(raw_response)}")
                print(f"🔍 DEBUG: Raw response preview: {raw_response[:200]}...")
                logger.info(f" Raw topic analysis response length: {len(raw_response)}")
                logger.info(f"🔍 Raw response preview: {raw_response[:200]}...")
                
                # Check if response is empty
                if not raw_response:
                    print("❌ DEBUG: Empty response from topic analysis")
                    logger.error("❌ Topic analysis returned empty response")
                    raise ValueError("Empty response from topic analysis")
                
                # Try to extract JSON if response contains extra text
                json_start = raw_response.find('{')
                json_end = raw_response.rfind('}') + 1
                
                if json_start == -1 or json_end == 0:
                    print(f"❌ DEBUG: No JSON found in response: {raw_response}")
                    logger.error(f"❌ No JSON found in response: {raw_response}")
                    raise ValueError("No JSON structure found in response")
                
                # Extract just the JSON part
                json_content = raw_response[json_start:json_end]
                print(f"🔍 DEBUG: Extracted JSON: {json_content[:200]}...")
                logger.info(f"🔍 Extracted JSON: {json_content[:200]}...")
                
                # Parse and validate topics
                topics_data = json.loads(json_content)
                
                if "topics" not in topics_data:
                    print(f"❌ DEBUG: No 'topics' key in JSON: {topics_data}")
                    logger.error(f"❌ No 'topics' key in JSON: {topics_data}")
                    raise ValueError("Invalid JSON structure - missing 'topics' key")
                
                topics = topics_data["topics"]
                
                if not topics:
                    print("❌ DEBUG: Empty topics list in JSON")
                    logger.error("❌ Empty topics list in JSON")
                    raise ValueError("Empty topics list")
                
                # ✅ NEW: Comprehensive topic logging with print statements
                print(f"✅ DEBUG: Successfully extracted {len(topics)} topics from transcript")
                logger.info(f"✅ Successfully extracted {len(topics)} topics from transcript")
                
                print("📋 DEBUG: TOPIC ANALYSIS RESULTS:")
                logger.info("📋 TOPIC ANALYSIS RESULTS:")
                print("=" * 60)
                logger.info("=" * 60)
                
                for i, topic in enumerate(topics, 1):
                    print(f"🎯 DEBUG: TOPIC {i}: {topic.get('title', 'No title')}")
                    logger.info(f"🎯 TOPIC {i}: {topic.get('title', 'No title')}")
                    
                    print(f"   DEBUG: Key Points: {len(topic.get('key_points', []))} points")
                    logger.info(f"   Key Points: {len(topic.get('key_points', []))} points")
                    
                    for j, point in enumerate(topic.get('key_points', []), 1):
                        print(f"      {j}. {point[:100]}{'...' if len(point) > 100 else ''}")
                        logger.info(f"      {j}. {point[:100]}{'...' if len(point) > 100 else ''}")
                    
                    print(f"   DEBUG: Quotes: {len(topic.get('quotes', []))} quotes")
                    logger.info(f"    Quotes: {len(topic.get('quotes', []))} quotes")
                    
                    for j, quote in enumerate(topic.get('quotes', []), 1):
                        print(f"      {j}. \"{quote[:80]}{'...' if len(quote) > 80 else ''}\"")
                        logger.info(f"      {j}. \"{quote[:80]}{'...' if len(quote) > 80 else ''}\"")
                    
                    print(f"   📅 DEBUG: Events: {len(topic.get('events', []))} events")
                    logger.info(f"    Events: {len(topic.get('events', []))} events")
                    
                    for j, event in enumerate(topic.get('events', []), 1):
                        print(f"      {j}. {event[:80]}{'...' if len(event) > 80 else ''}")
                        logger.info(f"      {j}. {event[:80]}{'...' if len(event) > 80 else ''}")
                    
                    print(f"   📖 DEBUG: Context: {topic.get('context', 'No context')[:100]}{'...' if len(topic.get('context', '')) > 100 else ''}")
                    logger.info(f"   📖 Context: {topic.get('context', 'No context')[:100]}{'...' if len(topic.get('context', '')) > 100 else ''}")
                    
                    print(f"   DEBUG: Controversy Level: {topic.get('controversy_level', 'unknown')}")
                    logger.info(f"   Controversy Level: {topic.get('controversy_level', 'unknown')}")
                    
                    print(f"   📈 DEBUG: Storytelling Potential: {topic.get('storytelling_potential', 'unknown')}")
                    logger.info(f"   📈 Storytelling Potential: {topic.get('storytelling_potential', 'unknown')}")
                    
                    print(f"   DEBUG: Emotional Hooks: {len(topic.get('emotional_hooks', []))} hooks")
                    logger.info(f"   Emotional Hooks: {len(topic.get('emotional_hooks', []))} hooks")
                    
                    for j, hook in enumerate(topic.get('emotional_hooks', []), 1):
                        print(f"      {j}. {hook[:60]}{'...' if len(hook) > 60 else ''}")
                        logger.info(f"      {j}. {hook[:60]}{'...' if len(hook) > 60 else ''}")
                    
                    print("-" * 40)
                    logger.info("-" * 40)
                
                print("=" * 60)
                logger.info("=" * 60)
                
                print(f"📊 DEBUG: TOPIC SUMMARY:")
                logger.info(f"📊 TOPIC SUMMARY:")
                print(f"   Total Topics: {len(topics)}")
                logger.info(f"   Total Topics: {len(topics)}")
                print(f"   High Controversy: {sum(1 for t in topics if t.get('controversy_level') == 'high')}")
                logger.info(f"   High Controversy: {sum(1 for t in topics if t.get('controversy_level') == 'high')}")
                print(f"   High Storytelling: {sum(1 for t in topics if t.get('storytelling_potential') == 'high')}")
                logger.info(f"   High Storytelling: {sum(1 for t in topics if t.get('storytelling_potential') == 'high')}")
                print(f"   Total Key Points: {sum(len(t.get('key_points', [])) for t in topics)}")
                logger.info(f"   Total Key Points: {sum(len(t.get('key_points', [])) for t in topics)}")
                print(f"   Total Quotes: {sum(len(t.get('quotes', [])) for t in topics)}")
                logger.info(f"   Total Quotes: {sum(len(t.get('quotes', [])) for t in topics)}")
                print(f"   Total Events: {sum(len(t.get('events', [])) for t in topics)}")
                logger.info(f"   Total Events: {sum(len(t.get('events', [])) for t in topics)}")
                print("=" * 60)
                logger.info("=" * 60)
                
                return topics
                
            except json.JSONDecodeError as e:
                print(f"❌ DEBUG: JSON parsing failed: {str(e)}")
                logger.error(f"❌ JSON parsing failed: {str(e)}")
                print(f"❌ DEBUG: Failed JSON content: {raw_response if 'raw_response' in locals() else 'No response'}")
                logger.error(f"❌ Failed JSON content: {raw_response if 'raw_response' in locals() else 'No response'}")
                # Fallback to basic topics
                return self._create_fallback_topics(transcript)
                
            except Exception as e:
                print(f"❌ DEBUG: Topic analysis attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    print("❌ DEBUG: All topic analysis attempts failed, using fallback")
                    return self._create_fallback_topics(transcript)
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def _create_fallback_topics(self, transcript: str) -> List[Dict]:
        """Create fallback topics if analysis fails."""
        logger.info("🔄 Creating fallback topics from transcript")
        
        # Simple fallback: split transcript into chunks
        words = transcript.split()
        chunk_size = max(1, len(words) // 8)  # Ensure chunk_size is at least 1
        topics = []
        
        for i in range(8):
            start = i * chunk_size
            end = start + chunk_size if i < 7 else len(words)
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            
            # Create a more descriptive title based on content
            title_words = chunk_text.split()[:5]  # First 5 words
            title = " ".join(title_words) + "..." if len(title_words) == 5 else chunk_text[:50] + "..."
            
            topic = {
                "title": f"Content Section {i+1}: {title}",
                "key_points": [chunk_text[:100] + "..."],
                "quotes": [],
                "events": [],
                "context": f"Content from section {i+1} of transcript",
                "controversy_level": "medium",
                "storytelling_potential": "medium",
                "emotional_hooks": []
            }
            topics.append(topic)
        
        # 🎯 NEW: Log fallback topics
        logger.info(f"✅ Created {len(topics)} fallback topics")
        logger.info("📋 FALLBACK TOPIC ANALYSIS RESULTS:")
        logger.info("=" * 60)
        
        for i, topic in enumerate(topics, 1):
            logger.info(f"🔄 FALLBACK TOPIC {i}: {topic['title']}")
            logger.info(f"    Key Points: {len(topic['key_points'])} points")
            for j, point in enumerate(topic['key_points'], 1):
                logger.info(f"      {j}. {point[:100]}{'...' if len(point) > 100 else ''}")
            
            logger.info(f"   📖 Context: {topic['context']}")
            logger.info(f"    Controversy Level: {topic['controversy_level']}")
            logger.info(f"    Storytelling Potential: {topic['storytelling_potential']}")
            logger.info("-" * 40)
        
        logger.info("=" * 60)
        logger.info(f"📊 FALLBACK TOPIC SUMMARY:")
        logger.info(f"   Total Fallback Topics: {len(topics)}")
        logger.info(f"   Average Words per Topic: {len(words) // len(topics)}")
        logger.info("=" * 60)
        
        return topics
    
    async def _plan_topic_expansions(self, topics: List[Dict], transcript: str) -> List[Dict]:
        """Plan how to expand each topic into a script section."""
        
        target_length_per_topic = self.target_length // len(topics)
        
        logger.info(f"📋 Planning topic expansions for {len(topics)} topics")
        logger.info(f"🎯 Target length per topic: {target_length_per_topic} characters")
        
        topic_plans = []
        for i, topic in enumerate(topics):
            
            # 🎯 REMOVED: Section roles (introduction, conclusion, etc.)
            # All sections are equal - just narrative flow
            
            plan = {
                "topic": topic,
                "target_length": target_length_per_topic,
                "section_number": i + 1,
                "total_sections": len(topics),
                "context": self._extract_topic_context(topic, transcript),
                
                # Apply ENGAGEMENT TECHNIQUES from original prompt
                "engagement_requirements": {
                    "hook_required": i == 0,  # Only first section needs hook
                    "transitions_required": i > 0,  # All except first need transitions
                    "curiosity_gaps": topic.get("storytelling_potential") == "high",
                    "emotional_moments": topic.get("emotional_hooks", []),
                    "dramatic_pacing": topic.get("controversy_level") == "high"
                },
                
                # Apply TONE AND STYLE from original prompt
                "style_requirements": {
                    "conversational": True,
                    "mysterious_intriguing": topic.get("controversy_level") in ["high", "medium"],
                    "authentic_relatable": True,
                    "dynamic_pacing": True
                }
            }
            topic_plans.append(plan)
            
            # 🎯 NEW: Log each topic plan
            logger.info(f"📝 TOPIC PLAN {i+1}: {topic['title']}")
            logger.info(f"   🎯 Target Length: {target_length_per_topic} chars")
            logger.info(f"   🪝 Hook Required: {plan['engagement_requirements']['hook_required']}")
            logger.info(f"   🔗 Transitions Required: {plan['engagement_requirements']['transitions_required']}")
            logger.info(f"   🔍 Curiosity Gaps: {plan['engagement_requirements']['curiosity_gaps']}")
            logger.info(f"   📈 Emotional Moments: {len(plan['engagement_requirements']['emotional_moments'])}")
            logger.info(f"   ⚡ Dramatic Pacing: {plan['engagement_requirements']['dramatic_pacing']}")
            logger.info(f"   🎭 Mysterious/Intriguing: {plan['style_requirements']['mysterious_intriguing']}")
            logger.info(f"   📖 Context Length: {len(plan['context'])} chars")
        
        logger.info(f" Planned {len(topic_plans)} topic expansions")
        return topic_plans
    
    def _extract_topic_context(self, topic: Dict, transcript: str) -> str:
        """Extract relevant context for a topic from the transcript."""
        # Simple context extraction - find sentences containing topic keywords
        topic_keywords = topic["title"].lower().split()
        sentences = transcript.split('.')
        
        relevant_sentences = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in topic_keywords):
                relevant_sentences.append(sentence.strip())
        
        return '. '.join(relevant_sentences[:3])  # Limit to 3 most relevant sentences
    
    async def _generate_topic_sections(self, topic_plans: List[Dict]) -> List[str]:
        """Generate each topic section in parallel."""
        
        # Generate sections concurrently for efficiency
        tasks = []
        for plan in topic_plans:
            task = self._generate_single_topic_section(plan)
            tasks.append(task)
        
        # Execute all topic generations in parallel
        sections = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any failed generations
        valid_sections = []
        for i, section in enumerate(sections):
            if isinstance(section, Exception):
                logger.error(f"Topic {i+1} generation failed: {section}")
                # Generate fallback section
                fallback = await self._generate_fallback_section(topic_plans[i])
                valid_sections.append(fallback)
            else:
                valid_sections.append(section)
        
        logger.info(f"✅ Generated {len(valid_sections)} topic sections")
        return valid_sections
    
    async def _generate_single_topic_section(self, plan: Dict) -> str:
        """Generate a single topic section optimized for TTS."""
        
        topic = plan["topic"]
        engagement = plan["engagement_requirements"]
        style = plan["style_requirements"]
        
        # 🎯 TTS-OPTIMIZED: Pre-define the problematic string to avoid f-string backslash issue
        transitions_text = "- Use smooth transitions that build anticipation for what's coming next"
        
        section_prompt = f"""
        CRITICAL REQUIREMENT: Create exactly {plan['target_length']} characters for this narrative section.
        
        SECTION: {plan['section_number']} of {plan['total_sections']}
        
        TOPIC: {topic['title']}
        KEY POINTS: {', '.join(topic['key_points'])}
        QUOTES: {', '.join(topic['quotes'])}
        EVENTS: {', '.join(topic['events'])}
        CONTEXT: {topic['context']}
        
        🎯 TTS OPTIMIZATION REQUIREMENTS:
        - Use simple, clear vocabulary that sounds natural when spoken
        - Keep sentences under 20 words for easy comprehension
        - Use active voice and direct language
        - Avoid complex sentence structures and run-on sentences
        - Use natural speech patterns and conversational flow
        - Break complex ideas into shorter, digestible sentences
        
        ENGAGEMENT TECHNIQUES TO INCLUDE:
        {'- Open with a powerful hook that immediately grabs attention' if engagement['hook_required'] else ''}
        {transitions_text if engagement['transitions_required'] else ''}
        {'- Include mysterious or intriguing elements that create curiosity gaps' if engagement['curiosity_gaps'] else ''}
        {'- Add emotional moments: ' + ', '.join(engagement['emotional_moments']) if engagement['emotional_moments'] else ''}
        - Use varied sentence lengths and rhythms to maintain interest
        {'- Include strategic pauses and emphasis points for dramatic effect' if engagement['dramatic_pacing'] else ''}
        
        TONE AND STYLE:
        - Conversational and engaging, like telling a story to a friend
        {'- Slightly mysterious and intriguing where appropriate' if style['mysterious_intriguing'] else ''}
        - Authentic and relatable, avoiding robotic or formulaic language
        {'- Dynamic pacing that speeds up and slows down for dramatic effect' if style['dynamic_pacing'] else ''}
        
        🎯 TTS LANGUAGE GUIDELINES:
        - Use simple words: "show" instead of "demonstrate", "help" instead of "facilitate"
        - Use natural speech patterns: "And then..." "But here's the thing..."
        - Avoid overly formal or academic language
        - Use repetition and emphasis for dramatic effect
        - Keep paragraphs short (2-3 sentences max)
        - Use direct, accessible language throughout
        
        PHRASING, DRAMATIC LANGUAGE, AND CENSORSHIP:
        - Use powerful, engaging language, like "shocking," "exposed," or "revealed," to hold the viewer's attention
        - Censor or reword sensitive topics to ensure compliance with YouTube's guidelines:
          - Avoid direct language for terms like "suicide," "overdose," or "criminal accusations"
          - Use indirect phrasing (e.g., "off'd himself" for "suicide," "O.D'd" for "overdose," "accusations surfaced" for legal issues)
          - Ensure any profanity is censored, e.g., "dmn" or "sht"
        - Don't repeat introductions or start each section with references to the title—just get straight to the point
        
        VARIED WORDING:
        - Avoid overusing specific phrases or descriptions (e.g., "shocking truth" or "exposed")
        - Vary the language to keep the script fresh and engaging
        - Ensure the script flows naturally and avoids a formulaic tone
        
        FORMAT: Write in paragraph form with no "movie director" language. Avoid phrases like "[Cut to shot of…]" or stage directions, and write as though it's a story told in a straightforward, engaging way. NO SECTION HEADINGS - just flowing narrative paragraphs.
        
        Write this TTS-optimized narrative section now:
        """
        
        # 🎯 FIXED: Calculate proper token limit for GPT-4o
        max_tokens_for_section = min(8000, plan["target_length"] // 3)  # Conservative estimate
        
        response = await self.openai_service.client.chat.completions.create(
            model=self.openai_service.model,
            messages=[
                {"role": "system", "content": "You are an expert script writer creating TTS-optimized, engaging narrative content without section headings."},
                {"role": "user", "content": section_prompt}
            ],
            temperature=0.7,
            max_tokens=max_tokens_for_section,  # 🎯 FIXED: Use calculated limit
            timeout=self.openai_service.timeout
        )
        
        return response.choices[0].message.content.strip()
    
    async def _generate_fallback_section(self, plan: Dict) -> str:
        """Generate a TTS-optimized fallback section if main generation fails."""
        topic = plan["topic"]
        
        fallback_prompt = f"""
        Create a TTS-optimized narrative section for this topic:
        
        TOPIC: {topic['title']}
        CONTEXT: {topic['context']}
        
        🎯 TTS OPTIMIZATION REQUIREMENTS:
        - Use simple, clear vocabulary that sounds natural when spoken
        - Keep sentences under 20 words for easy comprehension
        - Use active voice and direct language
        - Use natural speech patterns and conversational flow
        - Break complex ideas into shorter, digestible sentences
        
        🎯 TTS LANGUAGE GUIDELINES:
        - Use simple words: "show" instead of "demonstrate", "help" instead of "facilitate"
        - Use natural speech patterns: "And then..." "But here's the thing..."
        - Avoid overly formal or academic language
        - Keep paragraphs short (2-3 sentences max)
        
        Write a {plan['target_length']} character section in engaging, conversational style optimized for TTS.
        NO SECTION HEADINGS - just flowing narrative paragraphs.
        """
        
        # 🎯 FIXED: Calculate proper token limit
        max_tokens_for_fallback = min(6000, plan["target_length"] // 3)
        
        response = await self.openai_service.client.chat.completions.create(
            model=self.openai_service.model,
            messages=[
                {"role": "system", "content": "You are a script writer creating TTS-optimized, engaging narrative content without headings."},
                {"role": "user", "content": fallback_prompt}
            ],
            temperature=0.7,
            max_tokens=max_tokens_for_fallback,  # 🎯 FIXED: Use calculated limit
            timeout=self.openai_service.timeout
        )
        
        return response.choices[0].message.content.strip()
    
    async def _assemble_and_polish(self, sections: List[str], video_id: str = None) -> str:
        """Assemble sections into final script with dynamic post-processing."""
        
        logger.info("🔧 Phase 4: Assembling and polishing final script")
        
        # 🎯 FIXED: Validate sections before assembly
        if not sections:
            logger.error("❌ No sections to assemble")
            return ""
        
        # Combine all sections
        combined_script = ' '.join(sections)
        
        # 🎯 FIXED: Validate combined script
        if len(combined_script) < 1000:
            logger.warning("⚠️ Combined script too short, may indicate generation failure")
        
        # Check if we need to split for polishing due to size
        if len(combined_script) > 15000:  # If too large for single API call
            logger.info("📏 Large script detected, using chunked polishing")
            final_script = await self._polish_large_script(combined_script)
        else:
            # Standard polishing for smaller scripts
            polish_prompt = f"""
            FINAL POLISHING: Take this assembled script and make it flow perfectly.
            
            REQUIREMENTS:
            1. Ensure smooth transitions between sections
            2. Maintain consistent tone and style
            3. Fix any awkward phrasing or repetition
            4. Keep the engaging, dramatic style
            5. Ensure proper paragraph breaks for readability
            6. IMPORTANT: Return ONLY the polished script, no explanations
            
            SCRIPT TO POLISH:
            {combined_script}
            
            Polished script:
            """
            
            try:
                response = await self.openai_service.client.chat.completions.create(
                    model=self.openai_service.model,
                    messages=[
                        {"role": "system", "content": "You are an expert script editor. Polish the script to perfection. Return only polished text."},
                        {"role": "user", "content": polish_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=16000,  # Fixed: Use 16k limit for GPT-4o
                    timeout=self.openai_service.timeout
                )
                
                final_script = response.choices[0].message.content.strip()
                
                # 🎯 FIXED: Validate polishing worked
                if len(final_script) < len(combined_script) * 0.8:
                    logger.warning("⚠️ Polishing appears to have truncated script, using original")
                    final_script = combined_script
                
            except Exception as e:
                logger.warning(f"Polishing failed, using unpolished script: {e}")
                final_script = combined_script
        
        # 🎯 FIXED: Apply post-processing with validation
        original_length = len(final_script)
        final_script = await self._apply_post_processing(final_script, video_id)
        
        # 🎯 FIXED: Final validation
        if len(final_script) < original_length * 0.8:
            logger.warning("⚠️ Post-processing appears to have truncated script")
        
        logger.info(f"✅ Final script complete: {len(final_script)} characters")
        return final_script

    def _apply_rule_based_polish(self, script: str) -> str:
        """Apply rule-based polishing when AI polishing fails."""
        logger.info("🔧 Applying rule-based polish fallback")
        
        # Initialize TTS optimizer
        tts_optimizer = TTSScriptOptimizer()
        
        # Apply rule-based optimizations
        polished_script = tts_optimizer.optimize_for_tts(script)
        
        logger.info("✅ Rule-based polish complete")
        return polished_script

    async def _polish_large_script(self, script: str) -> str:
        """Polish large scripts by processing in chunks with improved error handling."""
        print("🔧 DEBUG: Starting chunked polish for large script")
        logger.info("🔧 Starting chunked polish for large script")
        
        # 🎯 IMPROVED: Better chunking strategy
        chunk_size = 8000  # Reduced for better reliability
        chunks = [script[i:i+chunk_size] for i in range(0, len(script), chunk_size)]
        
        print(f"🔧 DEBUG: Split into {len(chunks)} chunks of ~{chunk_size} characters each")
        logger.info(f" Split into {len(chunks)} chunks of ~{chunk_size} characters each")
        
        polished_chunks = []
        failed_chunks = 0
        
        for i, chunk in enumerate(chunks):
            print(f"🔧 DEBUG: Polishing chunk {i+1}/{len(chunks)} ({len(chunk)} characters)")
            logger.info(f" Polishing chunk {i+1}/{len(chunks)} ({len(chunk)} characters)")
            
            #  IMPROVED: Enhanced prompt for better AI polishing
            chunk_prompt = f"""
            Polish this script chunk to make it more engaging and TTS-friendly.
            
            SPECIFIC IMPROVEMENTS TO MAKE:
            1. Simplify complex sentences (break long sentences into shorter ones)
            2. Replace sophisticated vocabulary with simpler, more natural words
            3. Improve flow and transitions between ideas
            4. Make the language more conversational and engaging
            5. Ensure proper pacing for voice narration
            
            IMPORTANT: Return ONLY the polished text, no explanations or additional text.
            
            SCRIPT CHUNK TO POLISH:
            {chunk}
            
            POLISHED VERSION:
            """
            
            # 🎯 IMPROVED: Retry logic for failed chunks
            max_retries = 2
            polished_chunk = None
            
            for attempt in range(max_retries):
                try:
                    print(f" DEBUG: Chunk {i+1} attempt {attempt + 1}/{max_retries}")
                    
                    response = await self.openai_service.client.chat.completions.create(
                        model=self.openai_service.model,
                        messages=[
                            {"role": "system", "content": "You are an expert script editor specializing in TTS optimization. Your job is to make scripts more engaging, natural, and easy to read aloud. Return only the polished text."},
                            {"role": "user", "content": chunk_prompt}
                        ],
                        temperature=0.4,  # Slightly higher for more creative improvements
                        max_tokens=10000,  # Reduced for reliability
                        timeout=60.0  # Increased timeout significantly
                    )
                    
                    polished_chunk = response.choices[0].message.content.strip()
                    
                    # 🎯 IMPROVED: Better validation logic
                    if not polished_chunk or len(polished_chunk.strip()) == 0:
                        raise ValueError("Empty response from AI")
                    
                    if len(polished_chunk) < len(chunk) * 0.3:  # More reasonable minimum
                        raise ValueError(f"Chunk too short: {len(polished_chunk)} vs {len(chunk)}")
                    
                    if len(polished_chunk) > len(chunk) * 2.0:  # Prevent excessive expansion
                        print(f"⚠️ DEBUG: Chunk {i+1} too long, truncating")
                        polished_chunk = polished_chunk[:len(chunk) * 2]
                    
                    print(f"✅ DEBUG: Chunk {i+1} polished successfully ({len(polished_chunk)} characters)")
                    break
                    
                except Exception as e:
                    print(f"⚠️ DEBUG: Chunk {i+1} attempt {attempt + 1} failed: {str(e)}")
                    logger.warning(f"⚠️ Chunk {i+1} attempt {attempt + 1} failed: {str(e)}")
                    
                    if attempt == max_retries - 1:  # Last attempt
                        print(f"❌ DEBUG: Chunk {i+1} all attempts failed, using rule-based fallback")
                        logger.error(f"❌ Chunk {i+1} all attempts failed, using rule-based fallback")
                        polished_chunk = self._apply_rule_based_polish(chunk)
                        failed_chunks += 1
        
            polished_chunks.append(polished_chunk)
        
        # 🎯 IMPROVED: Better chunk assembly
        print(f"🔧 DEBUG: Assembling {len(polished_chunks)} chunks")
        logger.info(f"🔧 Assembling {len(polished_chunks)} chunks")
        
        # Simple concatenation with proper spacing
        final_script = " ".join(polished_chunks)
        
        # 🎯 IMPROVED: Final validation and cleanup
        if len(final_script) < len(script) * 0.7:  # More reasonable minimum
            print(f"⚠️ DEBUG: Final script too short ({len(final_script)} vs {len(script)}), using original")
            logger.warning(f"⚠️ Final script too short, using original")
            final_script = script
        
        # Clean up extra whitespace
        final_script = re.sub(r'\s+', ' ', final_script).strip()
        
        # Log results
        success_rate = ((len(chunks) - failed_chunks) / len(chunks)) * 100
        print(f"🔧 DEBUG: Chunked polish complete: {len(final_script)} characters, {success_rate:.1f}% success rate")
        logger.info(f"🔧 Chunked polish complete: {len(final_script)} characters, {success_rate:.1f}% success rate")
        
        if failed_chunks > 0:
            print(f"⚠️ DEBUG: {failed_chunks} chunks failed and used fallback polishing")
            logger.warning(f"⚠️ {failed_chunks} chunks failed and used fallback polishing")
        
        return final_script

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

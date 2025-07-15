#!/usr/bin/env python3
"""
Test script for chunked polishing system without using API quota.
This simulates the entire chunked polishing process to verify our fixes work.
"""

import sys
import os
import re
import asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.services.topic_driven_generator import TopicDrivenScriptGenerator, TTSScriptOptimizer

class MockOpenAIService:
    """Mock OpenAI service that simulates different scenarios without API calls."""
    
    def __init__(self, failure_scenario="success"):
        self.failure_scenario = failure_scenario
        self.model = "gpt-4o"
        self.timeout = 60.0
    
    async def client(self):
        return self
    
    async def chat(self):
        return self
    
    async def completions(self):
        return self
    
    async def create(self, **kwargs):
        """Simulate different API response scenarios."""
        
        # Extract the chunk content from the prompt
        prompt = kwargs.get('messages', [{}])[-1].get('content', '')
        chunk_match = re.search(r'SCRIPT CHUNK TO POLISH:\s*(.*?)(?=\s*POLISHED VERSION:)', prompt, re.DOTALL)
        chunk = chunk_match.group(1).strip() if chunk_match else ""
        
        # Simulate different scenarios
        if self.failure_scenario == "timeout":
            await asyncio.sleep(0.1)  # Simulate delay
            raise Exception("Request timeout")
        
        elif self.failure_scenario == "empty_response":
            return type('Response', (), {
                'choices': [type('Choice', (), {
                    'message': type('Message', (), {
                        'content': ''
                    })()
                })()]
            })()
        
        elif self.failure_scenario == "too_short":
            # Return a response that's too short
            return type('Response', (), {
                'choices': [type('Choice', (), {
                    'message': type('Message', (), {
                        'content': 'Short response.'
                    })()
                })()]
            })()
        
        elif self.failure_scenario == "partial_failure":
            # Fail on every other chunk
            if len(chunk) > 4000:  # Fail on larger chunks
                raise Exception("Partial failure simulation")
            else:
                return self._generate_mock_polished_response(chunk)
        
        else:  # success
            return self._generate_mock_polished_response(chunk)
    
    def _generate_mock_polished_response(self, chunk):
        """Generate a realistic mock polished response."""
        
        # Simulate AI polishing by applying some basic improvements
        polished = chunk
        
        # 1. Simplify some complex words
        word_replacements = {
            'consequently': 'so',
            'nevertheless': 'but',
            'furthermore': 'also',
            'subsequently': 'then',
            'approximately': 'about',
            'demonstrate': 'show',
            'utilize': 'use',
            'sophisticated': 'complex',
            'elaborate': 'detailed',
            'comprehensive': 'complete',
            'substantial': 'large',
            'significant': 'important',
            'remarkable': 'amazing'
        }
        
        for complex_word, simple_word in word_replacements.items():
            pattern = r'\b' + re.escape(complex_word) + r'\b'
            polished = re.sub(pattern, simple_word, polished, flags=re.IGNORECASE)
        
        # 2. Break some long sentences
        sentences = re.split(r'([.!?]+)', polished)
        improved_sentences = []
        
        for sentence in sentences:
            if len(sentence.split()) > 20:
                # Break at conjunctions
                sentence = re.sub(r'([,;])\s+(and|but|or|so|yet)', r'\1\2. ', sentence)
            improved_sentences.append(sentence)
        
        polished = ''.join(improved_sentences)
        
        # 3. Add some conversational improvements
        polished = re.sub(r'([.!?])\s+([A-Z])', r'\1 ... \2', polished)
        
        return type('Response', (), {
            'choices': [type('Choice', (), {
                'message': type('Message', (), {
                    'content': polished
                })()
            })()]
        })()

def create_test_script():
    """Create a large test script that would trigger chunked polishing."""
    
    test_script = """
    The sophisticated narrative surrounding the remarkable confrontation between Jean-Claude Van Damme and Steven Seagal demonstrates a comprehensive examination of martial arts cinema's most substantial rivalry. This elaborate feud, which subsequently evolved into a paradigm of professional animosity, facilitated numerous discussions regarding the methodology of action filmmaking and the considerable impact of personal conflicts on artistic expression.

    The approximately twenty-year period during which these two martial arts icons maintained their professional distance nevertheless produced an extensive body of work that utilized their respective talents in profoundly different ways. Van Damme's approach, characterized by its sophisticated choreography and elaborate stunt sequences, demonstrated a remarkable commitment to physical authenticity. Seagal's methodology, conversely, emphasized a more comprehensive understanding of martial arts philosophy, subsequently influencing the entire genre.

    The substantial financial implications of their rivalry, which approximately amounted to millions of dollars in potential revenue, facilitated numerous discussions regarding the economic aspects of action cinema. Furthermore, the considerable media attention generated by their ongoing conflict demonstrated the public's fascination with real-life drama in the entertainment industry. This elaborate narrative, which subsequently became a significant part of martial arts cinema history, nevertheless failed to produce the ultimate confrontation that fans had approximately anticipated for decades.

    The sophisticated analysis of their respective careers reveals a comprehensive pattern of professional development that utilized different approaches to martial arts cinema. Van Damme's elaborate choreography and sophisticated stunt work demonstrated a remarkable commitment to physical authenticity, while Seagal's methodology emphasized a more comprehensive understanding of martial arts philosophy. This substantial difference in approach, which approximately defined their respective contributions to the genre, nevertheless produced equally significant results in terms of commercial success and artistic achievement.

    The approximately twenty-year period during which these two martial arts icons maintained their professional distance nevertheless produced an extensive body of work that utilized their respective talents in profoundly different ways. Van Damme's approach, characterized by its sophisticated choreography and elaborate stunt sequences, demonstrated a remarkable commitment to physical authenticity. Seagal's methodology, conversely, emphasized a more comprehensive understanding of martial arts philosophy, subsequently influencing the entire genre.

    The substantial financial implications of their rivalry, which approximately amounted to millions of dollars in potential revenue, facilitated numerous discussions regarding the economic aspects of action cinema. Furthermore, the considerable media attention generated by their ongoing conflict demonstrated the public's fascination with real-life drama in the entertainment industry. This elaborate narrative, which subsequently became a significant part of martial arts cinema history, nevertheless failed to produce the ultimate confrontation that fans had approximately anticipated for decades.
    """
    
    return test_script

async def test_chunked_polishing_scenario(scenario_name, failure_scenario="success"):
    """Test chunked polishing with different scenarios."""
    
    print(f"\n🧪 TESTING SCENARIO: {scenario_name}")
    print("=" * 60)
    
    # Create test script
    original_script = create_test_script()
    print(f"📄 ORIGINAL SCRIPT LENGTH: {len(original_script)} characters")
    
    # Create mock generator with specific failure scenario
    mock_openai = MockOpenAIService(failure_scenario)
    generator = TopicDrivenScriptGenerator(mock_openai)
    
    # Test chunked polishing
    print("🔧 APPLYING CHUNKED POLISHING...")
    polished_script = await generator._polish_large_script(original_script)
    
    print(f"✅ POLISHED SCRIPT LENGTH: {len(polished_script)} characters")
    
    # Analyze results
    content_ratio = len(polished_script) / len(original_script)
    print(f"📊 CONTENT RATIO: {content_ratio:.2%}")
    
    # Check for improvements
    improvements = []
    if 'sophisticated' not in polished_script and 'sophisticated' in original_script:
        improvements.append("✅ Complex words simplified")
    if 'consequently' not in polished_script and 'consequently' in original_script:
        improvements.append("✅ Vocabulary improved")
    if '...' in polished_script:
        improvements.append("✅ Flow improvements added")
    
    if improvements:
        print(" IMPROVEMENTS DETECTED:")
        for improvement in improvements:
            print(f"  {improvement}")
    else:
        print("❌ No improvements detected")
    
    # Check for issues
    issues = []
    if content_ratio < 0.7:
        issues.append("⚠️ Content loss too high")
    if content_ratio > 2.0:
        issues.append("⚠️ Content expansion too high")
    if polished_script == original_script:
        issues.append("⚠️ No polishing applied")
    
    if issues:
        print(" ISSUES DETECTED:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ No issues detected")
    
    return polished_script, content_ratio, len(improvements), len(issues)

async def run_comprehensive_test():
    """Run comprehensive tests for all scenarios."""
    
    print("�� COMPREHENSIVE CHUNKED POLISHING TEST")
    print("=" * 80)
    
    scenarios = [
        ("SUCCESS SCENARIO", "success"),
        ("PARTIAL FAILURE SCENARIO", "partial_failure"),
        ("TIMEOUT SCENARIO", "timeout"),
        ("EMPTY RESPONSE SCENARIO", "empty_response"),
        ("TOO SHORT RESPONSE SCENARIO", "too_short")
    ]
    
    results = []
    
    for scenario_name, failure_scenario in scenarios:
        try:
            polished_script, content_ratio, improvements, issues = await test_chunked_polishing_scenario(
                scenario_name, failure_scenario
            )
            results.append({
                'scenario': scenario_name,
                'content_ratio': content_ratio,
                'improvements': improvements,
                'issues': issues,
                'success': issues == 0
            })
        except Exception as e:
            print(f"❌ SCENARIO FAILED: {str(e)}")
            results.append({
                'scenario': scenario_name,
                'content_ratio': 0,
                'improvements': 0,
                'issues': 1,
                'success': False
            })
    
    # Summary
    print(f"\n�� TEST SUMMARY")
    print("=" * 60)
    
    successful_scenarios = sum(1 for r in results if r['success'])
    total_scenarios = len(results)
    
    print(f"✅ SUCCESSFUL SCENARIOS: {successful_scenarios}/{total_scenarios}")
    print(f"📊 SUCCESS RATE: {(successful_scenarios/total_scenarios)*100:.1f}%")
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{status} {result['scenario']}: {result['improvements']} improvements, {result['issues']} issues")
    
    print(f"\n�� OVERALL ASSESSMENT:")
    if successful_scenarios >= 3:
        print("✅ CHUNKED POLISHING SYSTEM IS WORKING WELL")
    elif successful_scenarios >= 2:
        print("⚠️ CHUNKED POLISHING SYSTEM NEEDS MINOR IMPROVEMENTS")
    else:
        print("❌ CHUNKED POLISHING SYSTEM NEEDS MAJOR FIXES")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_test()) 
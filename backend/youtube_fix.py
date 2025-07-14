#!/usr/bin/env python3
"""
FINAL VICTORY - Use JSON3 format correctly
"""
import asyncio
import yt_dlp
import urllib.request
import json

async def victory_transcript_extraction(video_id: str):
    """FINAL working solution using JSON3."""
    video_url = f"https://youtube.com/watch?v={video_id}"
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        
        # Get subtitle URLs
        subtitles = info.get('subtitles', {})
        auto_captions = info.get('automatic_captions', {})
        
        subtitle_tracks = subtitles.get('en') or auto_captions.get('en')
        
        if not subtitle_tracks:
            raise Exception("No English subtitles found")
        
        # Find JSON3 format (best quality with timestamps)
        json3_track = None
        for track in subtitle_tracks:
            if track.get('ext') == 'json3':
                json3_track = track
                break
        
        if json3_track:
            print("🎯 Using JSON3 format (highest quality)")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            req = urllib.request.Request(json3_track['url'], headers=headers)
            with urllib.request.urlopen(req) as response:
                content = response.read().decode('utf-8')
                data = json.loads(content)
                
                transcript = []
                
                for event in data.get('events', []):
                    if 'segs' in event and event.get('tStartMs') is not None:
                        text_parts = []
                        for seg in event['segs']:
                            if 'utf8' in seg:
                                text_parts.append(seg['utf8'])
                        
                        if text_parts:
                            full_text = ''.join(text_parts).strip()
                            
                            # Less aggressive filtering - keep more content
                            if (full_text and 
                                len(full_text) > 1 and
                                not full_text.startswith('[♪♪♪]')):  # Only filter pure music symbols
                                
                                transcript.append({
                                    'text': full_text,
                                    'start': event['tStartMs'] / 1000.0,
                                    'duration': event.get('dDurationMs', 0) / 1000.0
                                })
                
                print(f"✅ JSON3 SUCCESS: {len(transcript)} segments extracted!")
                return transcript
        
        # Fallback to SRV1 with better parsing
        for track in subtitle_tracks:
            if track.get('ext') == 'srv1':
                print("🔄 Fallback to SRV1 format")
                
                req = urllib.request.Request(track['url'], headers=headers)
                with urllib.request.urlopen(req) as response:
                    content = response.read().decode('utf-8')
                    
                    import re
                    pattern = r'<text start="([^"]*)"[^>]*>([^<]*)</text>'
                    matches = re.findall(pattern, content, re.DOTALL)
                    
                    transcript = []
                    for start_str, text in matches:
                        clean_text = (text.replace('&amp;', '&')
                                    .replace('&#39;', "'")
                                    .strip())
                        
                        # Less aggressive filtering
                        if (clean_text and 
                            len(clean_text) > 1 and
                            not clean_text.startswith('[♪♪♪]')):
                            
                            try:
                                transcript.append({
                                    'text': clean_text,
                                    'start': float(start_str),
                                    'duration': 2.0
                                })
                            except ValueError:
                                continue
                    
                    print(f"✅ SRV1 SUCCESS: {len(transcript)} segments extracted!")
                    return transcript
        
        raise Exception("No working format found")

# Test final victory
if __name__ == "__main__":
    result = asyncio.run(victory_transcript_extraction("dQw4w9WgXcQ"))
    print(f"\n🏆 TOTAL VICTORY: {len(result)} segments")
    print(f"First 3 segments:")
    for i, seg in enumerate(result[:3]):
        print(f"  {i+1}: [{seg['start']:.1f}s] {seg['text']}")

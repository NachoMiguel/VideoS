import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import asyncio
import os
from pathlib import Path
import logging
from config import settings

# Import local services
from services.image_search import ImageSearchService
from services.face_detection import FaceDetector

logger = logging.getLogger(__name__)

class AdvancedVideoProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.image_search = ImageSearchService()
        self.face_detector = FaceDetector()
    
    async def extract_characters_from_script(self, script_content: str) -> List[str]:
        """Extract character names from the script content."""
        # For now, use known characters from the script
        known_characters = ["Jean Claude Van Damme", "Steven Seagal"]
        
        # Simple keyword matching
        extracted_characters = []
        for character in known_characters:
            if character.lower() in script_content.lower():
                extracted_characters.append(character)
        
        if not extracted_characters:
            # Fallback to known characters
            extracted_characters = known_characters
        
        self.logger.info(f"Extracted characters: {extracted_characters}")
        return extracted_characters
    
    async def search_character_images(self, characters: List[str]) -> Dict[str, List[str]]:
        """Search for character images using local ImageSearchService."""
        try:
            self.logger.info(f"Searching for images of: {characters}")
            character_images = await self.image_search.search_character_images(characters)
            self.logger.info(f"Found images for {len(character_images)} characters")
            return character_images
            
        except Exception as e:
            self.logger.error(f"Character image search failed: {str(e)}")
            return {}
    
    async def train_face_recognition(self, character_images: Dict[str, List[str]]) -> Dict:
        """Train face recognition model using local FaceDetector."""
        try:
            self.logger.info("Training face recognition model...")
            face_model = await self.face_detector.train_character_faces(character_images)
            self.logger.info(f"Face recognition trained for {len(face_model)} characters")
            return face_model
            
        except Exception as e:
            self.logger.error(f"Face recognition training failed: {str(e)}")
            return {}
    
    async def analyze_videos_with_character_detection(self, video_paths: List[str], face_model: Dict) -> List[Dict]:
        """Analyze videos using local face detection."""
        try:
            self.logger.info(f"Analyzing {len(video_paths)} videos with character detection")
            
            all_scenes = []
            
            for video_path in video_paths:
                self.logger.info(f"Processing video: {video_path}")
                
                # Extract scenes with face detection
                scenes = await self._extract_scenes_with_faces(video_path)
                
                # Enhance scenes with character detection
                character_scenes = []
                for scene in scenes:
                    detected_characters = await self._detect_characters_in_scene(
                        scene, face_model
                    )
                    
                    if detected_characters:
                        scene['detected_characters'] = detected_characters
                        scene['has_characters'] = True
                        character_scenes.append(scene)
                
                all_scenes.extend(character_scenes)
                self.logger.info(f"Found {len(character_scenes)} character scenes in {video_path}")
            
            return all_scenes
            
        except Exception as e:
            self.logger.error(f"Video analysis failed: {str(e)}")
            return []
    
    async def _extract_scenes_with_faces(self, video_path: str) -> List[Dict]:
        """Extract scenes with face detection."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Failed to open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            scenes = []
            current_scene = None
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_count / fps
                
                # Detect faces in frame
                faces = self.face_detector.detect_faces(frame)
                
                # Determine if this is a new scene
                if self._is_new_scene(faces, current_scene):
                    if current_scene:
                        scenes.append(current_scene)
                    
                    current_scene = {
                        "start_time": timestamp,
                        "end_time": timestamp,
                        "video_path": video_path,
                        "start_frame": frame_count,
                        "faces": faces
                    }
                else:
                    if current_scene:
                        current_scene["end_time"] = timestamp
                        current_scene["faces"].extend(faces)
                
                frame_count += 1
                
                # Progress update every 1000 frames
                if frame_count % 1000 == 0:
                    self.logger.info(f"Processed {frame_count}/{total_frames} frames")
            
            cap.release()
            
            # Add final scene
            if current_scene:
                scenes.append(current_scene)
            
            # Filter scenes by duration
            scenes = [s for s in scenes if 
                     s["end_time"] - s["start_time"] >= settings.min_scene_duration and
                     s["end_time"] - s["start_time"] <= settings.max_scene_duration]
            
            self.logger.info(f"Extracted {len(scenes)} scenes from {video_path}")
            return scenes
            
        except Exception as e:
            self.logger.error(f"Scene extraction failed: {str(e)}")
            return []
    
    def _is_new_scene(self, current_faces: List[Dict], current_scene: Optional[Dict]) -> bool:
        """Determine if current frame starts a new scene."""
        if not current_scene:
            return True
        
        # Check if face count changed significantly
        if len(current_faces) != len(current_scene["faces"]):
            return True
        
        # Check if faces moved significantly
        if current_faces and current_scene["faces"]:
            current_centers = [f["center"] for f in current_faces]
            scene_centers = [f["center"] for f in current_scene["faces"]]
            
            for curr_center in current_centers:
                for scene_center in scene_centers:
                    distance = np.sqrt((curr_center[0] - scene_center[0])**2 + 
                                     (curr_center[1] - scene_center[1])**2)
                    if distance > 100:  # Threshold for significant movement
                        return True
        
        return False
    
    async def _detect_characters_in_scene(self, scene: Dict, face_model: Dict) -> List[str]:
        """Detect specific characters in a scene."""
        try:
            # Sample frames from the scene for character detection
            video_path = scene["video_path"]
            start_time = scene["start_time"]
            end_time = scene["end_time"]
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            detected_characters = set()
            
            # Sample every 30 frames (1 second at 30fps)
            for frame_num in range(start_frame, end_frame, 30):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Detect faces and recognize characters
                faces = self.face_detector.detect_faces(frame)
                
                for face in faces:
                    character = await self.face_detector.recognize_character(face, face_model)
                    if character:
                        detected_characters.add(character)
            
            cap.release()
            return list(detected_characters)
            
        except Exception as e:
            self.logger.error(f"Character detection in scene failed: {str(e)}")
            return []
    
    async def select_best_scenes(self, scenes: List[Dict], script_content: str) -> List[Dict]:
        """Select the best scenes based on script content and character presence."""
        try:
            self.logger.info(f"Selecting best scenes from {len(scenes)} available scenes")
            
            # Extract characters from script
            script_characters = await self.extract_characters_from_script(script_content)
            
            # Score scenes based on character presence and duration
            scored_scenes = []
            for scene in scenes:
                score = 0
                
                # Score based on character presence
                detected_characters = scene.get('detected_characters', [])
                for script_char in script_characters:
                    if script_char in detected_characters:
                        score += 10
                
                # Score based on scene duration (prefer longer scenes)
                duration = scene['end_time'] - scene['start_time']
                if 5 <= duration <= 15:  # Optimal duration
                    score += 5
                elif duration > 15:
                    score += 2
                
                scene['score'] = score
                scored_scenes.append(scene)
            
            # Sort by score and select top scenes
            scored_scenes.sort(key=lambda x: x['score'], reverse=True)
            
            # Select scenes that cover the script duration
            selected_scenes = []
            total_duration = 0
            target_duration = len(script_content.split()) / 150  # Rough estimate: 150 words per minute
            
            for scene in scored_scenes:
                if scene['score'] > 0:  # Only scenes with characters
                    selected_scenes.append(scene)
                    total_duration += scene['end_time'] - scene['start_time']
                    
                    if total_duration >= target_duration:
                        break
            
            self.logger.info(f"Selected {len(selected_scenes)} scenes with total duration: {total_duration:.2f}s")
            return selected_scenes
            
        except Exception as e:
            self.logger.error(f"Scene selection failed: {str(e)}")
            return scenes[:10]  # Fallback to first 10 scenes
    
    async def assemble_video(self, scenes: List[Dict], audio_path: str, output_path: str) -> str:
        """Assemble final video from selected scenes with audio synchronization."""
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
            
            self.logger.info(f"Assembling video from {len(scenes)} scenes")
            
            if not scenes:
                raise Exception("No scenes to assemble")
            
            # Load audio
            audio_clip = AudioFileClip(audio_path)
            audio_duration = audio_clip.duration
            
            # Extract video clips
            video_clips = []
            current_duration = 0
            
            for scene in scenes:
                video_path = scene['video_path']
                start_time = scene['start_time']
                end_time = scene['end_time']
                
                # Load video clip
                video_clip = VideoFileClip(video_path)
                
                # Extract scene segment
                scene_clip = video_clip.subclip(start_time, end_time)
                video_clips.append(scene_clip)
                
                current_duration += scene_clip.duration
                video_clip.close()
                
                # Stop if we have enough content
                if current_duration >= audio_duration:
                    break
            
            if not video_clips:
                raise Exception("No video clips to assemble")
            
            # Concatenate video clips
            final_video = concatenate_videoclips(video_clips, method="compose")
            
            # Trim video to match audio duration
            if final_video.duration > audio_duration:
                final_video = final_video.subclip(0, audio_duration)
            
            # Set audio
            final_video = final_video.set_audio(audio_clip)
            
            # Write final video
            final_video.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile="temp-audio.m4a",
                remove_temp=True
            )
            
            # Cleanup
            final_video.close()
            audio_clip.close()
            
            self.logger.info(f"Video assembly completed: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Video assembly failed: {str(e)}")
            raise 
import pytest
from pureframe.pipeline.fuse import fuse
from pureframe.config import Config
from pureframe.pipeline.shots import Shot, Category, Action
from pureframe.pipeline.detect.nudity import Detection
from pureframe.pipeline.detect.scene_clip import ShotContext
from pureframe.pipeline.detect.audio import AudioContext

def get_shot():
    return Shot(index=0, start_frame=0, end_frame=100, start_time=0.0, end_time=4.0)

def test_fuse_nudity():
    config = Config(input_path="d", output_path="d")
    shot = get_shot()
    dets = [[Detection(label="FEMALE_BREAST_EXPOSED", score=0.99, box=(0,0,10,10))]]
    scene_ctx = ShotContext(explicit_act_score=0, implied_sex_score=0, kissing_score=0, safe_score=1)
    audio_ctx = AudioContext(moaning_score=0, sexual_audio_score=0, music_score=0, speech_score=0)
    
    verdict = fuse(shot, dets, scene_ctx, audio_ctx, config)
    assert verdict.category == Category.NUDITY_EXPLICIT
    assert verdict.action == Action.BLACK_BOX

def test_fuse_sexual_act():
    config = Config(input_path="d", output_path="d")
    shot = get_shot()
    dets = []
    scene_ctx = ShotContext(explicit_act_score=0.5, implied_sex_score=0, kissing_score=0, safe_score=0)
    audio_ctx = AudioContext(moaning_score=0.5, sexual_audio_score=0.5, music_score=0, speech_score=0)
    
    verdict = fuse(shot, dets, scene_ctx, audio_ctx, config)
    assert verdict.category == Category.SEXUAL_ACT_VISIBLE
    assert verdict.action == Action.FULL_FRAME_BLUR

def test_fuse_implied_sex():
    config = Config(input_path="d", output_path="d")
    shot = get_shot()
    dets = []
    scene_ctx = ShotContext(explicit_act_score=0, implied_sex_score=0.6, kissing_score=0, safe_score=0)
    audio_ctx = AudioContext(moaning_score=0.4, sexual_audio_score=0, music_score=0, speech_score=0)
    
    verdict = fuse(shot, dets, scene_ctx, audio_ctx, config)
    assert verdict.category == Category.SEXUAL_CONTEXT_NO_NUDITY
    assert verdict.action == Action.FULL_FRAME_BLUR

def test_fuse_kiss_intense():
    config = Config(input_path="d", output_path="d")
    shot = get_shot() 
    dets = []
    scene_ctx = ShotContext(explicit_act_score=0, implied_sex_score=0, kissing_score=0.6, safe_score=0)
    audio_ctx = AudioContext(moaning_score=0, sexual_audio_score=0, music_score=0, speech_score=0)
    
    verdict = fuse(shot, dets, scene_ctx, audio_ctx, config)
    assert verdict.category == Category.KISS_INTENSE
    assert verdict.action == Action.BLACK_BOX
    
def test_fuse_kiss_light():
    config = Config(input_path="d", output_path="d")
    shot = Shot(index=0, start_frame=0, end_frame=50, start_time=0.0, end_time=2.0)
    dets = []
    scene_ctx = ShotContext(explicit_act_score=0, implied_sex_score=0, kissing_score=0.6, safe_score=0)
    audio_ctx = AudioContext(moaning_score=0, sexual_audio_score=0, music_score=0, speech_score=0)
    
    verdict = fuse(shot, dets, scene_ctx, audio_ctx, config)
    assert verdict.category == Category.KISS_LIGHT
    assert verdict.action == Action.NONE
    
def test_fuse_safe():
    config = Config(input_path="d", output_path="d")
    shot = get_shot()
    dets = []
    scene_ctx = ShotContext(explicit_act_score=0, implied_sex_score=0, kissing_score=0, safe_score=0.9)
    audio_ctx = AudioContext(moaning_score=0, sexual_audio_score=0, music_score=0, speech_score=0)
    
    verdict = fuse(shot, dets, scene_ctx, audio_ctx, config)
    assert verdict.category == Category.SAFE
    assert verdict.action == Action.NONE

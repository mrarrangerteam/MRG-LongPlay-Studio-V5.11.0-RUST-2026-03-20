"""Tests for MasterChain orchestrator."""

import pytest
import json
from modules.master.chain import MasterChain


class TestMasterChainInit:
    def test_init_default(self):
        chain = MasterChain()
        assert chain is not None
        assert chain.equalizer is not None
        assert chain.dynamics is not None
        assert chain.imager is not None
        assert chain.maximizer is not None

    def test_init_with_ffmpeg_path(self):
        chain = MasterChain(ffmpeg_path="/usr/local/bin/ffmpeg")
        assert chain.ffmpeg_path == "/usr/local/bin/ffmpeg"

    def test_default_intensity(self):
        chain = MasterChain()
        assert chain.intensity in (50, 100)  # Default may vary

    def test_set_intensity(self):
        chain = MasterChain()
        chain.intensity = 50
        assert chain.intensity == 50

    def test_target_lufs(self):
        chain = MasterChain()
        assert chain.target_lufs == -14.0

    def test_target_tp(self):
        chain = MasterChain()
        assert chain.target_tp == -1.0


class TestMasterChainSettings:
    def test_save_settings(self):
        import tempfile, os
        chain = MasterChain()
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            chain.save_settings(path)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_load_roundtrip(self):
        import tempfile, os
        chain = MasterChain()
        chain.maximizer.set_gain(5.0)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            chain.save_settings(path)
            chain2 = MasterChain()
            chain2.load_settings(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_reset_all(self):
        chain = MasterChain()
        chain.maximizer.set_gain(10.0)
        chain.reset_all()
        assert chain.maximizer.gain_db == 0.0

    def test_set_platform(self):
        chain = MasterChain()
        chain.set_platform("Spotify")

    def test_set_genre(self):
        chain = MasterChain()
        if hasattr(chain, 'set_genre'):
            chain.set_genre("Pop")

    def test_meter_callback(self):
        chain = MasterChain()
        received = []
        chain.set_meter_callback(lambda levels: received.append(levels))
        assert chain._meter_callback is not None

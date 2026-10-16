"""Checkpoint-free tests for the whole-file conversion pass.

No weights and no audio device: a stub ASR records the windows it is handed, so
the equivalence claim ("the offline pass feeds the same windows in the same
order") is checked directly rather than through a waveform. Run in the isolated
MeanVC2 environment.
"""
import unittest
from unittest import mock
import numpy as np
try:
    import torch
    import torchaudio.compliance.kaldi as kaldi
except ImportError:
    raise unittest.SkipTest('Run offline pass tests in vc_models/meanvc2/.venv')
from src.vc.meanvc2 import MeanVC2Backend

HOP = 2560


class _RecordingAsr(torch.nn.Module):
    """Stands in for the frozen ASR trace and keeps every window it is given."""

    def __init__(self):
        super().__init__()
        self.windows = []
        self.offsets = []

    def forward(self, window, offset, cache_size, att_cache, cnn_cache):
        self.windows.append(window.clone())
        self.offsets.append(int(offset))
        return torch.zeros(1, 4, 256), att_cache, cnn_cache


class _CountingKaldi:
    def __init__(self):
        self.calls = 0
        self.samples = []

    def fbank(self, wave, *args, **kwargs):
        self.calls += 1
        self.samples.append(int(wave.shape[-1]))
        return kaldi.fbank(wave, *args, **kwargs)


class _CountingVocoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def decode(self, mels):
        self.calls += 1
        return torch.zeros(1, 1, mels.shape[2]*160)


class _SummingVocoder(torch.nn.Module):
    """A decode whose output follows the mel frames, frame index included.

    The all-zero stand-in makes a buffer handed over early, late or twice look
    the same as the right one; this one only stays equal when the frames land
    in the same places with the same values.
    """

    def __init__(self):
        super().__init__()
        self.calls = 0

    def decode(self, mels):
        self.calls += 1
        weights = torch.arange(mels.shape[2], dtype=mels.dtype)
        value = (mels.sum(dim=1)[0] * weights).sum()
        return torch.full((1, 1, mels.shape[2]*160), float(value))



class _FakeVcStage:
    """Stands in for a staged conversion call and keeps what was loaded into it.

    The checkpoint is not in this environment, so the schedule is checked by
    what it hands the stage: the noised input, the conditions and the cache.
    """

    def __init__(self, backend, frames, step, kv):
        self.frames = frames
        self.step = step
        self.graph = None
        self.u = None
        self.kv = None if kv is None else [(torch.zeros_like(k), torch.zeros_like(v)) for k, v in kv]
        # The checkpoint is not here, so the stage still has to hand back a cache
        # of the right shape; tagging it per step makes the next group's load
        # checkable, which is where a stage that keeps its first buffer's zeros
        # would give itself away.
        self.cache_out = [(torch.full((1, 2, 72, 256), 10.0+step), torch.full((1, 2, 72, 256), 20.0+step))
                          for _ in range(4)]
        self.loaded = []

    def fill(self, x, cond, kv):
        self.loaded.append((x.clone(), cond.clone(), None if kv is None else [(k.clone(), v.clone()) for k, v in kv]))
        if self.kv is not None:
            for (target_k, target_v), (source_k, source_v) in zip(self.kv, kv):
                target_k.copy_(source_k)
                target_v.copy_(source_v)

    def step_once(self):
        self.u = torch.zeros(1, self.frames, 80)


class _FakeVcModule:
    """Minimal stand-in for the DiT: a zero step and a cache of the right shape."""

    def __init__(self):
        self.calls = []

    def __call__(self, x, a, b, cache=None, cond=None, spks=None, offset=0,
                 is_inference=True, kv_cache=None):
        self.calls.append(dict(frames=int(x.shape[1]), offset=offset,
                               kv=None if kv_cache is None else int(kv_cache[0][0].shape[2])))
        cache_out = [(torch.full((1, 2, 72, 256), 10.0), torch.full((1, 2, 72, 256), 20.0))
                     for _ in range(4)]
        return torch.zeros(1, x.shape[1], 80), cache_out


class _RampingAsr(torch.nn.Module):
    """An ASR stand-in whose BN frames ramp with the window index."""

    def __init__(self):
        super().__init__()
        self.index = 0

    def forward(self, window, offset, cache_size, att_cache, cnn_cache):
        value = float(self.index)
        self.index += 1
        return torch.full((1, 4, 256), value), att_cache, cnn_cache


class _RecordingVcModule:
    """Minimal DiT stand-in that keeps the noised input and the conditions."""

    def __init__(self):
        self.calls = []

    def __call__(self, x, a, b, cache=None, cond=None, spks=None, offset=0,
                 is_inference=True, kv_cache=None):
        self.calls.append(dict(x=x.clone(), cond=cond.clone()))
        cache_out = [(torch.zeros(1, 2, 72, 256), torch.zeros(1, 2, 72, 256)) for _ in range(4)]
        return torch.zeros(1, x.shape[1], 80), cache_out


class BufferedScheduleTests(unittest.TestCase):
    """The schedule that captured graphs replay: buffers, not a growing concat."""

    def backend(self, hops=20):
        b = WholeFilePassTests().backend(group=6)
        b.offline_replay_enabled = True
        self.stages = []
        def factory(frames, step, kv):
            stage = _FakeVcStage(b, frames, step, kv)
            self.stages.append(stage)
            return stage
        b._offline_vc = factory
        return b

    def test_one_window_per_hop_and_one_stage_per_shape(self):
        b = self.backend()
        result = b.convert_offline(np.zeros(HOP*20, dtype=np.float32))
        replay = b.get_stats()['offline_replay']
        self.assertTrue(replay['used'] and not replay['captured'])
        # 20 hops x 16 frames is 320 condition frames; groups of 72 need 76 in
        # hand, so four groups are emitted and the trailing 32 stay unconverted.
        self.assertEqual([stage.frames for stage in self.stages], [76]*8)
        self.assertEqual([stage.step for stage in self.stages], [0, 1]*4)
        self.assertEqual([stage.kv is not None for stage in self.stages],
                         [False, False, True, True, True, True, True, True])
        self.assertEqual(b.kaldi.calls, 1)
        self.assertEqual(b.vocos.calls, 1)
        self.assertEqual(result.shape, (HOP*20,))

    def test_a_stage_that_cannot_be_staged_falls_back_to_its_eager_call(self):
        """A capture that the driver refuses must not cost the conversion.

        The two stages are independent: whichever one cannot be recorded keeps
        running eagerly while the other one replays, and the loop still hands
        both the same frames, offsets and caches as the scheme above.
        """
        b = WholeFilePassTests().backend(group=6)
        b.vc = _FakeVcModule()
        b.offline_replay_enabled = True
        b._offline_asr = lambda window_len: None
        b._offline_vc = lambda frames, step, kv: None
        result = b.convert_offline(np.zeros(HOP*20, dtype=np.float32))
        self.assertEqual(len(b.asr.windows), 20)
        self.assertEqual([call['frames'] for call in b.vc.calls], [76]*8)
        self.assertEqual([call['kv'] for call in b.vc.calls], [None, None] + [24]*6)
        self.assertEqual([call['offset'] for call in b.vc.calls], [0, 0] + [24]*6)
        self.assertTrue(np.isfinite(result).all())

    def test_a_refused_capture_is_attempted_once_per_shape(self):
        """A driver that refuses graphs must not be asked once per hop."""
        import src.vc.meanvc2 as module
        attempts = []
        b = WholeFilePassTests().backend(group=6, stub_convert=False)
        b.vc = _FakeVcModule()
        b.offline_replay_enabled = True
        def broken_asr(backend, window_len):
            attempts.append(('asr', window_len))
            raise RuntimeError('capture refused')
        def broken_vc(backend, frames, step, kv):
            attempts.append(('vc', frames, step,kv is None))
            raise RuntimeError('capture refused')
        with mock.patch.object(module, 'OfflineAsrStage', broken_asr), \
                mock.patch.object(module, 'OfflineVcStage', broken_vc):
            result = b.convert_offline(np.zeros(HOP*20, dtype=np.float32))
        # One window length, and one per step for each cache shape (absent and
        # 24 frames), never one per hop.
        self.assertEqual([name for name, *_ in attempts].count('asr'), 1)
        self.assertEqual(len(attempts) - 1, 4)
        self.assertEqual(b.get_stats()['offline_replay']['asr_graphs'], 0)
        self.assertIn('capture refused', b.get_stats()['offline_replay']['asr_error'])
        self.assertTrue(np.isfinite(result).all())

    def test_the_fallback_inside_the_buffered_loop_is_the_eager_loop(self):
        audio = (np.random.default_rng(5).normal(0, .05, HOP*20)).astype(np.float32)
        eager = WholeFilePassTests().backend(group=6, stub_convert=False, vc_module=_FakeVcModule())
        expected = eager.convert_offline(audio, replay=False)
        buffered = WholeFilePassTests().backend(group=6, stub_convert=False, vc_module=_FakeVcModule())
        buffered._offline_asr = lambda window_len: None
        buffered._offline_vc = lambda frames, step, kv: None
        np.testing.assert_array_equal(expected, buffered.convert_offline(audio))

    def test_every_group_after_the_first_is_handed_the_previous_cache(self):
        """A stage is reused per shape, so its cache has to be reloaded.

        Leaving the buffers at their initial zeros is a silent quality change:
        the arithmetic still runs and only the samples say what happened.
        """
        b = self.backend()
        b.convert_offline(np.zeros(HOP*20, dtype=np.float32))
        for stage in self.stages:
            for x, cond, kv in stage.loaded:
                self.assertEqual(x.shape, (1, 76, 80))
                if stage.kv is None:
                    self.assertIsNone(kv)
                    continue
                self.assertEqual([tuple(k.shape) for k, _ in kv], [(1, 2, 24, 256)]*4)
                # The previous group's last step, not a buffer that was never
                # written into and not the first step's cache either.
                for k, v in kv:
                    self.assertTrue(torch.all(k == 11.0), 'stale or unwritten cache keys')
                    self.assertTrue(torch.all(v == 21.0), 'stale or unwritten cache values')

    def test_the_frontend_thread_is_the_buffered_schedule_handed_over_by_cursor(self):
        """The CPU thread is scheduling, not arithmetic: same buffers, same order.

        The serial buffered route is the reference; the overlapped one may only
        differ in which thread runs the two stages. The summing decode makes the
        frames visible in the output, so a buffer handed over early, late or
        twice cannot pass as the right one.
        """
        rng = np.random.default_rng(7)
        audio = (rng.normal(0, .05, HOP*20)).astype(np.float32)

        def build(overlap):
            b = WholeFilePassTests().backend(group=6, stub_convert=False, vc_module=_FakeVcModule())
            b.vocos = _SummingVocoder()
            b.offline_overlap_enabled = overlap
            return b

        serial = build(False)
        expected = serial.convert_offline(audio, replay=True)
        overlapped = build(True)
        result = overlapped.convert_offline(audio, replay=True)
        np.testing.assert_array_equal(expected, result)
        self.assertTrue(np.any(result), 'the summing decode must move with the frames')
        self.assertEqual([int(window.shape[0]) for window in overlapped.asr.windows],
                         [int(window.shape[0]) for window in serial.asr.windows])
        self.assertEqual(overlapped.asr.offsets, serial.asr.offsets)
        self.assertEqual([call['frames'] for call in overlapped.vc.calls],
                         [call['frames'] for call in serial.vc.calls])
        self.assertEqual([call['offset'] for call in overlapped.vc.calls],
                         [call['offset'] for call in serial.vc.calls])
        self.assertEqual([call['kv'] for call in overlapped.vc.calls],
                         [call['kv'] for call in serial.vc.calls])
        self.assertTrue(overlapped.get_stats()['offline_replay']['overlap'])
        self.assertNotIn('overlap', serial.get_stats()['offline_replay'])


class WholeFilePassTests(unittest.TestCase):
    def backend(self, frontend='legacy', group=6, frames_per_call=0, stub_convert=True, vc_module=None):
        b = MeanVC2Backend(1)
        b.torch = torch
        b.device = torch.device('cpu')
        b.feature_frontend = frontend
        b.bn_interpolation = 'legacy'
        b.kaldi = _CountingKaldi()
        # A loaded backend, without loading anything: vc only has to be non-None
        # for reset(), and the conversion stage is replaced below.
        b.vc = vc_module if vc_module is not None else object()
        b.vocos = _CountingVocoder()
        b.asr = _RecordingAsr()
        b.position_table = torch.zeros(1, 5000, 256)
        b.original_positions = torch.zeros(1, 5000, 256)
        b.position_frequencies = torch.zeros(128)
        b.position_base = 0
        b.position_rolls = 0
        b.interpolation_weights = torch.arange(4)[None, None, :, None]/4
        b.cache_size_tensor = torch.tensor(8)
        b.speaker = torch.zeros(1, 256)
        b.vc_group_chunks = group
        b.steps = 2
        b.timesteps = [(torch.tensor([1.0]), torch.tensor([0.5])),
                       (torch.tensor([0.5]), torch.tensor([0.0]))]
        b.vocoder_batch_frames = 36
        b.vocoder_left = b.vocoder_right = 36
        b.max_kv_frames = 24
        b.kv = None
        b.offline_stages = {}
        b.reset()
        # The conversion stage is replaced: the tests here are about the
        # scheduling around it, and the checkpoint is not in this environment.
        # `frames_per_call` is what a real grouped call emits per invocation.
        if stub_convert:
            def fake_convert(blk=b, n=frames_per_call):
                blk.mels = torch.cat((blk.mels, torch.zeros(1, 80, n)), dim=2)
            b._convert_blocks = fake_convert
        # These tests stub the eager loop's conversion call; the buffered route
        # is covered by BufferedScheduleTests above.
        b.offline_replay_enabled = False
        return b

    def test_windows_match_the_streaming_frontend_hop_for_hop(self):
        """Same window per hop, both frontends, so the same frames reach the VC.

        This is the whole equivalence claim: the offline pass is allowed to change
        scheduling, not the neural inputs. A window that is shifted, reordered or
        dropped here would be a different voice, and no waveform comparison could
        say which hop went wrong.
        """
        rng = np.random.default_rng(11)
        audio = (rng.normal(0, .05, HOP*6)).astype(np.float32)
        for frontend in ('legacy', 'aligned'):
            streaming = self.backend(frontend)
            for index in range(6):
                streaming._frontend_fragments(audio[index*HOP:(index+1)*HOP])
            offline = self.backend(frontend)
            offline.convert_offline(audio)
            self.assertEqual(len(offline.asr.windows), len(streaming.asr.windows), frontend)
            for hop, (left, right) in enumerate(zip(streaming.asr.windows, offline.asr.windows)):
                self.assertEqual(tuple(left.shape), tuple(right.shape), f'{frontend} hop {hop}')
                # One kaldi call over the whole file rounds differently from many
                # short ones; the frames are the same to float32 noise, not to the bit.
                self.assertTrue(torch.allclose(left, right, atol=1e-5, rtol=1e-5),
                                f'{frontend} hop {hop}: max diff '
                                f'{(left-right).abs().max().item():.3e}')
            self.assertEqual(offline.asr.offsets, streaming.asr.offsets, frontend)

    def test_one_filterbank_call_and_one_decode_for_the_whole_input(self):
        """What the pass is for: no per-hop round trip to the compute device.

        Ten hops used to mean ten kaldi calls and (for a batch of 36 frames) ten
        device crossings. One call each is the difference between the device
        waiting and the device working.
        """
        audio = np.zeros(HOP*10, dtype=np.float32)
        b = self.backend(frames_per_call=16)
        result = b.convert_offline(audio)
        self.assertEqual(b.kaldi.calls, 1)
        self.assertEqual(b.vocos.calls, 1)
        self.assertEqual(b.get_stats()['offline_hops'], 10)
        # The caller trims by hop count, so the pass owes it exactly that many.
        self.assertEqual(result.shape, (HOP*10,))

    def test_legacy_grid_starts_on_the_carry_frame_the_streaming_loop_built(self):
        """The 240-sample carry is the reason the offline grid needs a prefix.

        Dropping it would move every legacy frame by 15ms, which is the same
        mistake as feeding the trace a different window; the first hop's window of
        each route has to agree, and that is where the prefix shows up.
        """
        audio = np.zeros(HOP*2, dtype=np.float32)
        legacy = self.backend('legacy')
        legacy.convert_offline(audio)
        aligned = self.backend('aligned')
        aligned.convert_offline(audio)
        self.assertEqual(legacy.kaldi.samples, [len(audio)+240])
        self.assertEqual(aligned.kaldi.samples, [len(audio)])

    def test_a_stream_that_cannot_fill_every_hop_fails_loudly(self):
        """The caller trims by hop count, so a short stream must not pass silently."""
        audio = np.zeros(HOP*40, dtype=np.float32)
        b = self.backend()
        with self.assertRaises(RuntimeError):
            b.convert_offline(audio)

    def test_offline_input_is_validated_like_the_streaming_one(self):
        b = self.backend()
        with self.assertRaises(ValueError):
            b.convert_offline(np.zeros(HOP, dtype=np.float64))
        with self.assertRaises(ValueError):
            b.convert_offline(np.full(HOP, np.nan, dtype=np.float32))
        b.vc = None
        with self.assertRaises(RuntimeError):
            b.convert_offline(np.zeros(HOP, dtype=np.float32))


class QualityRecipeTests(unittest.TestCase):
    """The quality recipe: wider step budget, aligned grid, upstream conditions.

    No checkpoint here either: the recipe keys are checked for what they
    change -- the steps budget, the grid the conditions come from, the
    whole-utterance interpolation and noise -- since those are exactly what
    the listening comparison decides between.
    """

    def test_the_steps_budget_is_the_callers_choice(self):
        b = WholeFilePassTests().backend()
        with self.assertRaises(ValueError):
            b.select_profile({'steps': 4})
        b.select_profile({'steps': 4}, steps_allowed=(2, 3, 4, 6, 8))
        self.assertEqual(b.steps, 4)
        self.assertEqual(len(b.timesteps), 4)
        self.assertFalse(b.stats['human_approved_offline'])

    def test_official_conditioning_requires_the_aligned_frontend(self):
        b = WholeFilePassTests().backend(frontend='legacy')
        with self.assertRaises(ValueError):
            b.select_profile({'offline_conditioning': 'official'})
        # The dispatch refuses a direct attribute flip as well.
        b.offline_conditioning = 'official'
        with self.assertRaises(ValueError):
            b.convert_offline(np.zeros(HOP*6, dtype=np.float32))

    def test_official_conditioning_interpolates_the_whole_utterance(self):
        """Conditions and noise come from one whole-file pass, not per hop.

        A ramping ASR makes each window's BN output different, so the expected
        interpolation is computable directly: it is the same
        ``F.interpolate(..., size=4B, align_corners=True)`` the upstream
        inference runs over the concatenated windows, and the noise is one
        seeded draw sliced by group.
        """
        hops = 6
        b = WholeFilePassTests().backend(frontend='aligned', stub_convert=False,
                                         vc_module=_RecordingVcModule())
        b.select_profile({'offline_conditioning': 'official'})
        b.asr = _RampingAsr()
        b.convert_offline(np.zeros(HOP*hops, dtype=np.float32))
        windows = hops-1                     # the aligned grid ends a hop short
        bn = torch.cat([torch.full((1, 4, 256), float(index)) for index in range(windows)], dim=1)
        expected_cond = torch.nn.functional.interpolate(bn.transpose(1, 2), size=bn.shape[1]*4,
                                                        mode='linear', align_corners=True).transpose(1, 2)
        expected_noise = torch.randn(1, bn.shape[1]*4, 80,
                                     generator=torch.Generator().manual_seed(110))
        # One group fits in 80 condition frames (76 needed): two Euler calls.
        self.assertEqual(len(b.vc.calls), 2)
        for call in b.vc.calls:
            self.assertEqual(tuple(call['x'].shape), (1, 76, 80))
            self.assertTrue(torch.allclose(call['x'], expected_noise[:, :76]))
            self.assertTrue(torch.equal(call['cond'], expected_cond[:, :76]))

    def test_official_route_replays_stages_on_the_buffered_schedule(self):
        """The staged variant is the same group schedule, not a different one.

        Captured stages key on (frames, step, cache length), so the replayed
        calls stay one per step per group while the conditions come from the
        whole-file pass; the stats say which conditioning ran.
        """
        owner = BufferedScheduleTests()
        b = owner.backend()
        b.select_profile({'feature_frontend': 'aligned', 'offline_conditioning': 'official'})
        result = b.convert_offline(np.zeros(HOP*20, dtype=np.float32))
        replay = b.get_stats()['offline_replay']
        self.assertTrue(replay['used'])
        self.assertTrue(replay['official'])
        self.assertEqual(len(b.asr.windows), 19)
        self.assertEqual([stage.frames for stage in owner.stages], [76]*8)
        self.assertEqual(result.shape, (HOP*20,))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

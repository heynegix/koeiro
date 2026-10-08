"""Checkpoint-free grouped inference tests in the isolated MeanVC2 runtime."""
import unittest
import numpy as np
try:
    import torch
except ImportError:
    raise unittest.SkipTest('Run in the isolated MeanVC2 environment')

from src.vc.meanvc2_continuity import MeanVC2ContinuityBackend
from src.vc.meanvc2 import MeanVC2Backend


class ContinuityTests(unittest.TestCase):
    def test_production_warmup_reaches_grouped_decode_then_resets(self):
        for frames, expected in ((1, 4), (36, 9)):
            b=MeanVC2Backend(1);b.feature_frontend='aligned';b.vocoder_batch_frames=frames
            calls=[];resets=[]
            b.process_chunk=lambda audio:calls.append(audio.copy())
            b.reset=lambda:resets.append(True)
            b.warmup()
            self.assertEqual(len(calls),expected)
            self.assertEqual(resets,[True])
            self.assertTrue(all(c.shape==(2560,) and not c.any() for c in calls))

    def test_shipped_voice_buffer_matches_profile_and_preserves_voice_assets(self):
        import json
        from pathlib import Path
        from src.vc.models import DEFAULT_VOICE_ID, profile
        root=Path(__file__).resolve().parents[1]
        selected=profile(DEFAULT_VOICE_ID)
        runtime=json.loads((root/'models'/selected['folder']/'runtime.json').read_text('utf-8'))
        b=MeanVC2Backend(1);b.feature_frontend=runtime['feature_frontend']
        b.vocoder_batch_frames=selected['vocoder_batch_frames']
        b.vc_group_chunks=selected['vc_group_chunks']
        self.assertEqual(b.vc_group_chunks,6)
        self.assertEqual(b.algorithmic_buffer_ms,selected['model_buffer_ms'])
        self.assertEqual(runtime['steps'],2)
        # The reference and its embedding must stay the ones registration recorded.
        folder=root/'models'/selected['folder']
        self.assertTrue((folder/'fixed_embedding.npy').is_file())
        self.assertTrue((folder/'reference.wav').is_file())
        assets={Path(a['path']).name for a in runtime['assets']}
        self.assertLessEqual({'reference.wav','fixed_embedding.npy'},assets)

    def test_shipped_profile_matches_its_declared_model_buffer(self):
        """The GUI shows model_buffer_ms; it must equal what the backend buffers."""
        import json
        from pathlib import Path
        from src.vc.models import DEFAULT_VOICE_ID, profile
        root=Path(__file__).resolve().parents[1]
        selected=profile(DEFAULT_VOICE_ID)
        runtime=json.loads((root/'models'/selected['folder']/'runtime.json').read_text('utf-8'))
        b=MeanVC2Backend(1)
        b.feature_frontend=runtime.get('feature_frontend','legacy')
        b.vc_group_chunks=selected.get('vc_group_chunks',runtime.get('vc_group_chunks',1))
        b.vocoder_batch_frames=selected.get('vocoder_batch_frames',runtime.get('vocoder_batch_frames',1))
        self.assertEqual(b.algorithmic_buffer_ms,1440)
        self.assertEqual(selected['model_buffer_ms'],1440)
        self.assertEqual(b.vc_group_chunks,6)
        self.assertEqual(b.vocoder_batch_frames,36)

    def test_shipped_profile_applies_grouping_on_a_loaded_backend(self):
        import json
        from pathlib import Path
        from src.vc.models import DEFAULT_VOICE_ID, profile
        root=Path(__file__).resolve().parents[1]
        selected=profile(DEFAULT_VOICE_ID)
        b=MeanVC2Backend(1)
        b.load(root/'models'/selected['folder'])
        self.assertEqual(b.algorithmic_buffer_ms,480)
        b.select_profile(selected)
        self.assertEqual(b.vc_group_chunks,6)
        self.assertEqual(b.vocoder_batch_frames,36)
        self.assertEqual(b.stats['algorithmic_buffer_ms'],b.algorithmic_buffer_ms)
        self.assertFalse(b.stats.get('human_approved_offline',False))
        self.assertEqual(len(b.pending_audio),b.algorithmic_buffer_ms*16)
        b.warmup()
        out=b.process_chunk(np.zeros(2560,dtype=np.float32))
        self.assertEqual(out.shape,(2560,))
        self.assertTrue(np.isfinite(out).all())
        b.unload()

    def test_shipped_profile_rejects_unbounded_grouping(self):
        from src.vc.models import DEFAULT_VOICE_ID, profile
        for key,value in (('vc_group_chunks',7),('vocoder_batch_frames',16),('vc_group_chunks',True)):
            b=MeanVC2Backend(1);b.feature_frontend='legacy';b.stats={}
            with self.assertRaises(ValueError):
                b.select_profile(dict(profile(DEFAULT_VOICE_ID),**{key:value}))

    def test_shipped_runtime_is_not_rewritten_by_the_profile_grouping(self):
        """Grouping lives in the profile, so the registered runtime keeps its own value."""
        import json
        from pathlib import Path
        from src.vc.models import DEFAULT_VOICE_ID, profile
        root=Path(__file__).resolve().parents[1]
        selected=profile(DEFAULT_VOICE_ID)
        runtime=json.loads((root/'models'/selected['folder']/'runtime.json').read_text('utf-8'))
        self.assertEqual(runtime.get('vc_group_chunks',1),1)
        self.assertEqual(runtime.get('vocoder_batch_frames',1),1)
        # A registered voice has not had a human audition, so it must not claim to.
        self.assertFalse(runtime['human_approved_offline'])
        self.assertEqual(runtime['reference_origin'],'User-added local audio')
        self.assertEqual(selected['vc_group_chunks'],6)
        self.assertEqual(selected['vocoder_batch_frames'],36)

    def test_frontend_refactor_is_behaviour_preserving(self):
        """_frontend_fragments must reproduce the inline append exactly."""
        b=MeanVC2Backend(1);b.torch=torch;b.feature_frontend='legacy';b.bn_interpolation='legacy'
        b.previous_bn=None;b.cond=torch.empty(1,0,256);b.noise=torch.empty(1,0,80)
        b.generator=torch.Generator(device='cpu').manual_seed(110)
        b.interpolation_weights=torch.arange(4,dtype=torch.float32)[None,None,:,None]/4
        frames=[torch.full((1,4,256),float(i)) for i in range(3)]
        for frame in frames:
            b._append_bn(frame)
        inline_cond,inline_noise,inline_bn=b.cond.clone(),b.noise.clone(),b.previous_bn.clone()
        b2=MeanVC2Backend(1);b2.torch=torch;b2.feature_frontend='legacy';b2.bn_interpolation='legacy'
        b2.previous_bn=None;b2.cond=torch.empty(1,0,256);b2.noise=torch.empty(1,0,80)
        b2.generator=torch.Generator(device='cpu').manual_seed(110)
        b2.interpolation_weights=b.interpolation_weights
        for frame in frames:
            cond,noise=b2._condition_frames(frame)
            b2.cond=torch.cat((b2.cond,cond),dim=1)
            b2.noise=torch.cat((b2.noise,noise),dim=1)
        self.assertTrue(torch.equal(inline_cond,b2.cond))
        self.assertTrue(torch.equal(inline_noise,b2.noise))
        self.assertTrue(torch.equal(inline_bn,b2.previous_bn))

    def test_parallel_backend_reproduces_the_serial_backend(self):
        """Same profile, same threads: the overlapped frontend must not change output."""
        from pathlib import Path
        from src.vc.meanvc2_parallel import MeanVC2ParallelBackend
        from src.vc.models import DEFAULT_VOICE_ID, profile
        root=Path(__file__).resolve().parents[1]
        selected=profile(DEFAULT_VOICE_ID)
        folder=root/'models'/selected['folder']
        rng=np.random.default_rng(5)
        blocks=[(rng.standard_normal(2560)*.05).astype(np.float32) for _ in range(24)]
        outputs=[]
        for cls,queue in ((MeanVC2Backend,None),(MeanVC2ParallelBackend,2)):
            b=cls(threads=1) if queue is None else cls(threads=1,queue_chunks=queue)
            b.load(folder);b.select_profile(selected);b.warmup()
            try:
                outputs.append(np.concatenate([b.process_chunk(x) for x in blocks]))
            finally:
                b.unload()
        np.testing.assert_array_equal(outputs[0],outputs[1])

    def test_parallel_backend_bounds_the_frontend_queue(self):
        from src.vc.meanvc2_parallel import MeanVC2ParallelBackend, MAX_QUEUE_CHUNKS
        for queue in (0,-1,True,MAX_QUEUE_CHUNKS+1):
            with self.assertRaises(ValueError):
                MeanVC2ParallelBackend(1,queue)
        for queue in (1,MAX_QUEUE_CHUNKS):
            self.assertEqual(MeanVC2ParallelBackend(1,queue).queue_chunks,queue)

    def test_parallel_backend_reports_a_failed_frontend_and_joins_its_thread(self):
        """A producer failure must reach the caller, and unload must join it."""
        from pathlib import Path
        from src.vc.meanvc2_parallel import MeanVC2ParallelBackend
        from src.vc.models import DEFAULT_VOICE_ID, profile
        root=Path(__file__).resolve().parents[1]
        b=MeanVC2ParallelBackend(1,1)
        b.load(root/'models'/profile(DEFAULT_VOICE_ID)['folder'])
        try:
            self.assertIsNotNone(b._thread)
            self.assertTrue(b._thread.is_alive())
            original=b._frontend_fragments
            def explode(_):
                raise ValueError('frontend boom')
            b._frontend_fragments=explode
            with self.assertRaises(RuntimeError):
                b.process_chunk(np.zeros(2560,dtype=np.float32))
        finally:
            b._frontend_fragments=getattr(b,'_frontend_fragments',None) or original
            b.unload()
        self.assertIsNone(b._thread)

    def test_phrase_alignment_follows_a_grouping_change(self):
        """Grouping moves the buffer; the source/voice alignment must move with it."""
        from pathlib import Path
        from src.vc.meanvc2_phrase import MeanVC2PhraseBackend
        from src.vc.models import DEFAULT_VOICE_ID, profile
        root=Path(__file__).resolve().parents[1]
        selected=profile(DEFAULT_VOICE_ID)
        b=MeanVC2PhraseBackend(1)
        b.load(root/'models'/selected['folder'])
        try:
            b.select_profile(selected)
            self.assertEqual(b.repair.alignment,b.algorithmic_buffer_ms*16)
            self.assertEqual(b.algorithmic_buffer_ms,1440)
            # Dropping to a single 120 ms block must shorten the buffer and the delay
            # the repair stage applies, so the two never disagree.
            b.select_profile(dict(selected,vc_group_chunks=1))
            self.assertEqual(b.algorithmic_buffer_ms,960)
            self.assertEqual(b.repair.alignment,960*16)
            self.assertEqual(len(b.repair.source_delay),960*16)
            b.select_profile(selected)
            self.assertEqual(b.algorithmic_buffer_ms,1440)
            self.assertEqual(b.repair.alignment,1440*16)
            self.assertEqual(len(b.repair.source_delay),1440*16)
        finally:
            b.unload()

    def test_delay_is_bounded_within_three_second_configuration(self):
        for group, frames in ((1, 1), (1, 36), (3, 1), (6, 1)):
            b = MeanVC2ContinuityBackend(1, group, frames)
            b.feature_frontend = 'aligned'
            self.assertLessEqual(b.algorithmic_buffer_ms + 40 + 1280, 3000)

    def test_invalid_unbounded_group_rejected(self):
        for group in (0, 2, 7, True):
            with self.assertRaises(ValueError):
                MeanVC2ContinuityBackend(1, group)

    def test_group_conversion_keeps_future_and_caps_cache(self):
        b = MeanVC2ContinuityBackend(1, 3)
        b.torch = torch
        b.cond = torch.zeros(1, 88, 256)
        b.noise = torch.arange(88.).reshape(1, 88, 1).expand(1, 88, 80).clone()
        b.mels = torch.empty(1, 80, 0)
        b.kv = None
        b.max_kv_frames = 24
        b.steps = 2
        b.timesteps = [(torch.tensor([1.]), torch.tensor([.5])),
                       (torch.tensor([.5]), torch.tensor([0.]))]
        b.speaker = torch.ones(1, 256)
        calls = []
        def vc(x, *times, **kwargs):
            calls.append((x.shape[1], kwargs['offset'], kwargs['kv_cache']))
            length = kwargs['offset'] + x.shape[1] - 4
            return torch.zeros_like(x), [(torch.zeros(1, 2, length, 4),
                                           torch.zeros(1, 2, length, 4))]
        b.vc = vc
        b._convert_blocks()
        self.assertEqual([c[:2] for c in calls], [(40, 0), (40, 0), (40, 24), (40, 24)])
        self.assertEqual(b.mels.shape, (1, 80, 72))
        self.assertEqual(b.cond.shape[1], 16)
        self.assertEqual(b.kv[0][0].shape[2], 24)
        self.assertTrue(torch.equal(b.noise[0, :, 0], torch.arange(72., 88.)))

    def test_production_group_matches_audition_algorithm(self):
        for group in (1,3,6):
            backends=[MeanVC2Backend(1),MeanVC2ContinuityBackend(1,group)]
            for b in backends:
                b.vc_group_chunks=group;b.torch=torch;b.cond=torch.zeros(1,160,256)
                b.noise=torch.arange(160.).reshape(1,160,1).expand(1,160,80).clone()
                b.mels=torch.empty(1,80,0);b.kv=None;b.max_kv_frames=24;b.steps=2
                b.timesteps=[(torch.tensor([1.]),torch.tensor([.5])),(torch.tensor([.5]),torch.tensor([0.]))]
                b.speaker=torch.ones(1,256)
                def vc(x,*times,**kwargs):
                    length=kwargs['offset']+x.shape[1]-4
                    return x*.125,[(torch.zeros(1,2,length,4),torch.zeros(1,2,length,4))]
                b.vc=vc;b._convert_blocks()
            for attr in ('mels','cond','noise'):
                self.assertTrue(torch.equal(getattr(backends[0],attr),getattr(backends[1],attr)))

    def test_production_grouped_warmup_reaches_decoder_and_resets(self):
        b=MeanVC2Backend(1);b.feature_frontend='aligned';b.vc_group_chunks=6
        calls=[];resets=[];b.process_chunk=lambda a:calls.append(a)
        b.reset=lambda:resets.append(True);b.warmup()
        self.assertEqual(len(calls),15);self.assertEqual(resets,[True])
        from src.vc.config import AIParameters
        from src.vc.models import DEFAULT_VOICE_ID, profile
        parameters=AIParameters(model=DEFAULT_VOICE_ID)
        # The profile's declared buffer must equal what this backend actually buffers,
        # and the total must fit inside the 2.3 s budget this route was tuned for.
        total=b.algorithmic_buffer_ms+40+parameters.startup_frames/48
        self.assertEqual(total,2280)
        self.assertLessEqual(total,2300)
        self.assertEqual(parameters.startup_frames,parameters.startup_chunks*parameters.chunk_frames)

    def test_decode_never_emits_unstable_future(self):
        b = MeanVC2ContinuityBackend(1, 1, 36)
        b.torch = torch
        b.mel_origin = b.decoded_frames = 0
        b.vocoder_left = b.vocoder_right = 36
        b.mels = torch.zeros(1, 80, 60)
        b.pending_audio = np.zeros(0, dtype=np.float32)
        class Vocoder:
            def decode(self, mels):
                return torch.arange(mels.shape[2] * 160.)
        b.vocos = Vocoder()
        b._decode_ready()
        self.assertEqual(b.decoded_frames, 0)
        b.mels = torch.zeros(1, 80, 72)
        b._decode_ready()
        self.assertEqual(b.decoded_frames, 36)
        self.assertEqual(len(b.pending_audio), 36 * 160)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

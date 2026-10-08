"""Same fixed neural voice with optional bounded asynchronous phrase repair."""
from .meanvc2 import MeanVC2Backend
from .phrase_prosody import PhraseRepair


class MeanVC2PhraseBackend(MeanVC2Backend):
    def __init__(self, threads=4):
        self.repair = None
        super().__init__(threads)

    def load(self, model_path):
        super().load(model_path)
        # Decode the same stable mel frames in fewer calls. Do not change
        # checkpoint attention, speaker conditioning or vocoder context.
        self.vocoder_batch_frames = 36
        self.reset()
        delay = self.algorithmic_buffer_ms + (40 if self.bn_interpolation == 'fixed_linear' else 0)
        self.repair = PhraseRepair(delay * 16)
        self.stats.update(phrase_repair=True, phrase_extra_delay_ms=1600,
                          phrase_source_alignment_ms=delay, phrase_quality_verified=False,
                          phrase_pitch_limit_st=.6, phrase_gain_limit_db=1.5)
        self.stats.update(vocoder_batch_frames=36, algorithmic_buffer_ms=self.algorithmic_buffer_ms)

    def process_chunk(self, audio):
        converted = super().process_chunk(audio)
        return self.repair.process(audio, converted) if self.repair is not None else converted

    def select_profile(self, selected):
        loaded = getattr(self, 'vc', None) is not None
        previous = self.algorithmic_buffer_ms if loaded else None
        super().select_profile(selected)
        if self.repair is None:
            return
        mode = selected.get('repair_mode', 'combined')
        if mode not in ('combined', 'energy'):
            raise ValueError('Unsupported phrase repair mode')
        self.repair.pitch = mode == 'combined'
        self.stats['phrase_repair_mode'] = mode
        self.stats['phrase_pitch_limit_st'] = .6 if self.repair.pitch else 0.
        if not loaded:
            return
        # Grouping moves the algorithmic buffer, and the repair aligns the source
        # against the converted waveform. Follow the buffer instead of keeping the
        # alignment captured at load time.
        delay = self.algorithmic_buffer_ms + (40 if self.bn_interpolation == 'fixed_linear' else 0)
        self.stats['phrase_source_alignment_ms'] = delay
        if previous != self.algorithmic_buffer_ms:
            self.repair.realign(delay * 16)

    def reset(self):
        super().reset()
        if self.repair is not None:
            self.repair.reset()

    def get_stats(self):
        stats = super().get_stats()
        if self.repair is not None:
            stats['phrase'] = dict(self.repair.stats)
        return stats

    def unload(self):
        if self.repair is not None:
            self.repair.close()
            self.repair = None
        super().unload()

"""Same fixed neural voice with optional bounded asynchronous phrase repair."""
from .meanvc2 import MeanVC2Backend
from .phrase_prosody import PhraseRepair


class MeanVC2PhraseBackend(MeanVC2Backend):
    def __init__(self, threads=4, device='cpu'):
        self.repair = None
        super().__init__(threads, device)

    def load(self, model_path):
        super().load(model_path)
        # Decode the same stable mel frames in fewer calls. Do not change
        # checkpoint attention, speaker conditioning or vocoder context.
        self.vocoder_batch_frames = 36
        self.reset()
        delay = self.algorithmic_buffer_ms + (40 if self.bn_interpolation == 'fixed_linear' else 0)
        self.repair = PhraseRepair(delay * 16)
        self.stats.update(phrase_repair=True, phrase_extra_delay_ms=self.repair.stats['extra_delay_ms'],
                          phrase_source_alignment_ms=delay, phrase_quality_verified=False,
                          phrase_pitch_limit_st=.6, phrase_gain_limit_db=1.5)
        self.stats.update(vocoder_batch_frames=36, algorithmic_buffer_ms=self.algorithmic_buffer_ms)

    def process_chunk(self, audio):
        converted = super().process_chunk(audio)
        return self.repair.process(audio, converted) if self.repair is not None else converted

    def convert_offline(self, audio, replay=None):
        """Whole-file conversion, then the same repair on the same hop pairs.

        The repair reads the source and the converted stream one 160ms hop at a
        time and pairs them by index, so replaying those pairs after the
        conversion gives it the identical input. What differs is the clock: the
        streaming route races its single analysis job against the hop cadence and
        bypasses corrections that are not ready, which is why its own stats carry
        late_bypass. Offline there is nothing to race, so each hop settles the
        previous analysis before moving on and every correction lands.
        """
        converted = super().convert_offline(audio, replay=replay)
        repair = self.repair
        if repair is None or repair.stats.get('disabled'):
            return converted
        hop = self.chunk_samples
        hops = len(converted) // hop
        for index in range(hops):
            repair.settle()
            slice_ = slice(index*hop, (index+1)*hop)
            converted[slice_] = repair.process(audio[slice_], converted[slice_])
        return converted

    def select_profile(self, selected, steps_allowed=(2,)):
        loaded = getattr(self, 'vc', None) is not None
        previous = self.algorithmic_buffer_ms if loaded else None
        super().select_profile(selected, steps_allowed=steps_allowed)
        if self.repair is None:
            return
        mode = selected.get('repair_mode', 'combined')
        if mode not in ('combined', 'energy'):
            raise ValueError('Unsupported phrase repair mode')
        self.repair.pitch = mode == 'combined'
        hops = selected.get('repair_lookahead_hops', 10)
        if type(hops) is not int or not 1 <= hops <= 10:
            raise ValueError('Repair lookahead must be 1..10 input hops')
        if hops != self.repair.lookahead_hops:
            # In-flight analysis windows belong to the old timing; rebuild so
            # a shorter lookahead never applies stale future context.
            delay = self.algorithmic_buffer_ms + (40 if self.bn_interpolation == 'fixed_linear' else 0)
            self.repair.close()
            self.repair = PhraseRepair(delay * 16, energy=self.repair.energy,
                                       pitch=self.repair.pitch, focus=self.repair.focus,
                                       lookahead_hops=hops)
            self.stats['phrase_extra_delay_ms'] = self.repair.stats['extra_delay_ms']
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

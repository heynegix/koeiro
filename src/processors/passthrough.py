from .base import AudioProcessor


class PassthroughProcessor(AudioProcessor):
    def process(self, audio, sample_rate):
        return audio

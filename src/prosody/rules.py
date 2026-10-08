"""Small corrections to observed intonation; no generated melody or semantics."""


class ContextRuleEngine:
    def targets(self, relative, velocity, deviation, onset_remaining, context, p):
        scale=p.amount/60
        pitch=relative*p.range_expansion*scale*(p.pitch_range/50)
        if velocity>0 and pitch>0:
            pitch*=1+p.rise_boost
        elif velocity<0 and pitch<0:
            pitch*=1+p.fall_boost
        if onset_remaining>0:
            pitch+=p.onset_lift*scale*min(1.,onset_remaining/.05)
        gain=deviation*p.energy_dynamics*scale*(p.energy/40)
        if context.state=='ONSET':
            gain+=.2*p.energy_dynamics*scale*(p.energy/40)
        if p.ending_emphasis and context.state=='ENDING_CANDIDATE' and abs(context.pitch_slope)>.6:
            pitch+=max(-.6,min(.6,context.pitch_slope*.06))*p.ending_strength*context.ending_probability*scale
            gain+=max(-.3,min(.3,context.energy_slope*.01))*p.energy_dynamics*scale*(p.energy/40)
        return pitch,gain

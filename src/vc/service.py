"""AI subprocess: model/resampling/FX/IPC; PortAudio is never imported here."""
import argparse
import json
from dataclasses import asdict
from .models import default_voice_id, is_meanvc2
from pathlib import Path
import sys
import time

from src.runtime_paths import asset_root, is_frozen
from src.audio.performance import TimingStats
from src.audio.diagnostics import AudioDiagnostics
from src.utils.windows import audio_scheduling,audio_process_policy
from .resampler import StreamingResampler
from .post_fx import LightPostFX
from .protocol import send, receive
from .realtime_gc import RealtimeGC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--factor', type=int, required=True)
    parser.add_argument('--threads', type=int, default=1)
    from .models import VOICE_PROFILES,profile
    from .config import EXPERIMENTS
    parser.add_argument('--model',choices=tuple(VOICE_PROFILES),default=default_voice_id())
    parser.add_argument('--delivery',choices=('streaming','utterance'),default='streaming')
    parser.add_argument('--enhancer',choices=('none','flashsr','mossformer','voicefixer','lavasr'),default='none')
    # Listening-comparison variant. Only the utterance+LavaSR route honours it.
    parser.add_argument('--experiment',choices=EXPERIMENTS,default='none')
    # Optional LavaSR upstream denoise. Off by default: no measured benefit on this
    # material, and it costs time inside an already RTF-sensitive path.
    parser.add_argument('--lavasr-denoise',action='store_true')
    # Voice tuning sliders. Only the natural comparison route honours them;
    # defaults reproduce the validated natural recipe exactly.
    parser.add_argument('--tune-sib-db',type=float,default=3.0)
    parser.add_argument('--tune-cons-db',type=float,default=3.0)
    parser.add_argument('--tune-caps',type=float,default=1.0)
    parser.add_argument('--tune-floor-db',type=float,default=3.0)
    parser.add_argument('--tune-excess-db',type=float,default=9.0)
    parser.add_argument('--tune-mid',type=float,default=0.8)
    parser.add_argument('--tune-match',type=float,default=0.25)
    parser.add_argument('--tune-ptrans',type=float,default=0.20)
    parser.add_argument('--tune-pcap',type=float,default=1.0)
    parser.add_argument('--tune-combined',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--tune-level-db',type=float,default=-20.0)
    args = parser.parse_args()
    import math as _math
    for _name, _low, _high in (('tune_sib_db', 0, 6), ('tune_cons_db', 0, 6),
                               ('tune_caps', 0.5, 1.5), ('tune_floor_db', 0, 6),
                               ('tune_excess_db', 3, 24), ('tune_mid', 0, 1),
                               ('tune_match', 0, 0.5), ('tune_ptrans', 0, 0.5),
                               ('tune_pcap', 0, 2), ('tune_level_db', -26, -14)):
        _value = getattr(args, _name)
        if not isinstance(_value, float) or not _math.isfinite(_value) \
                or not _low <= _value <= _high:
            raise ValueError(f'{_name} must be {_low}..{_high}')
    if not isinstance(args.tune_combined, bool):
        raise ValueError('tune_combined must be boolean')
    priority={}
    if is_meanvc2(args.model) and sys.platform=='win32':
        import psutil
        try:
            utterance=getattr(args,'delivery','streaming')=='utterance'
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if utterance else psutil.ABOVE_NORMAL_PRIORITY_CLASS)
            priority['compute_priority']='Below Normal' if utterance else 'Above Normal (process-local)'
        except (OSError,psutil.Error) as error:priority['priority_error']=str(error)
    # CPU neural inference is a sustained compute task. MMCSS can demote a
    # task that exhausts its CPU quota; retain timer/HighQoS but do not register
    # this MeanVC2 compute thread as Pro Audio. Callback/bridge remain audio tasks.
    with audio_process_policy() as policy,audio_scheduling(register_mmcss=not is_meanvc2(args.model)) as scheduling:
        scheduling['process_policy']=policy
        scheduling.update(priority)
        run(scheduling,args)


def run(scheduling,args):
    from .models import profile
    output, source = sys.stdout.buffer, sys.stdin.buffer
    selected=profile(args.model)
    folder = asset_root()/'models'/selected['folder']
    backend = None
    collection_policy = None
    diagnostics=AudioDiagnostics()
    diagnostics.attach()
    try:
        send(output, {'status': 'Loading'})
        runtime = json.loads((folder/'runtime.json').read_text(encoding='utf-8'))
        if runtime['backend'] == 'meanvc2':
            if selected.get('phrase_repair'):
                from .meanvc2_phrase import MeanVC2PhraseBackend as MeanVC2Backend
            else:
                from .meanvc2 import MeanVC2Backend
            backend = MeanVC2Backend(threads=args.threads)
        elif runtime['backend'] == 'beatrice_vst':
            from .beatrice_vst import BeatriceVSTBackend
            backend = BeatriceVSTBackend(args.factor)
        elif runtime['backend'] == 'llvc':
            from .llvc_onnx import LLVCOnnxBackend
            backend = LLVCOnnxBackend(args.factor, args.threads)
            folder=folder.parent/'research_llvc'
        else:
            raise ValueError('Unsupported AI backend')
        backend.load(folder)
        if hasattr(backend,'select_profile'):
            backend.select_profile(selected)
            experiment = getattr(args,'experiment','none')
            if experiment == 'all' and runtime.get('backend') == 'meanvc2':
                # Integrated comparison: 3-block grouping with energy-only repair
                # whose corrections ramp up toward the utterance ending.
                backend.select_profile(dict(selected, vc_group_chunks=3, repair_mode='energy'))
                if getattr(backend,'repair',None) is not None:
                    backend.repair.focus = 'ending'
            elif experiment == 'natural' and runtime.get('backend') == 'meanvc2':
                # Natural-leaning comparison: keep the tuned 720ms grouping while
                # bundling combined micro-pitch repair with an ending focus.
                # The tuning GUI can drop this to energy-only repair.
                repair = 'combined' if args.tune_combined else 'energy'
                backend.select_profile(dict(selected, vc_group_chunks=6, repair_mode=repair))
                if getattr(backend,'repair',None) is not None:
                    backend.repair.focus = 'ending'
        send(output, {'status': 'Warming Up'})
        backend.warmup()
        down = StreamingResampler(48000,16000) if backend.sample_rate==16000 else None
        up = StreamingResampler(16000,48000) if backend.sample_rate==16000 else None
        fx = LightPostFX()
        enhancer=None
        if getattr(args,'enhancer','none')=='flashsr':
            from .flashsr import FlashSR
            enhancer=FlashSR()
        elif getattr(args,'enhancer','none')=='mossformer':
            from .mossformer_sr import MossFormerSR
            enhancer=MossFormerSR(threads=args.threads)
        elif getattr(args,'enhancer','none')=='voicefixer':
            from .voicefixer_sr import VoiceFixerSR
            enhancer=VoiceFixerSR(threads=args.threads)
        elif getattr(args,'enhancer','none')=='lavasr':
            from .lavasr import LavaSR
            enhancer=LavaSR(denoise=bool(getattr(args,'lavasr_denoise',False)))
        inference, resample, post, total = TimingStats(), TimingStats(), TimingStats(), TimingStats()
        stats = {}
        next_stats = 0
        import psutil
        collection_policy = RealtimeGC()
        collection_policy.__enter__()
        load_gc_max_ms = diagnostics.gc_maximum_ms
        diagnostics.reset()  # streaming observations exclude model/import/collection
        send(output, {'status': 'Ready', 'model': dict(backend.get_stats(), scheduling=scheduling,
              delivery=getattr(args,'delivery','streaming'),enhancer=getattr(args,'enhancer','none'),
              enhancer_sha256=enhancer.sha256 if enhancer is not None else None,
              load_gc_max_ms=load_gc_max_ms, streaming_cyclic_gc_enabled=False,
              ram_bytes=psutil.Process().memory_info().rss, cpu_seconds=time.process_time())})
        while True:
            header, audio = receive(source)
            if header['op'] == 'stop':
                break
            if header['op'] == 'reset':
                backend.reset()
                if down:
                    down.reset(); up.reset()
                fx.reset()
                send(output, {'status': 'Ready'})
                continue
            if header['op']=='utterance':
                if runtime['backend']!='meanvc2' or getattr(args,'delivery','streaming')!='utterance':
                    raise ValueError('Utterance mode requires MeanVC2')
                if len(audio)>60*48000:
                    raise ValueError('Utterance exceeds 60 seconds')
                from .utterance import convert_utterance, MAX_SECONDS, utterance_limit
                t0=time.monotonic()
                guards={}
                experiment = getattr(args,'experiment','none')
                if experiment in ('all', 'natural'):
                    from .utterance import (clean_input, fry_fraction, level_utterance,
                                            lift_consonants, tame_plosives, tame_sibilance)
                    if experiment == 'natural':
                        guards['input_fry'] = fry_fraction(audio)
                        audio = clean_input(audio)
                        audio = level_utterance(audio, target_db=float(args.tune_level_db))
                        audio = tame_sibilance(audio, max_cut_db=float(args.tune_sib_db))
                        audio = lift_consonants(audio, max_lift_db=float(args.tune_cons_db))
                        audio = tame_plosives(audio)
                    else:
                        audio = level_utterance(audio)
                result=convert_utterance(backend,down,up,audio,
                    max_seconds=utterance_limit(getattr(args,'enhancer','none')),
                    statistics=guards,
                    refresh_pauses=(experiment in ('all', 'natural')))
                if experiment in ('all', 'natural'):
                    # Retrospective shaping: the whole utterance is known, so match
                    # its energy contour and ending against the source, then press
                    # the result's own noise floor back down. Metrics go to guards.
                    # 'natural' keeps 3 dB of room tone where 'all' cuts 6 dB.
                    from .phrase_prosody import (analyze_utterance_pair, retrospective_repair,
                        utterance_metrics)
                    from .utterance import suppress_floor
                    analysis = analyze_utterance_pair(audio, result)
                    guards['prosody_metrics'] = utterance_metrics(audio, result, analysis)
                    if experiment == 'natural':
                        result, shaping = retrospective_repair(audio, result, analysis,
                                                               adaptive=True,
                                                               cap_boost=float(args.tune_caps),
                                                               match_rate=float(args.tune_match))
                        from .phrase_prosody import retrospective_pitch
                        result, pitch_info = retrospective_pitch(audio, result, analysis,
                                                                 transfer=float(args.tune_ptrans),
                                                                 max_shift_st=float(args.tune_pcap))
                        guards['retro_pitch'] = pitch_info
                    else:
                        result, shaping = retrospective_repair(audio, result, analysis)
                    guards['retrospective'] = shaping
                    if experiment == 'natural':
                        from .phrase_prosody import breath_sample_mask
                        shield = breath_sample_mask(analysis, len(result))
                        guards['breath_frames'] = int((shield > 0.5).sum() / 480)
                        result = suppress_floor(result, max_cut_db=float(args.tune_floor_db),
                                                protect=shield)
                        guards['floor_max_cut_db'] = float(args.tune_floor_db)
                    else:
                        result = suppress_floor(result)
                        guards['floor_max_cut_db'] = 6.0
                vc_seconds=time.monotonic()-t0
                post_seconds=0.;enhancer_error=''
                if enhancer is not None:
                    # Only an entirely inaudible result is skipped; a partial skip would
                    # need a splice, and a seam costs more than the RTF saved.
                    from .runtime_audio import skip_enhancement
                    decision=skip_enhancement(result,48000)
                    guards['enhancer_skipped']=bool(decision['skip'])
                    guards['enhancer_skip_reason']=decision['reason']
                    if not decision['skip']:
                        post_start=time.monotonic()
                        try:
                            converted=result
                            result=enhancer.process(result)
                            if experiment in ('all', 'natural'):
                                from .utterance import blend_highs, detect_clicks, fade_edges
                                if experiment == 'natural':
                                    result = blend_highs(converted, result, guard_mode='excess',
                                                         mid_weight=float(args.tune_mid),
                                                         sib_ratio=0.5, sib_mix=0.35,
                                                         excess_db=float(args.tune_excess_db))
                                    guards['blend_guard'] = 'excess'
                                    guards['tune'] = dict(
                                        sib_db=float(args.tune_sib_db),
                                        cons_db=float(args.tune_cons_db),
                                        caps=float(args.tune_caps),
                                        floor_db=float(args.tune_floor_db),
                                        excess_db=float(args.tune_excess_db),
                                        mid=float(args.tune_mid),
                                        match=float(args.tune_match),
                                        ptrans=float(args.tune_ptrans),
                                        pcap=float(args.tune_pcap),
                                        combined=bool(args.tune_combined),
                                        level_db=float(args.tune_level_db))
                                else:
                                    result = blend_highs(converted, result)
                                    guards['blend_guard'] = 'default'
                                guards['clicks'] = detect_clicks(result)
                                if experiment == 'natural':
                                    # Gentle output finish only: 90 Hz high-pass
                                    # plus peak limiting at neutral brightness,
                                    # then 10 ms edge fades. No pitch/EQ change.
                                    # LightPostFX comes from the module import;
                                    # no local re-import (it would shadow the
                                    # global for the whole function).
                                    finish = LightPostFX(rate=48000)
                                    result = finish.process(result, brightness=50,
                                                            low_cut=True, limiter=True,
                                                            enabled=True)
                                    result = fade_edges(result)
                                    guards['natural_finish'] = dict(low_cut=True, limiter=True,
                                                                    edge_ms=10.0)
                        except Exception as error:enhancer_error=str(error)
                        post_seconds=time.monotonic()-post_start
                elapsed=time.monotonic()-t0
                send(output,dict(status='Ready',op='utterance',stats=dict(
                    rtf=elapsed/(len(audio)/48000),ram_bytes=psutil.Process().memory_info().rss,
                    utterance_vc_seconds=vc_seconds,utterance_post_seconds=post_seconds,
                    utterance_total_seconds=elapsed,utterance_input_seconds=len(audio)/48000,
                    utterance_max_seconds=utterance_limit(getattr(args,'enhancer','none')),
                    enhancer=getattr(args,'enhancer','none'),
                    experiment=experiment,
                    enhancer_error=enhancer_error,enhancer_fallback=bool(enhancer_error),
                    guards=guards)),result)
                continue
            expected_frames = backend.chunk_samples*(48000//backend.sample_rate)
            if header['op'] != 'process' or len(audio) != expected_frames:
                raise ValueError('Invalid AI processing request')
            t0 = time.perf_counter_ns()
            reduced = down.process(audio) if down else audio
            t1 = time.perf_counter_ns()
            if hasattr(backend,'set_pitch'):
                backend.set_pitch(header.get('pitch',4.0))
            converted = backend.process_chunk(reduced)
            t2 = time.perf_counter_ns()
            expanded = up.process(converted) if up else converted
            t3 = time.perf_counter_ns()
            result = expanded if runtime['backend']=='meanvc2' else fx.process(expanded, header['brightness'], header['low_cut'], header['limiter'], header['post_fx'],
                                dynamic_gain_db=header.get('dynamic_gain_db',0.))
            t4 = time.perf_counter_ns()
            inference.record(t2-t1, len(audio), 48000)
            resample.record(t1-t0+t3-t2 if down else 0, len(audio), 48000)
            post.record(t4-t3, len(audio), 48000)
            total.record(t4-t0, len(audio), 48000)
            update_stats=time.monotonic() >= next_stats
            if update_stats:
                stats = {key: asdict(value.snapshot()) for key, value in
                         [('inference', inference), ('resample', resample), ('postprocess', post), ('process_total', total)]}
                stats['rtf'] = (inference.total_ns+resample.total_ns+post.total_ns)/max(1,inference.count)/(len(audio)/48000*1e9)
                stats['ram_bytes'] = psutil.Process().memory_info().rss
                stats['cpu_seconds'] = time.process_time()
                stats['worker_gc_count'] = diagnostics.gc_count
                stats['worker_gc_max_ms'] = diagnostics.gc_maximum_ms
                stats['worker_cyclic_gc_enabled'] = False
                if runtime['backend']=='meanvc2':
                    stats['asr_position_rolls']=backend.position_rolls
                    stats['asr_position_base']=backend.position_base
                    if selected.get('phrase_repair'):
                        stats['phrase']=backend.get_stats()['phrase']
                next_stats = time.monotonic()+1
            response={'status':'Ready'}
            if update_stats:
                response['stats']=stats
            send(output, response, result)
    except Exception as error:
        send(output, {'status': 'Error', 'error': str(error)})
        raise
    finally:
        diagnostics.detach()
        try:
            if backend is not None:
                backend.unload()
        finally:
            if collection_policy is not None:
                collection_policy.__exit__(None, None, None)


if __name__ == '__main__':
    main()

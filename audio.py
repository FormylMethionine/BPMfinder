import essentia.standard
import essentia
import numpy as np


def create_analyzers():
    # From chrisdonahue/ddc
    # create the analyzers used by analyze function
    nffts = [1024, 2048, 4096]
    samplerate = 44100
    analyzers = []
    for nfft in nffts:
        win = essentia.standard.Windowing(size=nfft, type='hann')
        spec = essentia.standard.Spectrum(size=nfft)
        mel = essentia.standard.MelBands(inputSize=(nfft//2)+1,
                                         numberBands=80,
                                         lowFrequencyBound=27.5,
                                         highFrequencyBound=16000,
                                         sampleRate=samplerate)
        analyzers.append((nfft, win, spec, mel))
    return analyzers


def analyze(path):
    # returns the spectograms of a song
    samplerate = 44100
    loader = essentia.standard.MonoLoader(filename=path, sampleRate=samplerate)
    audiodata = loader()
    ret = []
    analyzers = create_analyzers()
    for nfft, win, spec, mel in analyzers:
        feats = []
        for frame in essentia.standard.FrameGenerator(audiodata, nfft, 512):
            frame_feats = mel(spec(win(frame)))
            feats.append(frame_feats)
        ret.append(feats)
    ret = np.transpose(np.stack(ret), (1, 2, 0))
    ret = np.log(ret + 1e-16)
    return ret


def analyze_with_time(path):
    # Like analyze but also returns duration of song
    # will one day be integrated into analyze
    samplerate = 44100
    loader = essentia.standard.MonoLoader(filename=path, sampleRate=samplerate)
    audiodata = loader()
    time = len(audiodata)/samplerate
    ret = []
    analyzers = create_analyzers()
    for nfft, win, spec, mel in analyzers:
        feats = []
        for frame in essentia.standard.FrameGenerator(audiodata, nfft, 512):
            frame_feats = mel(spec(win(frame)))
            feats.append(frame_feats)
        ret.append(feats)
    ret = np.transpose(np.stack(ret), (1, 2, 0))
    ret = np.log(ret + 1e-16)
    return ret, time


def time(path):
    # returns the duration of a song
    samplerate = 44100
    loader = essentia.standard.MonoLoader(filename=path, sampleRate=samplerate)
    audiodata = loader()
    time = len(audiodata)/samplerate
    return time

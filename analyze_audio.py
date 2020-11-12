import essentia.standard
import essentia
import numpy as np
import pickle as pkl
import os
import multiprocessing as mp


def create_analyzers():
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
    nffts = [1024, 2048, 4096]
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


def analyze_audio(f):
    print("Converting '" + f + "'")
    path = "./dataset_ddr/audiofiles/" + f
    audiodata = analyze(path)
    with open('dataset_ddr/'+f.split('.')[0]+'.pkl', 'wb') as fi:
        fi.write(pkl.dumps(audiodata))


if __name__ == "__main__":
    pool = mp.Pool(mp.cpu_count() - 1)
    pool.map_async(analyze_audio, os.listdir("./dataset_ddr/audiofiles/"))
    pool.close()
    pool.join()

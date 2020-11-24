import os
import json
import numpy as np
import pickle as pkl
import os.path
from audio import analyze_with_time
from math import ceil


def metadata_sm(path):
    keys = {"#TITLE",
            "#SUBTITLE",
            "#ARTIST",
            "#TITLETRANSLIT",
            "#ARTISTTRANSLIT",
            "#GENRE",
            "#CREDIT",
            "#BANNER",
            "#BACKGROUND",
            "#LYRICSPATH",
            "#CDTITLE",
            "#MUSIC",
            "#OFFSET",
            "#SAMPLESTART",
            "#SAMPLELENGTH",
            "#SELECTABLE",
            "#DISPLAYBPM",
            "#BPMS",
            "#STOPS",
            "#BGCHANGES",
            "#KEYSOUNDS"}
    ret = {}
    with open(path, "r") as f:
        for line in f:
            line = line.split(":")
            if line[0] in keys:
                ret[line[0]] = line[1][:len(line[1])-2].strip()
    return ret


def parse_bpm(bpm, dur):
    bpm = bpm.split(',')
    for i in bpm:
        bpm[bpm.index(i)] = i.split('=')
    ret = []
    j = 0
    for i in range(0, len(bpm)):
        if i < len(bpm)-1:
            while j < int(bpm[i+1][0].split('.')[0]):
                ret.append(int(bpm[i][1].split('.')[0]))
                j += 1
        else:
            while j < (float(bpm[i][1])/60)*dur:
                ret.append(int(bpm[i][1].split('.')[0]))
                j += 1
    return ret


def beats(metadata, bpm, dur):
    beats = []
    time = float(metadata["#OFFSET"]) * 1000
    beat = 0
    for beat, bpm in enumerate(bpm):
        bpmtmp = abs(bpm)
        if time > (dur*1000):
            break
        else:
            beats.append(time)
        time += (60*1000)/bpmtmp
    return beats


def vectorize(beats, audio, time):
    ret = np.zeros(len(audio))
    pos_percent = [i/(time*1000) for i in beats if i >= 0]
    for i in pos_percent:
        ret[int(i*(len(audio)-1)) - ceil(time/(100*len(audio))):
            int(i*(len(audio)-1)) + ceil(time/(100*len(audio))) + 1] = 1
    return ret.tolist()


def parse(f):
    path = "./dataset_ddr/stepcharts/" + f
    metadata = metadata_sm(path)
    metadata["#MUSIC"] = metadata["#MUSIC"].split('.')[0] + ".ogg"
    path_audio = "./dataset_ddr/audiofiles/" + metadata["#MUSIC"]
    if os.path.exists(path_audio):
        audio, time = analyze_with_time(path_audio)
        bpm = parse_bpm(metadata["#BPMS"], time)
        bpm = beats(metadata, bpm, time)
        bpm = vectorize(bpm, audio, time)
        if time < 150 and time > 60:
            with open('dataset_ddr/'+f.split('.')[0]+'.bpm', 'w') as fi:
                fi.write(json.dumps(bpm))
            with open('dataset_ddr/'+f.split('.')[0]+'.metadata', 'w') as fi:
                fi.write(json.dumps(metadata))
            with open('dataset_ddr/'+f.split('.')[0]+'.pkl', 'wb') as fi:
                fi.write(pkl.dumps(audio))


if __name__ == "__main__":
    files = os.listdir("./dataset_ddr/stepcharts")
    for i, f in enumerate(files):
        print(f"Converting '{f}\t{i+1}/{len(files)}'")
        parse(f)

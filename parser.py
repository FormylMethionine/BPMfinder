import os
import json
import numpy as np
import pickle as pkl
import multiprocessing as mp
import os.path
from analyze_audio import analyze


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


def metadata_ssc(path):
    keys = {"#VERSION",
            "#TITLE",
            "#SUBTITLE",
            "#ARTIST",
            "#TITLETRANSLIT",
            "#SUBTITLETRANSLIT",
            "#ARTISTTRANSLIT",
            "#GENRE",
            "#ORIGIN",
            "#CREDIT",
            "#BANNER",
            "#BACKGROUND",
            "#PREVIEWVID",
            "#CDTITLE",
            "#MUSIC",
            "#OFFSET",
            "#SAMPLESTART",
            "#SAMPLELENGTH",
            "#SELECTABLE",
            "#SONGTYPE",
            "#SONGCATEGORY",
            "#VOLUME",
            "#DISPLAYBPM",
            "#BPMS",
            "#TIMESIGNATURES",
            "#TICKCOUNTS",
            "#COMBOS",
            "#SPEEDS",
            "#SCROLLS",
            "#LABELS",
            "#LASTSECONDHINT",
            "#BGCHANGES"}
    ret = {}
    with open(path, "r") as f:
        for line in f:
            line = line.split(":")
            if line[0] == "#NOTEDATA":
                break
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
    time = -1 * float(metadata["#OFFSET"]) * 1000
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
        ret[int(i*(len(audio) - 1))] = 1
    return ret.tolist()


def parse(f):
    print("Converting '" + f + "'")
    path = "./dataset_ddr/stepcharts/" + f
    metadata = metadata_sm(path)
    metadata["#MUSIC"] = metadata["#MUSIC"].split('.')[0] + ".ogg"
    path_audio = "./dataset_ddr/audiofiles/" + metadata["#MUSIC"]
    errors = []
    if os.path.exists(path_audio):
        audio, time = analyze(path_audio)
        bpm = parse_bpm(metadata["#BPMS"], time)
        bpm = beats(metadata, bpm, time)
        bpm = vectorize(bpm, audio, time)
        with open('dataset_ddr/'+f.split('.')[0]+'.bpm', 'w') as fi:
            fi.write(json.dumps(bpm))
        with open('dataset_ddr/'+f.split('.')[0]+'.metadata', 'w') as fi:
            fi.write(json.dumps(metadata))
        with open('dataset_ddr/'+f.split('.')[0]+'.pkl', 'wb') as fi:
            fi.write(pkl.dumps(audio))
    else:
        errors.append(f)
    with open('errors.txt', 'wb') as fi:
        for error in errors:
            fi.write(errors)


if __name__ == "__main__":
    pool = mp.Pool(mp.cpu_count() - 1)
    pool.map_async(parse, os.listdir("./dataset_ddr/stepcharts/"))
    pool.close()
    pool.join()
    #for f in os.listdir("./dataset_ddr/stepcharts"):
        #parse(f)

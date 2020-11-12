import os
import json
import numpy as np
import pickle as pkl
import multiprocessing as mp
from tinytag import TinyTag
from time import perf_counter
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


def maps_sm(path):
    ret = []
    f = open(path, "r")
    line = f.readline()
    chart = []
    while line != "":
        while line != "#NOTES:\n":
            line = f.readline()
        line = f.readline()
        #while line[len(line)-2] == ':':
        while ':' in line:
            line = line.replace(' ', '')
            #line = line.replace(':', '')
            if line != "\n":
                #chart.append(line[:len(line)-1])
                chart.append(line[:line.find(':')])
            line = f.readline()
        while line == "\n":
            line = f.readline()
        while line != "" and line != "\n" and line[0] != ';':
            to_add = []
            while line != "" and \
                    line != "\n" and \
                    line[0] != ',' and \
                    line[0] != ';':
                line = line.replace('M', '0')
                line = line.replace(' ', '')
                if line[0] != "/":
                    to_add.append(line[:len(line)-1])
                line = f.readline()
            chart.append(to_add)
            to_add = []
            line = f.readline()
        ret.append(chart)
        chart = []
        line = f.readline()
    f.close()
    return ret


def filter(metadata, charts):
    ret = {}
    charts.sort(key=lambda x: int(x[3]), reverse=True)
    for chart in charts:
        i = 0
        while True:
            if type(charts[0][i]) == list:
                break
            i += 1
        ret[chart[3]] = chart[i:]
    bpm = parse_bpm(ret, metadata["#BPMS"])
    return ret, bpm


def parse_bpm(charts, bpm):
    bpm = bpm.split(',')
    for i in bpm:
        bpm[bpm.index(i)] = i.split('=')
    ret = {}
    for diff in charts:
        ret[diff] = []
        j = 0
        for i in range(0, len(bpm)):
            if i < len(bpm)-1:
                while j < int(bpm[i+1][0].split('.')[0]):
                    ret[diff].append(float(bpm[i][1]))
                    j += 1
            else:
                while j < len(charts[diff])*4:
                    ret[diff].append(float(bpm[i][1]))
                    j += 1
    return ret


def somme(note):
    ret = 0
    for char in note:
        ret += int(char)
    return ret


def onsets(metadata, charts, bpm, dur):
    ons = {}
    for diff in charts:
        time = -1 * float(metadata["#OFFSET"]) * 1000
        beat = 0
        ons[diff] = []
        for mes in charts[diff]:
            i = 0
            while i < 4:
                bpmtmp = abs(bpm[diff][beat])
                for note in mes[int(0.25*i*len(mes)):int(0.25*(i+1)*len(mes))]:
                    if somme(note) != 0:
                        if time > (dur*1000):
                            ons[diff].append(dur*1000)
                        else:
                            ons[diff].append(time)
                    time += 4/(bpmtmp*len(mes))*60*1000
                beat += 1
                i += 1
    return ons


def vectorize(ons, audio, time):
    # Onsets as a dictionnary of vectors of same size as audio
    ret = {}
    audio_ret = {}
    for diff in ons:
        ret[diff] = np.zeros(len(audio))
        pos_percent = [i/(time*1000) for i in ons[diff] if i >= 0]
        for i in pos_percent:
            ret[diff][int(i*(len(audio) - 1))] = 1
    #prob_ret = np.zeros(len(audio))
    # Onsets as probability vector
    #for frame in range(len(audio)):
        #prob = 0
        #for diff in ret:
            #prob += ret[diff][frame]
        #prob /= len(list(ret.keys()))
        #prob_ret[frame] = prob
    # Cleaning head of zeros to reduce sparcity
    for diff in ret:
        k = 0
        while True:
            if k+8 > len(ret[diff]) or ret[diff][k+8] != 0:
                break
            k += 1
        ret[diff] = ret[diff][k:].tolist()
        audio_ret[diff] = audio[k:, :, :]
    return ret, audio_ret


def parse(f):
    print("Converting '" + f + "'")
    path = "./dataset_ddr/stepcharts/" + f
    metadata = metadata_sm(path)
    music = metadata["#MUSIC"]
    tags = TinyTag.get(f"./dataset_ddr/audiofiles/{music}")
    time = tags.duration
    path_audio = "./dataset_ddr/audiofiles/" + metadata["#MUSIC"]
    chart = maps_sm(path)
    chart, bpm = filter(metadata, chart)
    chart = onsets(metadata, chart, bpm, time)
    audio = analyze(path_audio)
    chart, audio = vectorize(chart, audio, time)
    with open('dataset_ddr/'+f.split('.')[0]+'.chart', 'w') as fi:
        fi.write(json.dumps(chart))
    with open('dataset_ddr/'+f.split('.')[0]+'.metadata', 'w') as fi:
        fi.write(json.dumps(metadata))
    with open('dataset_ddr/'+f.split('.')[0]+'.pkl', 'wb') as fi:
        fi.write(pkl.dumps(audio))


if __name__ == "__main__":
    pool = mp.Pool(mp.cpu_count() - 1)
    pool.map_async(parse, os.listdir("./dataset_ddr/stepcharts/"))
    pool.close()
    pool.join()
    #for f in os.listdir("./dataset_ddr/stepcharts"):
        #parse(f)

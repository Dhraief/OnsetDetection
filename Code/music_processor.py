import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from glob import glob
from scipy import signal
from scipy.fftpack import fft
import librosa as lb 
from librosa.filters import mel
from librosa.display import specshow
from librosa import stft
from librosa.effects import pitch_shift
import pickle
import sys
import math 
from numba import jit, prange
from sklearn.preprocessing import normalize
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
import random
import argparse

class Audio:
    """
    audio class which holds music data and timestamp for notes.

    Args:
        filename: file name.
        stereo: True or False; wether you have Don/Ka streo file or not. normaly True.
    Variables:


    Example:
        >>>from music_processor import *
        >>>song = Audio(filename)
        >>># to get audio data
        >>>song.data
        >>># to import .tja files:
        >>>song.import_tja(filename)
        >>># to get data converted
        >>>song.data = (song.data[:,0]+song.data[:,1])/2
        >>>fft_and_melscale(song, include_zero_cross=False)
    """
    def __init__(self, music, stereo=True):

        self.data = music[0]
        self.samplerate= music[1]
        self.data=self.data.squeeze()
        if self.data.ndim==2 and stereo is False:
            self.data = (self.data[:, 0]+self.data[:, 1])/2
        self.timestamp = []



#allows to plot the audio signal of a song
    def plotaudio(self, start_t, stop_t):

        plt.plot(np.linspace(start_t, stop_t, stop_t-start_t), self.data[start_t:stop_t, 0])
        plt.show()


    def save(self, filename="./savedmusic.wav", start_t=0, stop_t=None):

        if stop_t is None:
            stop_t = self.data.shape[0]
        sf.write(filename, self.data[start_t:stop_t], self.samplerate)
   
#added method
#loads the timestamp 
    def set_timestamp_csv(self,label):
        self.timestamp = []
        for j in range (len(label.iloc[0])) :
            self.timestamp.append([label.iloc[0][j],1])
        self.timestamp=np.array(self.timestamp)
        #duration =lb.get_duration(filename=songplaces[i])
        #maxLabel=max(song.timestamp[:,0])
        #minLabel=min(song.timestamp[:,0])
        #if maxLabel> duration or minLabel<0:
        #    raise Exception('Wrong label', i )"      

#imports the label from the .tja files
    def import_tja(self, filename, verbose=False, diff=False, difficulty=None):
        """imports tja file and convert it into timestamp"""
        
        now = 0.0
        bpm = 100
        measure = [4, 4]  # hyousi
        self.timestamp = []
        skipflag = False

        with open(filename, "rb") as f:
            while True:
                line = f.readline()
                try:
                    line = line.decode('sjis')
                except UnicodeDecodeError:
                    line = line.decode('utf-8')
                if line.find('//') != -1:
                    line = line[:line.find('//')]
                if line[0:5] == "TITLE":
                    if verbose:
                        print("importing: ", line[6:])
                elif line[0:6] == "OFFSET":
                    now = -float(line[7:-2])
                elif line[0:4] == "BPM:":
                    bpm = float(line[4:-2])
                if line[0:6] == "COURSE":
                    if difficulty and difficulty > 0:
                        skipflag = True
                        difficulty -= 1
                elif line == "#START\r\n":
                    if skipflag:
                        skipflag = False
                        continue
                    break
            
            sound = []
            while True:
                line = f.readline()
                try:
                    line = line.decode('sjis')
                except UnicodeDecodeError:
                    line = line.decode('utf-8')

                if line.find('//') != -1:
                    line = line[:line.find('//')]
                if line[0] <= '9' and line[0] >= '0':
                    if line.find(',') != -1:
                        sound += line[0:line.find(',')]
                        beat = len(sound)
                        for i in range(beat):
                            if diff:
                                if int(sound[i]) in (1, 3, 5, 6, 7):
                                    self.timestamp.append([int(100*(now+i*60*measure[0]/bpm/beat))/100, 1])
                                elif int(sound[i]) in (2, 4):
                                    self.timestamp.append([int(100*(now+i*60*measure[0]/bpm/beat))/100, 2])
                            else:
                                if int(sound[i]) != 0:
                                    self.timestamp.append([int(100*(now+i*60*measure[0]/bpm/beat))/100, int(sound[i])])
                        now += 60/bpm*measure[0]
                        sound = []
                    else:
                        sound += line[0:-2]
                elif line[0] == ',':
                    now += 60/bpm*measure[0]
                elif line[0:10] == '#BPMCHANGE':
                    bpm = float(line[11:-2])
                elif line[0:8] == '#MEASURE':
                    measure[0] = int(line[line.find('/')-1])
                    measure[1] = int(line[line.find('/')+1])
                elif line[0:6] == '#DELAY':
                    now += float(line[7:-2])
                elif line[0:4] == "#END":
                    if(verbose):
                        print("import complete!")
                    self.timestamp = np.array(self.timestamp)
                    break

#adds the 'don' and 'ka' according to the timestamps
    def synthesize(self, diff=True, don="./data/don.wav", ka="./data/ka.wav"):
        
        donsound = sf.read(don)[0]
        donsound = (donsound[:, 0] + donsound[:, 1]) / 2
        kasound = sf.read(ka)[0]
        kasound = (kasound[:, 0] + kasound[:, 1]) / 2
        donlen = len(donsound)
        kalen = len(kasound)
        
        if diff is True:
            for stamp in self.timestamp:
                timing = int(stamp[0]*self.samplerate)
                try:
                    if stamp[1] in (1, 3, 5, 6, 7):
                        self.data[timing:timing+donlen] += donsound
                    elif stamp[1] in (2, 4):
                        self.data[timing:timing+kalen] += kasound
                except ValueError:
                    pass

        elif diff == 'don':
            if isinstance(self.timestamp[0], tuple):
                for stamp in self.timestamp:
                    if stamp*self.samplerate+donlen < self.data.shape[0]:
                        self.data[int(stamp[0]*self.samplerate):int(stamp[0]*self.samplerate)+donlen] += donsound
            else:
                for stamp in self.timestamp:
                    if stamp*self.samplerate+donlen < self.data.shape[0]:
                        self.data[int(stamp*self.samplerate):int(stamp*self.samplerate)+donlen] += donsound
        
        elif diff == 'ka':
            if isinstance(self.timestamp[0], tuple):
                for stamp in self.timestamp:
                    if stamp*self.samplerate+kalen < self.data.shape[0]:
                        self.data[int(stamp[0]*self.samplerate):int(stamp[0]*self.samplerate)+kalen] += kasound
            else:
                for stamp in self.timestamp:
                    if stamp*self.samplerate+kalen < self.data.shape[0]:
                        self.data[int(stamp*self.samplerate):int(stamp*self.samplerate)+kalen] += kasound

#helping function from fft_and_melscale
def make_frame(data, nhop, nfft):
    """
    helping function for fftandmelscale.
    細かい時間に切り分けたものを学習データとするため，nhop(512)ずつずらしながらnfftサイズのデータを配列として返す
    """
#    print(data.shape)
    length = data.shape[0]
    framedata = np.concatenate((data, np.zeros(nfft)))  # zero padding
    res=np.array([framedata[i*nhop:i*nhop+nfft] for i in range(length//nhop)])  
    return res

#performs the fourier transform and outputs the melspectograms for a single song
@jit
def fft_and_melscale(song, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False):
    """
    fft and melscale method.
    fft: nfft = [1024, 2048, 4096]; サンプルの切り取る長さを変えながらデータからnp.arrayを抽出して高速フーリエ変換を行う．
    melscale: 周波数の次元を削減するとともに，log10の値を取っている．
    """

    feat_channels = []
    for nfft in nffts:
        
        feats = []
        window = signal.blackmanharris(nfft)
        filt = mel(song.samplerate, nfft, mel_nband, mel_freqlo, mel_freqhi)
        
        # get normal frame
        frame = make_frame(song.data, nhop, nfft)
        # melscaling
        processedframe = fft(window*frame)[:, :nfft//2+1]
        processedframe = np.dot(filt, np.transpose(np.abs(processedframe)**2))
        processedframe = 20*np.log10(processedframe+0.1)
        feat_channels.append(processedframe)
    
    if include_zero_cross:
        song.zero_crossing = np.where(np.diff(np.sign(song.data)))[0]
        print(song.zero_crossing)
    res= np.array(feat_channels)
    return res

#eperfroms the melspectograms for all the songs
@jit(parallel=True)
def multi_fft_and_melscale(songs, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False,):
    
    for i in range(len(songs)):
        songs[i].feats = fft_and_melscale(songs[i], nhop, nffts, mel_nband, mel_freqlo, mel_freqhi)

def milden(data):
    """put smaller value(0.25) to plus minus 1 frame."""
    
    for i in range(data.shape[0]):
        
        if data[i] == 1:
            if i > 0:
                data[i-1] = 0.25
            if i < data.shape[0] - 1:
                data[i+1] = 0.25
        
        if data[i] == 0.26:
            if i > 0:
                data[i-1] = 0.1
            if i < data.shape[0] - 1:
                data[i+1] = 0.1
    
    return data

def smooth(x, window_len=11, window='hanning'):
    
    if x.ndim != 1:
        raise ValueError

    if x.size < window_len:
        raise ValueError

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    
    return y


def music_for_listening(serv, synthesize=True, difficulty=0):

    song = Audio(glob(serv+"/*.ogg")[0])
    if synthesize:
        song.import_tja(glob(serv+"/*.tja")[-1], difficulty=difficulty)
        song.synthesize()
    # plt.plot(song.data[1000:1512, 0])
    # plt.show()
    song.save("./data/saved_music.wav")


def music_for_validation(serv, deletemusic=True, verbose=False, difficulty=1):

    song = Audio(glob(serv+"/*.ogg")[0], stereo=False)
    song.import_tja(glob(serv+"/*.tja")[-1], difficulty=difficulty)
    song.feats = fft_and_melscale(song, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False)

    if deletemusic:
        song.data = None
    with open('./data/pickles/val_data.pickle', mode='wb') as f:
        pickle.dump(song, f)


def music_for_train(serv, deletemusic=True, verbose=False, difficulty=0, diff=False, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False):
    
    songplaces = glob(serv)
    songs = []
    
    for songplace in songplaces:
        
        if verbose:
            print(songplace)
        
        song = Audio(glob(songplace+"/*.ogg")[0])
        song.import_tja(glob(songplace+"/*.tja")[-1], difficulty=difficulty, diff=True)
        song.data = (song.data[:, 0]+song.data[:, 1])/2
        songs.append(song)
    multi_fft_and_melscale(songs, nhop, nffts, mel_nband, mel_freqlo, mel_freqhi, include_zero_cross=include_zero_cross)
    
    if deletemusic:
        for song in songs:
            song.data = None
    
    with open('./data/pickles/train_data.pickle', mode='wb') as f:
        pickle.dump(songs, f)
#checks that each songs match the label
def labelMatchSongs(list_of_songs, list_labels) :
        res=True
        for i in range(len(list_of_songs)):
            name_song=returnNameFile(list_of_songs[i])
            name_label=returnNameFile(list_labels[i])
            if (name_song!=name_label): 
                print("LABELS AND FILES WHOSE NAME DO NOT MATCH")
                print (name_song)
                print(name_label)
                print("********************")
                res= False
        return res
#returns the name of the song from the path 
def returnNameFile(s):
        return (os.path.splitext(os.path.basename(s))[0])
    
#returns the correct extention according to the dataset number
def ext(dataset_number):
    if (dataset_number==0): return  '*.ogg','*.tja'
    else: return  '*.wav','*.txt'
    

#def set_timestamp_txt( ):
def fileToLabel(filepath):
    res=[]
    with open(filepath,encoding='utf-8-sig') as fp:
           line = fp.readline().replace('ï»¿', '')
           while line.split()!=[]:
               #print(line.split())
               sec=float(line.split()[0])
               res.append([sec,1])
               line = fp.readline()
    return np.array(res)
    
def import_txt(list_labels):
    labels = []
    for filepath in list_labels:
        #print("******",filepath,"************")
        labels.append(fileToLabel(filepath))
    return np.array(labels)
#imports the labels for txt files 
def set_timestamp_txt(song,label):
    song.timestamp=np.array(label)
#DS0= tja files
#DS1 CSV
#DS2-DS5 : txt 
def music_for_train_mixed_data(serv,list, percentiles,deletemusic=True, verbose=False, difficulty=0, diff=False, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False):
    songs= []
    cnt=0
    for i in list:
        print("**************** DATASET NUMBER",i," *******************")
        serv_f= serv+"DS"+ str(i)+"/*"
        music_for_train_single_dataset (songs,serv_f,i,percentiles[cnt])
        cnt+=1 
    print("____________________________________________")
    print ("Total Songs", len(songs))
    print("Adding Spectograms")
    multi_fft_and_melscale(songs, nhop, nffts, mel_nband, mel_freqlo, mel_freqhi, include_zero_cross=include_zero_cross)
    print("Added Spectograms")

    with open('./data/pickles/train_reduced.pickle', mode='wb') as f:
        pickle.dump(songs, f)

    print("the end of reduced ")
#pre-prcess music for the transfer learning and returns two files for the two different datasets 
def music_for_train_transfer(serv,list, percentiles,dataset_type,deletemusic=True, verbose=False, difficulty=0, diff=False, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False):
    songs= []
    cnt=0
    for i in list:
        print("**************** DATASET NUMBER",i," *******************")
        serv_f= serv+"DS"+ str(i)+"/*"
        music_for_train_single_dataset (songs,serv_f,i,percentiles[cnt])
        cnt+=1 
    print("____________________________________________")
    print ("Total Songs", len(songs))
    print("Adding Spectograms")
    multi_fft_and_melscale(songs, nhop, nffts, mel_nband, mel_freqlo, mel_freqhi, include_zero_cross=include_zero_cross)
    print("Added Spectograms")

    with open('./data/pickles/'+dataset_type+'.pickle', mode='wb') as f:
        pickle.dump(songs, f)

def music_for_train_single_dataset (songs,serv,dataset_number, percentile, deletemusic=True, verbose=False, difficulty=0, diff=False, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False):  
    print("initially",songs)
    ext_data, ext_label= ext(dataset_number)
    songplaces= glob( os.path.join(serv, '**', ext_data), recursive=True)
    print("songs length",len(songplaces))
    list_labels= glob( os.path.join(serv, '**', ext_label), recursive=True)
    print("labels length",len(list_labels))

    if (dataset_number==1):
        label = [pd.read_csv(label,header=None) for label in list_labels ]
    if (dataset_number>=2):
        label=import_txt(list_labels)
    
    
    print ("label and file match ",labelMatchSongs(songplaces,list_labels))
    print("loading music")
    music = []
    for i in range(len(songplaces)):
        music.append(lb.load(songplaces[i]))
        print(1+i," /",len(songplaces), "loaded")
    #music = [print(song)  lb.load(song) for song in songplaces]
    print("music loaded")   
    print("loading labels")

    for i in range(len(songplaces)):
        song= Audio(music=music[i], stereo = False)
        if (dataset_number==0):
            song.import_tja(list_labels[i] , difficulty=difficulty, diff=True)
        elif (dataset_number==1):
            song.set_timestamp_csv(label[i])
        else :
            set_timestamp_txt(song,label[i])
        songs.append(song)

    print("before length",len(songs))
    print("percentile",percentile )
    songs=random.sample(songs,max(int(percentile*len(songs) /100 ),1) )
    print("after length",len(songs))
    print("label loaded")

    #print("number of songs ",len(songs)," for dataset number ",dataset_number)
    #test=randint(0,len(list_labels)-1)
    #print("test",list_labels[test], " ", len(songs[test].timestamp))


def music_for_train_reduced(serv,dataset_number, deletemusic=True, verbose=False, difficulty=0, diff=False, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False):
    songs=[]
    music_for_train_single_dataset(songs,serv,dataset_number)
    if deletemusic:
        for song in songs:
            song.data = None
    print("TOTAL number of songs ",len(songs))
    with open('./data/pickles/train_reduced.pickle', mode='wb') as f:
        pickle.dump(songs, f)
    
    print("the end of reduced ")

#1 test for now 
def music_for_test(serv,list,percentiles, deletemusic=True, verbose=False, difficulty=0, diff=False, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False):
    songs= []
    cnt=0
    for i in list:
        print("**************** DATASET NUMBER",i," *******************")
        serv_f= serv+"DS"+ str(i)+"/*"
        music_for_train_single_dataset (songs,serv_f,i,percentiles[cnt])
        cnt+=1 
    print("____________________________________________")
    print ("Total Songs", len(songs))
    print("Adding Spectograms")
    multi_fft_and_melscale(songs, nhop, nffts, mel_nband, mel_freqlo, mel_freqhi, include_zero_cross=include_zero_cross)
    print("Added Spectograms")

    with open('./data/pickles/test_data.pickle', mode='wb') as f:
        pickle.dump(songs, f)

def music_for_test_save(serv,list,percentiles, deletemusic=True, verbose=False, difficulty=0, diff=False, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False):
    for i in list:
        songs= []
        cnt=0
        print("**************** DATASET NUMBER",i," *******************")
        serv_f= serv+"DS"+ str(i)+"/*"
        music_for_train_single_dataset (songs,serv_f,i,percentiles[cnt])
        cnt+=1 
        print("____________________________________________")
        print ("Total Songs", len(songs))
        print("Adding Spectograms")
        multi_fft_and_melscale(songs, nhop, nffts, mel_nband, mel_freqlo, mel_freqhi, include_zero_cross=include_zero_cross)
        print("Added Spectograms")

        with open('./data/pickles/test_data'+str(i)+'.pickle', mode='wb') as f:
            pickle.dump(songs, f)

def save_file_train(serv,songs,n_dataset):
    with open('./data/pickles/test_save\train_reduced.pickle', mode='wb') as f:
        pickle.dump(songs, f)

def  music_for_train_save(serv, list_dataset,list_percentile,deletemusic=True, verbose=False, difficulty=0, diff=False, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False):
    for i in range(len(list_dataset)) :
            songs= []
            cnt=0
            print("**************** DATASET NUMBER",list_dataset[i]," *******************")
            serv_f= serv+"DS"+ str(list_dataset[i])+"/*"
            music_for_train_single_dataset (songs,serv_f,list_dataset[i],list_percentile[i])
            print("____________________________________________")
            print ("Total Songs", len(songs))
            print("Adding Spectograms")
            multi_fft_and_melscale(songs, nhop, nffts, mel_nband, mel_freqlo, mel_freqhi, include_zero_cross=include_zero_cross)
            print("Added Spectograms")
            print("nik omek size", len(songs))
            with open('./data/pickles/train_ds'+str(list_dataset[i])+' '+str(list_percentile[i])+'.pickle', mode='wb') as f:
                pickle.dump(songs, f)
            cnt+=1 


if __name__ == "__main__":

    if sys.argv[1] == 'train':
        print("preparing all train data processing...")
        serv = "./data/train/*"
        music_for_train(serv, verbose=True, difficulty=0, diff=True)
        print("all train data processing done!")    

    if sys.argv[1] == 'test':
        print("test data proccesing...")
        serv = "./data/test/*"
        warnings.filterwarnings("ignore")

        list_dataset=[]
        list_percentile=[]
        splitted_arg=sys.argv[2].split(',')
        for i in range(len(splitted_arg)) :
            if (i%2==0):
                list_dataset.append(int(splitted_arg[i]))
            else :
                list_percentile.append(int(splitted_arg[i]))
        print("list dataset: ", list_dataset)
        print("list percentile : ", list_percentile)
        
        music_for_test(serv,list_dataset,list_percentile)
        print("test data processing done!")

    if sys.argv[1] == 'val':
        print("validation data processing...")
        serv = "./data/validation"
        music_for_validation(serv)
        print("done!")

    if sys.argv[1] == 'reduced':
        serv = './data/train_reduced/'
        warnings.filterwarnings("ignore")
        list_dataset=[]
        list_percentile=[]
        splitted_arg=sys.argv[2].split(',')
        for i in range(len(splitted_arg)) :
            if (i%2==0):
                list_dataset.append(int(i))
            else :
                list_percentile.append(int(i))
        music_for_train_mixed_data(serv, list_dataset,list_percentile,verbose=True, difficulty=0, diff=True)
#added

    if sys.argv[1] == 'one_save':
        #go (list datasets train ) (percentile train)  (list datasets test ) (percentile test)  
        #ex: go 1,40,2,30 4,100 : preprocess on 40% of dataset 1 and 30% of dataset 2 for train then for test100 of 4
        serv = './data/train_reduced/'
        print("training data processing")
        list_dataset=[]
        list_percentile=[]
        splitted_arg=sys.argv[2].split(',')
        for i in range(len(splitted_arg)) :
            if (i%2==0):
                list_dataset.append(int(splitted_arg[i]))
            else :
                list_percentile.append(int(splitted_arg[i]))
        print("list dataset: ", list_dataset)
        print("list percentile : ", list_percentile)
        music_for_train_mixed_data(serv, list_dataset,list_percentile,verbose=True, difficulty=0, diff=True)
        print("training data processing done!")

        serv = "./data/test/*"
        print("test data proccesing...") 
        
        list_dataset=[]
        list_percentile=[]
        splitted_arg=sys.argv[3].split(',')
        for i in range(len(splitted_arg)) :
            if (i%2==0):
                list_dataset.append(int(splitted_arg[i]))
            else :
                list_percentile.append(int(splitted_arg[i])) 
        music_for_test(serv,list_dataset,list_percentile)
        print("test data processing done!")
   
    if sys.argv[1] == 'tl':
        # args: [datasets and percentile before freezing] [datasets and percentile to transfer] [test and percentile ]
        # 2,20 3,10 4,100 saves 20% of dataset 2 as dataset_a, 10% of dataset 3 as dataset_b and 100% of dataset 4
        # test
        serv = './data/train_reduced/'
        print("training dataset A  processing")
        list_dataset=[]
        list_percentile=[]
        splitted_arg=sys.argv[2].split(',')
        for i in range(len(splitted_arg)) :
            if (i%2==0):
                list_dataset.append(int(splitted_arg[i]))
            else :
                list_percentile.append(int(splitted_arg[i]))
        print("list dataset A: ", list_dataset)
        print("list percentile : ", list_percentile)
        music_for_train_transfer(serv, list_dataset,list_percentile,'dataset_a',verbose=True, difficulty=0, diff=True)
        print("training data A processing done!")

        list_dataset=[]
        list_percentile=[]
        splitted_arg=sys.argv[3].split(',')
        for i in range(len(splitted_arg)) :
            if (i%2==0):
                list_dataset.append(int(splitted_arg[i]))
            else :
                list_percentile.append(int(splitted_arg[i]))
        print("list dataset B: ", list_dataset)
        print("list percentile : ", list_percentile)
        music_for_train_transfer(serv, list_dataset,list_percentile,'dataset_b',verbose=True, difficulty=0, diff=True)
        print("training data B processing done!")

        serv = "./data/test/*"
        print("test data proccesing...")   
        list_dataset=[]
        list_percentile=[]
        print(sys.argv[4])
        splitted_arg=sys.argv[4].split(',')
        for i in range(len(splitted_arg)) :
            if (i%2==0):
                list_dataset.append(int(splitted_arg[i]))
            else :
                list_percentile.append(int(splitted_arg[i]))

        music_for_test(serv,list_dataset,list_percentile)
        print("test data processing done!")

    if sys.argv[1] == 'size_test':
        #checks that datasets were correclty loaded
        print("here")
        warnings.filterwarnings("ignore")

        for i in [0,2]:
            for j in [10,50,100]:
                with open('./data/pickles/train_ds'+str(i)+' '+str(j)+'.pickle', mode='rb') as f:
                     songs = pickle.load(f)
                print("train length",i," ",j, len(songs))
        for i in [1,3]:
            for j in [100]:
                with open('./data/pickles/train_ds'+str(i)+' '+str(j)+'.pickle', mode='rb') as f:
                     songs = pickle.load(f)
                print("train length",i," ",j, len(songs))
            """
            with open('./data/pickles/test_data'+str(i)+'.pickle', mode='rb') as f:
                songs = pickle.load(f)
            print("test length", len(songs))"""
#saves multiple dataset at once, 
    if sys.argv[1] == 'mul_save':
        #intput mul_save DS,Percentile,DS,Percentile..
        if (sys.argv[2] == 'train'):
            serv = './data/train_reduced/'
        if (sys.argv[2] == 'test'):
            serv = './data/test/*'

        warnings.filterwarnings("ignore")
        list_dataset=[]
        list_percentile=[]
        splitted_arg=sys.argv[3].split(',')
        for i in range(len(splitted_arg)) :
            if (i%2==0):
                list_dataset.append(int(splitted_arg[i]))
            else :
                list_percentile.append(int(splitted_arg[i]))
        print("training dataset A  processing")
        serv = './data/train_reduced/'
        print("ds",list_dataset)
        print("percentile",list_percentile)
        music_for_train_save(serv, list_dataset,list_percentile,verbose=True, difficulty=0, diff=True)

    if sys.argv[1] == 'save':
        print("there")
        warnings.filterwarnings("ignore")
        serv = './data/train_reduced/'
        print("training dataset A  processing")
        list_dataset=[0,3,4,5]
        list_percentile=[100,100,100,100]
        music_for_train_save(serv, list_dataset,list_percentile,verbose=True, difficulty=0, diff=True)
        serv = "./data/test/*"

        music_for_test_save(serv,list_dataset,list_percentile)



        



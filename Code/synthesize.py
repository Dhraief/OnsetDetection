import pickle
import numpy as np
from scipy.signal import argrelmax
from librosa.util import peak_pick
from librosa.onset import onset_detect
from music_processor import *
import mir_eval

debug=False
def from2Dto1D(res_np):
    f=[]
    for i in range(res_np.shape[0]):
        for j in range(len(res_np[i])):
            f.append(res_np[i][j])
    return np.array(f)

def detection_one_song_one_measure(don_inference, ka_inference, song):
    """detects notes disnotesiresultg don and ka"""
    don_reference=[x[0] for x in song.timestamp if x[1]==1 ]
    don_reference_np= np.array(don_reference)

    ka_reference=[x[0] for x in song.timestamp if x[1]==2 ]
    ka_reference_np=np.array(ka_reference)

    don_inference = smooth(don_inference, 5)

    ka_inference = smooth(ka_inference, 5)

    don_timestamp = (peak_pick(don_inference, 1, 2, 4, 5, 0.05, 3)+7)  # 実際は7フレーム目のところの音
    ka_timestamp = (peak_pick(ka_inference, 1, 2, 4, 5, 0.05, 3)+7)
    
    if (don_timestamp.shape==(0,)) :
            print("no don ")
            return 
    
    song.don_timestamp = don_timestamp[np.where(don_inference[don_timestamp] > ka_inference[don_timestamp])]
    song.timestamp = song.don_timestamp*512/song.samplerate
    print("don : min", min(song.timestamp ), " max :",max(song.timestamp ))

    print("shapes ref :",don_reference_np.shape, "timestamp",song.timestamp.shape )    
    print("f measure don", mir_eval.onset.f_measure(don_reference_np,song.timestamp,window=0.5)) 
  
    # print(len(song.timestamp))
    #song.synthesize(diff='don')

    if (ka_timestamp.shape==(0,)) :
                print("no ka")
                return

    # song.ka_timestamp = song.don_timestamp
    song.ka_timestamp = ka_timestamp[np.where(ka_inference[ka_timestamp] > don_inference[ka_timestamp])]
    song.timestamp=song.ka_timestamp*512/song.samplerate
    print("ka : min", min(song.timestamp ), " max :",max(song.timestamp ))


    print("shapes ref :",ka_reference_np.shape, "timestamp",song.timestamp.shape )    
    print("f measure ka", mir_eval.onset.f_measure(ka_reference_np,song.timestamp,window=0.5))

    # print(len(song.timestamp))
    #song.synthesize(diff='ka')

    song.save("./data/inference/inferred_music.wav")

#returns the evaluation metric per song 
def detection_multiple_measure(don_inferences, ka_inferences, songs):
    for i in range(len(songs)):
        print("******************song  ",i)
        detection_one_song_one_measure(don_inferences[i],ka_inferences[i],songs[i])

def detection(don_inferences, ka_inferences, songs, single_output):
    if (single_output==1):
        detection_one_measure(don_inferences, ka_inferences, songs)
    else :
        detection_multiple_measure(don_inferences, ka_inferences, songs)
#returns the evaluation metric as a whole
def detection_one_measure(don_inferences, ka_inferences, songs):
    """detects notes disnotesiresultg don and ka"""
    expected_lables= []
    output_lables=[]
    for i in range (len(songs)):
        song= songs[i]
        #refrence: input form the labels 
        don_reference=[x[0] for x in song.timestamp if x[1]==1 ]

        #inference: output of the model
        don_inference=don_inferences[i]
       # ka_inference= ka_inferences[i]

        don_inference = smooth(don_inference, 5)

        don_timestamp = (peak_pick(don_inference, 1, 2, 4, 5, 0.05, 3)+7)  # 実際は7フレーム目のところの音

        if (debug):
            print("don inference",don_inference.shape)        
            print("ka inference", ka_inference.shape)
            print("post smooth don inference", don_inference.shape)
            print("post smooth ka inference", ka_inference.shape)
            print("post peak pick ka inference", ka_timestamp.shape)
            print("timestamp ",song.timestamp.shape,"min",min(song.timestamp), "max", max(song.timestamp))
            print("don reference",don_reference_np.shape,"min",min(don_reference_np), "max", max(don_reference_np))


        if (don_timestamp.shape==(0,)) :
            print("no don ")
            continue
  
        #song.don_timestamp = don_timestamp[np.where(don_inference[don_timestamp] > ka_inference[don_timestamp])]
        song.don_timestamp=don_timestamp
        song.timestamp = song.don_timestamp*512/song.samplerate
        #print("expected shape", song.timestamp.shape)
        expected_lables.append(song.timestamp)
        #print("output_lables shape", len(don_reference))

        output_lables.append(don_reference)
        """
        song.ka_timestamp = ka_timestamp[np.where(ka_inference[ka_timestamp] > don_inference[ka_timestamp])]
        song.timestamp=song.ka_timestamp*512/song.samplerate
        print("timestamp ",song.timestamp.shape)
        print("ka reference ",ka_reference_np.shape)
        
        expected_lables.append(song.timestamp)
        output_lables.append(ka_reference)   
        """
        #song.timestamp = don_timestamp*512/song.samplerate
        
        #print("inference",song.timestamp)

    output_lables_np=np.array(output_lables,dtype=object)
    output_lables_np=from2Dto1D(output_lables_np)
#    print("output shape after ",output_lables_np.shape)


    expected_lables_np=np.array(expected_lables)
 #   print("output shape before",expected_lables_np.shape)
    expected_lables_np=from2Dto1D(expected_lables_np)
  #  print("output shape after ",expected_lables_np.shape)

    expected_lables_np.sort()
    output_lables_np.sort()
    print("f measure don", mir_eval.onset.f_measure(expected_lables_np,output_lables_np,window=0.07))
    
        # print(len(song.timestamp))
       
""" 
        song.synthesize(diff='don')
        song.save("./data/inference/inferred_music.wav")


        # song.ka_timestamp = song.don_timestamp
        print(ka_timestamp.shape)
        print("f measure ka", mir_eval.onset.f_measure(ka_reference_np,song.timestamp))

        # print(len(song.timestamp))
        song.synthesize(diff='ka')"""


def create_tja(filename, song, don_timestamp, ka_timestamp=None):

    if ka_timestamp is None:
        timestamp=don_timestamp*512/song.samplerate
        with open(filename, "w") as f:
            f.write('TITLE: xxx\nSUBTITLE: --\nBPM: 240\nWAVE:xxx.ogg\nOFFSET:0\n#START\n')
            i = 0
            time = 0
            while(i < len(timestamp)):
                if time/100 >= timestamp[i]:
                    f.write('1')
                    i += 1
                else:
                    f.write('0')
                if time % 100 == 99:
                    f.write(',\n')
                time += 1
            f.write('#END')

    else:
        don_timestamp=np.rint(don_timestamp*512/song.samplerate*100).astype(np.int32)
        ka_timestamp=np.rint(ka_timestamp*512/song.samplerate*100).astype(np.int32)
        with open(filename, "w") as f:
            f.write('TITLE: xxx\nSUBTITLE: --\nBPM: 240\nWAVE:xxx.ogg\nOFFSET:0\n#START\n')
            for time in range(np.max((don_timestamp[-1],ka_timestamp[-1]))):
                if np.isin(time,don_timestamp) == True:
                    f.write('1')
                elif np.isin(time,ka_timestamp) == True:
                    f.write('2')
                else:
                    f.write('0')
                if time%100==99:
                    f.write(',\n')
            f.write('#END')


if __name__ == "__main__":
#synthesize 1(one measure)/0 (not) (1 freeze)
    for i in [(3,0)]:

        for j in [1,2,3,4,5]:
            with open('./data/pickles/test_data'+str(i[0])+'.pickle', mode='rb') as f:
                songs = pickle.load(f)

            print("*************** NIK OM EL PROJET ",i[0],"  ", j)
            with open('./data/pickles/don_inference '+str(i[0])+' '+str(i[1])+' '+str(j)+'.pickle', mode='rb') as f:
                don_inference = pickle.load(f)

            with open('./data/pickles/ka_inference.pickle', mode='rb') as f:
                ka_inference = pickle.load(f)

            detection(don_inference, ka_inference, songs, single_output= int(sys.argv[1]))
            #create_tja("./data/inference/inferred_notes.tja",song, song.don_timestamp, song.ka_timestamp)


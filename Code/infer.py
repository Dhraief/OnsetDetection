from model import *
import pickle

if __name__ == '__main__':
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = convNet()
    net = net.to(device)
    """
    with open('./data/pickles/test_data.pickle', mode='rb') as f:
        songs = pickle.load(f)
"""
    if sys.argv[1] == 'experiement':
        with open('./data/pickles/test_data'+str(0)+'.pickle', mode='rb') as f:
                    songs = pickle.load(f)
        for j in [1,2,3,4,5]:
            print("***********************  TEST ",j)
            if torch.cuda.is_available():
                net.load_state_dict(torch.load('./models/M'+str(j)+'.pth'))
            else:
                net.load_state_dict(torch.load('./models/M'+str(j)+'.pth', map_location='cpu'))

            inference = []
            for song in songs :
                inference_song = net.infer(song.feats, device, minibatch=4192)
                inference_song = np.reshape(inference_song, (-1))
                inference.append(inference_song)
            inference_np= np.array(inference, dtype=object)
            print("don", inference_np.shape)
            print("ex", inference_np[0])

            with open('./data/pickles/don_inference 3 0 '+str(j)+'.pickle', mode='wb') as f:
                pickle.dump(inference_np, f)
    
    if sys.argv[1] == 'ka':
        
        if torch.cuda.is_available():
            net.load_state_dict(torch.load('./models/ka_model.pth'))
        else:
            net.load_state_dict(torch.load('./models/ka_model.pth', map_location='cpu'))

        inference = []
        for song in songs :
            inference_song = net.infer(song.feats, device, minibatch=4192)
            inference_song = np.reshape(inference_song, (-1))
            inference.append(inference_song)
        inference_np= np.array(inference,dtype=object)
        print("ka", inference_np.shape)

        with open('./data/pickles/ka_inference.pickle', mode='wb') as f:
            pickle.dump(inference_np, f)

    if sys.argv[1] == 'don':
        
        if torch.cuda.is_available():
            net.load_state_dict(torch.load('./models/don_model.pth'))
        else:
            net.load_state_dict(torch.load('./models/don_model.pth', map_location='cpu'))

        inference = []
        for song in songs :
            inference_song = net.infer(song.feats, device, minibatch=4192)
            inference_song = np.reshape(inference_song, (-1))
            inference.append(inference_song)
        inference_np= np.array(inference, dtype=object)
        print("don", inference_np.shape)
        print("ex", inference_np[0])

        with open('./data/pickles/don_inference.pickle', mode='wb') as f:
            pickle.dump(inference_np, f)
            
    if sys.argv[1] == 'all':
        for i in [0,3,4,5]:
            with open('./data/pickles/test_data'+str(i)+'.pickle', mode='rb') as f:
                      songs = pickle.load(f)
            for j in [0,3,4,5]:
                if (i==j): continue
                if torch.cuda.is_available():
                    net.load_state_dict(torch.load('./models/don_model'+str(j)+str(i)+'.pth'))
                else:
                    net.load_state_dict(torch.load('./models/don_model'+str(j)+str(i)+'.pth', map_location='cpu'))

                inference = []
                for song in songs :
                    inference_song = net.infer(song.feats, device, minibatch=4192)
                    inference_song = np.reshape(inference_song, (-1))
                    inference.append(inference_song)
                inference_np= np.array(inference, dtype=object)
                print("don", inference_np.shape)
                print("ex", inference_np[0])

                with open('./data/pickles/don_inference'+str(j)+str(i)+'.pickle', mode='wb') as f:
                    pickle.dump(inference_np, f)
    if sys.argv[1] == 'all_no':
        for i in [0,3,4,5]:
            with open('./data/pickles/test_data'+str(i)+'.pickle', mode='rb') as f:
                      songs = pickle.load(f)
            for j in [0,3,4,5]:
                if (i==j): continue
                if torch.cuda.is_available():
                    net.load_state_dict(torch.load('./models/no_freeze_don_model'+str(j)+str(i)+'.pth'))
                else:
                    net.load_state_dict(torch.load('./models/no_freeze_don_model'+str(j)+str(i)+'.pth', map_location='cpu'))

                inference = []
                for song in songs :
                    inference_song = net.infer(song.feats, device, minibatch=4192)
                    inference_song = np.reshape(inference_song, (-1))
                    inference.append(inference_song)
                inference_np= np.array(inference, dtype=object)
                print("don", inference_np.shape)
                print("ex", inference_np[0])

                with open('./data/pickles/no_freeze_don_inference'+str(j)+str(i)+'.pickle', mode='wb') as f:
                    pickle.dump(inference_np, f)
    if sys.argv[1] == 'mix_gen':
        for i in [0,3,4,5]:
            with open('./data/pickles/test_data'+str(i)+'.pickle', mode='rb') as f:
                      songs = pickle.load(f)
            for j in [0,3,4,5]:
                if (i==j): continue
                if torch.cuda.is_available():
                    net.load_state_dict(torch.load('./models/don_model'+str(j)+str(i)+'.pth'))
                else:
                    net.load_state_dict(torch.load('./models/don_model'+str(j)+str(i)+'.pth', map_location='cpu'))

                inference = []
                for song in songs :
                    inference_song = net.infer(song.feats, device, minibatch=4192)
                    inference_song = np.reshape(inference_song, (-1))
                    inference.append(inference_song)
                inference_np= np.array(inference, dtype=object)
                print("don", inference_np.shape)
                print("ex", inference_np[0])

                with open('./data/pickles/don_inference'+str(j)+str(i)+'.pickle', mode='wb') as f:
                    pickle.dump(inference_np, f)
        for i in [0,3,4,5]:
            with open('./data/pickles/test_data'+str(i)+'.pickle', mode='rb') as f:
                      songs = pickle.load(f)
            for j in [0,3,4,5]:
                if (i==j): continue
                if torch.cuda.is_available():
                    net.load_state_dict(torch.load('./models/no_freeze_don_model'+str(j)+str(i)+'.pth'))
                else:
                    net.load_state_dict(torch.load('./models/no_freeze_don_model'+str(j)+str(i)+'.pth', map_location='cpu'))

                inference = []
                for song in songs :
                    inference_song = net.infer(song.feats, device, minibatch=4192)
                    inference_song = np.reshape(inference_song, (-1))
                    inference.append(inference_song)
                inference_np= np.array(inference, dtype=object)
                print("don", inference_np.shape)
                print("ex", inference_np[0])

                with open('./data/pickles/no_freeze_don_inference'+str(j)+str(i)+'.pickle', mode='wb') as f:
                    pickle.dump(inference_np, f)
            
 
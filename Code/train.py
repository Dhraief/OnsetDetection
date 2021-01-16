from model import *
from music_processor import *


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = convNet()
    net = net.to(device)

    """
    try:
        with open('./data/pickles/train_data.pickle', mode='rb') as f:
            songs = pickle.load(f)
    except FileNotFoundError:
        with open('./data/pickles/train_reduced.pickle', mode='rb') as f:
            songs = pickle.load(f)
            print("song main len", len(songs))

    """
    minibatch = 128
    soundlen = 15
    epoch = 30

    if sys.argv[1] == 'don':
        net.train(songs=songs, minibatch=minibatch, val_song=None, epoch=int(sys.argv[2]), device=device, soundlen=soundlen, save_place='./models/don_model.pth', log='./data/log/don.txt', don_ka=1)
        print("the end")
    if sys.argv[1] == 'ka':
        net.train(songs=songs, minibatch=minibatch, val_song=None, epoch=epoch, device=device, soundlen=soundlen, save_place='./models/ka_model.pth', log='./data/log/ka.txt', don_ka=2)
    if sys.argv[1]=='freeze':
        list_dataset=[0,3,4,5]
        for i in list_dataset:
            print("************************************Training the model ",i )            
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            net = convNet()
            net = net.to(device)
            
            with open('./data/pickles/train_ds'+str(i)+'.pickle', mode='rb') as f:
                songs = pickle.load(f)
                #print("song open len", len(songs))

            net.train(songs=songs, minibatch=minibatch, val_song=None, epoch=int(sys.argv[2]),device=device, 
            soundlen=soundlen, save_place='./models/don_model'+str(i)+'.pth',log='./data/log/don'+str(i)+'.txt', 
            don_ka=1)
            
            for j in list_dataset: 
                if torch.cuda.is_available():
                    net.load_state_dict(torch.load('./models/don_model'+str(i)+'.pth'))
                else:
                    net.load_state_dict(torch.load('./models/don_model'+str(i)+'.pth', map_location='cpu'))
                if(i==j): continue 
                print("************************************STARTED IN ",i,"  ",j)
                with open('./data/pickles/train_ds'+str(j)+'.pickle', mode='rb') as f:
                    songs = pickle.load(f)
                #print("len songs", len(songs))
                
                cnt=0

                 
                for parma in net.parameters():
                    parma.requires_grad = False
                    cnt+=1
                    if (cnt==2) : break
                
                for parma in net.parameters():
                    print(parma.requires_grad)

                net.train(songs=songs, minibatch=minibatch, 
                val_song=None, epoch=int(sys.argv[2]), device=device, soundlen=soundlen, 
                save_place='./models/don_model'+str(i)+str(j)+'.pth', log='./data/log/don'+str(i)+str(j)+'.txt', don_ka=1)

    if sys.argv[1]=='experiement':
        for i in [(3,0),(1,2)]:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            net = convNet()
            net = net.to(device)
            with open('./data/pickles/train_ds'+str(i[0])+'.pickle', mode='rb') as f:
                    songs = pickle.load(f)
                    #print("song open len", len(songs))       

        
            net.train(songs=songs, minibatch=minibatch, 
            val_song=None, epoch=int(sys.argv[2]), device=device, soundlen=soundlen, 
            save_place='./models/don_model'+str(i[0])+'.pth', log='./data/log/don'+str(i[0])+str(i[1])+ 'percentile' +str(p)+'.txt', don_ka=1)

            cnt=0
            for parma in net.parameters():
                parma.requires_grad = False
                cnt+=1
                if (cnt==2) : break
            
            for parma in net.parameters():
                print(parma.requires_grad)
                
            for p in [10,50]:

                
                with open('./data/pickles/train_ds'+str(i[1])+' '+str(p)+'.pickle', mode='rb') as f:
                    songs = pickle.load(f)
                    #print("song open len", len(songs))       


                net.train(songs=songs, minibatch=minibatch, 
                val_song=None, epoch=int(sys.argv[2]), device=device, soundlen=soundlen, 
                save_place='./models/don_model'+str(i[0])+str(i[1])+ 'percentile' +str(p)+'.pth', log='./data/log/don'+str(i[0])+str(i[1])+ 'percentile' +str(p)+'.txt', don_ka=1)


    if sys.argv[1]=='all_no':
        list_dataset=[0,3,4,5]
        for i in list_dataset:
            print("************************************Training the model ",i )            
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            net = convNet()
            net = net.to(device)
            
            with open('./data/pickles/train_ds'+str(i)+'.pickle', mode='rb') as f:
                songs = pickle.load(f)
                #print("song open len", len(songs))

            net.train(songs=songs, minibatch=minibatch, val_song=None, epoch=int(sys.argv[2]),device=device, 
            soundlen=soundlen, save_place='./models/no_freeze_don_model'+str(i)+'.pth',log='./data/log/no_freeze_don'+str(i)+'.txt', 
            don_ka=1)
            
            for j in list_dataset: 
                if torch.cuda.is_available():
                    net.load_state_dict(torch.load('./models/no_freeze_don_model'+str(i)+'.pth'))
                else:
                    net.load_state_dict(torch.load('./models/no_freeze_don_model'+str(i)+'.pth', map_location='cpu'))
                if(i==j): continue 
                print("************************************STARTED IN ",i,"  ",j)
                with open('./data/pickles/train_ds'+str(j)+'.pickle', mode='rb') as f:
                    songs = pickle.load(f)
                #print("len songs", len(songs))
                
                cnt=0

                net.train(songs=songs, minibatch=minibatch, 
                val_song=None, epoch=int(sys.argv[2]), device=device, soundlen=soundlen, 
                save_place='./models/no_freeze_don_model'+str(i)+str(j)+'.pth', log='./data/log/no_freeze_don'+str(i)+str(j)+'.txt', don_ka=1)

    if sys.argv[1]=='mix_gen':
        list_dataset=[0,3,4,5]
        for i in list_dataset:
            print("************************************Training the model ",i )            
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            net = convNet()
            net = net.to(device)
            
            with open('./data/pickles/train_ds'+str(i)+'.pickle', mode='rb') as f:
                songs = pickle.load(f)
                #print("song open len", len(songs))

            net.train(songs=songs, minibatch=minibatch, val_song=None, epoch=int(sys.argv[2]),device=device, 
            soundlen=soundlen, save_place='./models/no_freeze_don_model'+str(i)+'.pth',log='./data/log/no_freeze_don'+str(i)+'.txt', 
            don_ka=1)
            
            for j in list_dataset: 
                if torch.cuda.is_available():
                    net.load_state_dict(torch.load('./models/no_freeze_don_model'+str(i)+'.pth'))
                else:
                    net.load_state_dict(torch.load('./models/no_freeze_don_model'+str(i)+'.pth', map_location='cpu'))
                if(i==j): continue 
                print("************************************STARTED IN ",i,"  ",j)
                with open('./data/pickles/train_ds'+str(j)+'.pickle', mode='rb') as f:
                    songs = pickle.load(f)
                #print("len songs", len(songs))
                
                cnt=0

                net.train(songs=songs, minibatch=minibatch, 
                val_song=None, epoch=int(sys.argv[2]), device=device, soundlen=soundlen, 
                save_place='./models/no_freeze_don_model'+str(i)+str(j)+'.pth', log='./data/log/no_freeze_don'+str(i)+str(j)+'.txt', don_ka=1)
        list_dataset=[0,3,4,5]
        for i in list_dataset:
            print("************************************Training the model ",i )            
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            net = convNet()
            net = net.to(device)
            
            with open('./data/pickles/train_ds'+str(i)+'.pickle', mode='rb') as f:
                songs = pickle.load(f)
                #print("song open len", len(songs))

            net.train(songs=songs, minibatch=minibatch, val_song=None, epoch=int(sys.argv[2]),device=device, 
            soundlen=soundlen, save_place='./models/don_model'+str(i)+'.pth',log='./data/log/don'+str(i)+'.txt', 
            don_ka=1)
            
            for j in list_dataset: 
                if torch.cuda.is_available():
                    net.load_state_dict(torch.load('./models/don_model'+str(i)+'.pth'))
                else:
                    net.load_state_dict(torch.load('./models/don_model'+str(i)+'.pth', map_location='cpu'))
                if(i==j): continue 
                print("************************************STARTED IN ",i,"  ",j)
                with open('./data/pickles/train_ds'+str(j)+'.pickle', mode='rb') as f:
                    songs = pickle.load(f)
                #print("len songs", len(songs))
                
                cnt=0

                 
                for parma in net.parameters():
                    parma.requires_grad = False
                    cnt+=1
                    if (cnt==2) : break
                
                for parma in net.parameters():
                    print(parma.requires_grad)

                net.train(songs=songs, minibatch=minibatch, 
                val_song=None, epoch=int(sys.argv[2]), device=device, soundlen=soundlen, 
                save_place='./models/don_model'+str(i)+str(j)+'.pth', log='./data/log/don'+str(i)+str(j)+'.txt', don_ka=1)

  
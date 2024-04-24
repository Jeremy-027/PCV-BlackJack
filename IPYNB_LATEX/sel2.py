def LoadCitraTraining(sDir,LabelKelas):  
  JumlahKelas=len(LabelKelas)
  TargetKelas = np.eye(JumlahKelas)
  X=[]
  T=[]
  for i in range(len(LabelKelas)):    
    DirKelas = os.path.join(sDir, LabelKelas[i])
    files = os.listdir(DirKelas)
    for f in files:
      ff=f.lower()  
      print(f)
      if (ff.endswith('.jpg')|ff.endswith('.jpeg')|ff.endswith('.png')):
         NmFile = os.path.join(DirKelas,f) 
         img= np.double(cv2.imread(NmFile,1))
         img=cv2.resize(img,(128,128));
         img= np.asarray(img)/255;
         img=img.astype('float32')
         X.append(img)
         T.append(TargetKelas[i])
  X=np.array(X)
  T=np.array(T)
  X=X.astype('float32')
  T=T.astype('float32')
  return X,T

def TrainingCNN(JumlahEpoh,DirektoriDataSet,LabelKelas,NamaFileBobot):
    X,D=LoadCitraTraining(DirektoriDataSet,LabelKelas)
    JumlahKelas = len(LabelKelas)
    ModelCNN =ModelDeepLearningCNN(JumlahKelas)
    history=ModelCNN.fit(X, D,epochs=JumlahEpoh,shuffle=True)
    ModelCNN.save(NamaFileBobot)
    return ModelCNN,history

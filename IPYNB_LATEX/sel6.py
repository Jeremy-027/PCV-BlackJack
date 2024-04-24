def ImageAugmentation(SPath,Kelas):
  parent_dir = SPath 
  directory = Kelas
  directoryExt =directory +"_ext"
  sdirExt = os.path.join(parent_dir, directoryExt)
  if not os.path.exists(sdirExt):
    os.mkdir(sdirExt)
  directory_image =directory
  sDir = os.path.join(parent_dir, directory_image)
  files = os.listdir(sDir)
  ii=0
  for f in files:
    ff=f.lower()
    if (ff.endswith('.jpg')|ff.endswith('.jpeg')|ff.endswith('.png')):
      print(ff)
      sfs= os.path.join(sDir,ff)
      img = load_img(sfs)
      img2 = np.array(img)
      sfn= os.path.join(sdirExt, ff)
      cv2.imwrite(sfn,img2)
      data = img_to_array(img)
      samples = expand_dims(data, 0)
      datagen = ImageDataGenerator(rotation_range=90,brightness_range=[0.2,2.0],zoom_range=[0.5,2.0],width_shift_range=0.2,height_shift_range=0.2)
      it = datagen.flow(samples, batch_size=1)
      for i in range(9):
        batch = it.next()
        image = batch[0].astype('uint8')
        now = datetime.now() 
        ii=ii+1
        sf = now.strftime("%Y%m%d%H%M%S")+"_"+str(ii)+".jpg"
        sfn= os.path.join(sdirExt, sf)
        cv2.imwrite(sfn,image)
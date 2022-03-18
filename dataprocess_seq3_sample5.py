import os 
import nibabel as nib 
import numpy as np
data_path ="./data/3D/"
save_path = "./data/2D_seq3_s5/"


def truncated_range(img):
    max_hu = 384
    min_hu = -384
    img[np.where(img > max_hu)] = max_hu
    img[np.where(img < min_hu)] = min_hu
    # img= (img - min_hu) / (max_hu - min_hu) * 255.

    return ((img - min_hu) / (max_hu - min_hu) * 255.).astype(float)


def pixelArrayToImage(pixel_array,savepath,imgname,idx):
    image_array = pixel_array
    idx = str(idx+1)
    idx =idx.zfill(4)
    imgname = imgname.split("mpr")[0]
    # image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min()))*255
    path_npy = savepath + '/'+ imgname + idx + '.npy'
    np.save(path_npy,image_array)


def labelArrayToImage(label_array,savepath,imgname,idx):

    image_array = label_array
    idx = str(idx+1)
    idx =idx.zfill(4)
    imgname = imgname.split("mpr")[0]
    path_npy = savepath + '/'+ imgname + idx + '.npy'
    np.save(path_npy,image_array)

def transdata(data_path,save_path):
    for datadir in sorted(os.listdir(data_path)):
        if datadir =="image":

            for imgname in sorted(os.listdir(os.path.join(data_path,datadir))):
                print("the processing data is data(%s)" %(imgname))
                save_slice_path = os.path.join(save_path,datadir)
                if not os.path.exists(save_slice_path):
                    os.makedirs(save_slice_path)
                image_data = nib.load(os.path.join(data_path,datadir,imgname))
                image_data = image_data.get_data()
                image_data = truncated_range(image_data)
                for i in range(0,image_data.shape[2],5):
                    if i == 0:
                        image_slice = image_data[:,:,[0,i,i+5]]
                    elif (i+5)>=image_data.shape[2]:
                        image_slice = image_data[:,:,[i-5,i,i]]
                    else:
                        image_slice = image_data[:,:,[i-5,i,i+5]]
                    pixelArrayToImage(image_slice,save_slice_path,imgname,i)

        elif datadir == "label":
            for imgname in sorted(os.listdir(os.path.join(data_path,datadir))):
                print("the processing data is data(%s)" %(imgname))
                save_slice_path = os.path.join(save_path,datadir)
                if not os.path.exists(save_slice_path):
                    os.makedirs(save_slice_path)
                image_data = nib.load(os.path.join(data_path,datadir,imgname))
                image_data = image_data.get_data()
                for i in range(0,image_data.shape[2],5):
                    if i == 0:
                        image_slice = image_data[:,:,[0,i,i+5]]
                    elif (i+5)>=image_data.shape[2]:
                        image_slice = image_data[:,:,[i-5,i,i]]
                    else:
                        image_slice = image_data[:,:,[i-5,i,i+5]]
                    labelArrayToImage(image_slice,save_slice_path,imgname,i)

if __name__== '__main__':
    rootpath ="./data/3D/"
    savepath = "./data/2D_seq3_s5/"

    for datadir in os.listdir(rootpath):
        data_path = rootpath+'/'+ datadir
        save_path = savepath + '/'+ datadir
        transdata(data_path,save_path)

import os 
import nibabel as nib 
import numpy as np
from skimage.morphology import dilation
from skimage.morphology import cube
from skimage import measure
import scipy.ndimage as nd





def pixelArrayToImage(pixel_array,savepath,imgname,idx):
    image_array = pixel_array
    idx = str(idx+1)
    idx =idx.zfill(4)
    imgname = imgname.split("mpr")[0]
    # if not (image_array.max() - image_array.min()) ==0:
    #     image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min()))*255
    # else:
    #     image_array=image_array
    path_npy = savepath + '/'+ imgname + idx + '.npy'
    np.save(path_npy,image_array)


def labelArrayToImage(label_array,savepath,imgname,idx):

    image_array = label_array
    idx = str(idx+1)
    idx =idx.zfill(4)
    imgname = imgname.split("mpr")[0]
    path_npy = savepath + '/'+ imgname + idx + '.npy'
    np.save(path_npy,image_array)

def truncated_range(img):
    max_hu = 750
    min_hu = -384
    img[np.where(img > max_hu)] = max_hu
    img[np.where(img < min_hu)] = min_hu
    img = ((img - min_hu) / (max_hu - min_hu) * 255.).astype(float)
    # img= (img - min_hu) / (max_hu - min_hu) * 255.

    return img


def post_processing(prediction):
    prediction = nd.binary_fill_holes(prediction)
    label_cc, num_cc = measure.label(prediction,return_num=True)
    total_cc = np.sum(prediction)
    measure.regionprops(label_cc)
    for cc in range(1,num_cc+1):
        single_cc = (label_cc==cc)
        single_vol = np.sum(single_cc)
        if single_vol/total_cc<0.4:
            prediction[single_cc]=0

    return prediction.astype(float)


def transdata(data_path,save_path):
    for datadir in sorted(os.listdir(data_path)):
        if datadir =="image":

            for imgname in sorted(os.listdir(os.path.join(data_path,datadir))):
                print("the processing data is data(%s)" %(imgname))
                # save_slice_path = os.path.join(save_path,imgname.split("_")[0]+'_'+imgname.split("_")[1],datadir)
                save_slice_path = os.path.join(save_path,datadir)
                if not os.path.exists(save_slice_path):
                    os.makedirs(save_slice_path)
                image_data = nib.load(os.path.join(data_path,datadir,imgname))
                image_data = image_data.get_data()
                image_data = truncated_range(image_data)
                label_path = os.path.join(data_path,'trans',imgname.split("_")[0]+'_cmpr_'+imgname.split("_")[1]+'_trans.nii.gz')
                if os.path.exists(label_path):
                    label_data = nib.load(label_path)
                    # label_data = nib.load(os.path.join(data_path,'label',imgname.split("_")[0]+'_segmentation_'+imgname.split("_")[1]+'_mpr.nii.gz'))
                    label_data = label_data.get_data()               
                    label_data = np.where(label_data >0.05,1,0)
                    label_data = dilation(label_data,cube(8)).astype(float)
                    label_data = post_processing(label_data)
                    image_data =image_data*label_data

                    for i in range(0,image_data.shape[2],2):
                        if i == 0:
                            image_slice = image_data[:,:,[0,0,i,i+2,i+4]]
                        if i == 2:
                            image_slice = image_data[:,:,[0,i-2,i,i+2,i+4]]
                        elif (i+2)>=image_data.shape[2]:
                            image_slice = image_data[:,:,[i-4,i-2,i,i,i]]
                        elif (i+4)>=image_data.shape[2]:
                            image_slice = image_data[:,:,[i-4,i-2,i,i+2,i+2]]
                        else:
                            image_slice = image_data[:,:,[i-4,i-2,i,i+2,i+4]]
                        pixelArrayToImage(image_slice,save_slice_path,imgname,i)
                else:
                    label_data = nib.load(os.path.join(data_path,'label',imgname.split("_")[0]+'_segmentation_'+imgname.split("_")[1]+'_mpr.nii.gz'))
                    label_data = label_data.get_data()               
                    label_data = np.where(label_data >0.5,1,0)
                    label_data = dilation(label_data,cube(10)).astype(float)
                    label_data = post_processing(label_data)
                    
                    image_data =image_data*label_data
                    for i in range(0,image_data.shape[2],4):
                        if i == 0:
                            image_slice = image_data[:,:,[0,0,i,i+4,i+8]]
                        if i == 4:
                            image_slice = image_data[:,:,[0,i-4,i,i+4,i+8]]
                        elif (i+4)>=image_data.shape[2]:
                            image_slice = image_data[:,:,[i-8,i-4,i,i,i]]
                        elif (i+8)>=image_data.shape[2]:
                            image_slice = image_data[:,:,[i-8,i-4,i,i+4,i+4]]
                        else:
                            image_slice = image_data[:,:,[i-8,i-4,i,i+4,i+8]]
                        pixelArrayToImage(image_slice,save_slice_path,imgname,i)

        elif datadir == "label":
            for imgname in sorted(os.listdir(os.path.join(data_path,datadir))):
                print("the processing data is data(%s)" %(imgname))
                # save_slice_path = os.path.join(save_path,imgname.split("_")[0]+'_'+imgname.split("_")[2],datadir)
                save_slice_path = os.path.join(save_path,datadir)
                if not os.path.exists(save_slice_path):
                    os.makedirs(save_slice_path)
                image_data = nib.load(os.path.join(data_path,datadir,imgname))
                image_data = image_data.get_data()
                label_data = np.where(image_data >0.5,1,0)        
                label_data = dilation(label_data,cube(5)).astype(float)
                label_data = post_processing(label_data)
                image_data = image_data*label_data
                label_path = os.path.join(data_path,'trans',imgname.split("_")[0]+'_cmpr_'+imgname.split("_")[2]+'_trans.nii.gz')
                if os.path.exists(label_path):
                    # label_data = np.where(label_data >0.5,1,0)
                    for i in range(0,image_data.shape[2],2):
                        if i == 0:
                            image_slice = image_data[:,:,[0,0,i,i+2,i+4]]
                        if i == 2:
                            image_slice = image_data[:,:,[0,i-2,i,i+2,i+4]]
                        elif (i+2)>=image_data.shape[2]:
                            image_slice = image_data[:,:,[i-4,i-2,i,i,i]]
                        elif (i+4)>=image_data.shape[2]:
                            image_slice = image_data[:,:,[i-4,i-2,i,i+2,i+2]]
                        else:
                            image_slice = image_data[:,:,[i-4,i-2,i,i+2,i+4]]
                        labelArrayToImage(image_slice,save_slice_path,imgname,i)
                else:
                    for i in range(0,image_data.shape[2],4):
                        if i == 0:
                            image_slice = image_data[:,:,[0,0,i,i+4,i+8]]
                        if i == 4:
                            image_slice = image_data[:,:,[0,i-4,i,i+4,i+8]]
                        elif (i+4)>=image_data.shape[2]:
                            image_slice = image_data[:,:,[i-8,i-4,i,i,i]]
                        elif (i+8)>=image_data.shape[2]:
                            image_slice = image_data[:,:,[i-8,i-4,i,i+4,i+4]]
                        else:
                            image_slice = image_data[:,:,[i-8,i-4,i,i+4,i+8]]
                        labelArrayToImage(image_slice,save_slice_path,imgname,i)

if __name__== '__main__':
    rootpath ="./data/3D/"
    savepath = "./data/2D_seq5_s5_nobg-3-10/"

    for datadir in os.listdir(rootpath):
        data_path = rootpath+'/'+ datadir
        save_path = savepath + '/'+ datadir
        transdata(data_path,save_path)
    # data_path = rootpath+'/'+ 'test/'
    # save_path = savepath + '/'+ 'test/'
    # transdata(data_path,save_path)


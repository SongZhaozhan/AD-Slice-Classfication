import os 
import nibabel as nib 
import numpy as np
data_path ="./data/3D/ori"
save_path = "./data/2D_patient/ori"



    # cur_dir_image = os.path.join(data_path,patient)
    # save_dir_image = os.path.join(save_path,"image")
    # save_dir_label = os.path.join(save_path,'label')
    # if not os.path.exists(save_dir_image):
    #     os.makedirs(save_dir_image)
    # if not os.path.exists(save_dir_label):
    #     os.makedirs(save_dir_label)
    # image_name = 'AD_' + patient + '_mpr.nii.gz'
    # img_path = os.path.join(cur_dir_image,image_name)
    # label_path = os.path.join(cur_dir_image,'AD_segmentation_' + patient + '_mpr.nii.gz')

    # save_img_mpr_path =os.path.join(save_dir_image,"AD_"+patient+"_mpr.nii.gz")
    # save_seg_mpr_path =os.path.join(save_dir_label,"AD_segmentation_"+patient+"_mpr.nii.gz")
    # label = sitk.ReadImage(label_path)
    # label = pick_largest_connected_component(label,[1,2])
    # sitk.WriteImage(label,save_seg_mpr_path)

    # img = sitk.ReadImage(img_path)
    # sitk.WriteImage(img,save_img_mpr_path)



def pixelArrayToImage(pixel_array,savepath,imgname,idx):
    # shape = pixel_array.shape
    # image_array = pixel_array.astype(float)
    # image_array = (np.maximum(image_array, 0) / image_array.max()) * 255.0
    # image_array = np.uint8(image_array)
    image_array = pixel_array
    idx = str(idx+1)
    idx =idx.zfill(4)
    imgname = imgname.split("mpr")[0]
    image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min()))*255
    path_npy = savepath + '/'+ imgname + idx + '.npy'
    np.save(path_npy,image_array)


def labelArrayToImage(label_array,savepath,imgname,idx):
    # shape = pixel_array.shape
    # image_array = pixel_array.astype(float)
    # image_array = (np.maximum(image_array, 0) / image_array.max()) * 255.0
    # image_array = np.uint8(image_array)
    image_array = label_array
    idx = str(idx+1)
    idx =idx.zfill(4)
    imgname = imgname.split("mpr")[0]
    # image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min()))*255.0
    image_array = image_array*50
    path_npy = savepath + '/'+ imgname + idx + '.npy'
    np.save(path_npy,image_array)


for datadir in sorted(os.listdir(data_path)):
    if datadir =="image":

        for imgname in sorted(os.listdir(os.path.join(data_path,datadir))):
            print("the processing data is data(%s)" %(imgname))
            save_slice_path = os.path.join(save_path,datadir)
            if not os.path.exists(save_slice_path):
                os.makedirs(save_slice_path)
            image_data = nib.load(os.path.join(data_path,datadir,imgname))
            image_data = image_data.get_data()
            for i in range(2,image_data.shape[2],5):
                image_slice = image_data[:,:,i-2:i+3]
                # image_slice_path = os.path.join(save_path,datadir)
                pixelArrayToImage(image_slice,save_slice_path,imgname,i)

    elif datadir == "label":
        for imgname in sorted(os.listdir(os.path.join(data_path,datadir))):
            print("the processing data is data(%s)" %(imgname))
            save_slice_path = os.path.join(save_path,datadir)
            if not os.path.exists(save_slice_path):
                os.makedirs(save_slice_path)
            image_data = nib.load(os.path.join(data_path,datadir,imgname))
            image_data = image_data.get_data()
            for i in range(2,image_data.shape[2],5):
                image_slice = image_data[:,:,i-2:i+3]
                image_slice_path = os.path.join(save_path,datadir)
                labelArrayToImage(image_slice,save_slice_path,imgname,i)
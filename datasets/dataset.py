import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class A4C_train_datasets(Dataset):
    def __init__(self, path_Data, config):
        super(A4C_train_datasets, self)
        self.num_classes = config.num_classes
        self.input_size_h = config.input_size_h
        self.input_size_w = config.input_size_w
        self.train_fold = config.train_list

        # images_path = path_Data + 'label_jpg.csv'
        # images_list = pd.read_csv(images_path, index_col='Name')
        # train_image = images_list[images_list['Split'].isin(self.train_fold)]

        masks_path = path_Data + 'label.csv'
        masks_list = pd.read_csv(masks_path, index_col='Name')
        train_mask_list = masks_list[masks_list['Split'].isin(self.train_fold)].index
        self.data = []
        for i, item in enumerate(train_mask_list):
            images = path_Data+'images/' + item[:-4]+ '.jpg'
            masks = path_Data + 'masks/' + item
            edges = path_Data + 'edges/' + item
            self.data.append([images, masks, edges])

        self.transformer = config.train_transformer
                        
    def __getitem__(self, indx):
        img_path, mask_path,edge_path = self.data[indx]
        img = np.array(Image.open(img_path))
        mask = Image.open(mask_path)
        png = np.array(mask)
        png = np.expand_dims(png,axis = 2)
        seg_labels  = np.eye(self.num_classes )[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_size_h), int(self.input_size_w), self.num_classes))
        edge = Image.open(edge_path)
        egs = np.array(edge)
        egs[egs >= self.num_classes] = self.num_classes-1
        egs = np.expand_dims(egs,axis = 2)
        edge_labels  = np.eye(self.num_classes )[egs.reshape([-1])]
        edge_labels  = edge_labels.reshape((int(self.input_size_h), int(self.input_size_w), self.num_classes))
        # img = np.array(Image.open(img_path).convert('RGB'))
        # msk = np.expand_dims(np.array(Image.open(msk_path)), axis=2) / 1.0
        img, seg_labels, edge_labels = self.transformer((img, seg_labels, edge_labels))# img:3,256,256, png:1,256,256
        return img, seg_labels, edge_labels

    def __len__(self):
        return len(self.data)
    

class A4C_test_datasets(Dataset):
    def __init__(self, path_Data, config, train=False, test = True):
        super(A4C_test_datasets, self)
        self.num_classes = config.num_classes
        self.input_size_h = config.input_size_h
        self.input_size_w = config.input_size_w
        self.test_fold = config.test_list

        masks_path = path_Data + 'label.csv'
        masks_list = pd.read_csv(masks_path, index_col='Name')
        test_mask_list = masks_list[masks_list['Split'].isin(self.test_fold)].index

        self.data = []
        for i, item in enumerate(test_mask_list):
            images = path_Data + 'images/' + item[:-4] + '.jpg'
            masks = path_Data + 'masks/' + item
            edges = path_Data + 'edges/' + item
            name = item
            self.data.append([images, masks, edges,name])

        self.transformer = config.test_transformer
                        
    def __getitem__(self, indx):
        img_path, mask_path,edge_path, fname = self.data[indx]
        img = np.array(Image.open(img_path))
        mask = Image.open(mask_path)
        png = np.array(mask)
        png = np.expand_dims(png,axis = 2)
        seg_labels  = np.eye(self.num_classes )[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_size_h), int(self.input_size_w), self.num_classes))
        edge = Image.open(edge_path)
        egs = np.array(edge)
        egs[egs >= self.num_classes] = self.num_classes - 1
        egs = np.expand_dims(egs, axis=2)
        edge_labels = np.eye(self.num_classes)[egs.reshape([-1])]
        edge_labels = edge_labels.reshape((int(self.input_size_h), int(self.input_size_w), self.num_classes))
        #
        img, seg_labels,edge_labels = self.transformer((img, seg_labels,edge_labels))# img:3,256,256, png:1,256,256
        return img, seg_labels, edge_labels, fname

    def __len__(self):
        return len(self.data)


class A4C_val_datasets(Dataset):
    def __init__(self, path_Data, config, train=False, test=True):
        super(A4C_val_datasets, self)
        self.num_classes = config.num_classes
        self.input_size_h = config.input_size_h
        self.input_size_w = config.input_size_w
        self.val_fold = config.val_list

        masks_path = path_Data + 'label.csv'
        masks_list = pd.read_csv(masks_path, index_col='Name')
        val_mask_list = masks_list[masks_list['Split'].isin(self.val_fold)].index

        self.data = []
        for i, item in enumerate(val_mask_list):
            images = path_Data + 'images/' + item[:-4] + '.jpg'
            masks = path_Data + 'masks/' + item
            edges = path_Data + 'edges/' + item
            name = item
            self.data.append([images, masks, edges,name])

        self.transformer = config.test_transformer

    def __getitem__(self, indx):
        img_path, mask_path, edge_path, fname = self.data[indx]
        img = np.array(Image.open(img_path))
        mask = Image.open(mask_path)
        png = np.array(mask)
        png = np.expand_dims(png, axis=2)
        seg_labels = np.eye(self.num_classes)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_size_h), int(self.input_size_w), self.num_classes))
        edge = Image.open(edge_path)
        egs = np.array(edge)
        egs[egs >= self.num_classes] = self.num_classes - 1
        egs = np.expand_dims(egs, axis=2)
        edge_labels = np.eye(self.num_classes)[egs.reshape([-1])]
        edge_labels = edge_labels.reshape((int(self.input_size_h), int(self.input_size_w), self.num_classes))
        #
        img, seg_labels, edge_labels = self.transformer((img, seg_labels, edge_labels))  # img:3,256,256, png:1,256,256
        return img, seg_labels, edge_labels, fname

    def __len__(self):
        return len(self.data)


# def random_rot_flip(image, label):
#     k = np.random.randint(0, 4)
#     image = np.rot90(image, k)
#     label = np.rot90(label, k)
#     axis = np.random.randint(0, 2)
#     image = np.flip(image, axis=axis).copy()
#     label = np.flip(label, axis=axis).copy()
#     return image, label


# def random_rotate(image, label):
#     angle = np.random.randint(-20, 20)
#     image = ndimage.rotate(image, angle, order=0, reshape=False)
#     label = ndimage.rotate(label, angle, order=0, reshape=False)
#     return image, label


# class RandomGenerator(object):
#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']

#         if random.random() > 0.5:
#             image, label = random_rot_flip(image, label)
#         elif random.random() > 0.5:
#             image, label = random_rotate(image, label)
#         x, y = image.shape
#         if x != self.output_size[0] or y != self.output_size[1]:
#             image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
#             label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
#         image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
#         label = torch.from_numpy(label.astype(np.float32))
#         sample = {'image': image, 'label': label.long()}
#         return sample

        
    
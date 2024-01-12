import torch


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def reverse_tuple(t):
    if len(t) == 0:
        return t
    else:
        return(t[-1],)+reverse_tuple(t[:-1])


def collate_fn(batch):
    # label = torch.LongTensor([item['label'] for item in batch])
    label = torch.IntTensor([item['label_encoded'] for item in batch])
    label_encoded = torch.FloatTensor([item['label_onehot_encoded'] for item in batch])
    # if np.any(np.array([item['seq'] for item in batch]) == None):
    #     fixation_sequence = None
    #     aoi_heatmap = None
    # else:
    fixation_sequence = [torch.FloatTensor(item['fix_seq']) for item in batch]
    aoi_heatmap = torch.stack([torch.FloatTensor(item['aoi']) for item in batch], dim=0) if 'aoi' in batch[0].keys() else None
    image_resized = torch.stack([torch.FloatTensor(item['image']) for item in batch], dim=0)
    # image_original = torch.stack([torch.FloatTensor(item['original_image']) for item in batch], dim=0)
    image_original = [torch.FloatTensor(item['original_image']) for item in batch]

    if 'sub_images' in batch[0].keys():
        img, subimage_positions = collate_subimages(batch)
        return img, label, label_encoded, fixation_sequence, aoi_heatmap, image_resized, image_original, subimage_positions
    else:
        img = torch.stack([torch.FloatTensor(item['image_z_normed']) for item in batch], dim=0)
        return img, label, label_encoded, fixation_sequence, aoi_heatmap, image_resized, image_original


def collate_subimages(batch):
    subimages = []
    subimage_masks = []
    n_subimages = len(batch[0]['sub_images'])
    subimage_positions = []
    for i in range(n_subimages):
        subimages.append(torch.stack([torch.FloatTensor(item['sub_images'][i]['image']) for item in batch], dim=0))
        subimage_masks.append(torch.stack([torch.BoolTensor(item['sub_images'][i]['mask']) for item in batch], dim=0))
        subimage_positions.append([item['sub_images'][i]['position'] for item in batch])
    return {'subimages': subimages, 'masks': subimage_masks}, subimage_positions
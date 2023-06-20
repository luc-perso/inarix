import tensorflow as tf


def build_dataset_base(data_path, label=0,
                       color_mode='rgb',
                       image_size=(256, 256),
                       shuffle=True,
                       seed=123):
    def make_label(img):
        return img, label

    data_tmp = tf.keras.utils.image_dataset_from_directory(
        directory=data_path,
        labels=None,
        label_mode=None,
        color_mode=color_mode,
        batch_size=None,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed
    )

    # file_paths_tmp = data_tmp.file_paths

    dataset = data_tmp.map(make_label)

    return dataset


# build dataset from data_paths
def build_dataset(data_paths, labels, **kwargs):

    dataset = None
    for data_path, label in zip(data_paths, labels):
        ds_tmp = build_dataset_base(data_path, label=label, **kwargs)

        if dataset is None:
            dataset = ds_tmp
        else:
            dataset = dataset.concatenate(ds_tmp)

    return dataset

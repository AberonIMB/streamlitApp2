from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.transforms import DivisiblePadd
import streamlit as st
import matplotlib.pylab as plt
import nibabel as nb
import torch
import monai
import numpy as np
from scipy.ndimage import label, zoom


def get_jaccar_index(pred, true):
    intersection = (pred * true).sum()
    union = pred.sum() + true.sum() - intersection
    jaccard = (intersection + 1e-8) / (union + 1e-8)
    return jaccard.item()


def find_best_threshold(pred, mask, thresholds):
    best_jaccard = 0
    best_threshold = 0
    for threshold in thresholds:
        binary_pred = (pred >= threshold).float()
        jaccard = get_jaccar_index(binary_pred, mask)
        if jaccard > best_jaccard:
            best_jaccard = jaccard
            best_threshold = threshold
    return best_threshold, best_jaccard


def reduce_resolution(image_data, factor=2):
    return zoom(image_data, (1/factor, 1/factor, 1/factor), order=1)


def normalize_data(data):
    min_val = torch.min(data)
    max_val = torch.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


def add_array(data):
    padder = DivisiblePadd(keys=['image'], k=16)
    data_tensor = torch.from_numpy(data.get_fdata()).unsqueeze(0)
    data_dict = {'image': data_tensor}
    return padder(data_dict)['image'].unsqueeze(0)


@st.cache_data
def load_file(file):
    nii_bytes = file.read()
    nii_image = nb.Nifti1Image.from_bytes(nii_bytes)
    result = add_array(nii_image)
    return result


def change_ct(scan, pred, value):
    copy_scan = np.copy(scan)
    copy_scan[pred > 0] = value
    return copy_scan


# def clean_pred(pred):
#     labels, num_labels = label(pred)
#     if num_labels == 0:
#         return pred
#     sizes = np.bincount(labels.ravel())
#     max_label = sizes[1:].argmax() + 1  # Пропускаем фон

#     largest_object = np.zeros_like(pred)
#     largest_object[labels == max_label] = 1

#     return largest_object


def show_slise(image, str):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, cmap='gray', aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    st.header(str)
    st.pyplot(fig)


learning_rate = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = UNet(
# spatial_dims=3,
# in_channels=1,
# out_channels=1,
# channels=(16, 32, 64, 128, 256),
# strides=(2, 2, 2, 2),
# num_res_units=2,
# norm=Norm.BATCH,
# bias=False,
# ).to(device).to(torch.float64)

model = UNet(
spatial_dims=3,
in_channels=1,
out_channels=1,
channels=(16, 32, 64, 128, 256),
strides=(2, 2, 2, 2),

).to(device).to(torch.float64)

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# loss_function = monai.losses.DiceLoss(sigmoid=True)

model.load_state_dict(torch.load("unet_model.pth"))

thresholds = np.concatenate((np.linspace(0.1, 0.9, 9), np.linspace(0.01, 0.09, 9)))
best_jaccard = 0

st.title("Liver Segmentation Service")

if st.button("Clean Cache"):
    st.cache_data.clear()
    st.success("Cache has been cleared")

ct_file = st.file_uploader("Upload a CT scan")
mask_file = st.file_uploader("Upload a mask (optional)")
mask_slices = []

if ct_file is not None:
    image = load_file(ct_file)
    model.eval()
    with torch.no_grad():
        pred = model(image)
    pred = normalize_data(pred)
    shape = pred.shape
    num_slices = shape[-1]

    if mask_file is not None:
        mask = load_file(mask_file)
        mask_slices = [mask[0, 0, :, :, i] for i in range(num_slices)]
        best_threshold, best_jaccard = find_best_threshold(pred, mask, thresholds)
        pred = (pred >= best_threshold).float()
        st.subheader("Индекс Жаккара: " + str(best_jaccard))
    else:
        pred[pred < 0.5] = 0

    ct_slices = [image[0, 0, :, :, i] for i in range(num_slices)]
    pred_slices = [pred[0, 0, :, :, i] for i in range(num_slices)]
    result_slices = [change_ct(ct_slice, pred_slice, np.max(ct_slice) * 1.5) for ct_slice, pred_slice in zip(ct_slices, pred_slices)]

    slice_index = st.slider("Select slice", 0, num_slices - 1)
    col1_mask, col2_mask, col3_mask = st.columns(3)
    col1, col2 = st.columns(2)
    if mask_file is not None:
        with col1_mask:
            show_slise(ct_slices[slice_index], "CT Scan")
        with col2_mask:
            show_slise(result_slices[slice_index], "Result")
        with col3_mask:
            show_slise(mask_slices[slice_index], "Mask")
    else:
        with col1:
            show_slise(ct_slices[slice_index], "CT Scan")
        with col2:
            show_slise(result_slices[slice_index], "Result")

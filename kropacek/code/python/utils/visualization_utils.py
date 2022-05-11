import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from functools import partial
from pathlib import Path

from utils.image_utils import segment2mask, contours_morphology, unique_indexes


def make_legend_elements(colors, labels, indexes):
    legend_elements = [
        Patch(
            facecolor=colors[i] / 255,
            label=labels[i],
            edgecolor='black') for i in indexes]

    return legend_elements


def show_class_coloring(colors, classes, **kwargs):
    legend_elements = make_legend_elements(
        colors,
        labels=classes,
        indexes=list(range(len(classes)))
    )

    show_legend(
        legend_elements,
        loc='center', ncol=2, frameon=False, fontsize=14,
        hide_axis=True,
        size_x=3.5,
        size_y=0.5,
        export=True,
        **kwargs
    )


def visualize_img_label(colors, classes, img, label, image_name, title='', output_path=Path('')):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    segmentation = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
    mask = segment2mask(segmentation, colors)
    image = cv2.resize(image, (mask.shape[0], mask.shape[1]), interpolation=cv2.INTER_NEAREST)
    contours = contours_morphology(mask, colors, 3, image)
    show_image(
        contours,
        legend_params={
            'handles': make_legend_elements(
                colors,
                labels=classes,
                indexes=unique_indexes(mask)),
            'framealpha': 1,
            'fontsize': 12},
        size=10,
        title= title + image_name,
        export=True,
        path=output_path / image_name,
        close_fig=True
    )


def tech_show_image(image):
    plt.imshow(image)
    plt.show()


def show_image(
        image,
        size=6,
        title=None,
        font_size=15,
        cmap=None,
        legend_params=None,
        export=False,
        path=Path("./untitled.png"),
        dpi=200,
        close_fig=False,
):
    fig, ax = plt.subplots(figsize=(size, size))

    if image.ndim == 3 and image.shape[2] == 1:
        ax.imshow(image[:, :, 0], cmap=cmap)  # show gray image hxwx1 as hxw
    else:
        ax.imshow(image, cmap=cmap)

    ax.axis('off')

    if title is not None:
        ax.set_title(title, fontsize=font_size)

    if legend_params is not None:
        ax.legend(**legend_params)

    if export:
        plt.savefig(path, dpi=dpi, pad_inches=0, bbox_inches='tight')

    if close_fig:
        plt.close(fig)
    else:
        plt.show()


def show_legend(
        legend_elements,
        size_x=6,
        size_y=6,
        hide_axis=True,
        export=False,
        path=Path("./legend.png"),
        dpi=200,
        close_fig=False,
        **kwargs
):
    fig, ax = plt.subplots(figsize=(size_x, size_y))
    ax.legend(handles=legend_elements, **kwargs)

    if hide_axis:
        ax.axis('off')

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)

    if export:
        plt.savefig(path, dpi=dpi, pad_inches=0, bbox_inches='tight')

    if close_fig:
        plt.close(fig)
    else:
        plt.show()


show_image_fine = partial(
    show_image,
    size=10,
    font_size=26)


def show_colors(colors, **kwargs):
    show_image(colors.reshape(1, -1, 3), **kwargs)


def show_image_list(
        image_list,
        size=6,
        title=None,
        font_size=15,
        cmap=None,
        legend_params=None,
        adjust_params=None,
        export=False,
        path=Path("./untitled.png"),
        dpi=200,
        close_fig=False,
):
    if adjust_params is None:
        adjust_params = {
            'left': 0.125,
            'right': 0.9,
            'top': 0.9,
            'bottom': 0.1,
            'hspace': 0.3,
            'wspace': 0,  # default 0.2
        }
    n_images = len(image_list)
    fig, ax = plt.subplots(ncols=n_images, figsize=(size, size))

    for i, image in enumerate(image_list):

        if image.ndim == 3 and image.shape[2] == 1:
            ax[i].imshow(image[:, :, 0], cmap=cmap)  # show gray image hxwx1 as hxw
        else:
            ax[i].imshow(image, cmap=cmap)

        ax[i].axis('off')

        if title is not None:
            if title[i] is not None:
                ax[i].set(title=title[i])
                ax[i].title.set_fontsize(fontsize=font_size)

        if legend_params is not None:
            if legend_params[i] is not None:
                ax[i].legend(**(legend_params[i]))

    fig.subplots_adjust(**adjust_params)

    if export:
        plt.savefig(path, dpi=dpi, pad_inches=0, bbox_inches='tight')

    if close_fig:
        plt.close(fig)
    else:
        plt.show()


def show_images_alpha(
        image1,
        image2,
        alpha,
        size=6,
        title=None,
        font_size=15,
        cmap1=None,
        cmap2=None,
        legend_params=None,
        export=False,
        path=Path("./untitled.png"),
        dpi=200,
        close_fig=False,
):
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(image1, alpha=alpha, cmap=cmap1)
    ax.imshow(image2, alpha=(1 - alpha), cmap=cmap2)
    ax.axis('off')

    if title is not None:
        ax.set_title(title, fontsize=font_size)

    if legend_params is not None:
        ax.legend(**legend_params)

    if export:
        plt.savefig(path, dpi=dpi, pad_inches=0, bbox_inches='tight')

    if close_fig:
        plt.close(fig)
    else:
        plt.show()

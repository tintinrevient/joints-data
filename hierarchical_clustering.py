import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.offsetbox import OffsetImage,AnnotationBbox


# CSV data
POSE_DATA_CSV = os.path.join('output', 'joint_angles.csv')

# all the artists
classical_artists = ['Artemisia Gentileschi', 'El Greco', 'Michelangelo', 'Pierre-Auguste Renoir', 'Pierre-Paul Prud\'hon']
modern_artists = ['Amedeo Modigliani', 'Felix Vallotton', 'Paul Delvaux', 'Paul Gauguin', 'Tamara de Lempicka']


def _read_data(artist):

    per_artist = True if artist else False

    df = pd.read_csv(POSE_DATA_CSV, index_col=0)
    columns = df.columns

    # filter for one artist
    if per_artist:
        df = df.loc[df.index.str.contains(artist), columns]

    print(df)
    return df


def _plot_setting(artist):

    per_artist = True if artist else False

    if per_artist:
        outfile = os.path.join('dendrogram-{}.png'.format(artist.replace(' ', '-').lower()))
    else:
        outfile = os.path.join('dendrogram.png')

    # size
    if artist == None:
        size = 30
    elif artist == 'Paul Delvaux':
        size = 26
    elif artist == 'El Greco':
        size = 16
    elif artist == 'Paul Gauguin':
        size = 20
    else:
        size = 10

    return outfile, size


def _get_norm_pose_img(name):
    '''show the corresponding images on xticks'''
    name_list = name.split('_')

    # artist
    artist = name_list[0]
    # fname
    fname = '%s_norm_%s.png' % (name_list[3], name_list[4])
    #category
    category = 'classical' if artist in classical_artists else 'modern'

    path = "keypoints/%s/%s/%s" % (category, artist, fname)
    im = plt.imread(path)
    return im


def _get_offset_image(xtick, label, ax):
    name = label.get_text()
    img = _get_norm_pose_img(name)  # img.shape = 200 x 200
    im = OffsetImage(img, zoom=0.2)
    im.image.axes = ax

    ab = AnnotationBbox(im, (xtick, 0),  xybox=(0., -20.), frameon=False,
                        xycoords='data',  boxcoords="offset points", pad=0)

    ax.add_artist(ab)


def _get_cropped_pose_img(name):
    '''show the corresponding images on xticks'''
    name_list = name.split('_')

    # artist
    artist = name_list[0]
    # fname
    fname = '%s_%s.png' % (name_list[3], name_list[4])
    # category
    category = 'classical' if artist in classical_artists else 'modern'

    path = "keypoints/%s/%s/%s" % (category, artist, fname)
    im = plt.imread(path)
    return im


def _get_cropped_image(xtick, label, ax):
    name = label.get_text()
    img = _get_cropped_pose_img(name)
    width, height, _ = img.shape
    zoom = 0.2 * 180 / width
    im = OffsetImage(img, zoom=zoom)
    im.image.axes = ax

    ab = AnnotationBbox(im, (xtick, 0), xybox=(0., -60.), frameon=False,
                        xycoords='data', boxcoords="offset points", pad=0)

    ax.add_artist(ab)


def generate_dendrogram(artist, show_pose):
    # read the data
    df = _read_data(artist)

    # setting of plot
    outfile, size = _plot_setting(artist)

    # linked
    Z = linkage(df, method='complete')
    print(Z.shape)

    # plot the dendrogram
    fig, ax = plt.subplots(figsize=(size, 5))

    dendrogram(Z, labels=list(df.index), color_threshold=0)

    # xticks
    plt.xticks(rotation=90)
    # ylabel
    ax.set_ylabel('distance')

    if show_pose:
        # xticks with images
        xticks = list(ax.get_xticks())
        xticklabels = list(ax.get_xticklabels())

        for xtick, label in zip(xticks, xticklabels):
            # 1st line: norm pose
            _get_offset_image(xtick, label, ax)
            # 2nd line: cropped pose from the rendered image with keypoints
            _get_cropped_image(xtick, label, ax)

        # don't show axis
        plt.axis('off')

    # save the plot
    plt.savefig(os.path.join('pix', outfile), bbox_inches='tight', pad_inches=0, dpi=227)


def generate_clusters(artist, num_cluster):

    # read the data
    df = _read_data(artist)

    # linked
    Z = linkage(df, method='complete')

    # show the clusters
    memb = fcluster(Z, num_cluster, criterion='maxclust')
    memb = pd.Series(memb, index=df.index)
    for key, item in memb.groupby(memb):
        print(f"{key} : {', '.join(item.index)}")


if __name__ == '__main__':

    # for all artists
    # python hierarchical_clustering.py
    # python hierarchical_clustering.py --cluster 10

    # for all COCO man or woman
    # python hierarchical_clustering.py --cluster 10

    # for one artist
    # python hierarchical_clustering.py --artist "Pierre-Paul Prud\'hon" --pose True

    parser = argparse.ArgumentParser(description='Extract the angles of keypoints')
    parser.add_argument("--artist", help="name of artist")
    parser.add_argument("--pose", default=False, help="whether to show the pose by xtick")
    parser.add_argument("--cluster", default=5, help="number of clusters")
    args = parser.parse_args()

    # input setting
    artist = args.artist if args.artist else None
    show_pose = True if args.pose == 'True' else False
    num_cluster = int(args.cluster)

    # step 1: generate the dendrogram
    generate_dendrogram(artist, show_pose)

    # step 2: generate the clusters
    generate_clusters(artist, num_cluster)

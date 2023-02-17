# Import des librairies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import plotly.express as px

# Pour les warnings
import warnings

warnings.filterwarnings("ignore")
# Pour les stats
from scipy import stats

# Pour la modélisation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score


def colonne(df):
    """Itération sur les colonnes du dataframe pour afficher le nombre unique des valeurs dans chaque colonne
    Exploration des colonnes"""
    for col in df.columns:
        print("La colonne ", col, " : contient", df[col].nunique(), "valeur unique")


def format_pourcentage(value):
    '''
    Format a percentage with 1 digit after comma 
    '''
    return "{0:.4f}%".format(value * 100)


def missing_data(df):
    """fonction qui retourne le nombre de nan total dans un df"""
    return df.isna().sum().sum()



def missing_percent(df):
    """ fonction qui retourne le nombre de nan total dans un df en pourcentage"""
    return df.isna().sum().sum() / (df.size)



def summary(df):
        """"Fonction summary du dataframe elle affciche la taille du df,nbre unique de la variable,nana et valeur minimale"""
        obs = df.shape[0]
        types = df.dtypes
        counts = df.apply(lambda x: x.count())
        #min = df.min()
        uniques = df.apply(lambda x: x.unique().shape[0])
        nulls = df.apply(lambda x: x.isnull().sum())
        print("Data shape:", df.shape)
        # cols = ["types", "counts", "uniques", "nulls","min","max"]
        cols = ["types", "counts", "uniques", "nulls"]
        str = pd.concat([types, counts, uniques, nulls], axis=1, sort=True)

        str.columns = cols
        dtypes = str.types.value_counts()
        print("___________________________\nData types:")
        print(str.types.value_counts())
        print("___________________________")
        return str



def missing_values(df):
    """ Fonction qui retourne un df avec nombre de nan et pourcentage"""
    nan = pd.DataFrame(columns=["Variable", "nan", "%nan"])
    nan["Variable"] = df.columns
    missing = []
    percent_missing = []
    for col in df.columns:
        nb_missing = missing_data(df[col])
        pc_missing = format_pourcentage(missing_percent(df[col]))
        missing.append(nb_missing)
        percent_missing.append(pc_missing)
    nan["nan"] = missing
    nan["%nan"] = percent_missing
    return nan.sort_values(by="%nan", ascending=False)

# Deux fonctions, pour calculer les scores quantiles pour la récence, la fréquence et le monétaire
#Le résultat contient mes colonnes Récence, Fréquence et Monétaire, ainsi que le quartile pour chaque valeur
def RScore(x, p, d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
         return 2
    elif x <= d[p][0.75]:
         return 3
    else:
         return 4


def FMScore(x, p, d):
    
    """ Renvoie la fréquence et le score monétaire d'une valeur par rapport à la valeur du quartile"""
    
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]:
        return 2
    else:
        return 1



def plot_nan(df):
    """ Fonction nan et plot"""
    fig = plt.figure(figsize=(22, 10))

    nan_p = df.isnull().sum().sum() / len(df) / len(df.columns) * 100
    plt.axhline(y=nan_p, linestyle="--", lw=2)
    plt.legend(["{:.2f}% Taux global de nan".format(nan_p)], fontsize=14)

    null = df.isnull().sum(axis=0).sort_values() / len(df) * 100
    sns.barplot(null.index, null.values)

    plt.ylabel("%")
    plt.title("Pourcentage de NAN pour chaque variable")
    plt.xticks(rotation=70)
    plt.show()
    

def plot_remp(df):
    """  Fonction remplissage et plot"""
    remplissage_df = df.count().sort_values(ascending=True)
    ax = remplissage_df.plot(kind="bar", figsize=(15, 15))
    ax.set_title("Remplissage des données")
    ax.set_ylabel("Nombre de données")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=14)
    plt.tight_layout()




def missing_rows(df):
    """ Fonction de nan par lignes"""
    lines_nan_info = []
    for index, row in df.iterrows():
        lines_nan_info.append((row.isna().sum().sum() / df.shape[1]) * 100)
        df_lines_nan_info = pd.DataFrame(np.array(lines_nan_info), columns=["nan %"])
    return df_lines_nan_info.sort_values(by=["nan %"], ascending=False)



def neg_to_zero(x):
    """ Fonction pour les valeurs en dessous de 0"""
    if x <= 0:
        return 1
    else:
        return x



def correlation_matrix(df):
    """ Affiche la matrice de corrélations"""
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(
        df.corr(),
        mask=mask,
        center=0,
        cmap="Reds",
        linewidths=1,
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
    )
    plt.title("Matrice des corrélations", fontsize=15, fontweight="bold")
    plt.show()



def outliers(df, str_columns):
    for i in range(len(str_columns)):
        col = str_columns[i]
        # Calcul du quantile 0,25 qui est le quartile q1 : Calcul de la borne inférieure
        q1 = df[col].quantile(0.25)
        # Calcul du quantile 0,75 qui est le quartile q3 : Calcul de la borne supérieure
        q3 = round(df[col].quantile(0.75), 2)
        # l'écart interquartile (IQR)
        iqr = q3 - q1
        # Mise en évidence de valeurs aberrante faibles
        low = q1 - (1.5 * iqr)
        # Mise en évidence de valeurs aberrante élevées
        high = q3 + (1.5 * iqr)
        # filter the dataset with the IQR
        df_outlier = df.loc[(df[col] > high) | (df[col] < low)]

        return df_outlier


#plot du dedogramm"""
def plot_dendrogram(Z):
    plt.figure(figsize=(20, 25))
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("distance")
    dendrogram(
        Z, orientation="left",
    )
    plt.show()


def visualizers(normalised_df_rfm, clusters_number):
    """ visualisation du coefficient de silhouette"""
    model = KMeans(n_clusters=clusters_number)
    visualizer = SilhouetteVisualizer(model, random_state=42)
    visualizer.fit(normalised_df_rfm)  # Fit the data to the visualizer
    visualizer.poof()


def kmeans(normalised_df_rfm, clusters_number, original_df_rfm):
    """ Fonction du clustering kmeans et visualisation TSNE"""
    kmeans = KMeans(n_clusters=clusters_number, random_state=42)
    kmeans.fit(normalised_df_rfm)
    # Extract cluster labels
    cluster_labels = kmeans.labels_

    # Create a cluster label column in original dataset
    df_new = original_df_rfm.assign(Cluster=cluster_labels)

    # Initialise TSNE
    model = TSNE(random_state=42)
    transformed = model.fit_transform(df_new)

    # Plot t-SNE
    plt.title("Flattened Graph of {} Clusters".format(clusters_number))
    sns.scatterplot(
        x=transformed[:, 0],
        y=transformed[:, 1],
        hue=cluster_labels,
        style=cluster_labels,
        palette="Set1",
    )

    return df_new



def snake_plot(normalised_df_rfm, df_rfm_kmeans, df_rfm_original):
    """ Fonction snake plot"""
    normalised_df_rfm = pd.DataFrame(
        normalised_df_rfm, index=df_rfm_original.index, columns=df_rfm_original.columns
    )
    normalised_df_rfm["Cluster"] = df_rfm_kmeans["Cluster"]
    # Melt data into long format
    df_melt = pd.melt(
        normalised_df_rfm.reset_index(),
        id_vars=["customer_unique_id", "Cluster"],
        value_vars=[
            "Recency",
            "Frequency",
            "Monetary",
            "Score_moyen",
            "Nb_Versement",
            "Moyen_paiement",
            "Fret_moyen",
            "day_diff_achat",
        ],
        var_name="Metric",
        value_name="Value",
    )
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.xticks(rotation=70)
    sns.pointplot(data=df_melt, x="Metric", y="Value", hue="Cluster")

    return


def heatmap(df):
    """heatmap des données"""
    # the mean value in total
    total_avg = df.iloc[:, 0:8].mean()
    total_avg
    # calculat proportion
    cluster_avg = df.groupby("Cluster").mean()
    prop_rfm = cluster_avg / total_avg
    # heatmap with RFM
    sns.heatmap(prop_rfm, cmap="Oranges", fmt=".2f", annot=True)
    plt.title("Heatmap des clusters")
    plt.plot()


def k_values(df):
    """Retourne dataframe agérégé"""
    df_new = (
        df.groupby(["Cluster"])
        .agg(
            {
                "Recency": "mean",
                "Frequency": "mean",
                "Monetary": "mean",
                "Score_moyen": "mean",
                "Nb_Versement": "mean",
                "Moyen_paiement": "mean",
                "Fret_moyen": "mean",
                "day_diff_achat": ["mean", "count"],
            }
        )
        .round(3)
    )

    return df_new


def dbscan(normalised_df_rfm, samples_number, eps_number, original_df_rfm):
    """fonction qui lance dbscan et visualisation tsne"""
    clusters = DBSCAN(eps=eps_number, min_samples=samples_number)
    clusters.fit(normalised_df_rfm)

    # Extract cluster labels
    clusters_labels = clusters.labels_

    # Create a cluster label column in original dataset
    df_new = original_df_rfm.assign(Cluster=clusters_labels)

    # Initialise TSNE
    model = TSNE(random_state=42)
    transformed = model.fit_transform(df_new)

    # Plot t-SNE
    plt.title("Flattened Graph of {} Clusters".format(set(clusters_labels)))
    sns.scatterplot(
        x=transformed[:, 0], y=transformed[:, 1], hue=clusters.labels_, palette="Set1"
    )
    return df_new


# ACP
def display_scree_plot(pca):
    """Display a scree plot for the pca"""

    scree = pca.explained_variance_ratio_ * 100
    plt.bar(np.arange(len(scree)) + 1, scree)
    plt.plot(np.arange(len(scree)) + 1, scree.cumsum(), c="red", marker="o")
    plt.xlabel("Number of principal components")
    plt.ylabel("Percentage explained variance")
    plt.title("Scree plot")
    plt.show(block=False)


def display_circles(
    pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None
):
    for (
        d1,
        d2,
    ) in (
        axis_ranks
    ):  # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7, 6))

            # détermination des limites du graphique
            if lims is not None:
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30:
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else:
                xmin, xmax, ymin, ymax = (
                    min(pcs[d1, :]),
                    max(pcs[d1, :]),
                    min(pcs[d2, :]),
                    max(pcs[d2, :]),
                )

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30:
                plt.quiver(
                    np.zeros(pcs.shape[1]),
                    np.zeros(pcs.shape[1]),
                    pcs[d1, :],
                    pcs[d2, :],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color="grey",
                )
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0, 0], [x, y]] for x, y in pcs[[d1, d2]].T]
                ax.add_collection(
                    LineCollection(lines, axes=ax, alpha=0.1, color="black")
                )

            # affichage des noms des variables
            if labels is not None:
                for i, (x, y) in enumerate(pcs[[d1, d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                        plt.text(
                            x,
                            y,
                            labels[i],
                            fontsize="14",
                            ha="center",
                            va="center",
                            rotation=label_rotation,
                            color="blue",
                            alpha=0.5,
                        )

            # affichage du cercle
            circle = plt.Circle((0, 0), 1, facecolor="none", edgecolor="b")
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color="grey", ls="--")
            plt.plot([0, 0], [-1, 1], color="grey", ls="--")

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel(
                "F{} ({}%)".format(
                    d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)
                )
            )
            plt.ylabel(
                "F{} ({}%)".format(
                    d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)
                )
            )

            plt.title("Cercle des corrélations (F{} et F{})".format(d1 + 1, d2 + 1))
            plt.show(block=False)


def display_factorial_planes(
    X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None
):
    for d1, d2 in axis_ranks:
        if d2 < n_comp:

            # initialisation de la figure
            fig = plt.figure(figsize=(12, 8))

            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(
                        X_projected[selected, d1],
                        X_projected[selected, d2],
                        alpha=alpha,
                        label=value,
                    )
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i, (x, y) in enumerate(X_projected[:, [d1, d2]]):
                    plt.text(x, y, labels[i], fontsize="14", ha="center", va="center")

            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1, d2]])) * 1.1
            plt.xlim([-boundary, boundary])
            plt.ylim([-boundary, boundary])

            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color="grey", ls="--")
            plt.plot([0, 0], [-100, 100], color="grey", ls="--")

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel(
                "F{} ({}%)".format(
                    d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)
                )
            )
            plt.ylabel(
                "F{} ({}%)".format(
                    d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)
                )
            )

            plt.title(
                "Projection des individus (sur F{} et F{})".format(d1 + 1, d2 + 1)
            )
            plt.show(block=False)

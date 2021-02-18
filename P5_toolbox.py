import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns
from matplotlib.collections import LineCollection
from scipy.cluster.hierarchy import dendrogram


def loss_ratio(vector, dataset , main_dataframe, min_value=0, max_value=0):
    
    if max_value == 0:
        max_value = dataset[vector].max()
        
    if min_value == 0:
        min_value = dataset[vector].min()

    # ratio with adjustables limits
    intra = dataset[dataset[vector] < min_value].shape[0] + dataset[dataset[vector] > max_value].shape[0]
    inter = dataset[dataset[vector] < min_value].shape[0] + dataset[dataset[vector] > max_value].shape[0]

    print("% intra loss  ", round(intra / dataset[vector].shape[0],4))
    print("% global loss ", round(inter / main_dataframe[vector].shape[0],4))

    
def drop_values(vector, dataset, main_dataframe, min_value, max_value):
    
    row_indexer = dataset[dataset[vector] >= max_value].index
    main_dataframe.drop(index=row_indexer, inplace=True)
    
    row_indexer = dataset[dataset[vector] <= min_value].index
    main_dataframe.drop(index=row_indexer, inplace=True)

    
def isolation_forest(matrix, contamination):
    # model selection
    from sklearn.ensemble import IsolationForest
    
    # feature set
    X = np.array(matrix)
    
    # model train
    model = IsolationForest(contamination=contamination, random_state=0)
    model.fit(X)
    
    # model prediction
    prediction = model.predict(X)
    
    return X[prediction == 1]

def kmeans_graph(matrix, n_clusters):
    # model selection
    from sklearn.cluster import KMeans
    matrix = np.array(matrix)
    
    # model train
    model = KMeans(n_clusters=n_clusters, random_state=0)
    model.fit(matrix)

    # model prediction
    prediction = model.predict(matrix)

    # visual result
    plt.figure(figsize=(20,10),dpi=200)
    plt.scatter(model.cluster_centers_[:, 1], model.cluster_centers_[:, 0], c='r')
    plt.scatter(matrix[:, 1], matrix[:, 0], c=prediction);

    
def elbow_method_graph(matrix, max_n_clusters):
    from sklearn.cluster import KMeans
    matrix = np.array(matrix)
    
    # inertia = cost function
    inertia = []

    # benchmark
    k_range = range(1,max_n_clusters)
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=0).fit(matrix)
        inertia.append(model.inertia_)
    
    # visual result
    plt.figure(dpi=100)
    plt.plot(k_range, inertia)
    plt.xlabel('Clusters number')
    plt.ylabel('Inertia cost')

   
def shapiro_ratio_categ_global(vector_name, target_vector, dataset):
    from scipy import stats

    for category_name in dataset[vector_name].unique():
        categ = dataset[dataset[vector_name] == category_name].copy()
        print(category_name + ' ' + str(round(stats.shapiro(categ[target_vector])[0],2)))

    print('global ' + str(round(stats.shapiro(dataset[target_vector])[0],2)))
    
def eta2(vector_quant, vector_quali, data):
    
    # function body
    from sklearn.feature_selection import f_classif

    # features set
    dataset = data[[vector_quant, vector_quali]].copy()

    # to set up n_samples, n_features
    dataset = one_hot_matrix(vector_quali, dataset)

    n_samples_n_features = np.array(dataset)

    # reshape(-1,1) : from scalar (1d) to vector(2d)
    n_samples = np.array(dataset.iloc[:,0]).reshape(-1,1)

    # matrix columns
    results_matrix = pd.DataFrame((f_classif(n_samples_n_features, n_samples)))
    results_matrix.columns = dataset.columns
    results_matrix.drop(vector_quant, axis=1, inplace=True)
    results_matrix = results_matrix.T
    
    results_matrix.iloc[:,0] = round((results_matrix.iloc[:,0]/results_matrix.iloc[:,0].sum())*100)
    results_matrix.iloc[:,1] = round(results_matrix.iloc[:,1],3)
    
    results_matrix.columns = ['Fisher %','p-value']

    # heatmpap global settings
    plt.style.use('default')
    sns.set(font_scale=2.5)
    plt.figure(figsize=(5,5))
    
    # eta value title
    title = 'η2 : ' + str(eta_squared(data[vector_quali], data[vector_quant]))
    plt.suptitle(title, fontsize=30)
    
    sns.heatmap(results_matrix, annot=True, cbar=False)

    plt.show()


def one_hot_matrix(vector,data):
    
    dataset = data.copy()
    
    one_hot_matrix = pd.get_dummies(dataset[vector],dtype='bool')

    # to merge new features
    dataset = dataset.merge(one_hot_matrix,how ='left', left_index=True, right_index=True)

    # to drop vector
    dataset = dataset.drop(vector, axis=1)

    return dataset


def eta_squared(x,y):
    moyenne_y = y.mean()
    classes = []
    
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
        
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    
    eta2 = round((SCE/SCT),2)
    
    if eta2==0:
        eta2=int(eta2)
    
    return eta2


def chi2(vector_x, vector_y, data):
    
    from sklearn.feature_selection import chi2
    
    contingency_table = data[[vector_x, vector_y]].copy()
    
    columns_names     = one_hot_matrix(vector_y, contingency_table).columns[1:]
    
    contingency_table = one_hot_matrix(vector_y, contingency_table)
    contingency_table = contingency_table.groupby(vector_x).sum()

    # matrix columns
    results_matrix = pd.DataFrame(chi2(contingency_table, contingency_table.index))
    results_matrix.columns = columns_names
    results_matrix = results_matrix.T
    
    results_matrix.iloc[:,0] = round((results_matrix.iloc[:,0]/results_matrix.iloc[:,0].sum())*100)
    results_matrix.iloc[:,1] = round(results_matrix.iloc[:,1],3)
    
    results_matrix.columns = ['Chi-2 %','p-value']

    # heatmpap global settings
    plt.style.use('default')
    sns.set(font_scale=2.5)
    plt.figure(figsize=(5,5))
    
    # xi_n value title
    title = 'ξn : ' + str(xi_n(vector_x, vector_y, data))
    plt.suptitle(title, fontsize=30)
    sns.heatmap(results_matrix, annot=True, cbar=False)

    plt.show()


def xi_n(X, y, data):

    # to create contingency table
    contingency_table = data[[X,y]].pivot_table(index=X,
                                                columns=y,
                                                aggfunc=len,
                                                margins=True,
                                                margins_name="Total",
                                                fill_value=0)


    # to create independence table
    tx = contingency_table.loc[:,["Total"]]
    ty = contingency_table.loc[["Total"],:]
    n = len(data)
    independence_table = tx.dot(ty) / n

    # to compute xi_n
    contingency_table = contingency_table.iloc[0:4,0:4]
    independence_table = independence_table.iloc[0:4,0:4]

    measure = (contingency_table-independence_table)**2/independence_table
    xi_n = int(measure.sum().sum().round())
    
    return xi_n


def lorenz_gini_graph_dataframe(vector_name, data):
    import quantecon as qe 

    # X = selected Feature
    X = np.array(data[vector_name])

    # to compute lorenz curve
    cum_frequences, cum_weights = qe.lorenz_curve(X)
    gini = qe.gini_coefficient(X)

    # visual result
    plt.style.use('default')
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(5,5),dpi=100)
    plt.axis(xmin=0, xmax=1 ,ymin=0, ymax=1)
    plt.title('Gini : ' + str(round(gini,2)))

    # to plot the equality line
    plt.plot([0, 1], lw=2, c='gray', ls='dotted', label='Equality')

    # to plot the Lorenz curve
    plt.plot(cum_frequences, cum_weights, lw=2, c='r', label='Lorenz')
    #plt.savefig('lorenz_gini.png')
    plt.legend(frameon=False);
    plt.show()

    # result into dataframe
    vectors = {vector_name     : np.sort(X),
               'cum_frequences': cum_frequences[1:],
               'cum_weights'   : cum_weights[1:]}

    dataset = pd.DataFrame(data=vectors)
    
    return dataset

def boxplot_method_cleaning(vector):
    q1 = vector.quantile(0.25)
    q2 = vector.quantile(0.75)
    iqr = q2 - q1

    b_min = q1 - np.multiply(iqr, 1.5)
    b_max = q2 + np.multiply(iqr, 1.5)

    vector = vector[vector>=b_min]
    vector = vector[vector<=b_max]
    
    return vector

def r2(target_vector, data):
    
    plt.figure(dpi=100)
    
    corr_vector = pd.DataFrame(((data.corr()**2).round(2)))[0:1]
    corr_vector = corr_vector.dropna(axis=1)
    one_level_column(corr_vector)

    sns.heatmap(corr_vector.iloc[:,1:], annot=True, cbar=False, square=True)

    plt.xlabel('')
    plt.ylabel('')
    plt.show();
    
    
def one_level_column(data):
    
    if data.columns.nlevels > 1 :
        names_col =[]
        for id_col in range(0,data.columns.shape[0]):
            name = (data.columns.get_level_values(0)[id_col] + '_' 
                  + data.columns.get_level_values(1)[id_col])
    
            names_col.append(name)

        data.columns = names_col
        

def pie_chart(lorenz_table, dataframe, color_list):

    data = dataframe.loc[lorenz_table.index]['categ'].value_counts(normalize=True).round(2).reset_index()
    data['index'] = data['index'].map({'c_1': 'catégorie 1', 'c_0':'catégorie 0', 'c_2':'catégorie 2'})
    
    plt.style.use('default')

    plt.pie(data['categ'], 
            labels=data['index'], 
            autopct=lambda x: str(int(round(x))) + ' %', 
            labeldistance=None, 
            wedgeprops={'edgecolor' : 'w', 'linewidth' : 2},
            textprops={'fontsize': 20, 'color':'w'},
            colors=color_list
           )

    plt.legend(loc='upper center', 
               fontsize=20, 
               bbox_to_anchor=(0.25, 0.75, 0.5, 0.5),
               frameon=False)

    plt.show()


def frequences_graph_stats(vector, data):

    frequences = pd.DataFrame(data[vector].value_counts(normalize=True).sort_values().reset_index())
    frequences = frequences.set_index(vector)
    sns.distplot(frequences);
    frequences.columns = [vector]

    print(frequences.describe().round(2))

    plt.figure()
    sns.boxplot(frequences[vector])
    plt.show()

    
def frequence_weight_ratios(lorenz_table, age):
    print((1 - lorenz_table[lorenz_table['age'] == age].iloc[0,1]).round(2))
    print((1 - lorenz_table[lorenz_table['age'] == age].iloc[0,2]).round(2))
    
    
def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: 
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,7))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='30', ha='left', va='center', rotation=label_rotation, color="red")
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, int(round(100*pca.explained_variance_ratio_[d1],0))))
            plt.ylabel('F{} ({}%)'.format(d2+1, int(round(100*pca.explained_variance_ratio_[d2],0))))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

            
def factorial_plane(matrix, groups, labels, centroids):
    
    names = matrix.index
    
    # data compression + pca projection
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler().fit_transform(matrix)
    pca = PCA(n_components=2)
    matrix = pca.fit_transform(scaler)
               
    # affichage des points
    fig = plt.figure(figsize=(7,7))
    
    plt.scatter(matrix[:, 0], matrix[:, 1], alpha=1, c=groups, cmap='Set2')  
    
    # centroids visual
    if centroids:
        centroids = np.concatenate([matrix,np.array(groups).reshape(-1, 1)], axis=1)
        centroids = pd.DataFrame(centroids).groupby(2).mean().round(2)
        
        # red dots
        plt.scatter(centroids.iloc[:, 0], centroids.iloc[:, 1], c='r', alpha=0.7)
        
        # text
        [plt.text(x=centroids.iloc[i, 0], 
                  y=centroids.iloc[i, 1], 
                  s=int(centroids.index[i]), 
                  c='r',
                  fontsize=20) for i in np.arange(0, centroids.shape[0])];
                
    # détermination des limites du graphique
    x_max = np.max(matrix[:, 0])* 1.1
    y_max = np.max(matrix[:, 1])* 1.1
    plt.xlim([-x_max,x_max])
    plt.ylim([-x_max,x_max])
    
    # naming dots
    if labels:
        [plt.text(x=matrix[i, 0], 
                  y=matrix[i, 1], 
                  s=names[i], 
                  c='r',
                  fontsize=8, alpha=0.7) for i in np.arange(0, names.shape[0])];
    
        
    # affichage des lignes horizontales et verticales
    plt.plot([-100, 100], [0, 0], color='grey', ls='--', alpha=0.3)
    plt.plot([0, 0], [-100, 100], color='grey', ls='--', alpha=0.3)

    # nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('criterion ({}%)'.format(int(round(100*pca.explained_variance_ratio_[0],0))))
    plt.ylabel('mean ({}%)'.format(int(round(100*pca.explained_variance_ratio_[1],0))))

    plt.title("Projection des groupes de pays")
    plt.show(block=False)
            
            
def pca_pareto_diagram_corr_circle(matrix):
    
    # data compression + pca projection
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler().fit_transform(matrix)

    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(scaler)
    
    # pareto diagram
    #scree = pca.explained_variance_ratio_*100
    #plt.bar(np.arange(len(scree))+1, scree)
    #plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    #plt.xlabel("Ordre des composantes")
    #plt.ylabel("% de Variance")
    #plt.title("Diagramme de Pareto")
    #plt.show(block=False)
    
    display_circles(pca.components_, 2, pca, [(0,1)], labels = np.array(matrix.columns))
    return pd.DataFrame(pca.components_, columns=matrix.columns, index=np.arange(1, pca.components_.shape[0]+1))
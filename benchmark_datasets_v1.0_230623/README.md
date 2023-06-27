# benchmark_datasets_v1.0_230623

Each dataset in this benchmark is based on a different underlying concept, mapping inputs to outputs.
The directories consist of different versions of these concepts, identified by a key, that describes the different
behaviour of each variation.

Each concept is named based on the following scheme concept _ "generation key" _ "name" _ "number of features" _ "number of targets"


An exemplary key could look like this:

gradual _ True _ faulty _ sensor _ False _ [0, 0, 0, 2, 0, 0] _ 500 _ [1, 0] _ 0

The variant described by this key has the following characteristics


| Characteristic                                                                                       | Realization in the exemplary key                                                          |
|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| **Type of Concept Drift**                                                                            | Gradual Concept Drift                                                                     |
| **Uninterrupted passage of Time**                                                                    | Time passes uninterrupted                                                                 |
| **Type of Data Drift**                                                                               | Faulty Sensor                                                                             |
| **Drifts block each other**                                                                          | Concept Drifts and Data Drift may overlap with other Concept and Data Drifts respectively |
| **Distributions of the Features:<br/>[uniform, gauss, constant, periodical, correlated, dependent]** | The features of this variant are 2 periodical distributed features                        |
| **Number of datapoints**                                                                             | 500                                                                                       |
| **Number of [Concept, Data] Drifts**                                                                 | This variant has one concept drift and no data drift                                      |
| **Factor of the multiplicative noise**                                                               | This variant has no noise, therfore 0                                                     |

The directory of each variant holds 6 files:

- x.csv (features)
- y.csv (targets)
- cd.csv (concept drift informations)
- dd.csv (data drift informations)
- pca_3d_proj.png (3D projection of a 2 feature 1 target system. If the dataset has more dimensions, pca is used for reduction)
- plot.png (Each feature and target dimension is plotted against each other)


## x.csv

Holding the feature data as a pd.DataFrame saved csv file. Feature dimensions are labeled feat_0, ..., feat_n.

## y.csv

Holding the target data as a pd.DataFrame saved csv file. Target dimensions are labeled label_0, ..., label_n.

## cd.csv
Holds information specifying the number and behaviour of concept drifts, including:

time_stamp(centre), radius, shift and class.

| Variable           | Meaning                                                      |
|--------------------|--------------------------------------------------------------|
| time_stamp(centre) | Epicentre of the drift                                       |
| radius             | Half of the duration of the drift, centered at the epicentre |
| shift              | By which values the concept mapping was altered              |
| class              | Of what class the concept drift is                           |

## dd.csv
Holds inofmration specifying the number and ehaviour of data drifts, including:

time_stamp(centre), affected_feature, radius, shift of distribution parameters and class

| Variable                         | Meaning                                                                                |
|----------------------------------|----------------------------------------------------------------------------------------|
| time_stamp(centre)               | Epicentre of the drift                                                                 |
| affected_feature                 | What feature dimension is affected by the drift                                        |
| radius                           | Half of the duration of the drift, centered at the epicentre                           |
| shift of distribution parameters | By which values the distribution parameters changed for the affected feature dimension |
| class                            | Of what class the data drift is                                                        |

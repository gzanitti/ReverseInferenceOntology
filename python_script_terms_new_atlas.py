# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: neurolang
#     language: python
#     name: neurolang
# ---

# %%

print('Starting')

import json

import pandas as pd
import nibabel as nib
import numpy as np
import xml.etree.ElementTree as ET

from nilearn import datasets, image, plotting
from neurolang.frontend import NeurolangPDL
from neurolang.exceptions import NeuroLangException
from typing import Iterable
from sklearn.model_selection import KFold
from rdflib import RDFS

from itertools import chain
import sys

import os
from glob import glob
import urllib

import datetime
import os.path
import gc

import argparse

# %%

print('Parsing args')
parser = argparse.ArgumentParser()
parser.add_argument("--id_region", nargs='?', type=int, default=1)
parser.add_argument("--n_folds", nargs='?', type=int, default=150)
parser.add_argument("--resample", nargs='?', type=int, default=1)
parser.add_argument("--random_state", nargs='?', type=int, default=42)
parser.add_argument("--frac_sample", nargs='?', type=int, default=0.7)
value = parser.parse_args()

# %%
id_region = value.id_region
n_folds = value.n_folds
resample = value.resample
random_state = value.random_state
frac = value.frac_sample

print(f'Running parameters: id_region={id_region}, n_folds={n_folds}, resample={resample}, random_state={random_state}')

start_time = datetime.datetime.now()
print(f'Starting at {start_time}')
if os.path.isfile(f'reverse_inference_results/neuro_paper_ri_term_probs_region{id_region}_{n_folds}folds_no_resample.hdf'):
    end_time = datetime.datetime.now()
    print(f'{id_region}: {end_time - start_time}')
    print('--------------')
    sys.exit()

regions_paths = 'JULICH_BRAIN_CYTOARCHITECTONIC_MAPS_2_9_MNI152_2009C_NONL_ASYM.txt'

lines = []
with open(regions_paths) as f:
    lines = f.readlines()

count = 0
res = []
for line in lines:
    if count == 0:
        count += 1
        continue
    splited = line.split(' ')
    res.append((splited[0], ' '.join(splited[1:-1])[1:], splited[-1][:-2]))

regions = pd.DataFrame(res, columns=['r_number', 'r_name', 'hemis'])
regions = regions[~regions.r_name.str.contains('GapMap')]
regions = regions.astype({'r_number': 'int64'})

## Atlas
mni_t1 = nib.load(datasets.fetch_icbm152_2009()['t1'])
mni_t1_4mm = image.resample_img(mni_t1, np.eye(3) * resample)

jubrain = 'JULICH_BRAIN_CYTOARCHITECTONIC_MAPS_2_9_MNI152_2009C_NONL_ASYM.pmaps.nii.gz'

pmaps_4d = image.resample_img(
    image.load_img(jubrain), mni_t1_4mm.affine, interpolation='nearest'
)

julich_regions_prob = []
for r in regions.r_number.values.astype(int):
    prob_region_data = pmaps_4d.dataobj[:,:,:,r-1]
    non_zero = np.nonzero(prob_region_data)
    for x, y, z in zip(*non_zero):
        p = float(prob_region_data[x][y][z])
        d = (p, x, y, z, r)
        julich_regions_prob.append(d)

# %%
# NeuroSynth
ns_database_fn, ns_features_fn = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('neurosynth'),
    [
        (
            'database.txt',
            'https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz',
            {'uncompress': True}
        ),
        (
            'features.txt',
            'https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz',
            {'uncompress': True}
        ),
    ]
)

ns_database = pd.read_csv(ns_database_fn, sep=f'\t')
ijk_positions = (
    nib.affines.apply_affine(
        np.linalg.inv(mni_t1_4mm.affine),
        ns_database[['x', 'y', 'z']]
    ).astype(int)
)
ns_database['i'] = ijk_positions[:, 0]
ns_database['j'] = ijk_positions[:, 1]
ns_database['k'] = ijk_positions[:, 2]

ns_features = pd.read_csv(ns_features_fn, sep=f'\t')
ns_terms = (
    pd.melt(
            ns_features,
            var_name='term', id_vars='pmid', value_name='TfIdf'
    )
    .query('TfIdf > 1e-3')[['pmid', 'term']]
)
ns_docs = ns_features[['pmid']].drop_duplicates()

# %%
# CogAt
cogAt = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('CogAt'),
    [
        (
            'cogat.xml',
            'http://data.bioontology.org/ontologies/COGAT/download?'
            'apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb&download_format=rdf',
            {'move': 'cogat.xml'}
        )
    ]
)[0]

gc.collect()

# %%
nl = NeurolangPDL()
nl.load_ontology(cogAt)

@nl.add_symbol
def agg_max(i: Iterable) -> float:
    return np.max(i)

@nl.add_symbol
def mean(iterable: Iterable) -> float:
    return np.mean(iterable)


@nl.add_symbol
def std(iterable: Iterable) -> float:
    return np.std(iterable)


subclass_of = nl.new_symbol(name='neurolang:subClassOf')
label = nl.new_symbol(name='neurolang:label')
hasTopConcept = nl.new_symbol(name='neurolang:hasTopConcept')

@nl.add_symbol
def word_lower(name: str) -> str:
    return name.lower()

ns_doc_folds = pd.concat(
    ns_docs.sample(frac=frac, random_state=i).assign(fold=[i] * (int((len(ns_docs)* frac))+1))
    for i in range(n_folds)
)
doc_folds = nl.add_tuple_set(ns_doc_folds, name='doc_folds')


activations = nl.add_tuple_set(ns_database.values, name='activations')
terms = nl.add_tuple_set(ns_terms.values, name='terms')
docs = nl.add_uniform_probabilistic_choice_over_set(
        ns_docs.values, name='docs'
)

terms_det = nl.add_tuple_set(
    ns_terms.term.unique(), name='terms_det'
)

j_brain = nl.add_tuple_set(
    julich_regions_prob,
    name='julich_brain_det'
)

j_regions = nl.add_tuple_set(
    regions.values,
    name='julich_regions'
)

# %%
with nl.scope as e:

    e.ontology_terms[e.onto_name] = (
    hasTopConcept[e.uri, e.cp] &
    label[e.uri, e.onto_name]
    )

    e.lower_terms[e.term] = (
        e.ontology_terms[e.onto_term] &
        (e.term == word_lower[e.onto_term])
    )

    e.filtered_terms[e.d, e.t] = (
        e.terms[e.d, e.t] &
        e.lower_terms[e.t]
    )

    f_term = nl.query((e.d, e.t), e.filtered_terms(e.d, e.t))

# %%
filtered = f_term.as_pandas_dataframe()
filtered_terms = nl.add_tuple_set(filtered.values, name='filtered_terms')


import neurolang
neurolang.config.disable_expression_type_printing()

# %%
start_time = datetime.datetime.now()
print(f'Starting query at {start_time}')
if os.path.isfile(f'reverse_inference_results/neuro_paper_ri_term_probs_region{id_region}_{n_folds}folds_no_resample.hdf'):
    end_time = datetime.datetime.now()
    print(f'{id_region}: {end_time - start_time}')
    print('--------------')
else:
    try:
        with nl.scope as e:

            (e.jbd @ e.p)[e.i, e.j, e.k, e.region, e.d] = (
                e.activations[
                    e.d, ..., ..., ..., ..., 'MNI', ..., ..., ..., ...,
                    ..., ..., ..., e.i, e.j, e.k
                ] &
                e.julich_brain_det[e.p, e.i, e.j, e.k, e.region] &
                (e.region == id_region)
            )

            e.act_regions[e.d, e.id] = e.jbd[..., ..., ..., e.id, e.d]


            e.no_act_regions[e.d, e.id] = (
                ~(e.act_regions[e.d, e.id]) &
                e.docs[e.d] &
                e.julich_regions[e.id, ..., ...]
            )

            e.term_prob[e.t, e.fold, e.PROB[e.t, e.fold]] = (
                (
                    e.filtered_terms[e.d, e.t]
                ) // (
                    e.act_regions[e.d, id_region] &
                    e.doc_folds[e.d, e.fold] &
                    e.docs[e.d]
                )
            )

            e.no_term_prob[e.t, e.fold, e.PROB[e.t, e.fold]] = (
                (
                    e.filtered_terms[e.d, e.t]
                ) // (
                    e.no_act_regions[e.d, id_region] &
                    e.doc_folds[e.d, e.fold] &
                    e.docs[e.d]
                )
            )

            e.ans[e.t, e.f, e.bf, e.p, e.pn] = (
                e.term_prob[e.t, e.f, e.p] &
                e.no_term_prob[e.t, e.f, e.pn] &
                (e.bf == (e.p / e.pn))
            )

            res = nl.query((e.t, e.f, e.bf, e.p, e.pn), e.ans[e.t, e.f, e.bf, e.p, e.pn])
            end_time = datetime.datetime.now()
            print(f'{id_region}: {end_time - start_time}')

            pss = res.as_pandas_dataframe()
            pss.to_hdf(f'reverse_inference_results/neuro_paper_ri_term_probs_region{id_region}_{n_folds}folds_no_resample.hdf', key=f'results')


    except Exception as e:
        print(f'Failed on region: {id_region}')
        print(e)
        #pss = pd.DataFrame([], columns=['t', 'fold', 'PROB'])
        #pss.to_hdf(f'reverse_inference_results/neuro_paper_ri_term_probs_region{id_region}_{n_folds}folds_no_resample.hdf', key=f'results')


# %%

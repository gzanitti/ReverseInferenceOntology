import pandas as pd
import numpy as np
import nibabel as nib

from nilearn import datasets, image
from neurolang.frontend import NeurolangPDL

import neurolang
neurolang.config.disable_expression_type_printing()
import argparse
import math
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

JULICH_REGIONS_PATH = 'JULICH_BRAIN_CYTOARCHITECTONIC_MAPS_2_9_MNI152_2009C_NONL_ASYM.txt'
JULICH_ATLAS_PATH = 'JULICH_BRAIN_CYTOARCHITECTONIC_MAPS_2_9_MNI152_2009C_NONL_ASYM.pmaps.nii.gz'
PEAKS_PATH = 'peaks_IBC.csv'

import time
start_time = time.time()
print(f'Ejecution started at {start_time}')

print('Parsing args')
parser = argparse.ArgumentParser()
parser.add_argument("--id_region", nargs='?', type=int, default=None)
parser.add_argument("--n_folds", nargs='?', type=int, default=150)
parser.add_argument("--resample", nargs='?', type=int, default=1)
parser.add_argument("--frac_sample", nargs='?', type=int, default=0.7)
value = parser.parse_args()

if value.id_region is None:
    print('You need to provide a region using the --id_region argument')
    exit

id_region = value.id_region
n_folds = value.n_folds
resample = value.resample
frac_sample = value.frac_sample

print('Loading data')
nl = NeurolangPDL()

print('    Loading Julich atlas')
lines = []
with open(JULICH_REGIONS_PATH) as f:
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

mni_t1 = nib.load(datasets.fetch_icbm152_2009()['t1'])
mni_t1_4mm = image.resample_img(mni_t1, np.eye(3) * resample)

pmaps_4d = image.resample_img(
    image.load_img(JULICH_ATLAS_PATH), mni_t1_4mm.affine, interpolation='nearest'
)

brain_regions_prob = []
prob_region_data = pmaps_4d.dataobj
non_zero = np.nonzero(pmaps_4d.dataobj)
for x, y, z, r in zip(*non_zero):
    p = prob_region_data[x, y, z, r]
    d = (p, x, y, z, r)
    brain_regions_prob.append(d)

print('    Loading CogAt')

cogAt = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('CogAt'),
    [
     	(
            'cogat.xml',
            'https://data.bioontology.org/ontologies/COGAT/submissions/7/download?'
            'apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb',
            {'move': 'cogat.xml'}
        )
    ]
)[0]

nl.load_ontology(cogAt)

print('    Loading IBC')
df = pd.read_csv(PEAKS_PATH)
df = df[df['Cluster Size (mm3)'] > 20]
#df = df[df['Cluster Size (mm3)'] < 200]


if not os.path.isfile('./removed_img.csv'):
    removed = df.img_id.sample(15, random_state=1)
    removed.to_csv('removed_img.csv', index=False)
    removed_imgs = removed.values
else:
    removed = pd.read_csv('removed_img.csv')
    removed_imgs = removed.values.flatten()

df = df[~df.img_id.isin(removed_imgs)]


def normalize_column(column_name):
    stim_mod_norm = set()
    for term in df[column_name].unique():
        if not isinstance(term, str) and math.isnan(term):
            stim_mod_norm.add('NaN')
            continue
        splitted = term.split(',')
        for s in splitted:
            stim_mod_norm.add(s.strip())

    new_column_name = ''.join(column_name.split(' '))
    stim_mod_norm = pd.DataFrame(stim_mod_norm).reset_index().rename(columns={'index': 'value', 0: new_column_name})

    norm_stim_mod = set()
    temp_df = df[['subject_id','img_id', column_name]].drop_duplicates()
    for index, row in temp_df.iterrows():
        if not isinstance(row[column_name], str) and math.isnan(row[column_name]):
            temp = ['NaN']
        else:
            temp = row[column_name].split(',')
        for t in temp:
            v = stim_mod_norm[stim_mod_norm[new_column_name] == t.strip()].value.values[0]
            norm_stim_mod.add((row.subject_id, row.img_id, v))

    return stim_mod_norm.set_index('value').join(
        pd.DataFrame(norm_stim_mod, columns=['subject_id','img_id', new_column_name]).set_index(new_column_name)
    ).reset_index(drop=True)


norm_tags = normalize_column('tag')
norm_tags['tag'] = norm_tags['tag'].apply(lambda x: x.strip().replace('_', ' ').replace('-', ' '))
#taskParadigm = normalize_column('task_paradigm')

norm_tags.loc[norm_tags.tag.str.contains('visual pseudo word recognition'), 'tag'] = 'visual pseudoword recognition'

@nl.add_symbol
def word_lower(name: str) -> str:
    return name.lower()

subclass_of = nl.new_symbol(name='neurolang:subClassOf')
label = nl.new_symbol(name='neurolang:label')
hasTopConcept = nl.new_symbol(name='neurolang:hasTopConcept')

df = df[['subject_id', 'img_id', 'X', 'Y', 'Z']]

ijk_positions = (
    nib.affines.apply_affine(
        np.linalg.inv(mni_t1_4mm.affine),
        df[['X', 'Y', 'Z']]
    ).astype(int)
)
df['i'] = ijk_positions[:, 0]
df['j'] = ijk_positions[:, 1]
df['k'] = ijk_positions[:, 2]

df = df[['subject_id', 'img_id', 'i', 'j', 'k']]

subject_id = df.subject_id.unique()
img_id = df.img_id.unique()

j_brain = nl.add_tuple_set(
    #(prob, x, y, z, region)
    brain_regions_prob,
    name='julich_brain'
)

j_regions = nl.add_tuple_set(
    #(r_number, r_name, hemis)
    regions.values,
    name='julich_regions'
)

#taskP = nl.add_tuple_set(
    #(task, subject, img)
#    taskParadigm,
#    name='taskParadigmIBC'
#)

tagsN = nl.add_tuple_set(
    #(tag, subject, img)
    norm_tags,
    name='tagsIBC'
)

dataIBC = nl.add_tuple_set(
    #(subject, img, i, j, k)
    df,
    name='dataIBC'
)

ns_docs = df[['subject_id', 'img_id']]
ns_terms = norm_tags[['subject_id', 'img_id', 'tag']]

subjects = nl.add_uniform_probabilistic_choice_over_set(
        pd.DataFrame(df.subject_id.unique(), columns=['subjects']), name='subjects'
)

images = nl.add_uniform_probabilistic_choice_over_set(
        pd.DataFrame(df.img_id.unique(), columns=['subjects']), name='images'
)

terms = nl.add_tuple_set(ns_terms.values, name='terms')
docs = nl.add_uniform_probabilistic_choice_over_set(
        ns_docs.values, name='docs'
)

with nl.scope as e:

    e.ontology_terms[e.cp, e.onto_name] = (
        hasTopConcept[e.uri, e.cp] &
        label[e.uri, e.onto_name]
    )

    e.lower_terms[e.cp, e.term] = (
        e.ontology_terms[e.cp, e.onto_term] &
        (e.term == word_lower[e.onto_term])
    )

    e.filtered_terms[e.cp, e.term] = (
        e.lower_terms[e.cp, e.term] &
        e.terms[..., ..., e.term]
    )

    f_term = nl.query((e.cp, e.t), e.filtered_terms[e.cp, e.t])

sam = ns_terms.join(f_term.as_pandas_dataframe().set_index('t'), on='tag')
sam = sam.fillna('')

#props = sam.groupby('cp').count() * frac_sample
#g = sam.groupby('cp')

#mapping = dict(zip(props.index, props.subject_id))

#ns_doc_folds = pd.DataFrame([])
#for i in range(n_folds):
#    w_sample = pd.concat(
#        group.sample(int(mapping[name]), random_state=i) for name, group in g
#    )
#    w_sample['fold'] = i
#    ns_doc_folds = pd.concat((ns_doc_folds, w_sample))

#g = sam.groupby('tag')

#mapping = dict(zip(props.index, props.subject_id))
#mean_amount = math.ceil(sam.groupby('tag').count().cp.mean())

g = sam.groupby('tag')

#mapping = dict(zip(props.index, props.subject_id))
#cnt = g.count()
#cnt = cnt[(cnt.cp < 1500) & (cnt.cp > 100)]
#valids = set(cnt.reset_index().tag.values)
#mean_amount = math.ceil(sam.groupby('tag').count().cp.mean())

ns_doc_folds = pd.DataFrame([])
for i in range(n_folds):
    w_sample = pd.concat(
        group.sample(g.count().cp.min(), random_state=i) for name, group in g
    )
    w_sample['fold'] = i
    ns_doc_folds = pd.concat((ns_doc_folds, w_sample))

ns_doc_folds = ns_doc_folds[['subject_id', 'img_id', 'fold']]
doc_folds = nl.add_tuple_set(ns_doc_folds, name='doc_folds')

print(f'Running query for region {id_region}')

try:
    with nl.scope as e:

        e.ontology_terms[e.onto_name] = (
            hasTopConcept[e.uri, e.cp] &
            label[e.uri, e.onto_name]
        )

        e.lower_terms[e.term] = (
            e.ontology_terms[e.onto_term] &
            (e.term == word_lower[e.onto_term])
        )

        e.filtered_terms[e.subject, e.image, e.term] = (
            e.lower_terms[e.term] &
            e.terms[e.subject, e.image, e.term]
        )

        (e.jbd @ e.p)[e.i, e.j, e.k, e.subject, e.image] = (
            e.dataIBC[e.subject, e.image, e.i, e.j, e.k] &
            e.julich_brain[e.p, e.i, e.j, e.k, e.region] &
            (e.region == id_region)
        )

        e.img_studies[e.subject, e.image] = e.jbd[..., ..., ..., e.subject, e.image]

        e.img_left_studies[e.subject, e.image] = e.docs[e.subject, e.image] & ~(e.img_studies[e.subject, e.image])


        e.term_prob[e.t, e.fold, e.PROB[e.t, e.fold]] = (
            (
                e.filtered_terms[e.subject, e.image, e.t]
            ) // (
                e.img_studies[e.subject, e.image] &
                e.doc_folds[e.subject, e.image, e.fold] &
                e.docs[e.subject, e.image] &
                e.subjects[e.subject] &
                e.images[e.image]
            )
        )

        e.no_term_prob[e.t, e.fold, e.PROB[e.t, e.fold]] = (
            (
                e.filtered_terms[e.subject, e.image, e.t]
            ) // (
                e.img_left_studies[e.subject, e.image] &
                e.doc_folds[e.subject, e.image, e.fold] &
                e.docs[e.subject, e.image] &
                e.subjects[e.subject] &
                e.images[e.image]
            )
        )

        e.ans[e.term, e.fold, e.region, e.p, e.pn, e.bf] = (
            e.term_prob[e.term, e.fold, e.p] &
            e.no_term_prob[e.term, e.fold, e.pn] &
            (e.bf == (e.p / e.pn)) &
            (e.region == id_region)
        )

        res = nl.query((e.term, e.fold, e.region, e.p, e.pn, e.bf), e.ans[e.term, e.fold, e.region, e.p, e.pn, e.bf])
        mean_bf = res.as_pandas_dataframe()#.groupby('term').mean().drop(['fold'], axis=1).sort_values('bf', ascending=False)

        mean_bf.to_hdf(f'./results/ibc_results_region_{id_region}_min_sample.hdf', key='results')
except Exception as e:
    print(f'Failed on region: {id_region}')
    print(e)

print(f'Total time: {(time.time() - start_time)}')
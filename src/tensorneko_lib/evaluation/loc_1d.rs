#![feature(iter_map_windows)]

extern crate serde_json;

use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs;
use env_logger::Target;
use ndarray::{arr2, Array, Array1, Array2, Axis, concatenate, OwnedRepr, s, stack, Zip};
use ndarray::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use log::{info};


#[derive(Serialize, Deserialize, Debug)]
struct Metadata {
    file: String,
    original: Option<String>,
    split: String,
    n_fakes: usize,
    duration: f32,
    fake_periods: Vec<Vec<f32>>,
    visual_fake_segments: Vec<Vec<f32>>,
    audio_fake_segments: Vec<Vec<f32>>,
    modify_type: String,
    modify_video: bool,
    modify_audio: bool,
    //audio_model: String,
    video_frames: i64,
    audio_frames: i64,
}

impl Metadata {
    fn new(
        file: String,
        original: Option<String>,
        split: String,
        fake_segments: Vec<Vec<f32>>,
        fps: i64,
        visual_fake_segments: Vec<Vec<f32>>,
        audio_fake_segments: Vec<Vec<f32>>,
        //audio_model: String,
        modify_type: String,
        video_frames: i64,
        audio_frames: i64,
    ) -> Metadata {
        Metadata {
            file,
            original,
            split,
            n_fakes: fake_segments.len(),
            duration: video_frames as f32 / fps as f32,
            fake_periods: fake_segments,
            visual_fake_segments,
            audio_fake_segments,
            modify_type: modify_type.clone(),
            modify_video: matches!(modify_type.as_str(), "both-modified" | "visual_modified"),
            modify_audio: matches!(modify_type.as_str(), "both-modified" | "audio_modified"),
            //audio_model: audio_model,
            video_frames,
            audio_frames,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct MetadataFileRecord {
    file: String,
    original: Option<String>,
    split: String,
    fake_segments: Vec<Vec<f32>>,
    visual_fake_segments: Vec<Vec<f32>>,
    audio_fake_segments: Vec<Vec<f32>>,
    modify_type: String,
    video_frames: i64,
    audio_frames: i64,
}

fn convert_metadata_info_to_metadata(metadata_info: MetadataFileRecord, fps: i64) -> Metadata {
    //let audio_model_default = "default_audio_model".to_string();
    Metadata::new(
        metadata_info.file,
        metadata_info.original,
        metadata_info.split,
        metadata_info.fake_segments,
        fps,
        metadata_info.visual_fake_segments,
        metadata_info.audio_fake_segments,
        //metadata_info.audio_model.unwrap_or(audio_model_default),
        metadata_info.modify_type,
        metadata_info.video_frames,
        metadata_info.audio_frames,
    )
}

fn iou_1d(proposal: Array2<f32>, target: &Array2<f32>) -> Array2<f32> {
    let m = proposal.nrows();
    let n = target.nrows();

    let mut ious = Array2::<f32>::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            let proposal_start = proposal[[i, 0]];
            let proposal_end = proposal[[i, 1]];
            let target_start = target[[j, 0]];
            let target_end = target[[j, 1]];

            let inner_begin = proposal_start.max(target_start);
            let inner_end = proposal_end.min(target_end);
            let outer_begin = proposal_start.min(target_start);
            let outer_end = proposal_end.max(target_end);

            let intersection = (inner_end - inner_begin).max(0.0);
            let union = outer_end - outer_begin;
            ious[[i, j]] = intersection / union;
        }
    }

    ious
}

fn calc_ap_curve(is_tp: Array1<bool>, n_labels: f32) -> Array2<f32> {
    let acc_tp = Array1::from_vec(
        is_tp.iter().scan(0.0, |state, &x| {
            if x { *state += 1.0 }
            Some(*state)
        }).collect()
    );

    let precision: Array1<f32> = acc_tp.iter().enumerate().map(|(i, &x)| x / (i as f32 + 1.0)).collect();
    let recall: Array1<f32> = acc_tp / n_labels;
    let binding = stack!(Axis(0), recall.view(), precision.view());
    let binding = binding.t();

    concatenate![
        Axis(0),
        arr2(&[[1., 0.]]).view(),
        binding.slice(s![..;-1, ..])
    ]
}

fn calculate_ap(curve: &Array2<f32>) -> f32 {
    let x = curve.column(0).to_owned();
    let y = curve.column(1).to_owned();

    let y_max = Array1::from(y.iter().scan(None, |state, &x| {
        if state.is_none() || x > state.unwrap() {
            *state = Some(x);
        }

        *state
    }).collect::<Vec<_>>());

    let x_diff: Array1<f32> = x
        .into_iter()
        .map_windows(|[x, y]| (y - x).abs())
        .collect();

    (x_diff * y_max.slice(s![..-1])).sum()
}

fn get_ap_values(
    iou_threshold: f32,
    proposals: &Array2<f32>,
    labels: &Array2<f32>,
    fps: f32,
) -> (Array1<f32>, Array1<bool>) {
    let n_labels = labels.len_of(Axis(0));
    let n_proposals = proposals.len_of(Axis(0));
    let local_proposals = if proposals.shape() != [0] {
        proposals.clone()
    } else {
        proposals.clone()
            .into_shape((0, 3))
            .unwrap()
    };

    let ious = if n_labels > 0 {
        iou_1d(local_proposals.slice(s![.., 1..]).mapv(|x| x / fps), labels)
    } else {
        Array::zeros((n_proposals, 0))
    };

    let confidence = local_proposals.column(0).to_owned();
    let potential_tp = ious.mapv(|x| x > iou_threshold);

    let mut is_tp = Array1::from_elem((n_proposals, ), false);

    for i in 0..n_labels {
        if let Some((index, _)) = potential_tp.column(i).iter().filter(|&&x| x).enumerate().next() {
            is_tp[index] = true;
        }
    };

    (confidence, is_tp)
}

fn calc_ap_scores(
    iou_thresholds: Vec<f32>,
    metadatas: &Vec<Metadata>,
    proposals_map: &Proposals,
) -> Vec<(f32, f32)> {
    iou_thresholds.par_iter().map(|iou| {
        let (values, labels): (Vec<_>, Vec<isize>) = metadatas
            .par_iter()
            .map(|meta| {
                let proposals = &proposals_map.content[&meta.file];
                let rows = meta.fake_periods.len();
                let x: Vec<f32> = meta.fake_periods.iter().flatten().copied().collect();
                let labels = Array2::from_shape_vec((rows, 2), x).unwrap().to_owned();
                let meta_value = get_ap_values(*iou, &proposals.row, &labels, 25.0);

                (meta_value, labels.len_of(Axis(0)) as isize)
            })
            .unzip();

        let n_labels = labels.iter().sum::<isize>() as f32;

        info!("{} completed, n_labels: {}", iou, n_labels);

        let (r, n): (Vec<_>, Vec<_>) = values.into_iter().unzip();
        let confidence = concatenate(
            Axis(0),
            &r.iter()
                .map(|x| x.view())
                .collect::<Vec<_>>(),
        ).unwrap();
        let is_tp = concatenate(
            Axis(0),
            &n.iter()
                .map(|x| x.view())
                .collect::<Vec<_>>(),
        ).unwrap();

        let mut indices: Vec<usize> = (0..confidence.len()).collect();
        indices.sort_by(|&a, &b| confidence[b].partial_cmp(&confidence[a]).unwrap());
        let is_tp = is_tp.select(Axis(0), &indices);
        let curve = calc_ap_curve(is_tp, n_labels);
        let ap = calculate_ap(&curve);

        (*iou, ap)
    }).collect::<Vec<_>>()
}


fn cummax_2d(array: &Array2<f32>) -> Array2<f32> {
    let mut result = array.clone();

    for mut column in result.axis_iter_mut(Axis(1)) {
        let mut cummax = column[0];

        for row in column.iter_mut().skip(1) {
            cummax = cummax.max(*row);
            *row = cummax;
        }
    }

    result
}

fn calc_ar_values(
    n_proposals: &Vec<usize>,
    iou_thresholds: &Vec<f32>,
    proposals: &Array2<f32>,
    labels: &Array2<f32>,
    fps: f32,
) -> ArrayBase<OwnedRepr<usize>, Ix3> {
    let max_proposals = *n_proposals.iter().max().unwrap();

    let mut proposals = proposals.slice(s![..max_proposals, ..]).to_owned();
    if proposals.is_empty() {
        proposals = Array2::zeros((0, 3)).into();
    }

    let n_proposals_clamped = n_proposals.iter().map(|&n| n.min(proposals.nrows())).collect::<Vec<_>>();
    let n_labels = labels.nrows();

    let ious = if n_labels > 0 {
        iou_1d(proposals.slice(s![.., 1..]).mapv(|x| x / fps), labels)
    } else {
        Array::zeros((max_proposals, 0))  // 这里还能再来个短路什么的
    };

    let mut values = Array3::zeros((iou_thresholds.len(), n_proposals_clamped.len(), 2));
    if !proposals.is_empty() {
        let iou_max = cummax_2d(&ious);  // (n_iou, n_labels)
        for (threshold_idx, &threshold) in iou_thresholds.iter().enumerate() {
            for (n_proposals_idx, &n_proposal) in n_proposals_clamped.iter().enumerate() {
                //dbg!(iou_max.row(n_proposal - 1));
                let tp = iou_max.row(n_proposal - 1).iter().filter(|&&iou| iou > threshold).count();
                values[[threshold_idx, n_proposals_idx, 0]] = tp;
                values[[threshold_idx, n_proposals_idx, 1]] = n_labels - tp;
            }
        }
    }
    values
}

fn calc_ar_scores(
    n_proposals: Vec<usize>,
    iou_thresholds: &Vec<f32>,
    metadata: &Vec<Metadata>,
    proposals_map: &Proposals,
) -> Vec<(usize, f32)> {
    let values = metadata.par_iter().map(|meta| {
        let proposals = &proposals_map.content[&meta.file];

        let rows = meta.fake_periods.len();
        let x: Vec<f32> = meta.fake_periods.iter().flatten().copied().collect();
        let labels = Array2::from_shape_vec((rows, 2), x).unwrap().to_owned();

        calc_ar_values(&n_proposals, iou_thresholds, &proposals.row, &labels, 25.0)
    }).collect::<Vec<_>>();

    let values = stack(
        Axis(0),
        &values
            .iter()
            .map(|x| x.view())
            .collect::<Vec<_>>()
    ).unwrap();

    let values_sum = values.sum_axis(Axis(0));
    let tp = values_sum.slice(s![.., .., 0]);
    let f_n = values_sum.slice(s![.., .., 1]);

    let recall = Zip::from(&tp).and(&f_n).map_collect(|&x, &y| {
        let div = x as f32 + y as f32;
        if div == 0. {
            0.
        } else {
            x as f32 / div
        }
    });

    n_proposals.iter().enumerate().map(|(ix, &prop)| {
        (prop, recall.column(ix).mean().unwrap())
    }).collect::<Vec<_>>()
}


#[derive(Deserialize, Debug)]
#[serde(transparent)]
struct ProposalRow {
    #[serde(with = "serde_ndim")]
    pub row: Array2<f32>,
}

#[derive(Deserialize, Debug)]
#[serde(transparent)]
struct Proposals {
    pub content: HashMap<String, ProposalRow>,
}

#[pyfunction]
pub fn ap_1d() {}

#[pyfunction]
pub fn ar_1d() {}
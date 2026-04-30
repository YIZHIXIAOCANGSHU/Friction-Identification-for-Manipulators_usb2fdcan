# Hardware Identification Summary

| Motor | accepted/total | tau_static | tau_c | viscous | inertia | friction RMSE | inertia RMSE | publish status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 01 motor_01 | 0/0 | nan | nan | nan | nan | nan | nan | not_run (piecewise_static_linear_v1) |
| 02 motor_02 | 0/0 | nan | nan | nan | nan | nan | nan | not_run (piecewise_static_linear_v1) |
| 03 motor_03 | 3/3 | 0.450000 | 0.605290 | 0.067810 | 0.062065 | 0.080813 | 0.159067 | published (piecewise_static_linear_v1) |
| 04 motor_04 | 0/0 | nan | nan | nan | nan | nan | nan | not_run (piecewise_static_linear_v1) |
| 05 motor_05 | 0/0 | nan | nan | nan | nan | nan | nan | not_run (piecewise_static_linear_v1) |
| 06 motor_06 | 0/0 | nan | nan | nan | nan | nan | nan | not_run (piecewise_static_linear_v1) |
| 07 motor_07 | 0/0 | nan | nan | nan | nan | nan | nan | not_run (piecewise_static_linear_v1) |

## Motor 01 motor_01

- publish_status: `not_run`
- publish_detail: ``
- model_kind: `piecewise_static_linear_v1`
- source_phases: `-`
- selected_rounds: `-`
- repeat_consistency_score: `nan`

## Motor 02 motor_02

- publish_status: `not_run`
- publish_detail: ``
- model_kind: `piecewise_static_linear_v1`
- source_phases: `-`
- selected_rounds: `-`
- repeat_consistency_score: `nan`

## Motor 03 motor_03

- publish_status: `published`
- publish_detail: `published 3/3 accepted rounds`
- model_kind: `piecewise_static_linear_v1`
- source_phases: `breakaway,low-speed,speed-hold,inertia`
- selected_rounds: `1,2,3`
- repeat_consistency_score: `0.136050`
- group=1, round=1, selected_for_publish=yes, model_kind=piecewise_static_linear_v1, validation_status=accepted, friction_rmse=0.054118, inertia_rmse=0.158000, detail=friction_rmse=0.054118, inertia_rmse=0.158000, inertia_savgol_window=31, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.047377, dynamic_mit_use_for_publish=no
- group=2, round=2, selected_for_publish=yes, model_kind=piecewise_static_linear_v1, validation_status=accepted, friction_rmse=0.142542, inertia_rmse=0.157918, detail=friction_rmse=0.142542, inertia_rmse=0.157918, inertia_savgol_window=21, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.047726, dynamic_mit_use_for_publish=no
- group=3, round=3, selected_for_publish=yes, model_kind=piecewise_static_linear_v1, validation_status=accepted, friction_rmse=0.045780, inertia_rmse=0.161284, detail=friction_rmse=0.045780, inertia_rmse=0.161284, inertia_savgol_window=21, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.078198, dynamic_mit_use_for_publish=no

## Motor 04 motor_04

- publish_status: `not_run`
- publish_detail: ``
- model_kind: `piecewise_static_linear_v1`
- source_phases: `-`
- selected_rounds: `-`
- repeat_consistency_score: `nan`

## Motor 05 motor_05

- publish_status: `not_run`
- publish_detail: ``
- model_kind: `piecewise_static_linear_v1`
- source_phases: `-`
- selected_rounds: `-`
- repeat_consistency_score: `nan`

## Motor 06 motor_06

- publish_status: `not_run`
- publish_detail: ``
- model_kind: `piecewise_static_linear_v1`
- source_phases: `-`
- selected_rounds: `-`
- repeat_consistency_score: `nan`

## Motor 07 motor_07

- publish_status: `not_run`
- publish_detail: ``
- model_kind: `piecewise_static_linear_v1`
- source_phases: `-`
- selected_rounds: `-`
- repeat_consistency_score: `nan`

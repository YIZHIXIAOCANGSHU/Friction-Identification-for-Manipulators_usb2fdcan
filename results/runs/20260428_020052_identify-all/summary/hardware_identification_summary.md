# Hardware Identification Summary

| Motor | accepted/total | tau_static | tau_c | viscous | inertia | friction RMSE | inertia RMSE | publish status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 01 motor_01 | 0/0 | nan | nan | nan | nan | nan | nan | not_run (piecewise_static_linear_v1) |
| 02 motor_02 | 2/3 | 0.675000 | 0.425237 | 0.011246 | 0.018147 | 0.016712 | 0.094244 | published (piecewise_static_linear_v1) |
| 03 motor_03 | 0/0 | nan | nan | nan | nan | nan | nan | not_run (piecewise_static_linear_v1) |
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

- publish_status: `published`
- publish_detail: `published 2/3 accepted rounds`
- model_kind: `piecewise_static_linear_v1`
- source_phases: `breakaway,low-speed,speed-hold,inertia`
- selected_rounds: `2,3`
- repeat_consistency_score: `0.081053`
- group=1, round=1, selected_for_publish=no, model_kind=piecewise_static_linear_v1, validation_status=rejected, friction_rmse=0.040581, inertia_rmse=0.103295, detail=breakaway_scan_limit_reached=both, inertia_savgol_window=81, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.022612, dynamic_mit_use_for_publish=no
- group=2, round=2, selected_for_publish=yes, model_kind=piecewise_static_linear_v1, validation_status=accepted, friction_rmse=0.024468, inertia_rmse=0.081600, detail=friction_rmse=0.024468, inertia_rmse=0.081600, inertia_savgol_window=81, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.024462, dynamic_mit_use_for_publish=no
- group=3, round=3, selected_for_publish=yes, model_kind=piecewise_static_linear_v1, validation_status=accepted, friction_rmse=0.008957, inertia_rmse=0.106888, detail=friction_rmse=0.008957, inertia_rmse=0.106888, inertia_savgol_window=81, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.022367, dynamic_mit_use_for_publish=no

## Motor 03 motor_03

- publish_status: `not_run`
- publish_detail: ``
- model_kind: `piecewise_static_linear_v1`
- source_phases: `-`
- selected_rounds: `-`
- repeat_consistency_score: `nan`

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

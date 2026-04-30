# Hardware Identification Summary

| Motor | accepted/total | tau_static | tau_c | viscous | inertia | friction RMSE | inertia RMSE | publish status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 01 motor_01 | 0/0 | nan | nan | nan | nan | nan | nan | not_run (static_v1) |
| 02 motor_02 | 1/3 | 0.815000 | 0.436553 | 0.011031 | 0.014928 | 0.016821 | 0.182135 | not_published (static_v1) |
| 03 motor_03 | 0/0 | nan | nan | nan | nan | nan | nan | not_run (static_v1) |
| 04 motor_04 | 0/0 | nan | nan | nan | nan | nan | nan | not_run (static_v1) |
| 05 motor_05 | 0/0 | nan | nan | nan | nan | nan | nan | not_run (static_v1) |
| 06 motor_06 | 0/0 | nan | nan | nan | nan | nan | nan | not_run (static_v1) |
| 07 motor_07 | 0/0 | nan | nan | nan | nan | nan | nan | not_run (static_v1) |

## Motor 01 motor_01

- publish_status: `not_run`
- publish_detail: ``
- model_kind: `static_v1`
- source_phases: `-`
- selected_rounds: `-`
- repeat_consistency_score: `nan`

## Motor 02 motor_02

- publish_status: `not_published`
- publish_detail: `accepted_round_count=1, required=2; previous published model retained for reference`
- model_kind: `static_v1`
- source_phases: `breakaway,speed-hold,inertia`
- selected_rounds: `3`
- repeat_consistency_score: `0.000000`
- group=1, round=1, selected_for_publish=no, model_kind=static_v1, validation_status=rejected, friction_rmse=0.027473, inertia_rmse=0.207976, detail=inertia_rmse=0.207976>0.200000, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.369223, dynamic_mit_use_for_publish=no
- group=2, round=2, selected_for_publish=no, model_kind=static_v1, validation_status=rejected, friction_rmse=0.032467, inertia_rmse=0.202184, detail=inertia_rmse=0.202184>0.200000, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.376546, dynamic_mit_use_for_publish=no
- group=3, round=3, selected_for_publish=yes, model_kind=static_v1, validation_status=accepted, friction_rmse=0.016821, inertia_rmse=0.182135, detail=friction_rmse=0.016821, inertia_rmse=0.182135, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.374196, dynamic_mit_use_for_publish=no

## Motor 03 motor_03

- publish_status: `not_run`
- publish_detail: ``
- model_kind: `static_v1`
- source_phases: `-`
- selected_rounds: `-`
- repeat_consistency_score: `nan`

## Motor 04 motor_04

- publish_status: `not_run`
- publish_detail: ``
- model_kind: `static_v1`
- source_phases: `-`
- selected_rounds: `-`
- repeat_consistency_score: `nan`

## Motor 05 motor_05

- publish_status: `not_run`
- publish_detail: ``
- model_kind: `static_v1`
- source_phases: `-`
- selected_rounds: `-`
- repeat_consistency_score: `nan`

## Motor 06 motor_06

- publish_status: `not_run`
- publish_detail: ``
- model_kind: `static_v1`
- source_phases: `-`
- selected_rounds: `-`
- repeat_consistency_score: `nan`

## Motor 07 motor_07

- publish_status: `not_run`
- publish_detail: ``
- model_kind: `static_v1`
- source_phases: `-`
- selected_rounds: `-`
- repeat_consistency_score: `nan`

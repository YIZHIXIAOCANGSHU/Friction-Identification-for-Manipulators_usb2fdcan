# Hardware Identification Summary

| Motor | accepted/total | tau_static | tau_c | viscous | inertia | friction RMSE | inertia RMSE | publish status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 01 motor_01 | 0/3 | 0.945000 | 0.183896 | 0.026801 | 0.020767 | 0.530138 | 0.673816 | retained_previous_model (joint_static_dynamic_v1) |
| 02 motor_02 | 0/3 | 0.755000 | 0.055718 | 0.008716 | 0.014107 | 0.500669 | 0.638772 | retained_previous_model (joint_static_dynamic_v1) |
| 03 motor_03 | 0/3 | 0.565000 | 0.226645 | 0.118985 | 0.056848 | 0.172470 | 0.355518 | retained_previous_model (joint_static_dynamic_v1) |
| 04 motor_04 | 0/3 | 0.535000 | 0.225699 | 0.134369 | 0.055393 | 0.166474 | 0.382356 | retained_previous_model (joint_static_dynamic_v1) |
| 05 motor_05 | 3/3 | 0.190000 | 0.095361 | 0.002100 | 0.000993 | 0.040702 | 0.089068 | published (joint_static_dynamic_v1) |
| 06 motor_06 | 3/3 | 0.180000 | 0.111298 | 0.003405 | 0.001431 | 0.044754 | 0.094508 | published (joint_static_dynamic_v1) |
| 07 motor_07 | 3/3 | 0.200000 | 0.111388 | 0.000708 | 0.000980 | 0.052601 | 0.100630 | published (joint_static_dynamic_v1) |

## Motor 01 motor_01

- publish_status: `retained_previous_model`
- publish_detail: `kept previous model; accepted_round_count=0, required=2`
- model_kind: `joint_static_dynamic_v1`
- source_phases: `breakaway,speed-hold,inertia,dynamic-mit`
- selected_rounds: `-`
- repeat_consistency_score: `0.227367`
- group=1, round=1, selected_for_publish=no, model_kind=joint_static_dynamic_v1, validation_status=rejected, friction_rmse=0.505883, inertia_rmse=0.653135, detail=friction_rmse=0.505883>0.150000; inertia_rmse=0.653135>0.200000, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.376160, dynamic_mit_use_for_publish=no
- group=2, round=8, selected_for_publish=no, model_kind=joint_static_dynamic_v1, validation_status=rejected, friction_rmse=0.430089, inertia_rmse=0.568992, detail=friction_rmse=0.430089>0.150000; inertia_rmse=0.568992>0.200000, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.391078, dynamic_mit_use_for_publish=no
- group=3, round=15, selected_for_publish=no, model_kind=joint_static_dynamic_v1, validation_status=rejected, friction_rmse=0.654442, inertia_rmse=0.799321, detail=friction_rmse=0.654442>0.150000; inertia_rmse=0.799321>0.200000, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.381220, dynamic_mit_use_for_publish=no

## Motor 02 motor_02

- publish_status: `retained_previous_model`
- publish_detail: `kept previous model; accepted_round_count=0, required=2`
- model_kind: `joint_static_dynamic_v1`
- source_phases: `breakaway,speed-hold,inertia,dynamic-mit`
- selected_rounds: `-`
- repeat_consistency_score: `1.294534`
- group=1, round=2, selected_for_publish=no, model_kind=joint_static_dynamic_v1, validation_status=rejected, friction_rmse=0.532209, inertia_rmse=0.681344, detail=friction_rmse=0.532209>0.150000; inertia_rmse=0.681344>0.200000, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.380191, dynamic_mit_use_for_publish=no
- group=2, round=9, selected_for_publish=no, model_kind=joint_static_dynamic_v1, validation_status=rejected, friction_rmse=0.397526, inertia_rmse=0.526026, detail=friction_rmse=0.397526>0.150000; inertia_rmse=0.526026>0.200000, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.368851, dynamic_mit_use_for_publish=no
- group=3, round=16, selected_for_publish=no, model_kind=joint_static_dynamic_v1, validation_status=rejected, friction_rmse=0.572271, inertia_rmse=0.708946, detail=friction_rmse=0.572271>0.150000; inertia_rmse=0.708946>0.200000, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.377455, dynamic_mit_use_for_publish=no

## Motor 03 motor_03

- publish_status: `retained_previous_model`
- publish_detail: `kept previous model; accepted_round_count=0, required=2`
- model_kind: `joint_static_dynamic_v1`
- source_phases: `breakaway,speed-hold,inertia,dynamic-mit`
- selected_rounds: `-`
- repeat_consistency_score: `0.051680`
- group=1, round=3, selected_for_publish=no, model_kind=joint_static_dynamic_v1, validation_status=rejected, friction_rmse=0.140068, inertia_rmse=0.305914, detail=inertia_rmse=0.305914>0.200000, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.260862, dynamic_mit_use_for_publish=no
- group=2, round=10, selected_for_publish=no, model_kind=joint_static_dynamic_v1, validation_status=rejected, friction_rmse=0.182783, inertia_rmse=0.401229, detail=friction_rmse=0.182783>0.150000; inertia_rmse=0.401229>0.200000, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.405387, dynamic_mit_use_for_publish=no
- group=3, round=17, selected_for_publish=no, model_kind=joint_static_dynamic_v1, validation_status=rejected, friction_rmse=0.194559, inertia_rmse=0.359409, detail=friction_rmse=0.194559>0.150000; inertia_rmse=0.359409>0.200000, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.291946, dynamic_mit_use_for_publish=no

## Motor 04 motor_04

- publish_status: `retained_previous_model`
- publish_detail: `kept previous model; accepted_round_count=0, required=2`
- model_kind: `joint_static_dynamic_v1`
- source_phases: `breakaway,speed-hold,inertia,dynamic-mit`
- selected_rounds: `-`
- repeat_consistency_score: `0.194448`
- group=1, round=4, selected_for_publish=no, model_kind=joint_static_dynamic_v1, validation_status=rejected, friction_rmse=0.250232, inertia_rmse=0.401597, detail=friction_rmse=0.250232>0.150000; inertia_rmse=0.401597>0.200000, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.314938, dynamic_mit_use_for_publish=no
- group=2, round=11, selected_for_publish=no, model_kind=joint_static_dynamic_v1, validation_status=rejected, friction_rmse=0.165411, inertia_rmse=0.376237, detail=friction_rmse=0.165411>0.150000; inertia_rmse=0.376237>0.200000, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.400599, dynamic_mit_use_for_publish=no
- group=3, round=18, selected_for_publish=no, model_kind=joint_static_dynamic_v1, validation_status=rejected, friction_rmse=0.083780, inertia_rmse=0.369235, detail=inertia_rmse=0.369235>0.200000, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.406894, dynamic_mit_use_for_publish=no

## Motor 05 motor_05

- publish_status: `published`
- publish_detail: `published 3/3 accepted rounds`
- model_kind: `joint_static_dynamic_v1`
- source_phases: `breakaway,speed-hold,inertia,dynamic-mit`
- selected_rounds: `1,2,3`
- repeat_consistency_score: `1.312707`
- group=1, round=5, selected_for_publish=yes, model_kind=joint_static_dynamic_v1, validation_status=accepted, friction_rmse=0.047670, inertia_rmse=0.085798, detail=joint_friction_rmse=0.047670, joint_inertia_rmse=0.085798, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.124677, dynamic_mit_use_for_publish=no
- group=2, round=12, selected_for_publish=yes, model_kind=joint_static_dynamic_v1, validation_status=accepted, friction_rmse=0.020127, inertia_rmse=0.076123, detail=joint_friction_rmse=0.020127, joint_inertia_rmse=0.076123, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.123050, dynamic_mit_use_for_publish=no
- group=3, round=19, selected_for_publish=yes, model_kind=joint_static_dynamic_v1, validation_status=accepted, friction_rmse=0.054308, inertia_rmse=0.105284, detail=joint_friction_rmse=0.054308, joint_inertia_rmse=0.105284, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.137507, dynamic_mit_use_for_publish=no

## Motor 06 motor_06

- publish_status: `published`
- publish_detail: `published 3/3 accepted rounds`
- model_kind: `joint_static_dynamic_v1`
- source_phases: `breakaway,speed-hold,inertia,dynamic-mit`
- selected_rounds: `1,2,3`
- repeat_consistency_score: `0.418277`
- group=1, round=6, selected_for_publish=yes, model_kind=joint_static_dynamic_v1, validation_status=accepted, friction_rmse=0.049131, inertia_rmse=0.096519, detail=joint_friction_rmse=0.049131, joint_inertia_rmse=0.096519, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.135410, dynamic_mit_use_for_publish=no
- group=2, round=13, selected_for_publish=yes, model_kind=joint_static_dynamic_v1, validation_status=accepted, friction_rmse=0.065613, inertia_rmse=0.110544, detail=joint_friction_rmse=0.065613, joint_inertia_rmse=0.110544, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.130085, dynamic_mit_use_for_publish=no
- group=3, round=20, selected_for_publish=yes, model_kind=joint_static_dynamic_v1, validation_status=accepted, friction_rmse=0.019518, inertia_rmse=0.076461, detail=joint_friction_rmse=0.019518, joint_inertia_rmse=0.076461, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.118999, dynamic_mit_use_for_publish=no

## Motor 07 motor_07

- publish_status: `published`
- publish_detail: `published 3/3 accepted rounds`
- model_kind: `joint_static_dynamic_v1`
- source_phases: `breakaway,speed-hold,inertia,dynamic-mit`
- selected_rounds: `1,2,3`
- repeat_consistency_score: `0.839840`
- group=1, round=7, selected_for_publish=yes, model_kind=joint_static_dynamic_v1, validation_status=accepted, friction_rmse=0.047785, inertia_rmse=0.103436, detail=joint_friction_rmse=0.047785, joint_inertia_rmse=0.103436, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.134491, dynamic_mit_use_for_publish=no
- group=2, round=14, selected_for_publish=yes, model_kind=joint_static_dynamic_v1, validation_status=accepted, friction_rmse=0.056058, inertia_rmse=0.096503, detail=joint_friction_rmse=0.056058, joint_inertia_rmse=0.096503, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.121411, dynamic_mit_use_for_publish=no
- group=3, round=21, selected_for_publish=yes, model_kind=joint_static_dynamic_v1, validation_status=accepted, friction_rmse=0.053960, inertia_rmse=0.101952, detail=joint_friction_rmse=0.053960, joint_inertia_rmse=0.101952, dynamic_mit_status=ok, dynamic_mit_valid_rmse=0.158375, dynamic_mit_use_for_publish=no

# Hardware Identification Summary

| Motor | accepted/total | tau_static | tau_c | viscous | inertia | friction RMSE | inertia RMSE | publish status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 01 motor_01 | 3/3 | 0.790000 | 0.492472 | 0.047701 | 0.021405 | 0.036945 | 0.142197 | published |
| 02 motor_02 | 2/3 | 0.660000 | 0.440561 | 0.006819 | 0.017760 | 0.019120 | 0.178715 | published |
| 03 motor_03 | 0/3 | 0.600000 | 0.544640 | 0.091720 | 0.060860 | nan | 0.152401 | retained_previous_model |
| 04 motor_04 | 0/3 | 0.600000 | 0.619013 | 0.079285 | 0.059799 | nan | 0.209758 | retained_previous_model |
| 05 motor_05 | 3/3 | 0.145000 | 0.100871 | 0.006976 | 0.002120 | 0.005621 | 0.028777 | published |
| 06 motor_06 | 3/3 | 0.170000 | 0.120925 | 0.008895 | 0.001198 | 0.006963 | 0.027854 | published |
| 07 motor_07 | 1/3 | 0.235000 | 0.126391 | 0.006206 | 0.001660 | 0.007727 | 0.042981 | retained_previous_model |

## Motor 01 motor_01

- publish_status: `published`
- publish_detail: `published 3/3 accepted rounds`
- selected_rounds: `1,2,3`
- repeat_consistency_score: `0.098258`
- group=1, round=1, selected_for_publish=yes, validation_status=accepted, friction_rmse=0.035795, inertia_rmse=0.151814, detail=friction_rmse=0.035795, inertia_rmse=0.151814
- group=2, round=8, selected_for_publish=yes, validation_status=accepted, friction_rmse=0.018974, inertia_rmse=0.143270, detail=friction_rmse=0.018974, inertia_rmse=0.143270
- group=3, round=15, selected_for_publish=yes, validation_status=accepted, friction_rmse=0.056066, inertia_rmse=0.131506, detail=friction_rmse=0.056066, inertia_rmse=0.131506

## Motor 02 motor_02

- publish_status: `published`
- publish_detail: `published 2/3 accepted rounds`
- selected_rounds: `2,3`
- repeat_consistency_score: `0.464618`
- group=1, round=2, selected_for_publish=no, validation_status=rejected, friction_rmse=0.015920, inertia_rmse=0.225676, detail=inertia_rmse=0.225676>0.200000
- group=2, round=9, selected_for_publish=yes, validation_status=accepted, friction_rmse=0.016307, inertia_rmse=0.158834, detail=friction_rmse=0.016307, inertia_rmse=0.158834
- group=3, round=16, selected_for_publish=yes, validation_status=accepted, friction_rmse=0.021932, inertia_rmse=0.198595, detail=friction_rmse=0.021932, inertia_rmse=0.198595

## Motor 03 motor_03

- publish_status: `retained_previous_model`
- publish_detail: `kept previous model; accepted_round_count=0, required=2`
- selected_rounds: `-`
- repeat_consistency_score: `0.216217`
- group=1, round=3, selected_for_publish=no, validation_status=rejected, friction_rmse=nan, inertia_rmse=0.146708, detail=accepted_valid_platform_count=0<2; friction_rmse=nan>0.150000
- group=2, round=10, selected_for_publish=no, validation_status=rejected, friction_rmse=nan, inertia_rmse=0.152202, detail=accepted_valid_platform_count=0<2; friction_rmse=nan>0.150000
- group=3, round=17, selected_for_publish=no, validation_status=rejected, friction_rmse=nan, inertia_rmse=0.158293, detail=accepted_valid_platform_count=0<2; friction_rmse=nan>0.150000

## Motor 04 motor_04

- publish_status: `retained_previous_model`
- publish_detail: `kept previous model; accepted_round_count=0, required=2`
- selected_rounds: `-`
- repeat_consistency_score: `0.128269`
- group=1, round=4, selected_for_publish=no, validation_status=rejected, friction_rmse=nan, inertia_rmse=0.222423, detail=accepted_valid_platform_count=0<2; friction_rmse=nan>0.150000; inertia_rmse=0.222423>0.200000
- group=2, round=11, selected_for_publish=no, validation_status=rejected, friction_rmse=nan, inertia_rmse=0.223470, detail=accepted_valid_platform_count=0<2; friction_rmse=nan>0.150000; inertia_rmse=0.223470>0.200000
- group=3, round=18, selected_for_publish=no, validation_status=rejected, friction_rmse=nan, inertia_rmse=0.183383, detail=accepted_valid_platform_count=0<2; friction_rmse=nan>0.150000

## Motor 05 motor_05

- publish_status: `published`
- publish_detail: `published 3/3 accepted rounds`
- selected_rounds: `1,2,3`
- repeat_consistency_score: `0.212138`
- group=1, round=5, selected_for_publish=yes, validation_status=accepted, friction_rmse=0.001821, inertia_rmse=0.027415, detail=friction_rmse=0.001821, inertia_rmse=0.027415
- group=2, round=12, selected_for_publish=yes, validation_status=accepted, friction_rmse=0.003095, inertia_rmse=0.028852, detail=friction_rmse=0.003095, inertia_rmse=0.028852
- group=3, round=19, selected_for_publish=yes, validation_status=accepted, friction_rmse=0.011947, inertia_rmse=0.030063, detail=friction_rmse=0.011947, inertia_rmse=0.030063

## Motor 06 motor_06

- publish_status: `published`
- publish_detail: `published 3/3 accepted rounds`
- selected_rounds: `1,2,3`
- repeat_consistency_score: `0.132262`
- group=1, round=6, selected_for_publish=yes, validation_status=accepted, friction_rmse=0.005995, inertia_rmse=0.028186, detail=friction_rmse=0.005995, inertia_rmse=0.028186
- group=2, round=13, selected_for_publish=yes, validation_status=accepted, friction_rmse=0.006213, inertia_rmse=0.030179, detail=friction_rmse=0.006213, inertia_rmse=0.030179
- group=3, round=20, selected_for_publish=yes, validation_status=accepted, friction_rmse=0.008681, inertia_rmse=0.025196, detail=friction_rmse=0.008681, inertia_rmse=0.025196

## Motor 07 motor_07

- publish_status: `retained_previous_model`
- publish_detail: `kept previous model; accepted_round_count=1, required=2`
- selected_rounds: `3`
- repeat_consistency_score: `0.000000`
- group=1, round=7, selected_for_publish=no, validation_status=rejected, friction_rmse=0.011902, inertia_rmse=0.038077, detail=invalid_inertia=-0.000110
- group=2, round=14, selected_for_publish=no, validation_status=rejected, friction_rmse=0.012466, inertia_rmse=0.039838, detail=invalid_inertia=-0.000952
- group=3, round=21, selected_for_publish=yes, validation_status=accepted, friction_rmse=0.007727, inertia_rmse=0.042981, detail=friction_rmse=0.007727, inertia_rmse=0.042981

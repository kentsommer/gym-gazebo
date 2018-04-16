SCRIPT_PATH="$(dirname "$(readlink -f "$0")")"

##########
## Main ##
##########
export GAZEBO_MODEL_PATH=$SCRIPT_PATH/gazeboschool-gym/gazeboschool/envs/assets/models

###############
## Turtlebot ##
###############
export GYM_GAZEBO_WORLD_CIRCUIT2=$SCRIPT_PATH/gazeboschool-gym/gazeboschool/envs/assets/worlds/circuit2.world

export GYM_GAZEBO_WORLD_OFFICE=$SCRIPT_PATH/gazeboschool-gym/gazeboschool/envs/assets/worlds/office.world

export GYM_GAZEBO_WORLD_TRAIN=$SCRIPT_PATH/gazeboschool-gym/gazeboschool/envs/assets/worlds/train.world

export GYM_GAZEBO_WORLD_TEST1=$SCRIPT_PATH/gazeboschool-gym/gazeboschool/envs/assets/worlds/test1.world

export GYM_GAZEBO_WORLD_TEST2=$SCRIPT_PATH/gazeboschool-gym/gazeboschool/envs/assets/worlds/test2.world
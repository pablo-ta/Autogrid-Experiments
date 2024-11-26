import grid2op
import os

env_name = "l2rpn_icaps_2021_large"  # or any other...
env = grid2op.make(env_name)

# retrieve the names of the chronics:
full_path_data = env.chronics_handler.subpaths
chron_names = [os.path.split(el)[-1] for el in full_path_data]
print(len(chron_names))

#total chronics: 2952
#steps per chronic: 8062
#total steps: 23_799_024
# splitting into training / eval, 150 chronics to the val set
nm_env_train, m_env_val = env.train_val_split_random(pct_val=5.11,add_for_val="val",add_for_train="train")
print(nm_env_train)
print(m_env_val)
env_train = grid2op.make(env_name + "_train")
env_val = grid2op.make(env_name + "_val")


print("TRAINING")

full_path_data = env_train.chronics_handler.subpaths
chron_names = [os.path.split(el)[-1] for el in full_path_data]
print(len(chron_names))

print("VALIDATION")

full_path_data = env_val.chronics_handler.subpaths
chron_names = [os.path.split(el)[-1] for el in full_path_data]
print(len(chron_names))
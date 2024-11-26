import os
import re
import json
import matplotlib.pyplot as plt

import pandas as pd

rootdir_glob = os.fsencode("./**/performance.txt")
"""
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    if filename.endswith(".asm") or filename.endswith(".py"):
        # print(os.path.join(directory, filename))
        continue
    else:
        continue"""

from glob import iglob

plots_folder = "./plots"
chronics_plot_folder = "./plots/chronics"
if not os.path.exists(plots_folder):
    os.mkdir(plots_folder)
if not os.path.exists(chronics_plot_folder):
    os.mkdir(chronics_plot_folder)


# This will return absolute paths
file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f)]
#print(file_list)
mean_score = {}
mean_timesteps={}
mean_score_timesteps={}
data = {}
data_each_chronic = {}
for file_name in file_list:
    #print(os.fsdecode(file_name))
    m = re.search('agents_(.+?)\\\\(.+?)\\\\', os.fsdecode(file_name))
    if m:
        reward = m.group(1).replace("Reward","")
        #print(reward)
        agent = m.group(2)
        agent = agent.replace("_discrete_singleaction","_S")
        agent = agent.replace("_singleaction","_S")
        agent = agent.replace("_Box","_B")
        agent = agent.replace("_box","_B")
        agent = agent.replace("_discrete","_D")
        agent = agent.replace("DoNothing","D_N")
        agent = agent.replace("Do_Nothing","S_R")
        #print(agent)
        if not data.get(reward,False):
            mean_score[reward]={}
            mean_timesteps[reward]={}
            mean_score_timesteps[reward]={}
            data[reward]={}
        if not data.get(reward).get(agent,False):
            mean_score[reward][agent]={}
            mean_timesteps[reward][agent]={}
            mean_score_timesteps[reward][agent]={}
            data[reward][agent]={}
        with open(file_name) as file:
            total_score=0
            total_timesteps=0
            total_chronics=0
            for line in file:
                #print(line.rstrip())
                m = re.search('at:\s(.+?)\ttotal score:\s(.+?)\ttime steps:\s(.+?)/', os.fsdecode(line.rstrip()))
                if m:
                    chronic = m.group(1)
                    score = float(m.group(2))
                    timesteps = int(m.group(3))
                    data[reward][agent][chronic] = {
                        "score":score,
                        "timesteps":timesteps
                    }

                    if not data_each_chronic.get(chronic, False):
                        data_each_chronic[chronic] = {"score":{},"timesteps":{},"score_timesteps":{}}

                    if not data_each_chronic.get(chronic).get("score").get(reward, False):
                        data_each_chronic[chronic]["score"][reward] = {}
                        data_each_chronic[chronic]["timesteps"][reward] = {}
                        data_each_chronic[chronic]["score_timesteps"][reward] = {}
                    if not data_each_chronic.get(chronic).get("score").get(reward).get(agent, False):
                        data_each_chronic[chronic]["score"][reward][agent] = {}
                        data_each_chronic[chronic]["timesteps"][reward][agent] = {}
                        data_each_chronic[chronic]["score_timesteps"][reward][agent] = {}

                    data_each_chronic[chronic]["score"][reward][agent] = score
                    data_each_chronic[chronic]["timesteps"][reward][agent] = timesteps
                    data_each_chronic[chronic]["score_timesteps"][reward][agent] = score/timesteps

                    total_score+=float(score)
                    total_timesteps+=int(timesteps)
                    total_chronics+=1

            mean_score[reward][agent] = round(total_score/total_chronics,2)

            mean_timesteps[reward][agent] = round(total_timesteps/total_chronics,2)

            mean_score_timesteps[reward][agent] = mean_score[reward][agent] / mean_timesteps[reward][agent]

# Graph MEAN SCORE
print("SCORE")
print(json.dumps(mean_score, indent=4))
print("timesteps")
print(json.dumps(mean_timesteps, indent=4))
print("score_timesteps")
print(json.dumps(mean_score_timesteps, indent=4))
df = pd.DataFrame(mean_score)
df = df.drop("S_R",axis="index")
df["average"] = df.mean(numeric_only=True,axis=1)
df.loc["average"] = df.mean()
mean_score_latex_table = df.round(2).to_latex(header="mean score")
with open("plots/mean_score_latex_table.tex","w") as latex_file:
    latex_file.write(mean_score_latex_table)

print (df)
plot = df.plot.bar(rot=-15)
plot.set_xlabel("SB3 Agents")
plot.set_ylabel("Mean Operation Cost")
plot.set_title("Mean Operation Cost by agent and trained reward")
#plt.ylim((0, 100000))
plt.show()
fig = plot.get_figure()
fig.savefig(os.path.join(plots_folder,F'Mean_score.png'))
plt.close("all")

# Graph MEAN timesteps
df = pd.DataFrame(mean_timesteps)
df = df.drop("S_R",axis="index")
df["average"] = df.mean(numeric_only=True,axis=1)
df.loc["average"] = df.mean()
mean_timesteps_latex_table = df.round(2).to_latex(header="mean timesteps")
with open("plots/mean_timesteps_latex_table.tex","w") as latex_file:
    latex_file.write(mean_timesteps_latex_table)

print (df)

plot = df.plot.bar(rot=-15)
plot.set_xlabel("SB3 Agents")
plot.set_ylabel("Mean timesteps")
plot.set_title("Mean timesteps by agent and trained reward")
plt.show()
fig = plot.get_figure()
fig.savefig(os.path.join(plots_folder,F'Mean_timesteps.png'))
plt.close("all")


# Graph MEAN SCORE/timesteps
df = pd.DataFrame(mean_score_timesteps)
df = df.drop("S_R",axis="index")
df["average"] = df.mean(numeric_only=True,axis=1)
df.loc["average"] = df.mean()
mean_timesteps_latex_table = df.round(2).to_latex(header="mean score/timesteps")
with open("plots/mean_scoretimesteps_latex_table.tex","w") as latex_file:
    latex_file.write(mean_timesteps_latex_table)

print (df)
plot = df.plot.bar(rot=-15)
plot.set_xlabel("SB3 Agents")
plot.set_ylabel("Mean Operation Cost/timesteps")
plot.set_title("Mean Operation Cost/timesteps by agent and trained reward")
plt.show()
fig = plot.get_figure()
fig.savefig(os.path.join(plots_folder,F'Mean_scoretimesteps.png'))
plt.close("all")

for chronic in data_each_chronic:
    print(F"Ploting chronic: {chronic}")

    #SCORES
    scores = data_each_chronic.get(chronic).get("score")
    df = pd.DataFrame(scores)
    df = df.astype(float)
    #print(df)

    plot = df.plot.bar(rot=-15)
    plot.set_xlabel("SB3 Agents")
    plot.set_ylabel("Score")
    plot.set_title(F"{chronic} Score by agent and trained reward")
    fig = plot.get_figure()
    fig.savefig(os.path.join(chronics_plot_folder,F'{chronic}_score.png'))
    plt.close("all")
    #plt.show()

    #TIMESTEPS
    timesteps = data_each_chronic.get(chronic).get("timesteps")
    df = pd.DataFrame(timesteps)
    df = df.astype(float)
    #print(df)
    plot = df.plot.bar(rot=-15)
    plt.ylim((0, 8062))
    plot.set_xlabel("SB3 Agents")
    plot.set_ylabel("Timesteps")
    plot.set_title(F"{chronic} Timesteps by agent and trained reward")
    fig = plot.get_figure()
    fig.savefig(os.path.join(chronics_plot_folder,F'{chronic}_timesteps.png'))
    plt.close("all")
    #plt.show()


    #SCORE OVER TIMESTEPS
    score_timesteps = data_each_chronic.get(chronic).get("score_timesteps")
    df = pd.DataFrame(score_timesteps)
    df = df.astype(float)
    #print(df)
    plot = df.plot.bar(rot=-15)
    #plt.ylim((0, 8062))
    plot.set_xlabel("SB3 Agents")
    plot.set_ylabel("Score over timesteps")
    plot.set_title(F"{chronic} Score over timesteps by agent and trained reward")
    fig = plot.get_figure()
    fig.savefig(os.path.join(chronics_plot_folder,F'{chronic}_score_timesteps.png'))
    plt.close("all")
    #plt.show()
